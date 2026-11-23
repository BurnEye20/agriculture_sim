"""
ROS 2 Bridge for Ground Station Backend
=======================================
Bidirectional ROS 2 bridge between the Ground Station web UI, SQLite database,
GStreamer 1.24 pipelines, and the live Gazebo / physical rover ROS 2 graph.

Key features:
  - Explicit robot operating states: running_mission | returning_to_base |
    manual_mode | at_base | estop | standby.
  - Automatic run completion when returning to base (-0.44, -1.75m) via either
    Auto-RTB or manual driving, so starting the next mission always creates a
    fresh run (#2, #3, ...) in SQLite instead of appending to Run #1.
  - Synchronized JPEG frame saving from /rover/yolo_detections (3-of-4 frame
    confirmed foreground detections) with per-panel spatial deduplication.
  - 10 Hz teleop & E-STOP /cmd_vel publisher with 0.55 s dead-man watchdog.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone
import json
import math
import os
import threading
import time
from typing import Any, TYPE_CHECKING

import cv2
import numpy as np

import database as db

if TYPE_CHECKING:
    from gstreamer_streamer import GStreamerManager

BASE_X = -0.4400
BASE_Y = -1.7500


class RosBridgeManager:
    def __init__(self, gst_manager: "GStreamerManager | None" = None):
        self.gst_manager = gst_manager
        self.lock = threading.Lock()
        self.ros_connected = False
        self.ros_available = False
        self.estop_active = False
        self.auto_align_mast = True
        self.robot_mode = "standby"
        self.status_detail = "Waiting for ROS 2 / Gazebo"
        self.active_run_id: int | None = None

        # Return-to-Base lifecycle tracking
        self._has_left_base = False
        self._returned_to_base = False

        existing_active = db.get_active_run()
        if existing_active is not None:
            self.active_run_id = int(existing_active["id"])
            if float(existing_active.get("distance_m", 0.0)) > 1.8:
                self._has_left_base = True

        self.telemetry: dict[str, Any] = {
            "ros_connected": False,
            "mission_state": "standby",
            "robot_mode": "standby",
            "status_detail": "Waiting for ROS 2 / Gazebo",
            "estop_active": False,
            "x_m": 0.0,
            "y_m": 0.0,
            "yaw_rad": 0.0,
            "yaw_deg": 0.0,
            "linear_vel": 0.0,
            "angular_vel": 0.0,
            "raw_odom_x": 0.0,
            "raw_odom_y": 0.0,
            "drift_m": 0.0,
            "mast_height_m": 0.0,
            "mast_target_m": 0.0,
            "auto_align_mast": True,
            "last_aruco_id": None,
            "aruco_dist_m": 0.0,
            "battery_pct": 100.0,
            "slope_deg": 2.0,
            "active_run_id": self.active_run_id,
            "timestamp": time.time(),
        }
        if self.gst_manager is not None:
            self.gst_manager.set_telemetry_ref(self.telemetry)

        # Teleop continuous 10 Hz publisher state + dead-man watchdog
        self._teleop_vx = 0.0
        self._teleop_wz = 0.0
        self._teleop_active = False
        self._teleop_expire_ts = 0.0
        self._last_cmd_pub_ts = 0.0

        # Live DB recording throttles & per-panel spatial deduplication
        self._last_telem_log_pose: tuple[float, float] | None = None
        self._last_telem_log_ts = 0.0
        self._last_aruco_logged: int | None = None
        # Maps row_index -> list of {"y": float, "cls": str, "conf": float, "det_id": int}
        self._captured_panels: dict[int, list[dict[str, Any]]] = {}

        self._node = None
        self._cmd_vel_pub = None
        self._mast_pub = None
        self._auto_align_pub = None
        self._mission_cmd_pub = None
        self._rclpy = None
        self._last_ros_msg_ts = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self._init_ros_on_main_thread()
        if self._node is not None:
            self._thread = threading.Thread(target=self._spin_loop, daemon=True)
            self._thread.start()

    def _start_new_trial_run(self, reason: str = "") -> int:
        """Create a brand-new SQLite run (#1, #2, #3...) and reset per-run trackers."""
        count = len(db.list_runs()) + 1
        run = db.create_run(
            f"Gazebo Trial #{count:02d}",
            reason or "Live ROS 2 / Gazebo greenhouse inspection trial.",
        )
        run_id = int(run["id"])
        with self.lock:
            self.active_run_id = run_id
            self.telemetry["active_run_id"] = run_id
            self._has_left_base = False
            self._returned_to_base = False
            self._last_telem_log_pose = None
            self._last_aruco_logged = None
            self._captured_panels.clear()
        return run_id

    def _archive_active_run_at_base(self, reason: str):
        """Complete & archive current run when rover returns to initial point (-0.44, -1.75)."""
        with self.lock:
            run_id = self.active_run_id
            if run_id is None:
                self._returned_to_base = True
                return
            self.active_run_id = None
            self.telemetry["active_run_id"] = None
            self._has_left_base = False
            self._returned_to_base = True
            self._last_telem_log_pose = None
            self._last_aruco_logged = None
            self._captured_panels.clear()
        try:
            db.add_event(run_id, "info", "mission", reason)
            db.complete_run(run_id, "completed")
        except Exception:
            pass

    def _mark_ros_alive(self, allow_auto_create: bool = False) -> int | None:
        now = time.time()
        should_create = False
        with self.lock:
            self._last_ros_msg_ts = now
            self.ros_connected = True
            self.telemetry["ros_connected"] = True
            if allow_auto_create and self.active_run_id is None and not self._returned_to_base:
                should_create = True
        if should_create:
            return self._start_new_trial_run("Auto-started upon live Gazebo mission departure.")
        return self.active_run_id

    def _init_ros_on_main_thread(self):
        if not os.environ.get("ROS_DISTRO") or not os.environ.get("AMENT_PREFIX_PATH"):
            self.ros_available = False
            self.ros_connected = False
            return

        try:
            import rclpy
            from rclpy.node import Node
            from geometry_msgs.msg import PointStamped, Twist
            from nav_msgs.msg import Odometry
            from sensor_msgs.msg import Image
            from std_msgs.msg import Bool, Float64, String

            self.ros_available = True
            self._rclpy = rclpy
            if not rclpy.ok():
                rclpy.init(args=None)

            bridge_self = self

            class GroundStationNode(Node):
                def __init__(self):
                    super().__init__("ground_station_web_bridge")
                    self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
                    self.mast_pub = self.create_publisher(
                        Float64, "/camera_mast/set_height", 10
                    )
                    self.auto_align_pub = self.create_publisher(
                        Bool, "/camera_mast/auto_align", 10
                    )
                    self.mission_cmd_pub = self.create_publisher(
                        String, "/ground_station/mission_cmd", 10
                    )

                    self.create_subscription(
                        Odometry, "/odometry/filtered", self._ekf_cb, 10
                    )
                    self.create_subscription(
                        Odometry, "/rover/wheel_odom", self._raw_odom_cb, 10
                    )
                    self.create_subscription(
                        Float64, "/camera_mast/current_height", self._mast_curr_cb, 10
                    )
                    self.create_subscription(
                        Float64, "/camera_mast/target_height", self._mast_tgt_cb, 10
                    )
                    self.create_subscription(
                        PointStamped, "/aruco_trigger", self._aruco_cb, 10
                    )
                    self.create_subscription(
                        String, "/ground_station/mission_status", self._mission_status_cb, 10
                    )
                    self.create_subscription(
                        String, "/rover/yolo_detections", self._yolo_det_cb, 10
                    )
                    self.create_subscription(
                        Image,
                        "/rover/left_camera/detections/image_raw",
                        lambda m: self._img_cb(m, "left"),
                        5,
                    )
                    self.create_subscription(
                        Image,
                        "/rover/right_camera/detections/image_raw",
                        lambda m: self._img_cb(m, "right"),
                        5,
                    )
                    self.create_subscription(
                        Image,
                        "/rover/bottom_camera/aruco_detections/image_raw",
                        lambda m: self._img_cb(m, "bottom"),
                        5,
                    )
                    self.create_subscription(
                        Image,
                        "/rover/left_camera/image_raw",
                        lambda m: self._raw_img_cb(m, "left"),
                        5,
                    )
                    self.create_subscription(
                        Image,
                        "/rover/right_camera/image_raw",
                        lambda m: self._raw_img_cb(m, "right"),
                        5,
                    )
                    self.create_subscription(
                        Image,
                        "/rover/bottom_camera/image_raw",
                        lambda m: self._raw_img_cb(m, "bottom"),
                        5,
                    )
                    self._last_det_img = {"left": 0.0, "right": 0.0, "bottom": 0.0}

                def _ekf_cb(self, msg: Odometry):
                    q = msg.pose.pose.orientation
                    yaw = math.atan2(
                        2.0 * (q.w * q.z + q.x * q.y),
                        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
                    )
                    now = time.time()
                    x = float(msg.pose.pose.position.x)
                    y = float(msg.pose.pose.position.y)
                    lv = float(msg.twist.twist.linear.x)
                    av = float(msg.twist.twist.angular.z)
                    dist_from_base = math.hypot(x - BASE_X, y - BASE_Y)

                    # Auto-start a fresh run when the rover leaves base after a previous RTB
                    if (
                        bridge_self._returned_to_base
                        and bridge_self.active_run_id is None
                        and dist_from_base > 1.20
                        and abs(lv) > 0.05
                    ):
                        bridge_self._start_new_trial_run(
                            "New trial auto-started after departing initial base."
                        )

                    # Auto-create first run when Gazebo is actively running/moving
                    allow_create = (
                        bridge_self.robot_mode == "running_mission"
                        or dist_from_base > 0.80
                        or abs(lv) > 0.05
                    )
                    run_id = bridge_self._mark_ros_alive(allow_auto_create=allow_create)

                    if run_id is not None and dist_from_base > 1.80:
                        bridge_self._has_left_base = True

                    # Detect manual or automatic arrival back at initial base (-0.44, -1.75)
                    if (
                        run_id is not None
                        and bridge_self._has_left_base
                        and dist_from_base <= 0.50
                        and abs(lv) < 0.08
                        and bridge_self.robot_mode in ("returning_to_base", "manual_mode", "at_base")
                    ):
                        bridge_self._archive_active_run_at_base(
                            "Rover returned to initial base (-0.44, -1.75m). Run completed & archived."
                        )
                        with bridge_self.lock:
                            if not bridge_self.estop_active:
                                bridge_self.robot_mode = "at_base"
                                bridge_self.status_detail = "Returned to Base — Ready for Next Mission"
                                bridge_self.telemetry["robot_mode"] = "at_base"
                                bridge_self.telemetry["mission_state"] = "at_base"
                                bridge_self.telemetry["status_detail"] = bridge_self.status_detail
                        run_id = None

                    with bridge_self.lock:
                        rx = bridge_self.telemetry["raw_odom_x"]
                        ry = bridge_self.telemetry["raw_odom_y"]
                        if rx == 0.0 and ry == 0.0:
                            rx, ry = x, y
                        drift = math.hypot(rx - x, ry - y)
                        mast_h = bridge_self.telemetry["mast_height_m"]
                        aruco_id = bridge_self.telemetry["last_aruco_id"]
                        bridge_self.telemetry.update(
                            {
                                "x_m": round(x, 3),
                                "y_m": round(y, 3),
                                "yaw_rad": round(yaw, 4),
                                "yaw_deg": round(math.degrees(yaw), 1),
                                "linear_vel": round(lv, 3),
                                "angular_vel": round(av, 3),
                                "drift_m": round(drift, 3),
                                "timestamp": now,
                            }
                        )

                    # Log telemetry to SQLite for the active run
                    if run_id is not None:
                        should_log = False
                        if bridge_self._last_telem_log_pose is None:
                            should_log = True
                        else:
                            lx, ly = bridge_self._last_telem_log_pose
                            if math.hypot(x - lx, y - ly) >= 0.20 or (
                                now - bridge_self._last_telem_log_ts >= 2.0 and abs(lv) > 0.02
                            ):
                                should_log = True
                        if should_log:
                            bridge_self._last_telem_log_pose = (x, y)
                            bridge_self._last_telem_log_ts = now
                            try:
                                db.add_telemetry(
                                    run_id=run_id,
                                    x_m=x,
                                    y_m=y,
                                    yaw_rad=yaw,
                                    linear_vel=lv,
                                    angular_vel=av,
                                    mast_height_m=mast_h,
                                    raw_odom_x=rx,
                                    raw_odom_y=ry,
                                    aruco_id=aruco_id,
                                )
                            except Exception:
                                pass

                def _raw_odom_cb(self, msg: Odometry):
                    bridge_self._mark_ros_alive(allow_auto_create=False)
                    with bridge_self.lock:
                        rx = float(msg.pose.pose.position.x)
                        ry = float(msg.pose.pose.position.y)
                        x = bridge_self.telemetry["x_m"]
                        y = bridge_self.telemetry["y_m"]
                        bridge_self.telemetry["raw_odom_x"] = round(rx, 3)
                        bridge_self.telemetry["raw_odom_y"] = round(ry, 3)
                        bridge_self.telemetry["drift_m"] = round(math.hypot(rx - x, ry - y), 3)

                def _mast_curr_cb(self, msg: Float64):
                    bridge_self._mark_ros_alive(allow_auto_create=False)
                    with bridge_self.lock:
                        bridge_self.telemetry["mast_height_m"] = round(float(msg.data), 4)

                def _mast_tgt_cb(self, msg: Float64):
                    with bridge_self.lock:
                        bridge_self.telemetry["mast_target_m"] = round(float(msg.data), 4)

                def _aruco_cb(self, msg: PointStamped):
                    run_id = bridge_self._mark_ros_alive(allow_auto_create=False)
                    aid = int(round(msg.point.x))
                    dist = round(float(msg.point.y), 2)
                    with bridge_self.lock:
                        bridge_self.telemetry["last_aruco_id"] = aid
                        bridge_self.telemetry["aruco_dist_m"] = dist
                    # Log EKF ArUco correction event only within actual EKF correction range (<= 1.10 m)
                    if dist <= 1.10 and bridge_self._last_aruco_logged != aid and run_id is not None:
                        bridge_self._last_aruco_logged = aid
                        try:
                            db.add_event(
                                run_id,
                                "info",
                                "nav",
                                f"ArUco #{aid:02d} locked (3D dist {dist:.2f} m) — EKF pose updated.",
                            )
                        except Exception:
                            pass

                def _mission_status_cb(self, msg: String):
                    bridge_self._mark_ros_alive(allow_auto_create=False)
                    raw = msg.data.strip()
                    if "|" in raw:
                        mode, detail = raw.split("|", 1)
                    else:
                        mode, detail = raw, raw
                    mode = mode.strip().lower()

                    if mode == "at_base" and bridge_self.active_run_id is not None and bridge_self._has_left_base:
                        bridge_self._archive_active_run_at_base(
                            "Autonomous RTB / mission complete at initial base (-0.44, -1.75m). Run archived."
                        )

                    with bridge_self.lock:
                        if not bridge_self.estop_active:
                            bridge_self.robot_mode = mode
                            bridge_self.status_detail = detail.strip()
                            bridge_self.telemetry["robot_mode"] = mode
                            bridge_self.telemetry["mission_state"] = mode
                            bridge_self.telemetry["status_detail"] = bridge_self.status_detail

                def _yolo_det_cb(self, msg: String):
                    """Record 3-of-4 confirmed foreground detections with synchronized capture frame."""
                    run_id = bridge_self._mark_ros_alive(allow_auto_create=False)
                    if run_id is None:
                        return
                    try:
                        data = json.loads(msg.data)
                    except Exception:
                        return

                    cam_side = str(data.get("camera", "right"))
                    if cam_side not in ("left", "right"):
                        return
                    dets = data.get("detections") or []
                    if not dets:
                        return

                    with bridge_self.lock:
                        x_r = float(bridge_self.telemetry["x_m"])
                        y_r = float(bridge_self.telemetry["y_m"])
                        yaw = float(bridge_self.telemetry["yaw_rad"])
                        ang_vel = float(bridge_self.telemetry["angular_vel"])
                        mode = bridge_self.robot_mode
                        mast_h = float(
                            data.get("mast_height_m", bridge_self.telemetry["mast_height_m"])
                        )

                    # Do not log blurry detections during sharp headland U-turns or at base
                    if abs(ang_vel) > 0.40 or mode in ("at_base", "estop"):
                        return

                    # Project 0.54 m laterally onto the CURRENT adjacent strawberry row only
                    side_sign = 1.0 if cam_side == "left" else -1.0
                    proj_x = x_r - side_sign * 0.54 * math.sin(yaw)
                    proj_y = y_r + side_sign * 0.54 * math.cos(yaw)

                    if proj_y < -1.35 or proj_y > 33.70:
                        return

                    rows = db.FIELD_GEOMETRY["rows"]
                    nearest_row = min(rows, key=lambda r: abs(r["x"] - proj_x))
                    # Must be aligned with an adjacent row (rejects outer-field side)
                    if abs(nearest_row["x"] - proj_x) > 0.65:
                        return
                    row_idx = int(nearest_row["index"])
                    row_x = float(nearest_row["x"])

                    lanes = db.FIELD_GEOMETRY["lanes"]
                    nearest_lane = min(lanes, key=lambda l: abs(l["x"] - x_r))["name"]

                    # Pick highest-confidence confirmed detection (prioritizing diseased classes)
                    mapped = []
                    for d in dets:
                        cls_key = db.map_yolo_class(
                            int(d.get("class_id", 3)), str(d.get("class_name", ""))
                        )
                        conf = float(d.get("confidence", 0.0))
                        if conf >= 0.45:
                            mapped.append((cls_key, conf, d.get("bbox", [])))
                    if not mapped:
                        return
                    mapped.sort(
                        key=lambda item: (0 if item[0] != "healthy" else 1, -item[1])
                    )
                    best_cls, best_conf, best_bbox = mapped[0]

                    # Per-panel spatial deduplication with DISEASE UPGRADE:
                    # Panels sit 0.955 m apart; if a panel was first seen as Healthy (or lower conf)
                    # and a diseased or higher-confidence view arrives 0.2s later, UPGRADE the capture!
                    row_slots = bridge_self._captured_panels.setdefault(row_idx, [])
                    existing_slot = None
                    for slot in row_slots:
                        if abs(proj_y - slot["y"]) < 0.72:
                            existing_slot = slot
                            break

                    if existing_slot is not None:
                        prev_is_healthy = existing_slot["cls"] == "healthy"
                        new_is_diseased = best_cls != "healthy"
                        should_upgrade = (prev_is_healthy and new_is_diseased) or (
                            existing_slot["cls"] == best_cls
                            and best_conf > existing_slot["conf"] + 0.06
                        )
                        if not should_upgrade:
                            return

                    # Use EXACT synchronized JPEG frame embedded by yolo_strawberry_detector.py
                    img_b64 = data.get("image_jpeg_b64") or ""
                    ts_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
                    fname = f"live_{cam_side}_r{row_idx}_{best_cls}_{ts_tag}.jpg"
                    out_path = os.path.join(db.SNAPSHOTS_DIR, fname)
                    os.makedirs(db.SNAPSHOTS_DIR, exist_ok=True)

                    saved = False
                    if img_b64:
                        try:
                            with open(out_path, "wb") as fh:
                                fh.write(base64.b64decode(img_b64))
                            saved = True
                        except Exception:
                            saved = False
                    if not saved and bridge_self.gst_manager is not None:
                        bgr = bridge_self.gst_manager.channels[cam_side].get_bgr()
                        if bgr is not None:
                            cv2.imwrite(out_path, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                            saved = True
                    if not saved:
                        return

                    try:
                        if existing_slot is not None:
                            db.replace_detection_capture(
                                det_id=existing_slot["det_id"],
                                run_id=run_id,
                                disease_class=best_cls,
                                confidence=best_conf,
                                x_m=row_x,
                                y_m=proj_y,
                                mast_height_m=mast_h,
                                image_path=f"snapshots/{fname}",
                                bbox=[best_bbox] if best_bbox else [],
                            )
                            was_healthy = existing_slot["cls"] == "healthy"
                            existing_slot["cls"] = best_cls
                            existing_slot["conf"] = best_conf
                            existing_slot["y"] = proj_y
                            if was_healthy and best_cls != "healthy":
                                label = db.DISEASE_META.get(best_cls, {}).get("short", best_cls)
                                db.add_event(
                                    run_id,
                                    "alert",
                                    "detection",
                                    f"{label} ({int(best_conf*100)}%) confirmed on Row {row_idx} at Y={proj_y:.2f}m ({cam_side} cam).",
                                )
                        else:
                            det_row = db.add_detection(
                                run_id=run_id,
                                disease_class=best_cls,
                                confidence=best_conf,
                                x_m=row_x,
                                y_m=proj_y,
                                row_index=row_idx,
                                lane=nearest_lane,
                                camera_side=cam_side,
                                mast_height_m=mast_h,
                                image_path=f"snapshots/{fname}",
                                bbox=[best_bbox] if best_bbox else [],
                            )
                            row_slots.append(
                                {
                                    "y": proj_y,
                                    "cls": best_cls,
                                    "conf": best_conf,
                                    "det_id": int(det_row["id"]),
                                }
                            )
                            if best_cls != "healthy":
                                label = db.DISEASE_META.get(best_cls, {}).get("short", best_cls)
                                db.add_event(
                                    run_id,
                                    "alert",
                                    "detection",
                                    f"{label} ({int(best_conf*100)}%) confirmed on Row {row_idx} at Y={proj_y:.2f}m ({cam_side} cam).",
                                )
                    except Exception:
                        pass

                def _decode_image(self, msg: Image) -> np.ndarray | None:
                    arr = np.frombuffer(msg.data, dtype=np.uint8)
                    expected = msg.height * msg.width * 3
                    if arr.size < expected:
                        return None
                    img = arr[:expected].reshape((msg.height, msg.width, 3))
                    if msg.encoding.lower() == "rgb8":
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    return img

                def _img_cb(self, msg: Image, side: str):
                    bridge_self._mark_ros_alive(allow_auto_create=False)
                    self._last_det_img[side] = time.monotonic()
                    if bridge_self.gst_manager is None:
                        return
                    img = self._decode_image(msg)
                    if img is not None:
                        bridge_self.gst_manager.channels[side].push_bgr_frame(img, from_ros=True)

                def _raw_img_cb(self, msg: Image, side: str):
                    bridge_self._mark_ros_alive(allow_auto_create=False)
                    if bridge_self.gst_manager is None or (
                        time.monotonic() - self._last_det_img[side]
                    ) < 1.5:
                        return
                    img = self._decode_image(msg)
                    if img is not None:
                        bridge_self.gst_manager.channels[side].push_bgr_frame(img, from_ros=True)

            node = GroundStationNode()
            self._node = node
            self._cmd_vel_pub = node.cmd_pub
            self._mast_pub = node.mast_pub
            self._auto_align_pub = node.auto_align_pub
            self._mission_cmd_pub = node.mission_cmd_pub
        except Exception:
            self.ros_available = False
            self.ros_connected = False

    def _publish_twist(self, lin: float, ang: float):
        if self._cmd_vel_pub is None:
            return
        try:
            from geometry_msgs.msg import Twist

            t = Twist()
            t.linear.x = float(lin)
            t.angular.z = float(ang)
            self._cmd_vel_pub.publish(t)
        except Exception:
            pass

    def _publish_mission_cmd(self, cmd: str):
        if self._mission_cmd_pub is None:
            return
        try:
            from std_msgs.msg import String

            self._mission_cmd_pub.publish(String(data=cmd))
        except Exception:
            pass

    def _spin_loop(self):
        try:
            while not self._stop.is_set() and self._rclpy and self._rclpy.ok():
                self._rclpy.spin_once(self._node, timeout_sec=0.05)

                now_m = time.monotonic()
                now_t = time.time()

                if now_m - self._last_cmd_pub_ts >= 0.10:
                    self._last_cmd_pub_ts = now_m
                    if self.estop_active:
                        self._publish_twist(0.0, 0.0)
                    elif self._teleop_active:
                        if now_m <= self._teleop_expire_ts:
                            self._publish_twist(self._teleop_vx, self._teleop_wz)
                        else:
                            self._teleop_active = False
                            self._teleop_vx = 0.0
                            self._teleop_wz = 0.0
                            self._publish_twist(0.0, 0.0)

                if self._last_ros_msg_ts > 0 and (now_t - self._last_ros_msg_ts) > 3.5:
                    with self.lock:
                        self.ros_connected = False
                        self.telemetry["ros_connected"] = False
                        self.robot_mode = "standby"
                        self.status_detail = "ROS 2 / Gazebo disconnected"
                        self.telemetry["robot_mode"] = "standby"
                        self.telemetry["status_detail"] = self.status_detail
        except Exception:
            pass

    def send_cmd_vel(self, linear_x: float, angular_z: float, duration_s: float = 0.55) -> dict[str, Any]:
        if self.estop_active and (abs(linear_x) > 1e-4 or abs(angular_z) > 1e-4):
            return {
                "ok": False,
                "ros_connected": self.ros_connected,
                "reason": "Emergency Stop is active. Release E-STOP first.",
            }

        lin = max(-0.75, min(0.75, float(linear_x)))
        ang = max(-1.20, min(1.20, float(angular_z)))
        is_moving = abs(lin) > 1e-4 or abs(ang) > 1e-4

        if is_moving and self.robot_mode in ("running_mission", "returning_to_base"):
            self._publish_mission_cmd("pause")
            with self.lock:
                self.robot_mode = "manual_mode"
                self.status_detail = "Manual Control Mode (Operator Teleop)"
                self.telemetry["robot_mode"] = "manual_mode"
                self.telemetry["mission_state"] = "manual_mode"
                self.telemetry["status_detail"] = self.status_detail

        with self.lock:
            self._teleop_vx = lin
            self._teleop_wz = ang
            self._teleop_active = is_moving
            self._teleop_expire_ts = time.monotonic() + max(0.25, min(1.5, duration_s))

        self._publish_twist(lin, ang)

        if not self.ros_connected:
            return {
                "ok": False,
                "ros_connected": False,
                "linear_x": lin,
                "angular_z": ang,
                "reason": "ROS 2 / Gazebo is not connected. Launch agriculture_world.launch.xml first.",
            }
        return {"ok": True, "ros_connected": True, "linear_x": lin, "angular_z": ang}

    def set_estop(self, active: bool) -> dict[str, Any]:
        self.estop_active = bool(active)
        with self.lock:
            self._teleop_active = False
            self._teleop_vx = 0.0
            self._teleop_wz = 0.0
            self.robot_mode = "estop" if self.estop_active else "manual_mode"
            self.status_detail = (
                "Emergency Stop Active"
                if self.estop_active
                else "Manual Control Mode (E-STOP Released)"
            )
            self.telemetry["estop_active"] = self.estop_active
            self.telemetry["robot_mode"] = self.robot_mode
            self.telemetry["mission_state"] = self.robot_mode
            self.telemetry["status_detail"] = self.status_detail
            if self.estop_active:
                self.telemetry["linear_vel"] = 0.0
                self.telemetry["angular_vel"] = 0.0

        self._publish_mission_cmd("estop" if self.estop_active else "release_estop")
        if self.estop_active:
            self._publish_twist(0.0, 0.0)

        return {
            "ok": True,
            "ros_connected": self.ros_connected,
            "estop_active": self.estop_active,
        }

    def set_mast_height(self, height_m: float, auto_align: bool | None = None) -> dict[str, Any]:
        clamped = max(0.7259, min(1.1050, float(height_m)))
        if auto_align is not None:
            self.auto_align_mast = bool(auto_align)
            if self._auto_align_pub is not None:
                try:
                    from std_msgs.msg import Bool

                    self._auto_align_pub.publish(Bool(data=self.auto_align_mast))
                except Exception:
                    pass

        if self._mast_pub is not None:
            try:
                from std_msgs.msg import Float64

                self._mast_pub.publish(Float64(data=clamped))
            except Exception:
                pass

        with self.lock:
            self.telemetry["mast_target_m"] = round(clamped, 4)
            self.telemetry["auto_align_mast"] = self.auto_align_mast

        return {
            "ok": self.ros_connected,
            "ros_connected": self.ros_connected,
            "mast_target_m": round(clamped, 4),
            "auto_align_mast": self.auto_align_mast,
            "reason": None if self.ros_connected else "ROS 2 / Gazebo not connected",
        }

    def handle_mission_action(self, action: str, explicit_run_id: int | None = None) -> dict[str, Any]:
        """
        Execute Start / Pause / RTB / Reset / Stop cleanly and manage SQLite Run transitions:
          - After RTB or full mission completion at base, clicking Start ALWAYS creates
            the next Run (#2, #3, ...) and resets Nav2 to Waypoint 1.
        """
        cmd = action.strip().lower()
        ros_cmd = cmd
        new_run_created = False

        if cmd == "start":
            self.estop_active = False
            self._teleop_active = False
            # If we finished/returned to base OR no active run exists -> create NEXT run!
            if self.active_run_id is None or self._returned_to_base or self.robot_mode in ("at_base", "returning_to_base"):
                if self.active_run_id is not None:
                    self._archive_active_run_at_base("Previous run archived before starting next mission.")
                self._start_new_trial_run("Started new mission from Ground Station.")
                new_run_created = True
                ros_cmd = "reset"
            elif explicit_run_id is not None and explicit_run_id != self.active_run_id:
                run = db.get_run(explicit_run_id)
                if run and run["status"] == "active":
                    self.active_run_id = explicit_run_id
            new_mode = "running_mission"
            detail = "Running Mission · Autonomous Row Inspection"

        elif cmd == "reset":
            self.estop_active = False
            self._teleop_active = False
            if self.active_run_id is not None:
                self._archive_active_run_at_base("Previous run archived on mission reset.")
            self._start_new_trial_run("Fresh mission started from Ground Station.")
            new_run_created = True
            new_mode = "running_mission"
            detail = "Running Mission · Restarted from Waypoint 1"

        elif cmd == "rtb":
            self.estop_active = False
            self._teleop_active = False
            new_mode = "returning_to_base"
            detail = "Returning to Base (-0.44, -1.75m)"

        elif cmd == "pause":
            self._teleop_active = False
            self._publish_twist(0.0, 0.0)
            new_mode = "manual_mode"
            detail = "Manual Control Mode (Nav2 Paused)"

        else:
            self._teleop_active = False
            self._publish_twist(0.0, 0.0)
            new_mode = "manual_mode"
            detail = "Stopped · Manual Control Mode"

        self._publish_mission_cmd(ros_cmd)
        with self.lock:
            self.robot_mode = new_mode
            self.status_detail = detail
            self.telemetry["robot_mode"] = new_mode
            self.telemetry["mission_state"] = new_mode
            self.telemetry["status_detail"] = detail
            self.telemetry["estop_active"] = self.estop_active
            self.telemetry["active_run_id"] = self.active_run_id

        if self.active_run_id is not None:
            try:
                db.add_event(self.active_run_id, "info", "mission", detail)
            except Exception:
                pass

        return {
            "ok": True,
            "ros_connected": self.ros_connected,
            "robot_mode": new_mode,
            "status_detail": detail,
            "active_run_id": self.active_run_id,
            "new_run_created": new_run_created,
            "message": (
                f"{detail} (Run #{self.active_run_id})"
                if self.active_run_id is not None
                else detail
            ),
        }

    def clear_active_run(self):
        with self.lock:
            self.active_run_id = None
            self.telemetry["active_run_id"] = None
            self._has_left_base = False
            self._returned_to_base = False
            self._last_telem_log_pose = None
            self._last_aruco_logged = None
            self._captured_panels.clear()

    def attach_gst_manager(self, gst_manager: "GStreamerManager"):
        self.gst_manager = gst_manager
        self.gst_manager.set_telemetry_ref(self.telemetry)

    def get_telemetry(self) -> dict[str, Any]:
        with self.lock:
            snap = dict(self.telemetry)
            snap["timestamp"] = time.time()
            snap["ros_available"] = self.ros_available
            return snap

    def shutdown(self):
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None
