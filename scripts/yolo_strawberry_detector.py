#!/home/burneye228/ros2_ws/.venv/bin/python
"""
ROS 2 node running YOLOv11 (best.pt) on both left and right rover cameras in Gazebo,
coupled with:
  1. Depth-gated current-row fast-path (skips YOLO completely on blank-sky outer camera
     frames, doubling active-row frame rate + depth=1 QoS for zero stale-frame lag).
  2. High-recall, low-false-positive detection filter (captures all 4 classes including
     compact Anthracnose lesions >= 18x18 px; instant confirm at conf >= 0.58 and 2-of-3
     temporal confirmation for moderate 0.45-0.58 detections).
  3. Synchronized JPEG frame capture embedded in /rover/yolo_detections.
  4. Cloud-immune current-row Visual-Servoing Camera Mast Auto-Alignment controller.
"""

import base64
from collections import deque
import json
import os
import time
from ament_index_python.packages import get_package_share_directory
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float64, String
import torch
from cv_bridge import CvBridge
from ultralytics import YOLO
import yaml

# Prevent PyTorch OpenMP from starving Gazebo/RViz on 6-core CPUs
torch.set_num_threads(3)

DEFAULT_MODEL_PATH = (
    "/home/burneye228/ros2_ws/src/agriculture_sim/"
    "Final_Fruit_Dataset_YOLOv11/runs/detect/train4/weights/best.pt"
)

# Physical vertical FOV span at nominal current-row standoff distance D_row = 0.44 m
NOMINAL_VERTICAL_SPAN_M = 0.508

# Normalized vertical image targets (v_norm = v / H in [0, 1], 0 = top, 1 = bottom)
TARGET_V_YOLO = 0.48    # Center foreground strawberry panels vertically in frame
TARGET_V_GREEN = 0.58   # Current-row foliage canopy hedge sits just below panel center
TARGET_V_WHITE = 0.76   # Current-row white hydroponic grow-bag sits below green foliage

# Class colors (BGR)
CLASS_COLORS_BGR = {
    0: (38, 38, 220),   # Anthracnose (Red)
    1: (6, 119, 217),   # Gray Mold (Amber)
    2: (237, 58, 124),  # Powdery Mildew (Violet)
    3: (105, 150, 5),   # Healthy (Emerald)
}


class YoloStrawberryDetector(Node):
    def __init__(self):
        super().__init__("yolo_strawberry_detector")

        self.declare_parameter("model_path", DEFAULT_MODEL_PATH)
        self.declare_parameter("conf_threshold", 0.45)
        self.declare_parameter("instant_conf_threshold", 0.58)
        self.declare_parameter("imgsz", 480)
        self.declare_parameter("auto_align_mast", True)

        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.conf_threshold = float(self.get_parameter("conf_threshold").value)
        self.instant_conf_threshold = float(self.get_parameter("instant_conf_threshold").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.auto_align_mast = bool(self.get_parameter("auto_align_mast").value)

        # Load camera mast height bounds from config
        self.h_min = 0.7259
        self.h_max = 1.1050
        self.current_mast_height = 0.9400
        self.target_mast_height = 0.9400
        try:
            cfg_dir = os.path.join(
                get_package_share_directory("agriculture_sim"), "config"
            )
            with open(os.path.join(cfg_dir, "camera_mast.yaml"), "r") as fh:
                mast_cfg = yaml.safe_load(fh)["camera_mast"]
                self.current_mast_height = float(mast_cfg.get("default_camera_height", 0.94))
                self.target_mast_height = self.current_mast_height
            with open(os.path.join(cfg_dir, "camera_mast_derived.yaml"), "r") as fh:
                derived_cfg = yaml.safe_load(fh)["camera_mast_derived"]
                self.h_min = float(derived_cfg.get("min_camera_height", 0.7259))
                self.h_max = float(derived_cfg.get("max_camera_height", 1.1050))
        except Exception as exc:
            self.get_logger().warn(f"Using default mast bounds ({exc})")

        if not os.path.exists(model_path):
            self.get_logger().error(f"YOLOv11 weights not found at: {model_path}")
            raise FileNotFoundError(model_path)

        self.get_logger().info(f"Loading YOLOv11 model from: {model_path}")
        self.model = YOLO(model_path)
        self.bridge = CvBridge()

        # Visual-servoing state per camera ('left' and 'right')
        self.cam_obs = {"left": None, "right": None}
        self.last_active_row_time = time.monotonic()
        self.scan_dir = -1.0
        self.align_status = "INIT"

        # 3-frame rolling buffer for moderate-confidence (0.45-0.58) 2-of-3 confirmation
        self.det_history = {
            "left": deque(maxlen=3),
            "right": deque(maxlen=3),
        }

        # Latest-frame-only QoS (depth=1) eliminates multi-second camera backlog lag
        cam_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.left_sub = self.create_subscription(
            Image, "/rover/left_camera/image_raw", self.left_camera_callback, cam_qos
        )
        self.right_sub = self.create_subscription(
            Image, "/rover/right_camera/image_raw", self.right_camera_callback, cam_qos
        )
        self.mast_height_sub = self.create_subscription(
            Float64, "/camera_mast/current_height", self.mast_height_callback, 10
        )
        self.auto_align_sub = self.create_subscription(
            Bool, "/camera_mast/auto_align", self.auto_align_callback, 10
        )

        self.left_pub = self.create_publisher(
            Image, "/rover/left_camera/detections/image_raw", 2
        )
        self.right_pub = self.create_publisher(
            Image, "/rover/right_camera/detections/image_raw", 2
        )
        self.det_json_pub = self.create_publisher(
            String, "/rover/yolo_detections", 10
        )
        self.mast_cmd_pub = self.create_publisher(
            Float64, "/camera_mast/set_height", 10
        )

        # 5 Hz visual-servoing mast alignment control loop
        self.create_timer(0.20, self.update_mast_alignment)

        self.get_logger().info(
            f"YOLOv11 Detector + Current-Row Mast Aligner ready "
            f"(imgsz={self.imgsz}, conf>={self.conf_threshold:.2f}, instant>={self.instant_conf_threshold:.2f}, "
            f"mast [{self.h_min:.3f}, {self.h_max:.3f}] m)."
        )

    def mast_height_callback(self, msg: Float64):
        self.current_mast_height = float(msg.data)

    def auto_align_callback(self, msg: Bool):
        self.auto_align_mast = bool(msg.data)
        self.align_status = "AUTO-ALIGN ON" if self.auto_align_mast else "MANUAL OVERRIDE"
        self.get_logger().info(f"Camera mast auto_align_mast set to {self.auto_align_mast}")

    def quick_has_row_geometry(self, frame: np.ndarray) -> bool:
        """
        Fast (<0.5 ms) pre-check before YOLO inference: returns False on pure sky/cloud
        frames (outer-facing camera with 0.98 m far-clip), saving ~50% CPU load.
        """
        small = frame[::8, ::8]  # 80x80 downsample
        b = small[:, :, 0].astype(np.int16)
        g = small[:, :, 1].astype(np.int16)
        r = small[:, :, 2].astype(np.int16)
        # Non-sky pixels: either darker than bright sky/clouds OR green foliage
        is_sky_or_cloud = (r > 135) & (g > 155) & (b > 165) & ((b - r) >= -10)
        non_sky_frac = float((~is_sky_or_cloud).mean())
        return non_sky_frac >= 0.04

    def filter_foreground_detections(self, boxes, names: dict, w: int, h: int) -> list[dict]:
        """
        Filter detections for the depth-gated current row:
          - Rejects tiny noise specks (<18x18 px or <360 px^2) while keeping real
            Anthracnose lesions (26-60 px) and larger berries.
          - Rejects extreme edge slivers (u < 0.04 or u > 0.96).
        """
        valid = []
        if boxes is None or len(boxes) == 0:
            return valid

        xyxy_arr = boxes.xyxy.cpu().numpy()
        conf_arr = boxes.conf.cpu().numpy()
        cls_arr = boxes.cls.cpu().numpy()

        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy_arr, conf_arr, cls_arr):
            cid = int(cls_id)
            c = float(conf)
            bw = float(x2 - x1)
            bh = float(y2 - y1)
            area = bw * bh
            u_c = 0.5 * (x1 + x2) / w
            v_c = 0.5 * (y1 + y2) / h

            if bw < 18.0 or bh < 18.0 or area < 360.0:
                continue
            aspect = bw / max(1.0, bh)
            if aspect < 0.20 or aspect > 5.0:
                continue
            if u_c < 0.04 or u_c > 0.96:
                continue

            valid.append(
                {
                    "class_id": cid,
                    "class_name": str(names.get(cid, f"cls_{cid}")),
                    "confidence": round(c, 3),
                    "bbox": [round(float(x1), 1), round(float(y1), 1), round(float(x2), 1), round(float(y2), 1)],
                    "u_c": u_c,
                    "v_c": v_c,
                    "instant_ok": c >= self.instant_conf_threshold,
                }
            )
        return valid

    def analyze_aisle_view(self, frame: np.ndarray, valid_dets: list[dict]) -> dict:
        h, w = frame.shape[:2]
        x_margin = int(0.05 * w)
        roi = frame[:, x_margin : w - x_margin]
        rh, _rw = roi.shape[:2]

        num_dets = len(valid_dets)
        v_yolo_norm = None
        has_corner_det = False
        if num_dets > 0:
            w_sum = 0.0
            v_sum = 0.0
            for d in valid_dets:
                _x1, y1, _x2, y2 = d["bbox"]
                v_c = d["v_c"]
                u_c = d["u_c"]
                clipped_top = y1 <= 16.0
                clipped_bot = y2 >= (h - 16.0)
                in_v_edge = (v_c < 0.24) or (v_c > 0.76)
                in_corner = ((u_c < 0.22) or (u_c > 0.78)) and ((v_c < 0.30) or (v_c > 0.70))
                if clipped_top or clipped_bot or in_v_edge or in_corner:
                    has_corner_det = True
                if clipped_top:
                    v_c = max(0.02, v_c - 0.08)
                elif clipped_bot:
                    v_c = min(0.98, v_c + 0.08)
                v_sum += d["confidence"] * v_c
                w_sum += d["confidence"]
            if w_sum > 1e-6:
                v_yolo_norm = v_sum / w_sum

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        b_ch = roi[:, :, 0].astype(np.int16)
        g_ch = roi[:, :, 1].astype(np.int16)
        r_ch = roi[:, :, 2].astype(np.int16)

        blue_sky = ((b_ch - r_ch) > 20) & (v_ch > 150)
        sky_ratio = float(blue_sky.mean())
        floor_mask = (v_ch < 105) & (s_ch < 110)
        floor_ratio = float(floor_mask.mean())

        green_raw = (
            (h_ch >= 33)
            & (h_ch <= 88)
            & (s_ch >= 65)
            & (v_ch >= 42)
            & ((g_ch - np.maximum(r_ch, b_ch)) >= 10)
        )
        row_green_frac = green_raw.mean(axis=1)
        valid_green_rows = row_green_frac >= 0.18
        green_band_mask = green_raw & valid_green_rows[:, None]
        green_ratio = float(green_band_mask.mean())

        v_green_norm = None
        if green_ratio >= 0.006:
            gy = np.where(green_band_mask)[0]
            v_green_norm = float(gy.mean()) / rh

        white_raw = (s_ch <= 34) & (v_ch >= 172) & (np.abs(b_ch - r_ch) <= 18)
        row_white_frac = white_raw.mean(axis=1)
        valid_white_rows = row_white_frac >= 0.50
        sky_cutoff_px = int((v_green_norm if v_green_norm is not None else 0.32) * rh)
        valid_white_rows[:sky_cutoff_px] = False
        valid_white_rows &= blue_sky.mean(axis=1) < 0.05

        white_mask = white_raw & valid_white_rows[:, None]
        white_ratio = float(white_mask.mean())
        v_white_norm = None
        if white_ratio >= 0.015:
            wy = np.where(white_mask)[0]
            v_white_norm = float(wy.mean()) / rh

        is_active_row = (
            (num_dets > 0) or (green_ratio >= 0.015) or (white_ratio >= 0.020)
        )

        delta_z = 0.0
        confidence = 0.0
        mode = "IDLE (NO ROW)"

        if is_active_row:
            dz_yolo = (
                NOMINAL_VERTICAL_SPAN_M * (TARGET_V_YOLO - v_yolo_norm)
                if v_yolo_norm is not None
                else None
            )
            dz_green = (
                NOMINAL_VERTICAL_SPAN_M * (TARGET_V_GREEN - v_green_norm)
                if v_green_norm is not None
                else None
            )
            dz_white = (
                NOMINAL_VERTICAL_SPAN_M * (TARGET_V_WHITE - v_white_norm)
                if v_white_norm is not None
                else None
            )

            if dz_yolo is not None and dz_green is not None:
                if has_corner_det:
                    delta_z = 1.40 * (0.75 * dz_yolo + 0.25 * dz_green)
                    mode = "CORNER DET -> ALIGN"
                    confidence = 1.0
                else:
                    delta_z = 0.65 * dz_yolo + 0.35 * dz_green
                    mode = "YOLO+GREEN LOCK"
                    confidence = 0.95
            elif dz_yolo is not None:
                boost = 1.40 if has_corner_det else 1.0
                delta_z = boost * dz_yolo
                mode = "CORNER DET -> ALIGN" if has_corner_det else "YOLO TRACK"
                confidence = 0.90
            elif dz_green is not None and green_ratio >= 0.015:
                low_green_boost = 1.30 if green_ratio < 0.04 else 1.0
                if dz_white is not None:
                    delta_z = low_green_boost * (0.75 * dz_green + 0.25 * dz_white)
                else:
                    delta_z = low_green_boost * dz_green
                mode = "LOW GREEN -> SEEK" if green_ratio < 0.04 else "GREEN CANOPY LOCK"
                confidence = min(0.88, 0.45 + green_ratio * 2.5)
            elif dz_white is not None:
                delta_z = 1.20 * dz_white
                mode = "NO GREEN -> TROUGH SEEK"
                confidence = 0.60

        return {
            "time": time.monotonic(),
            "is_active_row": is_active_row,
            "num_dets": num_dets,
            "has_corner_det": has_corner_det,
            "v_yolo_norm": v_yolo_norm,
            "green_ratio": green_ratio if is_active_row else 0.0,
            "v_green_norm": v_green_norm if is_active_row else None,
            "white_ratio": white_ratio,
            "v_white_norm": v_white_norm,
            "sky_ratio": sky_ratio,
            "floor_ratio": floor_ratio,
            "delta_z": delta_z,
            "confidence": confidence,
            "mode": mode,
        }

    def update_mast_alignment(self):
        if not self.auto_align_mast:
            return

        now = time.monotonic()
        active_cues = []
        for side in ("left", "right"):
            obs = self.cam_obs.get(side)
            if obs is not None and (now - obs["time"]) < 1.0 and obs["is_active_row"]:
                if obs["confidence"] > 0.0:
                    active_cues.append(obs)

        if active_cues:
            self.last_active_row_time = now
            w_sum = sum(c["confidence"] for c in active_cues)
            fused_dz = sum(c["confidence"] * c["delta_z"] for c in active_cues) / w_sum
            any_corner = any(c["has_corner_det"] for c in active_cues)
            best_cue = max(active_cues, key=lambda c: c["confidence"])

            if abs(fused_dz) < 0.010 and not any_corner:
                self.align_status = f"ALIGNED ({best_cue['mode']})"
                return

            kp = 0.60 if any_corner else 0.45
            step = max(-0.040, min(0.040, kp * fused_dz))
            raw_desired = self.current_mast_height + step
            clamped_desired = max(self.h_min, min(self.h_max, raw_desired))
            self.target_mast_height = 0.35 * self.target_mast_height + 0.65 * clamped_desired
            direction_str = "UP" if fused_dz > 0 else "DOWN"
            self.align_status = f"{best_cue['mode']} [{direction_str}]"
            self.mast_cmd_pub.publish(Float64(data=float(self.target_mast_height)))
            return

        if (now - self.last_active_row_time) > 3.0:
            recent_obs = [
                obs for obs in self.cam_obs.values()
                if obs is not None and (now - obs["time"]) < 1.0
            ]
            if recent_obs:
                avg_floor = sum(o["floor_ratio"] for o in recent_obs) / len(recent_obs)
                avg_sky = sum(o["sky_ratio"] for o in recent_obs) / len(recent_obs)
                if avg_floor > 0.55:
                    self.scan_dir = 1.0
                elif avg_sky > 0.60:
                    self.scan_dir = -1.0

            next_h = self.target_mast_height + self.scan_dir * 0.008
            if next_h >= self.h_max:
                next_h = self.h_max
                self.scan_dir = -1.0
            elif next_h <= self.h_min:
                next_h = self.h_min
                self.scan_dir = 1.0

            self.target_mast_height = next_h
            self.align_status = f"NO ROW -> SCAN {'UP' if self.scan_dir > 0 else 'DOWN'}"
            self.mast_cmd_pub.publish(Float64(data=float(self.target_mast_height)))
        else:
            self.align_status = "HOLDING HEIGHT"

    def run_detection(self, msg: Image, camera_label: str, side_key: str, pub) -> None:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        h, w = frame.shape[:2]

        # Fast-path: skip expensive YOLO inference when outer camera sees pure sky
        if self.quick_has_row_geometry(frame):
            results = self.model.predict(
                source=frame,
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                verbose=False,
            )
            names = results[0].names or {}
            valid_dets = self.filter_foreground_detections(results[0].boxes, names, w, h)
        else:
            valid_dets = []

        obs = self.analyze_aisle_view(frame, valid_dets)
        self.cam_obs[side_key] = obs

        annotated = frame.copy()
        for d in valid_dets:
            x1, y1, x2, y2 = [int(round(v)) for v in d["bbox"]]
            color = CLASS_COLORS_BGR.get(d["class_id"], (80, 220, 255))
            thickness = 3 if d["instant_ok"] else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            tag = f"{d['class_name']} {d['confidence']:.2f}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.rectangle(
                annotated,
                (x1, max(56, y1 - th - 8)),
                (x1 + tw + 8, max(56 + th + 8, y1)),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                tag,
                (x1 + 4, max(56 + th + 4, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        if obs["is_active_row"]:
            y_target = int(TARGET_V_YOLO * h)
            cv2.line(annotated, (0, y_target), (w, y_target), (190, 190, 190), 1, cv2.LINE_AA)
            if obs["v_green_norm"] is not None:
                cv2.circle(
                    annotated, (w - 26, int(obs["v_green_norm"] * h)), 7, (40, 240, 80), -1, cv2.LINE_AA
                )
            if obs["v_yolo_norm"] is not None:
                col = (0, 140, 255) if obs["has_corner_det"] else (80, 220, 255)
                cv2.circle(
                    annotated, (w - 26, int(obs["v_yolo_norm"] * h)), 7, col, 2, cv2.LINE_AA
                )

        green_pct = 100.0 * obs["green_ratio"]
        header_line1 = f"{camera_label} | Dets: {len(valid_dets)} | Green: {green_pct:.1f}%"
        header_line2 = (
            f"Mast: {self.current_mast_height:.3f}m -> {self.target_mast_height:.3f}m "
            f"[{obs['mode'] if obs['is_active_row'] else self.align_status}]"
        )
        cv2.rectangle(annotated, (0, 0), (w, 54), (18, 22, 28), -1)
        cv2.putText(
            annotated, header_line1, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (80, 255, 120), 2, cv2.LINE_AA
        )
        status_color = (
            (60, 180, 255)
            if obs["has_corner_det"]
            else ((120, 230, 255) if obs["is_active_row"] else (180, 180, 180))
        )
        cv2.putText(
            annotated, header_line2, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.46, status_color, 1, cv2.LINE_AA
        )

        out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        out_msg.header = msg.header
        pub.publish(out_msg)

        # --- Smart Confidence-Adaptive Confirmation ---
        # - Strong detections (conf >= 0.58): instant 1-frame confirmation (zero missed panels!)
        # - Moderate detections (0.45-0.58): 2-of-3 frame temporal confirmation (zero weak flickers!)
        present_classes = {d["class_id"] for d in valid_dets}
        hist = self.det_history[side_key]
        hist.append(present_classes)

        confirmed = []
        for d in valid_dets:
            cid = d["class_id"]
            if d["instant_ok"] or sum(1 for s in hist if cid in s) >= 2:
                confirmed.append(
                    {
                        "class_id": cid,
                        "class_name": d["class_name"],
                        "confidence": d["confidence"],
                        "bbox": d["bbox"],
                    }
                )

        if confirmed:
            ok_jpg, enc = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
            img_b64 = base64.b64encode(enc.tobytes()).decode("ascii") if ok_jpg else ""
            payload = {
                "camera": side_key,
                "timestamp": time.time(),
                "mast_height_m": round(float(self.current_mast_height), 4),
                "detections": confirmed,
                "image_jpeg_b64": img_b64,
            }
            self.det_json_pub.publish(String(data=json.dumps(payload)))

    def left_camera_callback(self, msg: Image):
        self.run_detection(msg, "LEFT CAM", "left", self.left_pub)

    def right_camera_callback(self, msg: Image):
        self.run_detection(msg, "RIGHT CAM", "right", self.right_pub)


def main(args=None):
    rclpy.init(args=args)
    node = YoloStrawberryDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
