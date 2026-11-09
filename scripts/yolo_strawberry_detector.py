#!/home/burneye228/ros2_ws/.venv/bin/python
"""
ROS 2 node running YOLOv11 (best.pt) on both left and right rover cameras in Gazebo,
coupled with a multi-cue Visual-Servoing Camera Mast Auto-Alignment controller:
  1. YOLOv11 Strawberry & Leaf Detections (with corner/edge out-of-scope boost)
  2. HSV + Excess-Green Foliage Canopy Area & Vertical Centroid Tracking
  3. White Hydroponic Grow-Bag Trough Reference & No-Green Vertical Search Recovery
"""

import os
import time
from ament_index_python.packages import get_package_share_directory
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge
from ultralytics import YOLO
import yaml


DEFAULT_MODEL_PATH = (
    "/home/burneye228/ros2_ws/src/agriculture_sim/"
    "Final_Fruit_Dataset_YOLOv11/runs/detect/train4/weights/best.pt"
)

# Physical vertical field-of-view span at nominal row standoff distance D_row = 0.44 m
# (VFOV = 1.0472 rad = 60 deg -> H_phys = 2 * 0.44 * tan(30 deg) = 0.508 m)
NOMINAL_VERTICAL_SPAN_M = 0.508

# Normalized vertical image targets (v_norm = v / H in [0, 1], 0 = top, 1 = bottom)
TARGET_V_YOLO = 0.50    # Center strawberry detections vertically in frame
TARGET_V_GREEN = 0.58   # Foliage canopy hedge sits just below panel center
TARGET_V_WHITE = 0.77   # White hydroponic grow-bag sits below green foliage


class YoloStrawberryDetector(Node):
    def __init__(self):
        super().__init__("yolo_strawberry_detector")

        self.declare_parameter("model_path", DEFAULT_MODEL_PATH)
        self.declare_parameter("conf_threshold", 0.25)
        self.declare_parameter("auto_align_mast", True)

        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.conf_threshold = (
            self.get_parameter("conf_threshold").get_parameter_value().double_value
        )
        self.auto_align_mast = (
            self.get_parameter("auto_align_mast").get_parameter_value().bool_value
        )

        # Load camera mast height bounds from config
        self.h_min = 0.7259
        self.h_max = 1.1050
        self.current_mast_height = 1.0900
        self.target_mast_height = 1.0900
        try:
            cfg_dir = os.path.join(
                get_package_share_directory("agriculture_sim"), "config"
            )
            with open(os.path.join(cfg_dir, "camera_mast.yaml"), "r") as fh:
                mast_cfg = yaml.safe_load(fh)["camera_mast"]
                self.current_mast_height = float(mast_cfg.get("default_camera_height", 1.09))
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

        # Subscribers for both opposite-facing rover cameras & mast height feedback
        self.left_sub = self.create_subscription(
            Image,
            "/rover/left_camera/image_raw",
            self.left_camera_callback,
            10,
        )
        self.right_sub = self.create_subscription(
            Image,
            "/rover/right_camera/image_raw",
            self.right_camera_callback,
            10,
        )
        self.mast_height_sub = self.create_subscription(
            Float64,
            "/camera_mast/current_height",
            self.mast_height_callback,
            10,
        )

        # Publishers for annotated YOLOv11 detection streams & mast height command
        self.left_pub = self.create_publisher(
            Image, "/rover/left_camera/detections/image_raw", 10
        )
        self.right_pub = self.create_publisher(
            Image, "/rover/right_camera/detections/image_raw", 10
        )
        self.mast_cmd_pub = self.create_publisher(
            Float64, "/camera_mast/set_height", 10
        )

        # 5 Hz visual-servoing mast alignment control loop
        self.create_timer(0.20, self.update_mast_alignment)

        self.get_logger().info(
            f"YOLOv11 Detector + Visual Mast Auto-Aligner ready "
            f"(range [{self.h_min:.3f}, {self.h_max:.3f}] m)."
        )

    def mast_height_callback(self, msg: Float64):
        self.current_mast_height = float(msg.data)

    def analyze_aisle_view(self, frame: np.ndarray, boxes) -> dict:
        """
        Analyze a 640x640 camera frame for:
          1. YOLOv11 strawberry/leaf bounding boxes (including corner/edge out-of-scope check)
          2. Green strawberry foliage canopy area & vertical centroid
          3. White hydroponic grow-bag trough band (rejects outer-field background grass)
        """
        h, w = frame.shape[:2]
        x_margin = int(0.05 * w)
        roi = frame[:, x_margin : w - x_margin]
        rh, rw = roi.shape[:2]

        # --- 1. YOLOv11 Detections & Corner/Edge Check ---
        num_dets = 0
        v_yolo_norm = None
        has_corner_det = False
        yolo_conf_sum = 0.0

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            num_dets = len(xyxy)
            weighted_v = 0.0
            for (x1, y1, x2, y2), conf in zip(xyxy, confs):
                u_c = 0.5 * (x1 + x2) / w
                v_c = 0.5 * (y1 + y2) / h

                # Check if strawberry is in the corner or clipped at top/bottom edge (out of scope)
                clipped_top = y1 <= 14.0
                clipped_bot = y2 >= (h - 14.0)
                in_vertical_edge = (v_c < 0.26) or (v_c > 0.74)
                in_corner = ((u_c < 0.24) or (u_c > 0.76)) and (
                    (v_c < 0.32) or (v_c > 0.68)
                )
                if clipped_top or clipped_bot or in_vertical_edge or in_corner:
                    has_corner_det = True

                # Compensate for clipped bounding boxes extending beyond top/bottom frame edge
                if clipped_top:
                    v_c = max(0.02, v_c - 0.08)
                elif clipped_bot:
                    v_c = min(0.98, v_c + 0.08)

                weighted_v += float(conf) * v_c
                yolo_conf_sum += float(conf)

            if yolo_conf_sum > 1e-6:
                v_yolo_norm = weighted_v / yolo_conf_sum

        # --- 2. HSV & Channel Segmentation for White Grow-Bag Trough, Sky, Floor, and Green Canopy ---
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        b_ch = roi[:, :, 0].astype(np.int16)
        g_ch = roi[:, :, 1].astype(np.int16)
        r_ch = roi[:, :, 2].astype(np.int16)

        # Warm-white hydroponic grow-bag trough: bright, low saturation, neutral B-R (rejects blue sky)
        white_raw = (s_ch <= 40) & (v_ch >= 162) & (np.abs(b_ch - r_ch) <= 22)
        row_white_frac = white_raw.mean(axis=1)
        valid_white_rows = row_white_frac >= 0.16
        white_mask = white_raw & valid_white_rows[:, None]
        white_ratio = float(white_mask.mean())

        v_white_norm = None
        v_white_top_norm = None
        if white_ratio >= 0.008:
            wy = np.where(white_mask)[0]
            v_white_norm = float(wy.mean()) / rh
            v_white_top_norm = float(np.percentile(wy, 10)) / rh

        # Sky & dark greenhouse floor masks (for out-of-scope recovery)
        sky_mask = ((b_ch - r_ch) > 28) & (v_ch > 165)
        sky_top_ratio = float(sky_mask[: int(0.45 * rh), :].mean())
        sky_ratio = float(sky_mask.mean())
        floor_mask = (v_ch < 105) & (s_ch < 110)
        floor_ratio = float(floor_mask.mean())

        # Green foliage mask (HSV + Excess Green)
        green_raw = (
            (h_ch >= 34)
            & (h_ch <= 86)
            & (s_ch >= 72)
            & (v_ch >= 48)
            & ((g_ch - np.maximum(r_ch, b_ch)) >= 12)
        )

        # If the white grow-bag trough is visible, only keep green foliage above/on the trough
        # (cleanly rejects any distant field grass visible underneath the steel stands)
        if v_white_norm is not None:
            cutoff_row = min(rh, int((v_white_norm + 0.05) * rh))
            green_mask = np.zeros_like(green_raw)
            green_mask[:cutoff_row, :] = green_raw[:cutoff_row, :]
        else:
            green_mask = green_raw

        green_ratio = float(green_mask.mean())
        v_green_norm = None
        if green_ratio >= 0.004:
            gy = np.where(green_mask)[0]
            v_green_norm = float(gy.mean()) / rh

        # --- 3. Active Strawberry Row Gate (Rejects Open-Field Grass on Outer Camera) ---
        # A camera faces an active strawberry row if it sees YOLO detections OR the white
        # hydroponic grow-bag trough OR close-up canopy without open-sky horizon grass.
        is_open_field_grass = (
            (num_dets == 0)
            and (white_ratio < 0.008)
            and (sky_top_ratio > 0.50)
        )
        is_active_row = (not is_open_field_grass) and (
            (num_dets > 0) or (white_ratio >= 0.008) or (green_ratio >= 0.025)
        )

        # --- 4. Compute Height Error delta_z (m) (+ = Move Mast UP, - = Move Mast DOWN) ---
        delta_z = 0.0
        confidence = 0.0
        mode = "IDLE (OUTER FIELD)"

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
                    delta_z = 1.45 * (0.75 * dz_yolo + 0.25 * dz_green)
                    mode = "CORNER DET -> ALIGN"
                    confidence = 1.0
                else:
                    delta_z = 0.65 * dz_yolo + 0.35 * dz_green
                    mode = "YOLO+GREEN LOCK"
                    confidence = 0.95
            elif dz_yolo is not None:
                boost = 1.45 if has_corner_det else 1.0
                delta_z = boost * dz_yolo
                mode = "CORNER DET -> ALIGN" if has_corner_det else "YOLO TRACK"
                confidence = 0.90
            elif dz_green is not None and green_ratio >= 0.008:
                # If green area is very small (<3.5%), canopy is near edge -> boost step
                low_green_boost = 1.35 if green_ratio < 0.035 else 1.0
                if dz_white is not None:
                    delta_z = low_green_boost * (0.75 * dz_green + 0.25 * dz_white)
                else:
                    delta_z = low_green_boost * dz_green
                mode = (
                    "LOW GREEN -> SEEK"
                    if green_ratio < 0.035
                    else "GREEN CANOPY LOCK"
                )
                confidence = min(0.85, 0.40 + green_ratio * 3.0)
            elif dz_white is not None:
                # No green leaves and no strawberries visible, but white trough is in view
                delta_z = 1.25 * dz_white
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
        """5 Hz control loop that fuses left & right camera visual cues and commands mast height."""
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

            # Deadband when well-aligned and no strawberry is in a corner/edge
            if abs(fused_dz) < 0.008 and not any_corner:
                self.align_status = f"ALIGNED ({best_cue['mode']})"
                return

            # Damped proportional visual-servoing update with EMA smoothing
            kp = 0.65 if any_corner else 0.50
            step = max(-0.045, min(0.045, kp * fused_dz))
            raw_desired = self.current_mast_height + step
            clamped_desired = max(self.h_min, min(self.h_max, raw_desired))
            self.target_mast_height = (
                0.35 * self.target_mast_height + 0.65 * clamped_desired
            )
            direction_str = "UP" if fused_dz > 0 else "DOWN"
            self.align_status = f"{best_cue['mode']} [{direction_str}]"
            self.mast_cmd_pub.publish(Float64(data=float(self.target_mast_height)))
            return

        # If no active row cues for > 2.5s, perform intelligent vertical search
        if (now - self.last_active_row_time) > 2.5:
            # Check sky vs dark floor dominance across available cameras
            recent_obs = [
                obs
                for obs in self.cam_obs.values()
                if obs is not None and (now - obs["time"]) < 1.0
            ]
            if recent_obs:
                avg_floor = sum(o["floor_ratio"] for o in recent_obs) / len(recent_obs)
                avg_sky = sum(o["sky_ratio"] for o in recent_obs) / len(recent_obs)
                if avg_floor > 0.52:
                    self.scan_dir = 1.0   # Looking too low at dark floor -> scan UP
                elif avg_sky > 0.58:
                    self.scan_dir = -1.0  # Looking too high at sky -> scan DOWN

            next_h = self.target_mast_height + self.scan_dir * 0.010
            if next_h >= self.h_max:
                next_h = self.h_max
                self.scan_dir = -1.0
            elif next_h <= self.h_min:
                next_h = self.h_min
                self.scan_dir = 1.0

            self.target_mast_height = next_h
            scan_str = "UP" if self.scan_dir > 0 else "DOWN"
            self.align_status = f"NO GREEN -> SCAN {scan_str}"
            self.mast_cmd_pub.publish(Float64(data=float(self.target_mast_height)))
        else:
            self.align_status = "HOLDING HEIGHT"

    def run_detection(self, msg: Image, camera_label: str, side_key: str) -> Image:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        results = self.model.predict(
            source=frame,
            imgsz=640,
            conf=self.conf_threshold,
            verbose=False,
        )
        boxes = results[0].boxes
        obs = self.analyze_aisle_view(frame, boxes)
        self.cam_obs[side_key] = obs

        annotated = results[0].plot(line_width=2, font_size=12)
        h, w = annotated.shape[:2]

        # Draw visual-servoing target alignment guides when facing an active row
        if obs["is_active_row"]:
            y_target_yolo = int(TARGET_V_YOLO * h)
            cv2.line(
                annotated, (0, y_target_yolo), (w, y_target_yolo), (180, 180, 180), 1, cv2.LINE_AA
            )
            if obs["v_green_norm"] is not None:
                y_green = int(obs["v_green_norm"] * h)
                cv2.circle(annotated, (w - 28, y_green), 7, (40, 240, 80), -1, cv2.LINE_AA)
            if obs["v_yolo_norm"] is not None:
                y_yolo = int(obs["v_yolo_norm"] * h)
                color_yolo = (0, 140, 255) if obs["has_corner_det"] else (80, 220, 255)
                cv2.circle(annotated, (w - 28, y_yolo), 7, color_yolo, 2, cv2.LINE_AA)

        # Draw two-line HUD overlay with detection count, green leaf %, and mast auto-alignment state
        num_dets = obs["num_dets"]
        green_pct = 100.0 * obs["green_ratio"]
        header_line1 = (
            f"{camera_label} | Dets: {num_dets} | Green: {green_pct:.1f}%"
        )
        header_line2 = (
            f"Mast: {self.current_mast_height:.3f}m -> {self.target_mast_height:.3f}m "
            f"[{ obs['mode'] if obs['is_active_row'] else self.align_status }]"
        )

        cv2.rectangle(annotated, (0, 0), (w, 54), (20, 20, 20), -1)
        cv2.putText(
            annotated,
            header_line1,
            (10, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (80, 255, 120),
            2,
            cv2.LINE_AA,
        )
        status_color = (
            (60, 180, 255)
            if obs["has_corner_det"]
            else ((120, 230, 255) if obs["is_active_row"] else (180, 180, 180))
        )
        cv2.putText(
            annotated,
            header_line2,
            (10, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            status_color,
            1,
            cv2.LINE_AA,
        )

        out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        out_msg.header = msg.header
        return out_msg

    def left_camera_callback(self, msg: Image):
        out_msg = self.run_detection(msg, "LEFT CAM", "left")
        self.left_pub.publish(out_msg)

    def right_camera_callback(self, msg: Image):
        out_msg = self.run_detection(msg, "RIGHT CAM", "right")
        self.right_pub.publish(out_msg)


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
