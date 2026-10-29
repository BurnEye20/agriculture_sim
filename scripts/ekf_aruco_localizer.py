#!/home/burneye228/ros2_ws/.venv/bin/python
"""
EKF Localization + IMU & Wheel Encoder Odometry + ArUco Absolute Correction + Dual-Loop PID Controller + PGM Map Server.

Logic:
- Between ArUco codes (every 5m): Robot estimates its position using IMU (gyro yaw rate) + Wheel Encoders (/joint_states).
- Over ArUco codes: Downward bottom camera detects the ArUco marker ID, looks up its true coordinates on the PGM map,
  and performs an EKF measurement update to correct drift and lock onto the absolute map location.
- Closed-Loop PID Control:
  1. Steering / Lane-Keeping PID Controller: Computes cross-track error (CTE) to the lane centerline + heading error
     from the EKF pose and applies P + I + D (with anti-windup and derivative filtering) to keep the rover centered
     over the ArUco markers even under wheel slip, bumps, and rough field disturbances.
  2. Longitudinal Velocity PID Controller: Regulates forward velocity against terrain bumps/resistance using
     wheel encoder feedback and smooth approach deceleration at row ends.
"""

import math
import os
import cv2
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from cv_bridge import CvBridge

from geometry_msgs.msg import PoseStamped, TransformStamped, Twist, Point
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import Image, Imu, JointState
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster


PACKAGE_DIR = "/home/burneye228/ros2_ws/src/agriculture_sim"
MAP_YAML_PATH = os.path.join(PACKAGE_DIR, "maps/farming_rows_map.yaml")
COSTMAP_PGM_PATH = os.path.join(PACKAGE_DIR, "maps/farming_rows_costmap.pgm")
ARUCO_YAML_PATH = os.path.join(PACKAGE_DIR, "config/aruco_landmarks.yaml")


def yaw_to_quat(yaw: float):
    return (0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5))


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class PIDController:
    """Discrete PID controller with integral anti-windup and low-pass filtered derivative."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        integral_limit: float,
        output_limit: float,
        deriv_alpha: float = 0.7,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        self.deriv_alpha = deriv_alpha

        self.integral = 0.0
        self.prev_error = None
        self.filtered_deriv = 0.0
        self.last_p = 0.0
        self.last_i = 0.0
        self.last_d = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = None
        self.filtered_deriv = 0.0

    def compute(self, error: float, dt: float) -> float:
        if dt <= 1e-5:
            return 0.0

        # Proportional term
        self.last_p = self.kp * error

        # Integral term with anti-windup clamping
        self.integral += error * dt
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        self.last_i = self.ki * self.integral

        # Derivative term with low-pass filter to reject high-frequency bump noise
        if self.prev_error is None:
            raw_deriv = 0.0
        else:
            raw_deriv = (error - self.prev_error) / dt
        self.prev_error = error
        self.filtered_deriv = (
            self.deriv_alpha * raw_deriv + (1.0 - self.deriv_alpha) * self.filtered_deriv
        )
        self.last_d = self.kd * self.filtered_deriv

        output = self.last_p + self.last_i + self.last_d
        return max(-self.output_limit, min(self.output_limit, output))


class EkfArucoLocalizer(Node):
    def __init__(self):
        super().__init__("ekf_aruco_localizer")

        # Motion & PID ROS 2 Parameters (tunable for bumpy/rough field terrain)
        self.declare_parameter("auto_drive", True)
        self.declare_parameter("target_speed", 0.35)
        self.declare_parameter("target_lane", "center")

        # Steering (Lane-Keeping + Heading) PID gains
        self.declare_parameter("steer_kp", 2.20)
        self.declare_parameter("steer_ki", 0.25)
        self.declare_parameter("steer_kd", 0.65)
        self.declare_parameter("cte_weight", 1.80)

        # Longitudinal Velocity PID gains
        self.declare_parameter("speed_kp", 1.40)
        self.declare_parameter("speed_ki", 0.45)
        self.declare_parameter("speed_kd", 0.12)

        self.auto_drive = self.get_parameter("auto_drive").get_parameter_value().bool_value
        self.target_speed = self.get_parameter("target_speed").get_parameter_value().double_value
        self.target_lane = self.get_parameter("target_lane").get_parameter_value().string_value
        self.cte_weight = self.get_parameter("cte_weight").get_parameter_value().double_value

        # Initialize Steering PID & Longitudinal Speed PID Controllers
        self.steering_pid = PIDController(
            kp=self.get_parameter("steer_kp").get_parameter_value().double_value,
            ki=self.get_parameter("steer_ki").get_parameter_value().double_value,
            kd=self.get_parameter("steer_kd").get_parameter_value().double_value,
            integral_limit=0.50,
            output_limit=1.20,
        )
        self.speed_pid = PIDController(
            kp=self.get_parameter("speed_kp").get_parameter_value().double_value,
            ki=self.get_parameter("speed_ki").get_parameter_value().double_value,
            kd=self.get_parameter("speed_kd").get_parameter_value().double_value,
            integral_limit=0.30,
            output_limit=0.25,
        )

        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)

        # Rover physical parameters
        self.wheel_radius = 0.13
        self.wheel_separation = 0.50
        self.bottom_cam_offset_x = 0.30
        self.bottom_cam_height = 0.192
        self.bottom_cam_fov = 1.65  # rad (640x640 wide-angle ground view)

        # Lane X centerlines
        self.lane_x_centers = {
            "left": -0.4400,
            "center": 1.4000,
            "right": 3.2400,
        }
        self.target_lane_x = self.lane_x_centers.get(self.target_lane, 1.4000)
        self.target_yaw = math.pi / 2.0  # +Y direction along the row
        self.goal_y = 34.00

        # Initial robot pose in map frame (Center lane start: X=1.40, Y=-1.35, Yaw=+pi/2)
        self.init_x = self.target_lane_x
        self.init_y = -1.35
        self.init_yaw = math.pi / 2.0

        # 1. Pure IMU + Wheel Encoder Dead-Reckoning state (accumulates drift between ArUco codes)
        self.odom_x = self.init_x
        self.odom_y = self.init_y
        self.odom_yaw = self.init_yaw

        # 2. EKF State [x, y, yaw] and 3x3 Covariance P in map frame
        self.ekf_x = self.init_x
        self.ekf_y = self.init_y
        self.ekf_yaw = self.init_yaw
        self.P = np.diag([0.02, 0.02, 0.005])

        # Process noise (grows between ArUco markers) & ArUco Measurement noise (small -> strong absolute lock)
        self.Q_base = np.diag([0.08, 0.08, 0.02])
        self.R_aruco = np.diag([0.002, 0.002, 0.001])

        # 2.0-degree Field Slope Rotation Matrix (Roll = +1.732 deg, Pitch = -1.000 deg)
        roll_s = 0.0302300
        pitch_s = -0.0174533
        rx = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, math.cos(roll_s), -math.sin(roll_s)],
                [0.0, math.sin(roll_s), math.cos(roll_s)],
            ]
        )
        ry = np.array(
            [
                [math.cos(pitch_s), 0.0, math.sin(pitch_s)],
                [0.0, 1.0, 0.0],
                [-math.sin(pitch_s), 0.0, math.cos(pitch_s)],
            ]
        )
        self.R_slope = ry @ rx

        # Ground-truth & Odometry Accuracy Verification Metrics
        self.gt_x = self.init_x
        self.gt_y = self.init_y
        self.gt_received = False
        self.ekf_err_current = 0.0
        self.raw_err_current = 0.0
        self.ekf_sq_err_sum = 0.0
        self.raw_sq_err_sum = 0.0
        self.ekf_err_max = 0.0
        self.raw_err_max = 0.0
        self.err_sample_count = 0
        self.ekf_rmse = 0.0
        self.raw_rmse = 0.0

        # Sensor & PID tracking state
        self.prev_wheel_pos = None
        self.prev_pid_time = None
        self.latest_imu_gz = 0.0
        self.current_vx = 0.0
        self.current_wz = 0.0

        self.last_cte_error = 0.0
        self.last_heading_error = 0.0
        self.last_speed_error = 0.0
        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0

        # Active ArUco detection state for visualization
        self.last_detected_id = None
        self.last_detection_time = -10.0
        self.total_aruco_corrections = 0
        self.last_correction_magnitude = 0.0
        self.dist_since_last_aruco = 0.0

        # Load ArUco landmark database
        self.landmarks = self.load_aruco_landmarks()
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        # Load PGM map & costmap
        self.map_msg, self.costmap_msg = self.load_pgm_maps()

        # QoS for latched Map topics
        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.map_pub = self.create_publisher(OccupancyGrid, "/map", map_qos)
        self.costmap_pub = self.create_publisher(OccupancyGrid, "/costmap", map_qos)

        # Publishers for EKF Odometry, Paths, Markers, Annotated ArUco Camera, and PID Diagnostics
        self.ekf_odom_pub = self.create_publisher(Odometry, "/ekf_odom", 10)
        self.raw_odom_pub = self.create_publisher(Odometry, "/wheel_imu_odom", 10)
        self.ekf_path_pub = self.create_publisher(Path, "/ekf_path", 10)
        self.raw_path_pub = self.create_publisher(Path, "/odom_imu_encoder_path", 10)
        self.gt_path_pub = self.create_publisher(Path, "/ground_truth_path", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/aruco_localization_markers", 10)
        self.aruco_img_pub = self.create_publisher(
            Image, "/rover/bottom_camera/aruco_detections/image_raw", 10
        )
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.cte_pub = self.create_publisher(Float64, "/pid/cross_track_error", 10)
        self.heading_err_pub = self.create_publisher(Float64, "/pid/heading_error", 10)
        self.speed_err_pub = self.create_publisher(Float64, "/pid/speed_error", 10)
        self.ekf_err_pub = self.create_publisher(Float64, "/odometry_accuracy/ekf_error_m", 10)
        self.raw_err_pub = self.create_publisher(Float64, "/odometry_accuracy/raw_odom_error_m", 10)

        self.ekf_path = Path()
        self.ekf_path.header.frame_id = "map"
        self.raw_path = Path()
        self.raw_path.header.frame_id = "map"
        self.gt_path = Path()
        self.gt_path.header.frame_id = "map"

        # Subscribers: IMU, Wheel Encoders (/joint_states), Ground Truth Odom, and Bottom ArUco Camera
        self.imu_sub = self.create_subscription(Imu, "/rover/imu", self.imu_callback, 20)
        self.joint_sub = self.create_subscription(
            JointState, "/joint_states", self.joint_state_callback, 20
        )
        self.gt_sub = self.create_subscription(
            Odometry, "/rover/ground_truth_odom", self.gt_odom_callback, 20
        )
        self.bottom_cam_sub = self.create_subscription(
            Image, "/rover/bottom_camera/image_raw", self.bottom_camera_callback, 10
        )

        # Publish static map immediately & periodically
        self.publish_maps()
        self.map_timer = self.create_timer(2.0, self.publish_maps)
        # 20 Hz closed-loop PID control timer + 10 Hz visualization timer + 2.5s accuracy log timer
        self.pid_timer = self.create_timer(0.05, self.pid_control_loop)
        self.vis_timer = self.create_timer(0.10, self.publish_visualization)
        self.log_timer = self.create_timer(2.5, self.log_verification_status)

        self.get_logger().info(
            f"EKF + IMU/Encoders + ArUco + Dual PID Controller started on '{self.target_lane}' lane (X={self.target_lane_x:.2f}m)."
        )

    def log_verification_status(self):
        if not self.gt_received:
            return
        self.get_logger().info(
            f"[Odom Verification] GT=({self.gt_x:.3f}, {self.gt_y:.2f}m) | "
            f"RawOdom Err={self.raw_err_current*100:.2f}cm (RMSE={self.raw_rmse*100:.2f}cm, Max={self.raw_err_max*100:.2f}cm) | "
            f"EKF Err={self.ekf_err_current*100:.2f}cm (RMSE={self.ekf_rmse*100:.2f}cm, Max={self.ekf_err_max*100:.2f}cm) | "
            f"PID CTE={self.last_cte_error*100:+.2f}cm, v={self.current_vx:.2f}m/s"
        )

    def load_aruco_landmarks(self):
        with open(ARUCO_YAML_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        landmarks = {}
        for item in data.get("markers", []):
            pose = item["pose"]
            landmarks[int(item["id"])] = {
                "id": int(item["id"]),
                "lane": item["lane"],
                "x": float(pose[0]),
                "y": float(pose[1]),
                "z": float(pose[2]),
                "yaw": float(pose[5]),
            }
        return landmarks

    def load_pgm_maps(self):
        with open(MAP_YAML_PATH, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)

        pgm_file = os.path.join(os.path.dirname(MAP_YAML_PATH), meta["image"])
        occ_img = cv2.imread(pgm_file, cv2.IMREAD_GRAYSCALE)
        cost_img = cv2.imread(COSTMAP_PGM_PATH, cv2.IMREAD_GRAYSCALE)

        occ_flipped = np.flipud(occ_img)
        cost_flipped = np.flipud(cost_img)

        h, w = occ_flipped.shape
        res = float(meta["resolution"])
        origin_x, origin_y, _ = meta["origin"]

        def make_grid(arr_u8, is_costmap=False):
            msg = OccupancyGrid()
            msg.header.frame_id = "map"
            msg.info.resolution = res
            msg.info.width = w
            msg.info.height = h
            msg.info.origin.position.x = float(origin_x)
            msg.info.origin.position.y = float(origin_y)
            msg.info.origin.position.z = 0.0
            msg.info.origin.orientation.w = 1.0

            grid = np.zeros_like(arr_u8, dtype=np.int8)
            if not is_costmap:
                grid[arr_u8 < 128] = 100
                grid[arr_u8 >= 128] = 0
            else:
                scaled = ((254.0 - arr_u8.astype(np.float32)) / 254.0 * 100.0).astype(np.int8)
                grid = np.clip(scaled, 0, 100)
            msg.data = grid.flatten().tolist()
            return msg

        return make_grid(occ_flipped, False), make_grid(cost_flipped, True)

    def publish_maps(self):
        now = self.get_clock().now().to_msg()
        self.map_msg.header.stamp = now
        self.costmap_msg.header.stamp = now
        self.map_pub.publish(self.map_msg)
        self.costmap_pub.publish(self.costmap_msg)

    def imu_callback(self, msg: Imu):
        self.latest_imu_gz = msg.angular_velocity.z

    def gt_odom_callback(self, msg: Odometry):
        """Project 3D Gazebo world ground-truth pose onto the 2.0-degree sloped field map frame and compute RMSE."""
        pw = np.array(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ]
        )
        pm = self.R_slope.T @ pw
        self.gt_x = float(pm[0])
        self.gt_y = float(pm[1])
        self.gt_received = True

        self.ekf_err_current = math.hypot(self.ekf_x - self.gt_x, self.ekf_y - self.gt_y)
        self.raw_err_current = math.hypot(self.odom_x - self.gt_x, self.odom_y - self.gt_y)

        self.err_sample_count += 1
        self.ekf_sq_err_sum += self.ekf_err_current ** 2
        self.raw_sq_err_sum += self.raw_err_current ** 2
        self.ekf_err_max = max(self.ekf_err_max, self.ekf_err_current)
        self.raw_err_max = max(self.raw_err_max, self.raw_err_current)

        self.ekf_rmse = math.sqrt(self.ekf_sq_err_sum / self.err_sample_count)
        self.raw_rmse = math.sqrt(self.raw_sq_err_sum / self.err_sample_count)

        self.ekf_err_pub.publish(Float64(data=float(self.ekf_err_current)))
        self.raw_err_pub.publish(Float64(data=float(self.raw_err_current)))

    def joint_state_callback(self, msg: JointState):
        name_to_pos = dict(zip(msg.name, msg.position))
        required = [
            "front_left_wheel_joint",
            "rear_left_wheel_joint",
            "front_right_wheel_joint",
            "rear_right_wheel_joint",
        ]
        if not all(k in name_to_pos for k in required):
            return

        left_pos = 0.5 * (
            name_to_pos["front_left_wheel_joint"] + name_to_pos["rear_left_wheel_joint"]
        )
        right_pos = 0.5 * (
            name_to_pos["front_right_wheel_joint"] + name_to_pos["rear_right_wheel_joint"]
        )
        now_sec = self.get_clock().now().nanoseconds * 1e-9

        if self.prev_wheel_pos is None:
            self.prev_wheel_pos = (left_pos, right_pos, now_sec)
            return

        prev_l, prev_r, prev_t = self.prev_wheel_pos
        dt = now_sec - prev_t
        if dt <= 1e-4:
            return
        self.prev_wheel_pos = (left_pos, right_pos, now_sec)

        # Wheel encoder displacements (meters)
        dl = (left_pos - prev_l) * self.wheel_radius
        dr = (right_pos - prev_r) * self.wheel_radius
        ds = 0.5 * (dl + dr)
        dtheta_enc = (dr - dl) / self.wheel_separation

        # Fuse IMU gyroscope yaw rate with wheel encoder differential yaw
        dtheta_imu = self.latest_imu_gz * dt
        dtheta = 0.80 * dtheta_imu + 0.20 * dtheta_enc

        self.current_vx = ds / dt
        self.current_wz = dtheta / dt
        self.dist_since_last_aruco += abs(ds)

        # --- 1. Update Pure IMU + Encoder Dead-Reckoning Odometry (accumulates slip/drift on 2-deg bumpy slope) ---
        ds_raw = ds * 1.018  # Uncompensated uphill wheel spin over 2-deg slope & bumps
        slip_yaw_bias = -0.012 * ds
        mid_odom_yaw = self.odom_yaw + 0.5 * (dtheta + slip_yaw_bias)
        self.odom_x += ds_raw * math.cos(mid_odom_yaw)
        self.odom_y += ds_raw * math.sin(mid_odom_yaw)
        self.odom_yaw = normalize_angle(self.odom_yaw + dtheta + slip_yaw_bias)

        # --- 2. EKF Prediction Step (IMU + Wheel Encoders assume position between ArUco codes) ---
        ds_ekf = ds
        mid_ekf_yaw = self.ekf_yaw + 0.5 * dtheta
        self.ekf_x += ds_ekf * math.cos(mid_ekf_yaw)
        self.ekf_y += ds_ekf * math.sin(mid_ekf_yaw)
        self.ekf_yaw = normalize_angle(self.ekf_yaw + dtheta)

        F = np.array(
            [
                [1.0, 0.0, -ds_ekf * math.sin(mid_ekf_yaw)],
                [0.0, 1.0, ds_ekf * math.cos(mid_ekf_yaw)],
                [0.0, 0.0, 1.0],
            ]
        )
        Q = self.Q_base * max(abs(ds_ekf), 0.002)
        self.P = F @ self.P @ F.T + Q

        self.broadcast_tf_and_odom()

    def pid_control_loop(self):
        """20 Hz Closed-Loop Dual PID Controller (Steering Lane-Keeping PID + Longitudinal Speed PID)."""
        if not self.auto_drive:
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if self.prev_pid_time is None:
            self.prev_pid_time = now_sec
            return
        dt = now_sec - self.prev_pid_time
        if dt <= 1e-4:
            return
        self.prev_pid_time = now_sec

        cmd = Twist()
        dist_to_goal = self.goal_y - self.ekf_y

        if dist_to_goal <= 0.08:
            # Reached end of row: stop and reset PID integrators
            self.steering_pid.reset()
            self.speed_pid.reset()
            self.last_cmd_v = 0.0
            self.last_cmd_w = 0.0
            self.cmd_vel_pub.publish(cmd)
            return

        # 1. Longitudinal Velocity PID (with smooth deceleration near the end of the row)
        v_ref = min(self.target_speed, max(0.10, 0.5 * dist_to_goal))
        speed_error = v_ref - self.current_vx
        v_pid_adj = self.speed_pid.compute(speed_error, dt)
        cmd_v = max(0.0, min(self.target_speed * 1.20, v_ref + v_pid_adj))

        # 2. Lateral Cross-Track Error (CTE) + Heading PID for Lane-Keeping over bumpy terrain
        # Driving along +Y (target_yaw = +pi/2): if ekf_x > target_lane_x (drifted right/+X), steer left (+wz)
        cte_error = (self.ekf_x - self.target_lane_x) * math.sin(self.target_yaw)
        heading_error = normalize_angle(self.target_yaw - self.ekf_yaw)
        combined_steer_error = (self.cte_weight * cte_error) + heading_error
        cmd_w = self.steering_pid.compute(combined_steer_error, dt)

        self.last_cte_error = cte_error
        self.last_heading_error = heading_error
        self.last_speed_error = speed_error
        self.last_cmd_v = cmd_v
        self.last_cmd_w = cmd_w

        cmd.linear.x = float(cmd_v)
        cmd.angular.z = float(cmd_w)
        self.cmd_vel_pub.publish(cmd)

        # Publish PID diagnostic topics for plotting (rqt_plot / PlotJuggler)
        self.cte_pub.publish(Float64(data=float(cte_error)))
        self.heading_err_pub.publish(Float64(data=float(heading_error)))
        self.speed_err_pub.publish(Float64(data=float(speed_error)))

    def bottom_camera_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        h, w, _ = frame.shape
        corners, ids, _ = self.aruco_detector.detectMarkers(frame)

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        aruco_locked = False

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))

            for idx, marker_id_arr in enumerate(ids):
                marker_id = int(marker_id_arr[0])
                if marker_id not in self.landmarks:
                    continue

                lm = self.landmarks[marker_id]
                pts = corners[idx][0]
                cx = float(np.mean(pts[:, 0]))
                cy = float(np.mean(pts[:, 1]))

                cv2.circle(frame, (int(cx), int(cy)), 6, (0, 0, 255), -1)
                cv2.line(frame, (int(cx) - 15, int(cy)), (int(cx) + 15, int(cy)), (0, 255, 255), 2)
                cv2.line(frame, (int(cx), int(cy) - 15), (int(cx), int(cy) + 15), (0, 255, 255), 2)

                focal_px = (w * 0.5) / math.tan(self.bottom_cam_fov * 0.5)
                dx_cam_body = -((cy - (h * 0.5)) / focal_px) * self.bottom_cam_height
                dy_cam_body = -((cx - (w * 0.5)) / focal_px) * self.bottom_cam_height

                marker_rel_x = self.bottom_cam_offset_x + dx_cam_body
                marker_rel_y = dy_cam_body

                top_vec = pts[1] - pts[0]
                marker_angle_img = math.atan2(top_vec[1], top_vec[0])
                yaw_err_cam = (
                    math.atan2(math.sin(4.0 * marker_angle_img), math.cos(4.0 * marker_angle_img))
                    / 4.0
                )
                z_yaw = normalize_angle(self.target_yaw + yaw_err_cam)

                z_x = lm["x"] - (
                    marker_rel_x * math.cos(self.ekf_yaw)
                    - marker_rel_y * math.sin(self.ekf_yaw)
                )
                z_y = lm["y"] - (
                    marker_rel_x * math.sin(self.ekf_yaw)
                    + marker_rel_y * math.cos(self.ekf_yaw)
                )

                # --- EKF Measurement Update (Absolute Correction from ArUco Landmark) ---
                pre_ekf_err = math.hypot(self.ekf_x - self.gt_x, self.ekf_y - self.gt_y)
                is_first_frame_of_marker = (self.last_detected_id != marker_id) or (
                    (now_sec - self.last_detection_time) > 1.5
                )

                z = np.array([z_x, z_y, z_yaw])
                x_pred = np.array([self.ekf_x, self.ekf_y, self.ekf_yaw])
                y_innov = z - x_pred
                y_innov[2] = normalize_angle(y_innov[2])

                H = np.eye(3)
                S = H @ self.P @ H.T + self.R_aruco
                K = self.P @ H.T @ np.linalg.inv(S)

                x_new = x_pred + K @ y_innov
                self.last_correction_magnitude = float(
                    np.linalg.norm([z_x - self.odom_x, z_y - self.odom_y])
                )
                self.ekf_x = float(x_new[0])
                self.ekf_y = float(x_new[1])
                self.ekf_yaw = normalize_angle(float(x_new[2]))
                self.P = (np.eye(3) - K @ H) @ self.P

                self.last_detected_id = marker_id
                self.last_detection_time = now_sec
                self.dist_since_last_aruco = 0.0
                self.total_aruco_corrections += 1
                aruco_locked = True

                post_ekf_err = math.hypot(self.ekf_x - self.gt_x, self.ekf_y - self.gt_y)
                if is_first_frame_of_marker:
                    self.get_logger().info(
                        f"[ArUco #{marker_id:02d} Lock @ Y={lm['y']:.1f}m] "
                        f"Raw Odom Err: {self.raw_err_current*100:.2f}cm (RMSE: {self.raw_rmse*100:.2f}cm) | "
                        f"EKF Err: {pre_ekf_err*100:.2f}cm -> {post_ekf_err*100:.2f}cm (RMSE: {self.ekf_rmse*100:.2f}cm)"
                    )

                label = f"ArUco ID:{marker_id} ({lm['lane'].upper()}) Map:({lm['x']:.2f}, {lm['y']:.1f}m)"
                cv2.putText(
                    frame,
                    label,
                    (12, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 120),
                    2,
                    cv2.LINE_AA,
                )
                break

        # Top HUD Banner with EKF + PID Status
        cv2.rectangle(frame, (0, 0), (w, 86), (20, 20, 25), -1)
        if aruco_locked or (now_sec - self.last_detection_time < 0.8):
            status_txt = f"MODE: ARUCO ABSOLUTE LOCK [ID #{self.last_detected_id}]"
            color = (60, 255, 100)
        else:
            status_txt = (
                f"MODE: IMU+ENC ODOM ({self.dist_since_last_aruco:.2f}m / 5.0m)"
            )
            color = (0, 200, 255)

        cv2.putText(
            frame, status_txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA
        )
        pose_txt = (
            f"EKF Err:{self.ekf_err_current * 100:.1f}cm (RMSE:{self.ekf_rmse * 100:.1f}cm) | "
            f"RawOdom Err:{self.raw_err_current * 100:.1f}cm (RMSE:{self.raw_rmse * 100:.1f}cm)"
        )
        cv2.putText(
            frame,
            pose_txt,
            (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        pid_txt = (
            f"PID Control | CTE:{self.last_cte_error * 100:+.1f}cm "
            f"YawErr:{math.degrees(self.last_heading_error):+.1f}deg -> "
            f"v:{self.last_cmd_v:.2f}m/s w:{self.last_cmd_w:+.2f}rad/s"
        )
        cv2.putText(
            frame,
            pid_txt,
            (10, 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (120, 220, 255),
            1,
            cv2.LINE_AA,
        )

        out_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        out_msg.header = msg.header
        self.aruco_img_pub.publish(out_msg)

    def broadcast_tf_and_odom(self):
        now = self.get_clock().now().to_msg()

        # 1. odom -> base_link (pure IMU + Wheel Encoder odometry)
        t_odom_base = TransformStamped()
        t_odom_base.header.stamp = now
        t_odom_base.header.frame_id = "odom"
        t_odom_base.child_frame_id = "base_link"
        t_odom_base.transform.translation.x = self.odom_x
        t_odom_base.transform.translation.y = self.odom_y
        t_odom_base.transform.translation.z = 0.0
        qx, qy, qz, qw = yaw_to_quat(self.odom_yaw)
        t_odom_base.transform.rotation.x = qx
        t_odom_base.transform.rotation.y = qy
        t_odom_base.transform.rotation.z = qz
        t_odom_base.transform.rotation.w = qw

        # 2. map -> odom (ArUco EKF correction transform so map -> base_link == ekf_pose)
        dyaw = normalize_angle(self.ekf_yaw - self.odom_yaw)
        dx = self.ekf_x - (self.odom_x * math.cos(dyaw) - self.odom_y * math.sin(dyaw))
        dy = self.ekf_y - (self.odom_x * math.sin(dyaw) + self.odom_y * math.cos(dyaw))

        t_map_odom = TransformStamped()
        t_map_odom.header.stamp = now
        t_map_odom.header.frame_id = "map"
        t_map_odom.child_frame_id = "odom"
        t_map_odom.transform.translation.x = dx
        t_map_odom.transform.translation.y = dy
        t_map_odom.transform.translation.z = 0.0
        mqx, mqy, mqz, mqw = yaw_to_quat(dyaw)
        t_map_odom.transform.rotation.x = mqx
        t_map_odom.transform.rotation.y = mqy
        t_map_odom.transform.rotation.z = mqz
        t_map_odom.transform.rotation.w = mqw

        self.tf_broadcaster.sendTransform([t_map_odom, t_odom_base])

        # Publish /ekf_odom in map frame
        ekf_odom = Odometry()
        ekf_odom.header.stamp = now
        ekf_odom.header.frame_id = "map"
        ekf_odom.child_frame_id = "base_link"
        ekf_odom.pose.pose.position.x = self.ekf_x
        ekf_odom.pose.pose.position.y = self.ekf_y
        ekf_odom.pose.pose.position.z = 0.0
        eqx, eqy, eqz, eqw = yaw_to_quat(self.ekf_yaw)
        ekf_odom.pose.pose.orientation.x = eqx
        ekf_odom.pose.pose.orientation.y = eqy
        ekf_odom.pose.pose.orientation.z = eqz
        ekf_odom.pose.pose.orientation.w = eqw
        ekf_odom.pose.covariance[0] = float(self.P[0, 0])
        ekf_odom.pose.covariance[7] = float(self.P[1, 1])
        ekf_odom.pose.covariance[35] = float(self.P[2, 2])
        ekf_odom.twist.twist.linear.x = self.current_vx
        ekf_odom.twist.twist.angular.z = self.current_wz
        self.ekf_odom_pub.publish(ekf_odom)

        # Publish /wheel_imu_odom in map frame for comparison
        raw_odom = Odometry()
        raw_odom.header.stamp = now
        raw_odom.header.frame_id = "map"
        raw_odom.child_frame_id = "base_link"
        raw_odom.pose.pose.position.x = self.odom_x
        raw_odom.pose.pose.position.y = self.odom_y
        raw_odom.pose.pose.orientation.x = qx
        raw_odom.pose.pose.orientation.y = qy
        raw_odom.pose.pose.orientation.z = qz
        raw_odom.pose.pose.orientation.w = qw
        self.raw_odom_pub.publish(raw_odom)

    def publish_visualization(self):
        now_clock = self.get_clock().now()
        now = now_clock.to_msg()
        now_sec = now_clock.nanoseconds * 1e-9

        self.broadcast_tf_and_odom()

        # Append to EKF Path (Green) and Raw IMU+Encoder Path (Orange)
        ekf_pose = PoseStamped()
        ekf_pose.header.stamp = now
        ekf_pose.header.frame_id = "map"
        ekf_pose.pose.position.x = self.ekf_x
        ekf_pose.pose.position.y = self.ekf_y
        ekf_pose.pose.position.z = 0.05
        _, _, eqz, eqw = yaw_to_quat(self.ekf_yaw)
        ekf_pose.pose.orientation.z = eqz
        ekf_pose.pose.orientation.w = eqw
        self.ekf_path.header.stamp = now
        self.ekf_path.poses.append(ekf_pose)
        if len(self.ekf_path.poses) > 600:
            self.ekf_path.poses.pop(0)
        self.ekf_path_pub.publish(self.ekf_path)

        raw_pose = PoseStamped()
        raw_pose.header.stamp = now
        raw_pose.header.frame_id = "map"
        raw_pose.pose.position.x = self.odom_x
        raw_pose.pose.position.y = self.odom_y
        raw_pose.pose.position.z = 0.03
        _, _, qz, qw = yaw_to_quat(self.odom_yaw)
        raw_pose.pose.orientation.z = qz
        raw_pose.pose.orientation.w = qw
        self.raw_path.header.stamp = now
        self.raw_path.poses.append(raw_pose)
        if len(self.raw_path.poses) > 600:
            self.raw_path.poses.pop(0)
        self.raw_path_pub.publish(self.raw_path)

        if self.gt_received:
            gt_pose = PoseStamped()
            gt_pose.header.stamp = now
            gt_pose.header.frame_id = "map"
            gt_pose.pose.position.x = self.gt_x
            gt_pose.pose.position.y = self.gt_y
            gt_pose.pose.position.z = 0.04
            gt_pose.pose.orientation.w = 1.0
            self.gt_path.header.stamp = now
            self.gt_path.poses.append(gt_pose)
            if len(self.gt_path.poses) > 600:
                self.gt_path.poses.pop(0)
            self.gt_path_pub.publish(self.gt_path)

        ma = MarkerArray()
        recently_locked = (now_sec - self.last_detection_time) < 1.0

        # 1. All 24 ArUco Landmarks on the PGM Map
        for mid, lm in self.landmarks.items():
            is_active = recently_locked and (self.last_detected_id == mid)

            m = Marker()
            m.header.stamp = now
            m.header.frame_id = "map"
            m.ns = "aruco_landmarks"
            m.id = mid
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = lm["x"]
            m.pose.position.y = lm["y"]
            m.pose.position.z = 0.015
            m.pose.orientation.w = 1.0
            m.scale.x = 0.24 if is_active else 0.18
            m.scale.y = 0.24 if is_active else 0.18
            m.scale.z = 0.02
            if is_active:
                m.color.r, m.color.g, m.color.b, m.color.a = (0.1, 1.0, 0.2, 0.95)
            else:
                m.color.r, m.color.g, m.color.b, m.color.a = (0.0, 0.75, 1.0, 0.80)
            ma.markers.append(m)

            txt = Marker()
            txt.header.stamp = now
            txt.header.frame_id = "map"
            txt.ns = "aruco_labels"
            txt.id = 100 + mid
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = lm["x"] + 0.22
            txt.pose.position.y = lm["y"]
            txt.pose.position.z = 0.15
            txt.scale.z = 0.14
            txt.color.r, txt.color.g, txt.color.b, txt.color.a = (1.0, 1.0, 1.0, 0.9)
            txt.text = f"ArUco #{mid:02d}"
            ma.markers.append(txt)

        # 2. EKF Uncertainty Covariance Circle/Ellipse around Robot
        cov_m = Marker()
        cov_m.header.stamp = now
        cov_m.header.frame_id = "map"
        cov_m.ns = "ekf_uncertainty"
        cov_m.id = 500
        cov_m.type = Marker.CYLINDER
        cov_m.action = Marker.ADD
        cov_m.pose.position.x = self.ekf_x
        cov_m.pose.position.y = self.ekf_y
        cov_m.pose.position.z = 0.01
        cov_m.pose.orientation.w = 1.0
        sx = max(0.25, min(1.8, 3.5 * math.sqrt(abs(self.P[0, 0]))))
        sy = max(0.25, min(1.8, 3.5 * math.sqrt(abs(self.P[1, 1]))))
        cov_m.scale.x = sx
        cov_m.scale.y = sy
        cov_m.scale.z = 0.01
        if recently_locked:
            cov_m.color.r, cov_m.color.g, cov_m.color.b, cov_m.color.a = (0.1, 1.0, 0.3, 0.45)
        else:
            cov_m.color.r, cov_m.color.g, cov_m.color.b, cov_m.color.a = (1.0, 0.65, 0.0, 0.38)
        ma.markers.append(cov_m)

        # 3. Floating 3D Status HUD above the Rover in RViz2 (with live PID telemetry)
        hud = Marker()
        hud.header.stamp = now
        hud.header.frame_id = "map"
        hud.ns = "ekf_hud"
        hud.id = 501
        hud.type = Marker.TEXT_VIEW_FACING
        hud.action = Marker.ADD
        hud.pose.position.x = self.ekf_x
        hud.pose.position.y = self.ekf_y
        hud.pose.position.z = 1.55
        hud.scale.z = 0.16
        if recently_locked:
            hud.color.r, hud.color.g, hud.color.b, hud.color.a = (0.2, 1.0, 0.4, 1.0)
            hud.text = (
                f"ARUCO LOCK [ID #{self.last_detected_id:02d}] | "
                f"PID w={self.last_cmd_w:+.2f} rad/s, v={self.last_cmd_v:.2f} m/s"
            )
        else:
            hud.color.r, hud.color.g, hud.color.b, hud.color.a = (0.3, 0.9, 1.0, 1.0)
            hud.text = (
                f"IMU+ENC ODOM ({self.dist_since_last_aruco:.1f}m/5m) | "
                f"PID CTE:{self.last_cte_error * 100:+.1f}cm w:{self.last_cmd_w:+.2f}"
            )
        ma.markers.append(hud)

        # 4. Active Detection Ray from Bottom Camera to Ground ArUco Marker
        ray = Marker()
        ray.header.stamp = now
        ray.header.frame_id = "map"
        ray.ns = "aruco_detection_ray"
        ray.id = 502
        ray.type = Marker.LINE_STRIP
        if recently_locked and self.last_detected_id in self.landmarks:
            ray.action = Marker.ADD
            lm = self.landmarks[self.last_detected_id]
            p_cam = Point(
                x=self.ekf_x + self.bottom_cam_offset_x * math.cos(self.ekf_yaw),
                y=self.ekf_y + self.bottom_cam_offset_x * math.sin(self.ekf_yaw),
                z=self.bottom_cam_height,
            )
            p_tag = Point(x=lm["x"], y=lm["y"], z=0.01)
            ray.points = [p_cam, p_tag]
            ray.scale.x = 0.03
            ray.color.r, ray.color.g, ray.color.b, ray.color.a = (0.1, 1.0, 0.2, 0.95)
        else:
            ray.action = Marker.DELETE
        ma.markers.append(ray)

        self.marker_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = EkfArucoLocalizer()
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
