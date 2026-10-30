#!/home/burneye228/ros2_ws/.venv/bin/python
"""
Compact ArUco Marker Localizer Node for robot_localization (ekf_node)
=====================================================================
- Detects ground ArUco markers (DICT_4X4_50) from the tilted upper-frame camera
  (/rover/bottom_camera/image_raw).
- Computes 3D camera-to-marker distance and perspective-rectified ground offset
  via 4-corner ray-plane homography + cv2.solvePnP (IPPE_SQUARE).
- Cross-references detected marker IDs with config/aruco_landmarks.yaml.
- Publishes geometry_msgs/msg/PoseWithCovarianceStamped on /aruco_pose (in 'map'
  frame) consumed by robot_localization's ekf_node.
- Publishes annotated tilted camera view on /rover/bottom_camera/aruco_detections/image_raw
  for RViz and Gazebo visualization.
"""

import math
import os

from ament_index_python.packages import get_package_share_directory
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
import yaml


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_to_quaternion(yaw: float):
    half = 0.5 * yaw
    return 0.0, 0.0, math.sin(half), math.cos(half)


def quaternion_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class ArucoLocalizerNode(Node):
    def __init__(self):
        super().__init__("aruco_localizer_node")

        self.declare_parameter("camera_height", 0.445)
        self.declare_parameter("camera_offset_x", 0.365)
        self.declare_parameter("camera_pitch", 0.8500)
        self.declare_parameter("camera_fov", 1.3500)
        self.declare_parameter("max_detection_dist_3d", 1.05)

        self.cam_height = self.get_parameter("camera_height").value
        self.cam_offset_x = self.get_parameter("camera_offset_x").value
        self.cam_pitch = self.get_parameter("camera_pitch").value
        self.cam_fov = self.get_parameter("camera_fov").value
        self.max_dist_3d = self.get_parameter("max_detection_dist_3d").value

        # Current EKF state from robot_localization (/odometry/filtered)
        self.ekf_x = -0.4400
        self.ekf_y = -1.7500
        self.ekf_yaw = math.pi / 2.0
        self.current_wz = 0.0
        self.last_detected_id = None
        self.last_dist_3d = 0.0
        self.last_rel_x = 0.0
        self.last_rel_y = 0.0

        # Load ArUco landmarks from config/aruco_landmarks.yaml
        pkg_share = get_package_share_directory("agriculture_sim")
        landmarks_path = os.path.join(pkg_share, "config", "aruco_landmarks.yaml")
        with open(landmarks_path, "r") as f:
            data = yaml.safe_load(f)

        self.marker_size_m = float(data.get("marker_size_m", 0.28))
        self.landmarks = {}
        for m in data.get("markers", []):
            pose = m["pose"]
            self.landmarks[int(m["id"])] = {
                "lane": m["lane"],
                "x": float(pose[0]),
                "y": float(pose[1]),
                "z": float(pose[2]),
                "yaw": float(pose[5]),
            }

        # OpenCV ArUco detector setup
        self.bridge = CvBridge()
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        else:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
            self.aruco_detector = None

        # Publishers
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, "/aruco_pose", 10)
        self.trigger_pub = self.create_publisher(PointStamped, "/aruco_trigger", 10)
        self.rviz_img_pub = self.create_publisher(
            Image, "/rover/bottom_camera/aruco_detections/image_raw", 10
        )
        self.annotated_img_pub = self.create_publisher(Image, "/aruco_detection/image_annotated", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/aruco_localization_markers", 10)

        # Subscribers
        self.create_subscription(Odometry, "/odometry/filtered", self.ekf_callback, 10)
        self.create_subscription(Image, "/rover/bottom_camera/image_raw", self.image_callback, 10)
        self.create_timer(0.25, self.publish_landmark_markers)

        self.get_logger().info(
            f"ArUco Localizer Node started with {len(self.landmarks)} landmarks "
            f"(marker_size={self.marker_size_m:.2f}m). Publishing /aruco_pose & RViz camera feed."
        )

    def ekf_callback(self, msg: Odometry):
        self.ekf_x = msg.pose.pose.position.x
        self.ekf_y = msg.pose.pose.position.y
        self.ekf_yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        self.current_wz = msg.twist.twist.angular.z

    def image_callback(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        h, w, _ = frame.shape

        if self.aruco_detector is not None:
            corners, ids, _ = self.aruco_detector.detectMarkers(frame)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                frame, self.aruco_dict, parameters=self.aruco_params
            )

        locked_now = False
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))
            focal_px = (w * 0.5) / math.tan(self.cam_fov * 0.5)
            cos_p = math.cos(self.cam_pitch)
            sin_p = math.sin(self.cam_pitch)
            h_eff = self.cam_height - 0.004

            half_s = self.marker_size_m * 0.5
            obj_pts = np.array(
                [
                    [-half_s, half_s, 0.0],
                    [half_s, half_s, 0.0],
                    [half_s, -half_s, 0.0],
                    [-half_s, -half_s, 0.0],
                ],
                dtype=np.float64,
            )
            cam_matrix = np.array(
                [[focal_px, 0.0, w * 0.5], [0.0, focal_px, h * 0.5], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)

            candidates = []
            for i, marker_id_arr in enumerate(ids):
                marker_id = int(marker_id_arr[0])
                if marker_id not in self.landmarks:
                    continue

                pts = corners[i][0]
                ground_corners = []
                valid_proj = True
                for u_px, v_px in pts:
                    xn = (float(u_px) - w * 0.5) / focal_px
                    yn = (float(v_px) - h * 0.5) / focal_px
                    rx = cos_p - yn * sin_p
                    ry = -xn
                    rz = -sin_p - yn * cos_p
                    if rz >= -0.05:
                        valid_proj = False
                        break
                    t_ray = h_eff / (-rz)
                    ground_corners.append((self.cam_offset_x + t_ray * rx, t_ray * ry))

                if not valid_proj or len(ground_corners) != 4:
                    continue

                x_rel_hom = sum(p[0] for p in ground_corners) * 0.25
                y_rel_hom = sum(p[1] for p in ground_corners) * 0.25

                ok_pnp, rvec, tvec = cv2.solvePnP(
                    obj_pts,
                    pts.astype(np.float64),
                    cam_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
                if ok_pnp:
                    tx_c, ty_c, tz_c = float(tvec[0][0]), float(tvec[1][0]), float(tvec[2][0])
                    x_rel_pnp = self.cam_offset_x + (tz_c * cos_p - ty_c * sin_p)
                    y_rel_pnp = -tx_c
                    marker_rel_x = 0.75 * x_rel_hom + 0.25 * x_rel_pnp
                    marker_rel_y = 0.75 * y_rel_hom + 0.25 * y_rel_pnp
                else:
                    rvec, tvec = None, None
                    marker_rel_x = x_rel_hom
                    marker_rel_y = y_rel_hom

                dx_cam = marker_rel_x - self.cam_offset_x
                dy_cam = marker_rel_y
                dist_3d = math.sqrt(dx_cam**2 + dy_cam**2 + h_eff**2)
                if dist_3d <= 2.10:
                    candidates.append(
                        {
                            "id": marker_id,
                            "pts": pts,
                            "ground_corners": ground_corners,
                            "marker_rel_x": marker_rel_x,
                            "marker_rel_y": marker_rel_y,
                            "dist_3d": dist_3d,
                            "rvec": rvec,
                            "tvec": tvec,
                        }
                    )

            candidates.sort(key=lambda c: c["dist_3d"])
            if candidates:
                cand = candidates[0]
                marker_id = cand["id"]
                lm = self.landmarks[marker_id]
                dist_3d = cand["dist_3d"]
                marker_rel_x = cand["marker_rel_x"]
                marker_rel_y = cand["marker_rel_y"]

                self.last_dist_3d = dist_3d
                self.last_rel_x = marker_rel_x
                self.last_rel_y = marker_rel_y

                if cand["rvec"] is not None and cand["tvec"] is not None:
                    cv2.drawFrameAxes(
                        frame, cam_matrix, dist_coeffs, cand["rvec"], cand["tvec"], 0.10, 2
                    )
                cx = int(np.mean(cand["pts"][:, 0]))
                cy = int(np.mean(cand["pts"][:, 1]))
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                # Publish trigger message for mission monitor
                trig = PointStamped()
                trig.header = msg.header
                trig.header.frame_id = "base_link"
                trig.point.x = float(marker_id)
                trig.point.y = float(dist_3d)
                trig.point.z = float(marker_rel_x)
                self.trigger_pub.publish(trig)

                # Publish PoseWithCovarianceStamped (/aruco_pose) for robot_localization ekf_node
                if dist_3d <= self.max_dist_3d and abs(marker_rel_y) <= 0.55 and abs(self.current_wz) < 0.35:
                    ground_corners = cand["ground_corners"]
                    edge_dx = ground_corners[1][0] - ground_corners[0][0]
                    edge_dy = ground_corners[1][1] - ground_corners[0][1]
                    # Directed vector P0 -> P1 of all markers points along +X_map (0 rad),
                    # so body edge angle = -robot_yaw over the full [-pi, pi] range.
                    measured_yaw = normalize_angle(-math.atan2(edge_dy, edge_dx))
                    yaw_innov = max(-0.12, min(0.12, normalize_angle(measured_yaw - self.ekf_yaw)))
                    z_yaw = normalize_angle(self.ekf_yaw + yaw_innov)

                    proj_yaw = normalize_angle(self.ekf_yaw + 0.5 * yaw_innov)
                    z_x = lm["x"] - (
                        marker_rel_x * math.cos(proj_yaw) - marker_rel_y * math.sin(proj_yaw)
                    )
                    z_y = lm["y"] - (
                        marker_rel_x * math.sin(proj_yaw) + marker_rel_y * math.cos(proj_yaw)
                    )

                    pose_msg = PoseWithCovarianceStamped()
                    pose_msg.header.stamp = msg.header.stamp
                    pose_msg.header.frame_id = "map"
                    pose_msg.pose.pose.position.x = float(z_x)
                    pose_msg.pose.pose.position.y = float(z_y)
                    pose_msg.pose.pose.position.z = 0.0
                    qx, qy, qz, qw = yaw_to_quaternion(z_yaw)
                    pose_msg.pose.pose.orientation.x = qx
                    pose_msg.pose.pose.orientation.y = qy
                    pose_msg.pose.pose.orientation.z = qz
                    pose_msg.pose.pose.orientation.w = qw

                    dist_scale = max(0.75, (dist_3d / 0.60) ** 2)
                    var_xy = 0.0009 * dist_scale
                    var_yaw = 0.0020 * dist_scale
                    cov = [0.0] * 36
                    cov[0] = var_xy
                    cov[7] = var_xy
                    cov[35] = var_yaw
                    pose_msg.pose.covariance = cov
                    self.pose_pub.publish(pose_msg)
                    locked_now = True

                    if self.last_detected_id != marker_id:
                        self.last_detected_id = marker_id
                        self.get_logger().info(
                            f"[ArUco #{marker_id:02d}] Published /aruco_pose=({z_x:.2f}, {z_y:.2f}m) | "
                            f"3D Dist={dist_3d:.2f}m (Fwd={marker_rel_x:.2f}m, Lat={marker_rel_y*100:+.1f}cm)"
                        )

                cv2.putText(
                    frame,
                    f"ArUco #{marker_id:02d} ({lm['lane'].upper()}) | 3D Dist:{dist_3d:.2f}m Fwd:{marker_rel_x:.2f}m Lat:{marker_rel_y*100:+.1f}cm",
                    (10, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (0, 255, 120),
                    2,
                    cv2.LINE_AA,
                )

        # Draw Top HUD Banner for RViz Tilted Camera View
        cv2.rectangle(frame, (0, 0), (w, 72), (22, 24, 28), -1)
        status_color = (50, 255, 100) if locked_now else (80, 210, 255)
        status_str = f"ARUCO #{self.last_detected_id:02d} LOCKED" if locked_now and self.last_detected_id is not None else "TILTED UPPER-FRAME ARUCO CAMERA (48.7 deg)"
        cv2.putText(
            frame,
            f"{status_str} | EKF=({self.ekf_x:.2f}, {self.ekf_y:.2f}m, {math.degrees(self.ekf_yaw):.1f}deg)",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            status_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Last Cam->ArUco 3D Dist: {self.last_dist_3d:.2f}m | Ground Fwd: {self.last_rel_x:.2f}m | Lat: {self.last_rel_y*100:+.1f}cm",
            (10, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        out_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        out_msg.header = msg.header
        self.rviz_img_pub.publish(out_msg)
        self.annotated_img_pub.publish(out_msg)

    def publish_landmark_markers(self):
        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for marker_id, lm in self.landmarks.items():
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = stamp
            m.ns = "aruco_landmarks"
            m.id = marker_id
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = lm["x"]
            m.pose.position.y = lm["y"]
            m.pose.position.z = 0.01
            m.pose.orientation.w = 1.0
            m.scale.x = self.marker_size_m
            m.scale.y = self.marker_size_m
            m.scale.z = 0.015
            if self.last_detected_id == marker_id:
                m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 1.0, 0.3, 0.95
            else:
                m.color.r, m.color.g, m.color.b, m.color.a = 0.95, 0.85, 0.15, 0.85
            ma.markers.append(m)
        self.marker_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoLocalizerNode()
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
