#!/home/burneye228/ros2_ws/.venv/bin/python
"""
Nav2 Multi-Row Agricultural Inspection Mission Node
===================================================
Uses the standard Nav2 `NavigateToPose` action server (/navigate_to_pose)
to navigate the rover across all 3 lanes and return to the starting point:
  1. Left Lane North       -> (-0.44, 34.00, +90 deg)  [ArUco #07]
  2. North Headland East   -> ( 1.40, 34.00, -90 deg)  [ArUco #15, 1st Rotation into Center Lane]
  3. Center Lane South     -> ( 1.40, -1.75,   0 deg)  [ArUco #08]
  4. South Headland East   -> ( 3.24, -1.75, +90 deg)  [ArUco #16, 2nd Rotation into Right Lane]
  5. Right Lane North      -> ( 3.24, 34.00, 180 deg)  [ArUco #23]
  6. Return Headland West  -> (-0.44, 34.00, -90 deg)  [ArUco #07]
  7. Return to Start Point -> (-0.44, -1.75, +90 deg)  [ArUco #00]
Also publishes /ekf_path, /odom_imu_encoder_path, and /ground_truth_path for RViz.
"""

import math

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node


def yaw_to_quaternion(yaw: float):
    half = 0.5 * yaw
    return 0.0, 0.0, math.sin(half), math.cos(half)


class Nav2RowMissionNode(Node):
    def __init__(self):
        super().__init__("nav2_row_mission")

        self.nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

        self.waypoints = [
            ("1. Left Lane North (ArUco #07)", -0.44, 34.00, math.pi / 2.0),
            ("2. 1st Rotation -> Center Lane Headland (ArUco #15)", 1.40, 34.00, -math.pi / 2.0),
            ("3. Center Lane South (ArUco #08)", 1.40, -1.75, 0.0),
            ("4. 2nd Rotation -> Right Lane Headland (ArUco #16)", 3.24, -1.75, math.pi / 2.0),
            ("5. Right Lane North (ArUco #23)", 3.24, 34.00, math.pi),
            ("6. Return Headland West (ArUco #07)", -0.44, 34.00, -math.pi / 2.0),
            ("7. Return to Starting Point (ArUco #00)", -0.44, -1.75, math.pi / 2.0),
        ]
        self.wp_idx = 0
        self.goal_active = False
        self.mission_done = False

        # RViz trajectory paths
        self.ekf_path = Path()
        self.ekf_path.header.frame_id = "map"
        self.raw_path = Path()
        self.raw_path.header.frame_id = "map"
        self.gt_path = Path()
        self.gt_path.header.frame_id = "map"

        self.ekf_path_pub = self.create_publisher(Path, "/ekf_path", 10)
        self.raw_path_pub = self.create_publisher(Path, "/odom_imu_encoder_path", 10)
        self.gt_path_pub = self.create_publisher(Path, "/ground_truth_path", 10)

        self.create_subscription(Odometry, "/odometry/filtered", self.ekf_callback, 10)
        self.create_subscription(Odometry, "/rover/wheel_odom", self.wheel_odom_callback, 10)
        self.create_subscription(Odometry, "/rover/ground_truth_odom", self.gt_odom_callback, 10)

        self.create_timer(1.0, self.mission_tick)
        self.get_logger().info("Nav2 Row Mission Node initialized. Waiting for /navigate_to_pose action server...")

    def mission_tick(self):
        if self.mission_done or self.goal_active:
            return

        if not self.nav_client.server_is_ready():
            return

        if self.wp_idx >= len(self.waypoints):
            self.mission_done = True
            self.get_logger().info("All Nav2 multi-row waypoints completed! Rover returned to start.")
            return

        desc, gx, gy, gyaw = self.waypoints[self.wp_idx]
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(gx)
        goal_msg.pose.pose.position.y = float(gy)
        goal_msg.pose.pose.position.z = 0.0
        qx, qy, qz, qw = yaw_to_quaternion(gyaw)
        goal_msg.pose.pose.orientation.x = qx
        goal_msg.pose.pose.orientation.y = qy
        goal_msg.pose.pose.orientation.z = qz
        goal_msg.pose.pose.orientation.w = qw

        self.goal_active = True
        self.get_logger().info(
            f"[Nav2 Goal {self.wp_idx + 1}/{len(self.waypoints)}] Sending '{desc}' -> "
            f"({gx:.2f}, {gy:.2f}m, yaw={math.degrees(gyaw):.1f}deg)"
        )
        send_future = self.nav_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Nav2 goal rejected, retrying on next tick...")
            self.goal_active = False
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        status = future.result().status
        desc, gx, gy, _ = self.waypoints[self.wp_idx]
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f"[Nav2 Goal Reached] Completed '{desc}' at ({gx:.2f}, {gy:.2f}m).")
            self.wp_idx += 1
        else:
            self.get_logger().warn(f"[Nav2 Goal Status={status}] Retrying '{desc}'...")
        self.goal_active = False

    def _append_path(self, path_msg: Path, pub, x: float, y: float, stamp):
        if len(path_msg.poses) == 0 or (
            math.hypot(x - path_msg.poses[-1].pose.position.x, y - path_msg.poses[-1].pose.position.y) > 0.08
        ):
            ps = PoseStamped()
            ps.header.stamp = stamp
            ps.header.frame_id = "map"
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            path_msg.header.stamp = stamp
            path_msg.poses.append(ps)
            if len(path_msg.poses) > 1500:
                path_msg.poses.pop(0)
            pub.publish(path_msg)

    def ekf_callback(self, msg: Odometry):
        self._append_path(
            self.ekf_path,
            self.ekf_path_pub,
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.header.stamp,
        )

    def wheel_odom_callback(self, msg: Odometry):
        ox = msg.pose.pose.position.x
        oy = msg.pose.pose.position.y
        self._append_path(
            self.raw_path,
            self.raw_path_pub,
            -0.44 - oy,
            -1.75 + ox,
            msg.header.stamp,
        )

    def gt_odom_callback(self, msg: Odometry):
        self._append_path(
            self.gt_path,
            self.gt_path_pub,
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.header.stamp,
        )


def main(args=None):
    rclpy.init(args=args)
    node = Nav2RowMissionNode()
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
