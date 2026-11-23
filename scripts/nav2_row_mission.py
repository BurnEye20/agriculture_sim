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
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String


def yaw_to_quaternion(yaw: float):
    half = 0.5 * yaw
    return 0.0, 0.0, math.sin(half), math.cos(half)


class Nav2RowMissionNode(Node):
    def __init__(self):
        super().__init__("nav2_row_mission")

        self.declare_parameter("auto_start", True)
        auto_start = bool(self.get_parameter("auto_start").value)

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
        self.paused = not auto_start
        self.estop = False
        self.returning_to_base = False
        self.current_goal_handle = None

        # Ground Station mission command & status interface
        self.status_pub = self.create_publisher(String, "/ground_station/mission_status", 10)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(
            String, "/ground_station/mission_cmd", self.mission_cmd_callback, 10
        )

        self.create_timer(1.0, self.mission_tick)
        self.create_timer(0.5, self.publish_status)
        self.get_logger().info(
            f"Nav2 Row Mission Node initialized (auto_start={auto_start}). "
            "Listening for Ground Station commands on /ground_station/mission_cmd."
        )

    def publish_status(self):
        if self.estop:
            payload = "estop|Emergency Stop Active"
        elif self.mission_done:
            payload = "at_base|Returned to Base — Ready for Next Mission"
        elif self.paused:
            payload = "manual_mode|Manual Control Mode (Nav2 Paused)"
        elif self.returning_to_base or self.wp_idx >= 5:
            desc = (
                self.waypoints[self.wp_idx][0]
                if self.wp_idx < len(self.waypoints)
                else "Base (-0.44, -1.75)"
            )
            payload = f"returning_to_base|Returning to Base · {desc}"
        else:
            desc = (
                self.waypoints[self.wp_idx][0]
                if self.wp_idx < len(self.waypoints)
                else "Inspecting Rows"
            )
            payload = f"running_mission|Running Mission · WP {self.wp_idx + 1}/{len(self.waypoints)} ({desc})"
        self.status_pub.publish(String(data=payload))

    def _cancel_active_goal(self):
        if self.current_goal_handle is not None:
            try:
                self.current_goal_handle.cancel_goal_async()
            except Exception:
                pass
            self.current_goal_handle = None
        self.goal_active = False

    def mission_cmd_callback(self, msg: String):
        cmd = msg.data.strip().lower()
        self.get_logger().info(f"[Ground Station Command] '{cmd}'")
        if cmd in ("pause", "stop"):
            self.paused = True
            self.estop = False
            self.returning_to_base = False
            self._cancel_active_goal()
            self.cmd_vel_pub.publish(Twist())
        elif cmd == "estop":
            self.paused = True
            self.estop = True
            self.returning_to_base = False
            self._cancel_active_goal()
            self.cmd_vel_pub.publish(Twist())
        elif cmd == "release_estop":
            self.estop = False
            self.paused = True
        elif cmd in ("start", "resume"):
            self.estop = False
            if self.mission_done or self.returning_to_base:
                self._cancel_active_goal()
                self.wp_idx = 0
                self.mission_done = False
                self.returning_to_base = False
            self.paused = False
        elif cmd == "reset":
            self.estop = False
            self.returning_to_base = False
            self._cancel_active_goal()
            self.wp_idx = 0
            self.mission_done = False
            self.paused = False
        elif cmd == "rtb":
            self.estop = False
            self.returning_to_base = True
            self._cancel_active_goal()
            self.wp_idx = len(self.waypoints) - 1
            self.mission_done = False
            self.paused = False
        self.publish_status()

    def mission_tick(self):
        if self.paused or self.estop or self.mission_done or self.goal_active:
            return

        if not self.nav_client.server_is_ready():
            return

        if self.wp_idx >= len(self.waypoints):
            self.mission_done = True
            self.get_logger().info("All Nav2 multi-row waypoints completed! Rover returned to start.")
            self.publish_status()
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
            self.current_goal_handle = None
            return
        self.current_goal_handle = goal_handle
        if self.paused or self.estop:
            # Cancelled while waiting for acceptance
            self._cancel_active_goal()
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        status = future.result().status
        self.current_goal_handle = None
        self.goal_active = False
        if self.paused or self.estop or status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info("[Nav2 Goal Canceled] Yielding control to Ground Station operator.")
            return
        if self.wp_idx < len(self.waypoints):
            desc, gx, gy, _ = self.waypoints[self.wp_idx]
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info(f"[Nav2 Goal Reached] Completed '{desc}' at ({gx:.2f}, {gy:.2f}m).")
                self.wp_idx += 1
                if self.wp_idx >= len(self.waypoints):
                    self.mission_done = True
                    self.returning_to_base = False
                    self.get_logger().info("Returned to initial base (-0.44, -1.75m). Ready for next mission.")
                    self.publish_status()
            else:
                self.get_logger().warn(f"[Nav2 Goal Status={status}] Retrying '{desc}'...")


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
