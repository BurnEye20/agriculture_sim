#!/usr/bin/env python3
"""
Camera Mast Controller (telescopic linear-actuator lifting column)
==================================================================
Single interface for setting the row-camera height. A future computer-vision
node only needs to publish the desired camera height; this node clamps it to the
mechanical range and converts it into one per-stage extension command that all
stages follow synchronously (like a real multi-stage lifting column driven by a
single linear motor).

Subscribes:
  /camera_mast/set_height       std_msgs/Float64  desired camera height above ground [m]
  /joint_states                 sensor_msgs/JointState (mast_stage2_joint .. mast_stageN_joint)
Publishes:
  /camera_mast/stage_cmd        std_msgs/Float64  per-stage extension -> Gazebo JointPositionControllers
  /camera_mast/current_height   std_msgs/Float64  measured camera height [m]
  /camera_mast/target_height    std_msgs/Float64  active (clamped) target [m]
  /camera_mast/in_position      std_msgs/Bool     True when within height_tolerance

Range and stage count come from config/camera_mast_derived.yaml (written by
scripts/generate_camera_mast.py); speed/startup values from config/camera_mast.yaml.
"""

import os

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64
import yaml


class CameraMastController(Node):
    def __init__(self):
        super().__init__("camera_mast_controller")

        cfg_dir = os.path.join(get_package_share_directory("agriculture_sim"), "config")
        with open(os.path.join(cfg_dir, "camera_mast.yaml"), "r") as fh:
            cfg = yaml.safe_load(fh)["camera_mast"]
        with open(os.path.join(cfg_dir, "camera_mast_derived.yaml"), "r") as fh:
            derived = yaml.safe_load(fh)["camera_mast_derived"]

        self.stage_joints = list(derived["stage_joints"])
        self.declare_parameter("min_camera_height", float(derived["min_camera_height"]))
        self.declare_parameter("max_camera_height", float(derived["max_camera_height"]))
        self.declare_parameter("default_camera_height", float(cfg["default_camera_height"]))
        self.declare_parameter("height_tolerance", float(cfg.get("height_tolerance", 0.005)))

        self.h_min = float(self.get_parameter("min_camera_height").value)
        self.h_max = float(self.get_parameter("max_camera_height").value)
        self.tol = float(self.get_parameter("height_tolerance").value)
        self.n_stages = len(self.stage_joints)

        self.target = self.clamp(float(self.get_parameter("default_camera_height").value))
        self.current = None
        self.was_in_position = False

        self.stage_pub = self.create_publisher(Float64, "/camera_mast/stage_cmd", 10)
        self.current_pub = self.create_publisher(Float64, "/camera_mast/current_height", 10)
        self.target_pub = self.create_publisher(Float64, "/camera_mast/target_height", 10)
        self.in_pos_pub = self.create_publisher(Bool, "/camera_mast/in_position", 10)

        self.create_subscription(Float64, "/camera_mast/set_height", self.set_height_cb, 10)
        self.create_subscription(JointState, "/joint_states", self.joint_state_cb, 20)

        # Re-send the target periodically so the command survives bridge/Gazebo restarts.
        self.create_timer(0.5, self.send_target)
        self.create_timer(0.1, self.publish_status)

        self.get_logger().info(
            f"{int(derived['num_sections'])}-section camera mast, range {self.h_min:.2f}-"
            f"{self.h_max:.2f} m, startup target {self.target:.3f} m. "
            f"Command height on /camera_mast/set_height (std_msgs/Float64)."
        )

    def clamp(self, h: float) -> float:
        return max(self.h_min, min(self.h_max, h))

    def set_height_cb(self, msg: Float64):
        requested = float(msg.data)
        self.target = self.clamp(requested)
        if abs(requested - self.target) > 1e-6:
            self.get_logger().warn(
                f"Requested camera height {requested:.3f} m outside "
                f"[{self.h_min:.2f}, {self.h_max:.2f}] m -> clamped to {self.target:.3f} m"
            )
        else:
            self.get_logger().info(f"New camera height target: {self.target:.3f} m")
        self.was_in_position = False
        self.send_target()

    def send_target(self):
        self.stage_pub.publish(Float64(data=(self.target - self.h_min) / self.n_stages))

    def joint_state_cb(self, msg: JointState):
        pos = dict(zip(msg.name, msg.position))
        if all(j in pos for j in self.stage_joints):
            self.current = self.h_min + sum(pos[j] for j in self.stage_joints)

    def publish_status(self):
        self.target_pub.publish(Float64(data=self.target))
        if self.current is None:
            return
        self.current_pub.publish(Float64(data=self.current))
        in_pos = abs(self.current - self.target) <= self.tol
        self.in_pos_pub.publish(Bool(data=in_pos))
        if in_pos and not self.was_in_position:
            self.get_logger().info(f"Camera mast in position: {self.current:.3f} m")
        self.was_in_position = in_pos


def main(args=None):
    rclpy.init(args=args)
    node = CameraMastController()
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
