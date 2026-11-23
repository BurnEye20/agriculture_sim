#!/home/burneye228/ros2_ws/.venv/bin/python
"""
Realistic Wheel Odometry & IMU Drift/Noise Node for Agricultural Terrain
========================================================================
Models realistic 4-wheel agricultural rover dead-reckoning errors over uneven soil:
  1. Longitudinal wheel-slip drift: +2.5% baseline on the 2.0 deg sloped field,
     increasing to +4.5% (3-5% drift) over the 65 field bumps (detected live from
     IMU vertical acceleration & roll/pitch jolts).
  2. Integrates dead-reckoning heading from the noisy IMU gyroscope (NOT skid-steer
     wheel difference, which falsely overestimates turns by 2x), with realistic MEMS
     gyro bias (~0.0007 rad/m) and gentle bump yaw jolts so raw odometry drifts
     smoothly and physically by 3-5% across the greenhouse instead of spinning wildly.
  3. Resets dead-reckoning origin cleanly when a new Ground Station mission starts.
"""

import math
import random

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import String


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_to_quaternion(yaw: float):
    half = 0.5 * yaw
    return 0.0, 0.0, math.sin(half), math.cos(half)


def quat_to_yaw(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class WheelOdomNoiseNode(Node):
    def __init__(self):
        super().__init__("wheel_odom_noise_node")

        # Physically realistic agricultural terrain drift (3-5% over bumps & 2 deg slope)
        self.declare_parameter("base_scale_drift", 0.025)          # +2.5% baseline slip on 2 deg slope
        self.declare_parameter("bump_extra_scale_drift", 0.020)    # +2.0% extra over bumps -> 4.5% total
        self.declare_parameter("slope_lateral_drift_rate", -0.003) # Gentle downhill cross-slope creep
        self.declare_parameter("yaw_drift_per_meter", 0.0007)      # Realistic MEMS gyro/traction drift (~1.4 deg / 35 m)
        self.declare_parameter("bump_yaw_slip_std", 0.007)         # Gentle asymmetric yaw jolt over bumps [rad/s]
        self.declare_parameter("linear_vel_noise_std", 0.012)      # Gaussian encoder noise [m/s]
        self.declare_parameter("angular_vel_noise_std", 0.003)     # Gaussian MEMS gyro noise [rad/s]
        self.declare_parameter("bump_accel_threshold", 0.32)       # IMU |az - g| bump detection threshold [m/s^2]
        self.declare_parameter("random_seed", 42)

        self.base_scale_drift = float(self.get_parameter("base_scale_drift").value)
        self.bump_extra_scale_drift = float(self.get_parameter("bump_extra_scale_drift").value)
        self.slope_lat_drift = float(self.get_parameter("slope_lateral_drift_rate").value)
        self.yaw_drift_per_m = float(self.get_parameter("yaw_drift_per_meter").value)
        self.bump_yaw_slip_std = float(self.get_parameter("bump_yaw_slip_std").value)
        self.lin_noise_std = float(self.get_parameter("linear_vel_noise_std").value)
        self.ang_noise_std = float(self.get_parameter("angular_vel_noise_std").value)
        self.bump_thresh = float(self.get_parameter("bump_accel_threshold").value)

        seed = int(self.get_parameter("random_seed").value)
        self.rng = random.Random(seed)

        # Dead-reckoning state initialized at Left Lane start (-0.44, -1.75, yaw = pi/2)
        self.dr_x = -0.4400
        self.dr_y = -1.7500
        self.dr_yaw = math.pi / 2.0
        self.dr_initialized_from_gt = False
        self.last_gt_pose: tuple[float, float, float] | None = None
        self.last_odom_time = None

        # Live IMU yaw rate (replaces erroneous 4-wheel skid-steer DiffDrive angular.z)
        self.latest_noisy_imu_wz = 0.0
        self.imu_received = False

        # Bump state detected from IMU vertical acceleration & roll/pitch rate
        self.over_bump = False
        self.bump_intensity = 0.0
        self.bump_count = 0
        self.current_vx = 0.0

        # RViz Path accumulators
        self.raw_path = Path()
        self.raw_path.header.frame_id = "map"
        self.ekf_path = Path()
        self.ekf_path.header.frame_id = "map"
        self.gt_path = Path()
        self.gt_path.header.frame_id = "map"
        self.max_path_poses = 2500

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, "/rover/wheel_odom", 20)
        self.imu_pub = self.create_publisher(Imu, "/rover/imu_noisy", 20)
        self.raw_path_pub = self.create_publisher(Path, "/odom_imu_encoder_path", 10)
        self.ekf_path_pub = self.create_publisher(Path, "/ekf_path", 10)
        self.gt_path_pub = self.create_publisher(Path, "/ground_truth_path", 10)

        # Subscribers
        self.create_subscription(
            Odometry, "/rover/wheel_odom_raw", self.raw_odom_callback, 20
        )
        self.create_subscription(Imu, "/rover/imu", self.raw_imu_callback, 30)
        self.create_subscription(Odometry, "/odometry/filtered", self.ekf_callback, 10)
        self.create_subscription(
            Odometry, "/rover/ground_truth_odom", self.gt_callback, 10
        )
        self.create_subscription(
            String, "/ground_station/mission_cmd", self.mission_cmd_callback, 10
        )

        self.get_logger().info(
            f"Wheel Odom Noise & Drift Node started (IMU-gyro + wheel-slip DR): "
            f"base_drift={self.base_scale_drift*100:.1f}%, "
            f"bump_drift={(self.base_scale_drift + self.bump_extra_scale_drift)*100:.1f}%, "
            f"yaw_drift={self.yaw_drift_per_m:.4f} rad/m."
        )

    def mission_cmd_callback(self, msg: String):
        """Reset dead-reckoning origin cleanly when starting a fresh Ground Station mission."""
        if msg.data.strip().lower() == "reset":
            if self.last_gt_pose is not None:
                self.dr_x, self.dr_y, self.dr_yaw = self.last_gt_pose
            else:
                self.dr_x, self.dr_y, self.dr_yaw = -0.4400, -1.7500, math.pi / 2.0
            self.raw_path.poses.clear()
            self.ekf_path.poses.clear()
            self.gt_path.poses.clear()
            self.get_logger().info(
                f"Reset raw dead-reckoning origin to ({self.dr_x:.2f}, {self.dr_y:.2f}m, {math.degrees(self.dr_yaw):.1f}deg)."
            )

    def raw_imu_callback(self, msg: Imu):
        """Detect soil bumps from vertical acceleration & roll/pitch jolts and corrupt IMU yaw rate."""
        az = msg.linear_acceleration.z
        wx = msg.angular_velocity.x
        wy = msg.angular_velocity.y
        wz = msg.angular_velocity.z

        vert_jolt = abs(az - 9.794)
        rp_rate = math.hypot(wx, wy)
        moving = abs(self.current_vx) > 0.02

        if moving and (vert_jolt > self.bump_thresh or rp_rate > 0.065):
            if not self.over_bump:
                self.bump_count += 1
            self.over_bump = True
            self.bump_intensity = min(2.0, max(vert_jolt / self.bump_thresh, rp_rate / 0.065))
        else:
            self.over_bump = False
            self.bump_intensity = max(0.0, self.bump_intensity * 0.85)

        speed = abs(self.current_vx)
        if speed > 0.008 or abs(wz) > 0.008:
            sys_wz_drift = speed * self.yaw_drift_per_m
            bump_wz_slip = (
                self.rng.gauss(0.0, self.bump_yaw_slip_std * self.bump_intensity)
                if self.over_bump
                else 0.0
            )
            gauss_wz = self.rng.gauss(0.0, self.ang_noise_std)
            noisy_wz = float(wz + sys_wz_drift + bump_wz_slip + gauss_wz)
        else:
            noisy_wz = float(wz)

        self.latest_noisy_imu_wz = noisy_wz
        self.imu_received = True

        noisy_imu = Imu()
        noisy_imu.header = msg.header
        noisy_imu.orientation = msg.orientation
        noisy_imu.orientation_covariance = msg.orientation_covariance
        noisy_imu.angular_velocity.x = wx
        noisy_imu.angular_velocity.y = wy
        noisy_imu.angular_velocity.z = noisy_wz
        noisy_imu.angular_velocity_covariance = list(msg.angular_velocity_covariance)
        noisy_imu.angular_velocity_covariance[8] = 0.0012
        noisy_imu.linear_acceleration = msg.linear_acceleration
        noisy_imu.linear_acceleration_covariance = msg.linear_acceleration_covariance
        self.imu_pub.publish(noisy_imu)

    def raw_odom_callback(self, msg: Odometry):
        """Inject realistic 3-5% bump/slope slip into wheel odometry using IMU-assisted yaw rate."""
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_odom_time is None:
            self.last_odom_time = stamp_sec
            return

        dt = stamp_sec - self.last_odom_time
        self.last_odom_time = stamp_sec
        if dt <= 0.0 or dt > 0.5:
            return

        true_vx = msg.twist.twist.linear.x
        true_vy = msg.twist.twist.linear.y
        self.current_vx = true_vx

        # Use noisy IMU gyro yaw rate (avoids 2x skid-steer wheel-difference turn error)
        wz_source = self.latest_noisy_imu_wz if self.imu_received else msg.twist.twist.angular.z

        moving = abs(true_vx) > 0.005 or abs(wz_source) > 0.005
        if moving:
            # 2.5% baseline scale drift on 2 deg slope, rising to 4.5% over field bumps
            scale_drift = self.base_scale_drift + (
                self.bump_extra_scale_drift * min(1.3, self.bump_intensity)
                if self.over_bump
                else 0.0
            )
            noise_mult = 1.8 if self.over_bump else 1.0

            noisy_vx = true_vx * (1.0 + scale_drift) + self.rng.gauss(
                0.0, self.lin_noise_std * noise_mult
            )
            noisy_vy = (
                true_vy
                + abs(true_vx) * self.slope_lat_drift
                + self.rng.gauss(0.0, 0.35 * self.lin_noise_std * noise_mult)
            )
            noisy_wz = wz_source
        else:
            noisy_vx = 0.0
            noisy_vy = 0.0
            noisy_wz = 0.0

        # Integrate smooth, physically authentic dead-reckoning pose (no ArUco resets)
        mid_yaw = self.dr_yaw + 0.5 * noisy_wz * dt
        self.dr_x += (noisy_vx * math.cos(mid_yaw) - noisy_vy * math.sin(mid_yaw)) * dt
        self.dr_y += (noisy_vx * math.sin(mid_yaw) + noisy_vy * math.cos(mid_yaw)) * dt
        self.dr_yaw = normalize_angle(self.dr_yaw + noisy_wz * dt)

        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = "odom"
        out.child_frame_id = "base_link"
        out.pose.pose.position.x = float(self.dr_x)
        out.pose.pose.position.y = float(self.dr_y)
        out.pose.pose.position.z = 0.0
        qx, qy, qz, qw = yaw_to_quaternion(self.dr_yaw)
        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw

        out.twist.twist.linear.x = float(noisy_vx)
        out.twist.twist.linear.y = float(noisy_vy)
        out.twist.twist.angular.z = float(noisy_wz)

        pose_cov = [0.0] * 36
        pose_cov[0] = 0.03
        pose_cov[7] = 0.03
        pose_cov[35] = 0.02
        out.pose.covariance = pose_cov

        twist_cov = [0.0] * 36
        twist_cov[0] = 0.005 if self.over_bump else 0.002
        twist_cov[7] = 0.005 if self.over_bump else 0.002
        twist_cov[35] = 0.003 if self.over_bump else 0.001
        out.twist.covariance = twist_cov

        self.odom_pub.publish(out)

        self._append_pose_to_path(
            self.raw_path, self.raw_path_pub, msg.header.stamp, self.dr_x, self.dr_y, self.dr_yaw
        )

    def ekf_callback(self, msg: Odometry):
        yaw = quat_to_yaw(msg.pose.pose.orientation)
        self._append_pose_to_path(
            self.ekf_path,
            self.ekf_path_pub,
            msg.header.stamp,
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            yaw,
        )

    def gt_callback(self, msg: Odometry):
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        yaw = quat_to_yaw(msg.pose.pose.orientation)
        self.last_gt_pose = (x, y, yaw)
        if not self.dr_initialized_from_gt:
            self.dr_x, self.dr_y, self.dr_yaw = x, y, yaw
            self.dr_initialized_from_gt = True
        self._append_pose_to_path(
            self.gt_path, self.gt_path_pub, msg.header.stamp, x, y, yaw
        )

    def _append_pose_to_path(
        self, path_msg: Path, pub, stamp, x: float, y: float, yaw: float
    ):
        if path_msg.poses:
            last = path_msg.poses[-1].pose.position
            if math.hypot(x - last.x, y - last.y) < 0.04:
                return
        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = "map"
        ps.pose.position.x = float(x)
        ps.pose.position.y = float(y)
        ps.pose.position.z = 0.02
        qx, qy, qz, qw = yaw_to_quaternion(yaw)
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        path_msg.poses.append(ps)
        if len(path_msg.poses) > self.max_path_poses:
            path_msg.poses.pop(0)
        path_msg.header.stamp = stamp
        pub.publish(path_msg)


def main(args=None):
    rclpy.init(args=args)
    node = WheelOdomNoiseNode()
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
