#!/home/burneye228/ros2_ws/.venv/bin/python
"""ROS 2 node running YOLOv11 (best.pt) on both left and right rover cameras in Gazebo."""

import os
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO


DEFAULT_MODEL_PATH = (
    "/home/burneye228/ros2_ws/src/agriculture_sim/"
    "Final_Fruit_Dataset_YOLOv11/runs/detect/train4/weights/best.pt"
)


class YoloStrawberryDetector(Node):
    def __init__(self):
        super().__init__("yolo_strawberry_detector")

        self.declare_parameter("model_path", DEFAULT_MODEL_PATH)
        self.declare_parameter("conf_threshold", 0.25)

        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        self.conf_threshold = (
            self.get_parameter("conf_threshold").get_parameter_value().double_value
        )

        if not os.path.exists(model_path):
            self.get_logger().error(f"YOLOv11 weights not found at: {model_path}")
            raise FileNotFoundError(model_path)

        self.get_logger().info(f"Loading YOLOv11 model from: {model_path}")
        self.model = YOLO(model_path)
        self.bridge = CvBridge()

        # Subscribers for both opposite-facing rover cameras
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

        # Publishers for annotated YOLOv11 detection streams
        self.left_pub = self.create_publisher(
            Image, "/rover/left_camera/detections/image_raw", 10
        )
        self.right_pub = self.create_publisher(
            Image, "/rover/right_camera/detections/image_raw", 10
        )

        self.get_logger().info(
            "YOLOv11 Strawberry Detector ready for Left and Right cameras."
        )

    def run_detection(self, msg: Image, camera_label: str) -> Image:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        results = self.model.predict(
            source=frame,
            imgsz=640,
            conf=self.conf_threshold,
            verbose=False,
        )
        annotated = results[0].plot(line_width=2, font_size=12)

        num_dets = len(results[0].boxes) if results[0].boxes is not None else 0
        header_text = f"{camera_label} | Detections: {num_dets}"
        cv2.rectangle(annotated, (0, 0), (360, 32), (20, 20, 20), -1)
        cv2.putText(
            annotated,
            header_text,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (80, 255, 120),
            2,
            cv2.LINE_AA,
        )

        out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        out_msg.header = msg.header
        return out_msg

    def left_camera_callback(self, msg: Image):
        out_msg = self.run_detection(msg, "LEFT ROW CAM (YOLOv11)")
        self.left_pub.publish(out_msg)

    def right_camera_callback(self, msg: Image):
        out_msg = self.run_detection(msg, "RIGHT ROW CAM (YOLOv11)")
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
