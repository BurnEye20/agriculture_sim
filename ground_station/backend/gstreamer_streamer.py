"""
GStreamer 1.24 Live Video Streaming Engine for Rover Ground Station
=================================================================
Uses native GStreamer (gi.repository.Gst) pipelines to ingest live camera streams
from:
  1. Live ROS 2 / Gazebo camera topics (via GStreamer `appsrc -> videoconvert -> jpegenc -> appsink`)
  2. Custom GStreamer source pipelines (UDP/RTP, RTSP, V4L2, videotestsrc) for physical hardware
  3. Clean "Awaiting Live Camera Feed" slate (no fake/hardcoded footage) when Gazebo is offline.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import cv2
import numpy as np

try:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    GST_AVAILABLE = True
    GST_VERSION = Gst.version_string()
except Exception:
    GST_AVAILABLE = False
    GST_VERSION = "Unavailable (OpenCV fallback)"


DEFAULT_PIPELINES = {
    "left": "appsrc name=src is-live=true format=time ! videoconvert ! jpegenc quality=82 ! appsink name=sink emit-signals=false max-buffers=2 drop=true",
    "right": "appsrc name=src is-live=true format=time ! videoconvert ! jpegenc quality=82 ! appsink name=sink emit-signals=false max-buffers=2 drop=true",
    "bottom": "appsrc name=src is-live=true format=time ! videoconvert ! jpegenc quality=82 ! appsink name=sink emit-signals=false max-buffers=2 drop=true",
}

CHANNEL_TOPICS = {
    "left": "/rover/left_camera/detections/image_raw",
    "right": "/rover/right_camera/detections/image_raw",
    "bottom": "/rover/bottom_camera/aruco_detections/image_raw",
}


class GstChannel:
    """Single camera channel backed by a real GStreamer 1.24 pipeline."""

    def __init__(self, name: str, width: int = 640, height: int = 480):
        self.name = name
        self.width = width
        self.height = height
        self.lock = threading.Lock()
        self.latest_jpeg: bytes | None = None
        self.latest_bgr: np.ndarray | None = None
        self.last_ros_frame_ts = 0.0
        self.last_frame_ts = 0.0
        self.fps = 0.0
        self._frame_times: list[float] = []
        self.mode = "ros2_appsrc"  # "ros2_appsrc" | "custom_gst"
        self.pipeline_desc = DEFAULT_PIPELINES[name]
        self.custom_error: str | None = None

        self._pipeline = None
        self._appsrc = None
        self._appsink = None
        self._custom_thread: threading.Thread | None = None
        self._stop_custom = threading.Event()
        self._pts_ns = 0

        if GST_AVAILABLE:
            self._init_appsrc_pipeline()

    def _init_appsrc_pipeline(self):
        self._teardown_pipeline()
        caps_str = (
            f"video/x-raw,format=BGR,width={self.width},height={self.height},"
            f"framerate=15/1"
        )
        desc = (
            f"appsrc name=src is-live=true block=false format=time caps={caps_str} ! "
            f"videoconvert ! jpegenc quality=82 ! "
            f"appsink name=sink emit-signals=false sync=false max-buffers=2 drop=true"
        )
        try:
            pipe = Gst.parse_launch(desc)
            self._pipeline = pipe
            self._appsrc = pipe.get_by_name("src")
            self._appsink = pipe.get_by_name("sink")
            pipe.set_state(Gst.State.PLAYING)
            self.pipeline_desc = desc
            self.mode = "ros2_appsrc"
            self.custom_error = None
        except Exception as exc:
            self.custom_error = str(exc)
            self._pipeline = None

    def _teardown_pipeline(self):
        self._stop_custom.set()
        if self._custom_thread and self._custom_thread.is_alive():
            self._custom_thread.join(timeout=1.0)
        self._custom_thread = None
        if self._pipeline is not None:
            try:
                self._pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
        self._pipeline = None
        self._appsrc = None
        self._appsink = None

    def set_custom_pipeline(self, source_pipeline: str) -> tuple[bool, str]:
        cleaned = source_pipeline.strip()
        if not cleaned or cleaned.lower() == "ros2":
            if GST_AVAILABLE:
                self._init_appsrc_pipeline()
            return True, "Switched to ROS 2 -> GStreamer appsrc pipeline."

        if not GST_AVAILABLE:
            return False, "GStreamer Python bindings unavailable on host."

        self._teardown_pipeline()
        if "appsink" not in cleaned:
            full_desc = (
                f"{cleaned} ! videoconvert ! videoscale ! "
                f"video/x-raw,width={self.width},height={self.height} ! "
                f"jpegenc quality=80 ! appsink name=sink emit-signals=false sync=false max-buffers=2 drop=true"
            )
        else:
            full_desc = cleaned

        try:
            pipe = Gst.parse_launch(full_desc)
            sink = pipe.get_by_name("sink")
            if sink is None:
                raise RuntimeError("Pipeline must include an appsink named 'sink'.")
            ret = pipe.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                pipe.set_state(Gst.State.NULL)
                raise RuntimeError("GStreamer failed to transition pipeline to PLAYING.")

            self._pipeline = pipe
            self._appsink = sink
            self.pipeline_desc = full_desc
            self.mode = "custom_gst"
            self.custom_error = None
            self._stop_custom.clear()
            self._custom_thread = threading.Thread(
                target=self._pull_custom_loop, daemon=True
            )
            self._custom_thread.start()
            return True, f"Active GStreamer pipeline: {full_desc}"
        except Exception as exc:
            self.custom_error = str(exc)
            self._init_appsrc_pipeline()
            return False, f"Pipeline error: {exc}"

    def _pull_custom_loop(self):
        while not self._stop_custom.is_set() and self._appsink is not None:
            sample = self._appsink.emit("try-pull-sample", int(0.2 * Gst.SECOND))
            if sample is None:
                continue
            buf = sample.get_buffer()
            ok, map_info = buf.map(Gst.MapFlags.READ)
            if ok:
                jpeg_bytes = bytes(map_info.data)
                buf.unmap(map_info)
                arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                self._record_encoded(jpeg_bytes, bgr, from_ros=False)

    def push_bgr_frame(self, frame: np.ndarray, from_ros: bool = True):
        if self.mode == "custom_gst":
            return

        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)

        jpeg_bytes: bytes | None = None
        if GST_AVAILABLE and self._appsrc is not None and self._appsink is not None:
            try:
                raw_bytes = np.ascontiguousarray(frame).tobytes()
                gst_buf = Gst.Buffer.new_allocate(None, len(raw_bytes), None)
                gst_buf.fill(0, raw_bytes)
                gst_buf.pts = self._pts_ns
                gst_buf.duration = int(Gst.SECOND / 15)
                self._pts_ns += gst_buf.duration
                self._appsrc.emit("push-buffer", gst_buf)
                sample = self._appsink.emit("try-pull-sample", int(0.08 * Gst.SECOND))
                if sample is not None:
                    out_buf = sample.get_buffer()
                    ok, map_info = out_buf.map(Gst.MapFlags.READ)
                    if ok:
                        jpeg_bytes = bytes(map_info.data)
                        out_buf.unmap(map_info)
            except Exception:
                jpeg_bytes = None

        if jpeg_bytes is None:
            ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok:
                jpeg_bytes = enc.tobytes()

        if jpeg_bytes is not None:
            self._record_encoded(jpeg_bytes, frame, from_ros=from_ros)

    def _record_encoded(self, jpeg_bytes: bytes, bgr: np.ndarray | None, from_ros: bool):
        now = time.monotonic()
        with self.lock:
            self.latest_jpeg = jpeg_bytes
            if bgr is not None:
                self.latest_bgr = bgr
            self.last_frame_ts = now
            if from_ros:
                self.last_ros_frame_ts = now
                self._frame_times.append(now)
                cutoff = now - 2.0
                self._frame_times = [t for t in self._frame_times if t >= cutoff]
                if len(self._frame_times) >= 2:
                    span = self._frame_times[-1] - self._frame_times[0]
                    self.fps = round((len(self._frame_times) - 1) / max(0.05, span), 1)
            elif self.mode == "custom_gst":
                self._frame_times.append(now)
                cutoff = now - 2.0
                self._frame_times = [t for t in self._frame_times if t >= cutoff]
                if len(self._frame_times) >= 2:
                    span = self._frame_times[-1] - self._frame_times[0]
                    self.fps = round((len(self._frame_times) - 1) / max(0.05, span), 1)
            else:
                self.fps = 0.0

    def get_jpeg(self) -> bytes | None:
        with self.lock:
            return self.latest_jpeg

    def get_bgr(self) -> np.ndarray | None:
        with self.lock:
            return None if self.latest_bgr is None else self.latest_bgr.copy()

    def is_ros_live(self) -> bool:
        return (time.monotonic() - self.last_ros_frame_ts) < 2.5

    def status(self) -> dict[str, Any]:
        return {
            "channel": self.name,
            "mode": (
                "Custom GStreamer Pipeline"
                if self.mode == "custom_gst"
                else ("ROS 2 -> GStreamer Live" if self.is_ros_live() else "Awaiting Live Feed")
            ),
            "ros_live": self.is_ros_live(),
            "fps": self.fps,
            "resolution": f"{self.width}x{self.height}",
            "codec": "MJPEG (jpegenc)",
            "gst_version": GST_VERSION,
            "pipeline": self.pipeline_desc,
            "error": self.custom_error,
        }


class GStreamerManager:
    """Manages Left, Right, and Bottom GStreamer channels."""

    def __init__(self):
        self.channels = {
            "left": GstChannel("left", 640, 480),
            "right": GstChannel("right", 640, 480),
            "bottom": GstChannel("bottom", 640, 480),
        }
        self._stop = threading.Event()
        self._telemetry_ref: dict[str, Any] = {}
        self._standby_thread = threading.Thread(target=self._standby_loop, daemon=True)
        self._standby_thread.start()

    def set_telemetry_ref(self, telemetry_ref: dict[str, Any]):
        self._telemetry_ref = telemetry_ref

    def _render_waiting_slate(self, side: str, tick: int) -> np.ndarray:
        frame = np.full((480, 640, 3), (18, 24, 34), dtype=np.uint8)
        # Subtle minimalist grid
        for gx in range(0, 640, 40):
            cv2.line(frame, (gx, 0), (gx, 480), (26, 34, 46), 1)
        for gy in range(0, 480, 40):
            cv2.line(frame, (0, gy), (640, gy), (26, 34, 46), 1)

        dots = "." * ((tick % 3) + 1)
        cv2.putText(
            frame,
            f"AWAITING LIVE {side.upper()} CAMERA STREAM{dots}",
            (118, 224),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (226, 232, 240),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"ROS 2 topic: {CHANNEL_TOPICS[side]}",
            (132, 256),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (148, 163, 184),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"{GST_VERSION} pipeline ready | Launch agriculture_world.launch.xml",
            (115, 286),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (100, 116, 139),
            1,
            cv2.LINE_AA,
        )
        return frame

    def _standby_loop(self):
        tick = 0
        while not self._stop.is_set():
            for name, ch in self.channels.items():
                if ch.mode == "ros2_appsrc" and not ch.is_ros_live():
                    ch.push_bgr_frame(self._render_waiting_slate(name, tick), from_ros=False)
            tick += 1
            time.sleep(0.5)

    def get_all_status(self) -> dict[str, Any]:
        return {
            "gst_available": GST_AVAILABLE,
            "gst_version": GST_VERSION,
            "channels": {k: ch.status() for k, ch in self.channels.items()},
        }

    def shutdown(self):
        self._stop.set()
        if self._standby_thread.is_alive():
            self._standby_thread.join(timeout=1.0)
        for ch in self.channels.values():
            ch._teardown_pipeline()
