#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

NUMPY_MAJOR = int(np.__version__.split(".", maxsplit=1)[0])

if NUMPY_MAJOR < 2:
    try:
        from cv_bridge import CvBridge
    except Exception:  # pragma: no cover - runtime environment dependent
        CvBridge = None
else:  # pragma: no cover - runtime environment dependent
    CvBridge = None


def ros_time_to_float(msg: Image) -> float:
    return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) / 1_000_000_000.0


def convert_common_encoding(msg: Image) -> np.ndarray:
    data = np.frombuffer(msg.data, dtype=np.uint8)
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)
    row_data = data.reshape((height, step))
    encoding = msg.encoding.lower()

    if encoding == "rgb8":
        trimmed = row_data[:, : width * 3]
        return cv2.cvtColor(trimmed.reshape((height, width, 3)), cv2.COLOR_RGB2BGR)
    if encoding == "bgr8":
        trimmed = row_data[:, : width * 3]
        return trimmed.reshape((height, width, 3))
    if encoding == "rgba8":
        trimmed = row_data[:, : width * 4]
        rgba = trimmed.reshape((height, width, 4))
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    if encoding == "bgra8":
        trimmed = row_data[:, : width * 4]
        bgra = trimmed.reshape((height, width, 4))
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    if encoding == "mono8":
        trimmed = row_data[:, :width]
        return cv2.cvtColor(trimmed.reshape((height, width)), cv2.COLOR_GRAY2BGR)

    raise ValueError(f"Unsupported encoding without cv_bridge: {msg.encoding}")


class TimedVideoRecorder(Node):
    def __init__(self, topic: str, output_path: Path, duration: float, fps: float) -> None:
        super().__init__("record_ros_camera_video")
        self._topic = topic
        self._output_path = output_path
        self._duration = duration
        self._fps = fps
        self._bridge = CvBridge() if CvBridge is not None else None
        self._writer: cv2.VideoWriter | None = None
        self._first_frame_monotonic: float | None = None
        self._first_stamp: float | None = None
        self._last_stamp: float | None = None
        self._frame_count = 0
        self._done = False
        self._encoding = ""
        self._shape: tuple[int, int] | None = None
        self.create_subscription(Image, topic, self._callback, 10)

    @property
    def done(self) -> bool:
        return self._done

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def finalize(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def summary(self) -> str:
        if self._shape is None:
            return "No frames recorded."

        height, width = self._shape
        duration_ros = None
        if self._first_stamp is not None and self._last_stamp is not None:
            duration_ros = max(0.0, self._last_stamp - self._first_stamp)

        lines = [
            f"Saved video: {self._output_path}",
            f"width={width}",
            f"height={height}",
            f"encoding={self._encoding}",
            f"frames={self._frame_count}",
            f"writer_fps={self._fps}",
        ]
        if duration_ros is not None:
            lines.append(f"ros_duration_s={duration_ros:.3f}")
            if duration_ros > 0.0 and self._frame_count > 1:
                lines.append(f"observed_fps={(self._frame_count - 1) / duration_ros:.3f}")
        return "\n".join(lines)

    def _convert(self, msg: Image) -> np.ndarray:
        if self._bridge is not None:
            return self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        return convert_common_encoding(msg)

    def _ensure_writer(self, frame: np.ndarray) -> None:
        if self._writer is not None:
            return

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        height, width = frame.shape[:2]
        self._shape = (height, width)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self._output_path), fourcc, self._fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {self._output_path}")

    def _callback(self, msg: Image) -> None:
        if self._done:
            return

        frame = self._convert(msg)
        self._ensure_writer(frame)

        now = time.monotonic()
        if self._first_frame_monotonic is None:
            self._first_frame_monotonic = now
            self._first_stamp = ros_time_to_float(msg)
            self._encoding = msg.encoding

        assert self._writer is not None
        self._writer.write(frame)
        self._frame_count += 1
        self._last_stamp = ros_time_to_float(msg)

        if now - self._first_frame_monotonic >= self._duration:
            self._done = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True, help="ROS 2 image topic to subscribe to")
    parser.add_argument("--out", required=True, help="MP4 output path")
    parser.add_argument("--duration", type=float, default=6.0, help="Seconds of video to record after the first frame")
    parser.add_argument("--fps", type=float, default=16.0, help="Output video FPS")
    parser.add_argument("--timeout", type=float, default=30.0, help="Maximum seconds to wait overall")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.out).expanduser().resolve()

    rclpy.init()
    node = TimedVideoRecorder(args.topic, output_path, args.duration, args.fps)
    deadline = time.monotonic() + args.timeout

    try:
        while rclpy.ok() and not node.done and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        node.finalize()
        print(node.summary())
        node.destroy_node()
        rclpy.shutdown()

    if not output_path.exists() or node.frame_count == 0:
        print(f"No video recorded on {args.topic} within {args.timeout} seconds.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
