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
        return trimmed.reshape((height, width))

    raise ValueError(f"Unsupported encoding without cv_bridge: {msg.encoding}")


class OneShotFrameSaver(Node):
    def __init__(self, topic: str, output_path: Path) -> None:
        super().__init__("verify_ros_camera_frame")
        self._topic = topic
        self._output_path = output_path
        self._bridge = CvBridge() if CvBridge is not None else None
        self._saved = False
        self.create_subscription(Image, topic, self._callback, 10)

    @property
    def saved(self) -> bool:
        return self._saved

    def _callback(self, msg: Image) -> None:
        if self._saved:
            return

        try:
            if self._bridge is not None:
                frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            else:
                frame = convert_common_encoding(msg)
        except Exception as exc:
            self.get_logger().error(f"Failed to convert image: {exc}")
            raise

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(self._output_path), frame):
            raise RuntimeError(f"Failed to save image to {self._output_path}")

        timestamp = ros_time_to_float(msg)
        print(f"Saved frame: {self._output_path}")
        print(f"width={msg.width}")
        print(f"height={msg.height}")
        print(f"encoding={msg.encoding}")
        print(f"timestamp={timestamp:.9f}")
        self._saved = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True, help="ROS 2 image topic to subscribe to")
    parser.add_argument("--out", required=True, help="PNG output path")
    parser.add_argument("--timeout", type=float, default=20.0, help="Seconds to wait for one frame")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.out).expanduser().resolve()

    rclpy.init()
    node = OneShotFrameSaver(args.topic, output_path)
    deadline = time.monotonic() + args.timeout

    try:
        while rclpy.ok() and not node.saved and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if not output_path.exists():
        print(f"No image received on {args.topic} within {args.timeout} seconds.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
