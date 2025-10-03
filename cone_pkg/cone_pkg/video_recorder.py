#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from pathlib import Path
from datetime import datetime

class ImageRecorder(Node):
    def __init__(self):
        super().__init__('video_recorder')

        # Paraméterek
        self.declare_parameter('topic', '/ultralytics/detection/image')
        self.declare_parameter('path', '')          # ha üres, ~/.ros alá ment
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('codec', 'MJPG')     # konténer-kompatibilis fallback
        self.declare_parameter('ext', 'avi')        # MP4-hez próbáld: codec=mp4v, ext=mp4

        topic = self.get_parameter('topic').get_parameter_value().string_value
        self.path_param = self.get_parameter('path').get_parameter_value().string_value
        self.fps = float(self.get_parameter('fps').value)
        self.codec = self.get_parameter('codec').get_parameter_value().string_value
        self.ext = self.get_parameter('ext').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.writer = None
        self.size = None
        self.color_mode = None  # 'mono' vagy 'bgr'

        self.sub = self.create_subscription(Image, topic, self.cb, 10)
        self.get_logger().info(f"Feliratkozva: {topic}")

    def _open_writer(self, width, height):
        out_dir = Path(self.path_param) if self.path_param else (Path.home() / '.ros')
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = out_dir / f"topic_record_{ts}.{self.ext}"

        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(str(out_path), fourcc, self.fps, (width, height))
        if not self.writer.isOpened():
            self.get_logger().warn(f"Nem tudtam megnyitni a videóírót: {out_path} (codec={self.codec})")
            return False
        self.get_logger().info(f"Írás: {out_path}  {width}x{height}@{self.fps}  codec={self.codec}")
        self.out_path = out_path
        return True

    def cb(self, msg: Image):
        # Első frame-nél detektáljuk a méretet és csatornákat
        if self.size is None:
            w, h = msg.width, msg.height
            self.size = (w, h)

            # Ha mono8 érkezik, videóhoz 3 csatornává konvertálunk
            self.color_mode = 'mono' if msg.encoding in ('mono8', '8UC1') else 'bgr'
            if not self._open_writer(w, h):
                return

        # ROS Image -> OpenCV
        try:
            if self.color_mode == 'mono':
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
                # A legtöbb VideoWriter 3 csatornát szeret → konvertáljunk BGR-re
                frame = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                # Ha bgr8 jön, hagyjuk úgy
                frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"Konverziós hiba: {e}")
            return

        if self.writer:
            try:
                self.writer.write(frame)
            except Exception as e:
                self.get_logger().warn(f"Írási hiba: {e}")

    def destroy_node(self):
        try:
            if self.writer:
                self.writer.release()
                self.get_logger().info(f"Lezárva: {getattr(self, 'out_path', '')}")
        finally:
            super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ImageRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
