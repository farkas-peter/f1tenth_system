import cv2
import requests

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class SpeechToObjAgent(Node):
    def __init__(self):
        super().__init__('speech_to_obj_agent')

        self.last_received_value = None

        self.timer = self.create_timer(0.033, self.camera_callback)  # 30 FPS

        self.subscription = self.create_subscription(String, '/my_topic', self.listener_callback, 10)

        self.get_logger().info("SpeechToObjAgent node started.")

    def camera_callback(self):
        pass

    def tracking(self, frame, bbox):
        tracker = cv2.legacy.TrackerCSRT_create()
        initialized = tracker.init(frame, bbox)  # bbox = (x, y, w, h)

        if initialized is not None and tracker is not None:
            ok, bbox = tracker.update(frame)

            if ok:
                # Bbox can be drawn
                pass
            else:
                # Tracking lost
                # Tracker should be initialized again
                pass


    def listener_callback(self, msg):
        self.last_received_value = msg.data  # store message
        self.get_logger().info(f"Received: {self.last_received_value}")

    def shutdown(self):
        self.pipeline.stop()
        self.get_logger().info("SpeechToObjAgent node stopped.")


def main(args=None):
    rclpy.init(args=args)
    node = SpeechToObjAgent()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.shutdown()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()