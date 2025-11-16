import cv2
import pyrealsense2 as rs
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class ObjTrackerNode(Node):
    def __init__(self):
        super().__init__('obj_tracker_node')

        # Camera variables
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        # ROS2 variables
        self.subscription = self.create_subscription(String, '/my_topic', self.listener_callback, 10)
        self.timer = self.create_timer(0.033, self.camera_callback)  # 30 FPS

        self.get_logger().info("ObjTrackerNode node started.")

    def camera_callback(self):
        # Wait for frame and align the color and depth frames
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return

        # Visualize
        color_np_image = np.asanyarray(color_frame.get_data())
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_np_image)
        cv2.waitKey(1)

        # todo: When the node started, let's save an image for obj detection


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
        data = msg.data  # store message
        self.get_logger().info(f"Received: {data}")

    def shutdown(self):
        self.pipeline.stop()
        self.get_logger().info("ObjTrackerNode node stopped.")


def main(args=None):
    rclpy.init(args=args)
    node = ObjTrackerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.shutdown()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()