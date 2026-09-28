import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import Image, Imu
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import copy
from cv_bridge import CvBridge

class PublisherNode(Node):
    def __init__(self):
        super().__init__('publisher_node')
        self.bridge = CvBridge()

        self.enable_color_image = True
        self.enable_depth_image = True
        self.enable_imu = True

        self.width = 848
        self.height = 480
        self.fps = 30.0
        
        # ROS 2 Publisher
        self.image_pub = self.create_publisher(Image, "/camera/image", 1)
        self.depth_pub = self.create_publisher(Image, "/camera/depth", 1)
        self.imu_pub = self.create_publisher(Imu,'/camera/imu', 1)
        
        #Stereo Camera
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)
        config.enable_stream(rs.stream.accel)
        config.enable_stream(rs.stream.gyro)

        self.pipeline.start(config)

        #color frame align to depth frame
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        
        self.timer = self.create_timer((1.0/self.fps), self.capture_frames)
        self.get_logger().info("RealSense node started.")
    

    def capture_frames(self):
        frames = self.pipeline.wait_for_frames()
        stamp = self.get_clock().now().to_msg()

        if self.enable_imu:
            self._imu_pub(frames, stamp)    

        aligned_frames = self.align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not aligned_depth_frame or not color_frame:
            return

        if self.image_pub.get_subscription_count() >= 0 and self.enable_color_image:
            self._image_pub(np.asanyarray(color_frame.get_data()), stamp)

        if self.depth_pub.get_subscription_count() >= 0 and self.enable_depth_image:
            self._depth_pub(np.asanyarray(aligned_depth_frame.get_data()), stamp)


    def _imu_pub(self,frames, stamp):
        accel_frame = frames.first_or_default(rs.stream.accel)
        gyro_frame = frames.first_or_default(rs.stream.gyro)

        imu_msg = Imu()
        imu_msg.header.stamp = stamp
        imu_msg.header.frame_id = 'camera_imu_frame'

        if accel_frame:
            accel = accel_frame.as_motion_frame().get_motion_data()
            imu_msg.linear_acceleration.x = accel.x
            imu_msg.linear_acceleration.y = accel.y
            imu_msg.linear_acceleration.z = accel.z

        if gyro_frame:
            gyro = gyro_frame.as_motion_frame().get_motion_data()
            imu_msg.angular_velocity.x = gyro.x
            imu_msg.angular_velocity.y = gyro.y
            imu_msg.angular_velocity.z = gyro.z

        self.imu_pub.publish(imu_msg)
        
            
    def _image_pub(self, color_image, stamp):
        #scaled_image = cv2.resize(color_image, (480, 240), interpolation=cv2.INTER_AREA)
        gray_scaled_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY) #MONO16
        msg = self.bridge.cv2_to_imgmsg(gray_scaled_image, encoding="mono8")
        msg.header.stamp = stamp
        msg.header.frame_id = "camera_image_frame"
        self.image_pub.publish(msg)

    def _depth_pub(self, depth_image, stamp):
        #scaled_depth = cv2.resize(depth_image, (480, 240), interpolation=cv2.INTER_NEAREST)
        msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="16UC1")
        msg.header.stamp = stamp
        msg.header.frame_id = "camera_image_frame"
        self.depth_pub.publish(msg)
    
    def shutdown(self):
        self.pipeline.stop()
        self.get_logger().info("RealSense node stopped.")


def main(args=None):
    rclpy.init(args=args)
    node = PublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.shutdown()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

