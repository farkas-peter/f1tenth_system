#!/usr/bin/python3

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class WebcamPublisher(Node):
    def __init__(self, port=0):
        super().__init__('Webcam_Pub')
        
        # Kép publikálásához szükséges beállítások
        self.pub = self.create_publisher(Image, '/camera/rgb/image_raw', 10)
        self.cap = cv2.VideoCapture(port)
        self.bridge = CvBridge()

        # Publikálási frekvencia beállítása (30 Hz)
        self.timer = self.create_timer(1.0 / 30.0, self.publish_image)
    
    def publish_image(self):
        """Kép olvasása és publikálása"""
        val, frame = self.cap.read()
        
        if val:
            # Kép konvertálása ROS üzenetté
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.pub.publish(img_msg)
            self.get_logger().info("Publishing image...")

def main(args=None):
    rclpy.init(args=args)
    webcam_publisher = WebcamPublisher()
    rclpy.spin(webcam_publisher)
    webcam_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

