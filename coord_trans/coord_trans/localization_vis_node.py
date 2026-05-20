#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import PoseStamped
import math


class LocalizationVisNode(Node):
    def __init__(self):
        super().__init__('localization_vis_node')

        self.utm_sub = self.create_subscription(Odometry, '/gps_odom', self.utm_callback, 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.loc_vis_pub = self.create_publisher(MarkerArray, '/trajectory_vis', 10)
        self.marker_array = MarkerArray()
        self.id = 0
        self.prev_x = None
        self.prev_y = None
        self.x = None
        self.y = None
        self.dist_threshold = 0.2
        self.latest_goal = None

        self.timer = self.create_timer(0.1, self.publish_markers)
        
        self.get_logger().info("Localization Vis Node started.")

    def goal_callback(self, msg: PoseStamped):
        self.marker_array.markers.clear()

        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y

        goal_radius = Marker()
        goal_radius.header.frame_id = "map"
        goal_radius.ns = "goal_radius"
        goal_radius.id = 0
        goal_radius.type = Marker.CYLINDER
        goal_radius.action = Marker.ADD

        #Position
        goal_radius.pose.position.x = goal_x
        goal_radius.pose.position.y = goal_y
        goal_radius.pose.position.z = 0.0

        #Orientation
        goal_radius.pose.orientation.x = 0.0
        goal_radius.pose.orientation.y = 0.0
        goal_radius.pose.orientation.z = 0.0
        goal_radius.pose.orientation.w = 1.0 

        #Size
        goal_radius.scale.x = 0.5
        goal_radius.scale.y = 0.5
        goal_radius.scale.z = 0.01
        
        #Color
        goal_radius.color.a = 1.0
        goal_radius.color.r = 0.0
        goal_radius.color.g = 1.0
        goal_radius.color.b = 0.0

        goal_radius.lifetime = Duration(seconds=0.1).to_msg()

        self.marker_array.markers.append(goal_radius)
        
    def utm_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        if self.prev_x is not None and self.prev_y is not None:
            dist = math.sqrt((x - self.prev_x) ** 2 + (y - self.prev_y) ** 2)
            if dist < self.dist_threshold:
                return

        marker = Marker()
        marker.header.frame_id = "map"
        marker.ns = "trajectory"
        marker.id = self.id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        #Position
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0

        #Orientation
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0 

        #Size
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        
        #Color
        v_x = msg.twist.twist.linear.x
        v_y = msg.twist.twist.linear.y
        v_z = msg.twist.twist.linear.z
        speed = math.sqrt(v_x**2 + v_y**2 + v_z**2)
        ratio = max(0.0, min(speed / 3.0, 1.0))
        
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        marker.lifetime = Duration(seconds=0.1).to_msg()

        self.marker_array.markers.append(marker)
                
        self.id += 1
        self.prev_x = x
        self.prev_y = y

    def publish_markers(self):
        self.loc_vis_pub.publish(self.marker_array)
        

def main(args=None):
    rclpy.init(args=args)
    node = LocalizationVisNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
