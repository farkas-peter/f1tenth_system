#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import PoseWithCovarianceStamped
from ublox_msgs.msg import NavPVT
import utm

class CoordTransNode(Node):
    def __init__(self):
        super().__init__('coord_trans_node')
        
        self.subscription = self.create_subscription(NavSatFix, '/fix', self.fix_callback, 10)
        self.navpvt_sub = self.create_subscription(NavPVT, '/navpvt', self.navpvt_callback, 10)
            
        self.publisher_ = self.create_publisher(PoseWithCovarianceStamped, '/utm_pose', 10)
            
        self.origin_x = None
        self.origin_y = None
        self.origin_z = None
        
        self.prev_x = None
        self.prev_y = None
        self.current_yaw = 0.0  # Initial yaw
        self.has_rtk_fix = False
        
        self.get_logger().info("Coord Trans Node started.")

    def navpvt_callback(self, msg: NavPVT):
        # FLAGS_CARRIER_PHASE_MASK = 192 (0b11000000)
        # CARRIER_PHASE_FIXED = 128      (0b10000000)
        if (msg.flags & NavPVT.FLAGS_CARRIER_PHASE_MASK) == NavPVT.FLAGS_CARRIER_PHASE_FIXED:
            self.has_rtk_fix = True
        else:
            self.has_rtk_fix = False

    def fix_callback(self, msg: NavSatFix):
        if math.isnan(msg.latitude) or math.isnan(msg.longitude):
            self.get_logger().warn("Received NaN coordinates in /fix")
            return
            
        try:
            # Convert lat/lon to UTM
            # utm.from_latlon returns (easting, northing, zone_number, zone_letter)
            easting, northing, zone_num, zone_letter = utm.from_latlon(msg.latitude, msg.longitude)
            alt = float(msg.altitude) if not math.isnan(msg.altitude) else 0.0
            
            if self.origin_x is None:
                if not self.has_rtk_fix:
                    self.get_logger().info("Waiting for RTK Fix before setting origin...", throttle_duration_sec=2.0)
                    return
                
                self.origin_x = easting
                self.origin_y = northing
                self.origin_z = alt
                self.get_logger().info(f"Origin set at UTM E:{easting:.2f}, N:{northing:.2f} (RTK Fix achieved)")
                
            # Distance check for heading update (>= 5 cm)
            if self.prev_x is not None and self.prev_y is not None:
                dx = easting - self.prev_x
                dy = northing - self.prev_y
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist >= 0.05:
                    self.current_yaw = math.atan2(dy, dx)
                    self.prev_x = easting
                    self.prev_y = northing
            else:
                # Initialization
                self.prev_x = easting
                self.prev_y = northing
                
            # Create PoseWithCovarianceStamped message
            pose_msg = PoseWithCovarianceStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "map"
            
            pose_msg.pose.pose.position.x = float(easting - self.origin_x)
            pose_msg.pose.pose.position.y = float(northing - self.origin_y)
            pose_msg.pose.pose.position.z = float(alt - self.origin_z)
            
            # Convert yaw to quaternion (Euler to Quaternion where roll=0, pitch=0)
            pose_msg.pose.pose.orientation.x = 0.0
            pose_msg.pose.pose.orientation.y = 0.0
            pose_msg.pose.pose.orientation.z = math.sin(self.current_yaw / 2.0)
            pose_msg.pose.pose.orientation.w = math.cos(self.current_yaw / 2.0)
            
            # EKF requires non-zero covariance for variables it uses! Index 35 is Yaw.
            pose_msg.pose.covariance[35] = 0.1
            
            self.publisher_.publish(pose_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error converting coordinates: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = CoordTransNode()
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
