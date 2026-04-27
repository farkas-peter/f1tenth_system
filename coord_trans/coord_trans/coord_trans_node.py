#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Odometry
from ublox_msgs.msg import NavPVT
import utm

class CoordTransNode(Node):
    def __init__(self):
        super().__init__('coord_trans_node')
        
        self.subscription = self.create_subscription(NavSatFix, '/fix', self.fix_callback, 10)
        self.navpvt_sub = self.create_subscription(NavPVT, '/navpvt', self.navpvt_callback, 10)
            
        self.publisher_ = self.create_publisher(Odometry, '/gps_odom', 10)
            
        self.origin_x = None
        self.origin_y = None
        self.origin_z = None
        
        self.prev_x = None
        self.prev_y = None
        self.current_yaw = 0.0  # Initial yaw
        self.has_rtk_fix = False
        
        self.vel_e = 0.0
        self.vel_n = 0.0
        self.vel_d = 0.0
        
        self.get_logger().info("Coord Trans Node started.")

    def navpvt_callback(self, msg: NavPVT):
        # FLAGS_CARRIER_PHASE_MASK = 192 (0b11000000)
        # CARRIER_PHASE_FIXED = 128      (0b10000000)
        if (msg.flags & 192) == 128:
            self.has_rtk_fix = True
        else:
            self.has_rtk_fix = False
            
        self.vel_e = msg.vel_e / 1000.0
        self.vel_n = msg.vel_n / 1000.0
        self.vel_d = msg.vel_d / 1000.0

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
                
            # Create Odometry message
            odom_msg = Odometry()
            odom_msg.header.stamp = self.get_clock().now().to_msg()
            odom_msg.header.frame_id = "odom"
            odom_msg.child_frame_id = "base_link"
            
            odom_msg.pose.pose.position.x = float(easting - self.origin_x)
            odom_msg.pose.pose.position.y = float(northing - self.origin_y)
            odom_msg.pose.pose.position.z = float(alt - self.origin_z)
            
            # Convert yaw to quaternion (Euler to Quaternion where roll=0, pitch=0)
            odom_msg.pose.pose.orientation.x = 0.0
            odom_msg.pose.pose.orientation.y = 0.0
            odom_msg.pose.pose.orientation.z = math.sin(self.current_yaw / 2.0)
            odom_msg.pose.pose.orientation.w = math.cos(self.current_yaw / 2.0)
            
            # Fill velocity (twist) in base_link frame
            # Rotate global East/North velocities with -yaw to get long/lat velocities
            v_x = self.vel_e * math.cos(self.current_yaw) + self.vel_n * math.sin(self.current_yaw)
            v_y = -self.vel_e * math.sin(self.current_yaw) + self.vel_n * math.cos(self.current_yaw)
            
            odom_msg.twist.twist.linear.x = v_x
            odom_msg.twist.twist.linear.y = v_y
            odom_msg.twist.twist.linear.z = -self.vel_d  # velD is down, z is up
            
            self.publisher_.publish(odom_msg)
            
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
