#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from ublox_msgs.msg import NavPVT
import math

class GpsEvalNode(Node):
    def __init__(self):
        super().__init__('gps_eval_node')
        self.subscription_fix = self.create_subscription(NavSatFix, '/fix', self.fix_callback, 10)
        self.subscription_navpvt = self.create_subscription(NavPVT, '/navpvt', self.navpvt_callback, 10)
        
        self.rtk_status_str = "UNKNOWN (Waiting for NavPVT...)"
        self.is_rtk_fixed = False
        
        self.get_logger().info('GPS Eval Node started, listening on /fix and /navpvt')

    def navpvt_callback(self, msg: NavPVT):
        # FLAGS_CARRIER_PHASE_MASK = 192 (0b11000000)
        # 0 = No carrier phase solution
        # 64 (0b01000000) = Float solution
        # 128 (0b10000000) = Fixed solution
        carr_soln = msg.flags & 192
        
        if carr_soln == 128:
            self.rtk_status_str = "RTK FIXED"
            self.is_rtk_fixed = True
        elif carr_soln == 64:
            self.rtk_status_str = "RTK FLOAT"
            self.is_rtk_fixed = False
        else:
            self.rtk_status_str = "NO RTK (3D FIX or NO FIX)"
            self.is_rtk_fixed = False

    def fix_callback(self, msg):
        lat = msg.latitude
        lon = msg.longitude
        alt = msg.altitude

        # Calculate 2D accuracy
        if len(msg.position_covariance) == 9:
            var_east = msg.position_covariance[0]
            var_north = msg.position_covariance[4]
            accuracy_2d = math.sqrt(var_east + var_north)
        else:
            accuracy_2d = -1.0 # Unknown

        self.get_logger().info('-----------------------------')
        if self.is_rtk_fixed:
            self.get_logger().info(f'RTK STATUS   : [\033[92mOK\033[0m] {self.rtk_status_str}')
        else:
            self.get_logger().info(f'RTK STATUS   : [\033[91mNOT OK\033[0m] {self.rtk_status_str}')
            
        if accuracy_2d >= 0:
            self.get_logger().info(f'2D ACCURACY  : \033[93m{accuracy_2d:.4f} m\033[0m')
        else:
            self.get_logger().info(f'2D ACCURACY  : Unknown')
            
        self.get_logger().info(f'COORDINATES  : Lat: {lat:.7f}, Lon: {lon:.7f}, Alt: {alt:.2f}')

def main(args=None):
    rclpy.init(args=args)
    node = GpsEvalNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
