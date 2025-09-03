import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool
from sensor_msgs.msg import Joy
from rclpy.time import Time
import math


class PurePursuitLocal(Node):
    def __init__(self):
        super().__init__('pure_pursuit_local_node')
        self.cnt = 1.0
        self.lookahead_parameter = 1.25
        self.wheelbase = 0.33
        self.constant_speed = 2.0

        self.target_point = None
        self.last_target_time = Time()

        self.enabled = False
        self.subscription = self.create_subscription(Bool, '/autonomous_enable', self.enable_cb, 10)
        self.joy_sub =  self.create_subscription(Joy, '/joy', self.joy_cb, 10)
        self.create_subscription(Point, '/target_point', self.target_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.create_timer(0.033, self.control_loop)
        self.get_logger().info("PPC node started.")

    def enable_cb(self, msg):
        self.enabled = msg.data

        if not self.enabled:
            self.stop_vehicle()

    def joy_cb(self, msg: Joy):
        if msg.axes[7] == 1.0:
            if self.cnt != 5.0:
                self.cnt += 0.5
        elif msg.axes[7] == -1.0:
            if self.cnt != 1.0:
                self.cnt -= 0.5

    def stop_vehicle(self):
        msg = AckermannDriveStamped()
        msg.drive.speed = 0.0
        self.drive_pub.publish(msg)

    def target_callback(self, msg):
        self.target_point = msg
        self.last_target_time = self.get_clock().now()

    def control_loop(self):
        drive_msg = AckermannDriveStamped()
        if self.target_point is None:
            return
        
        
        if self.last_target_time is None or (self.get_clock().now() - self.last_target_time).nanoseconds > 2e8:
            #self.get_logger().warn("Nincs friss target_point üzenet!")
            drive_msg.drive.speed = 0.0
            drive_msg.drive.steering_angle = 0.0

            self.drive_pub.publish(drive_msg)
            return

        if self.enabled:
            x = self.target_point.x
            y = self.target_point.y
            distance = math.sqrt(x ** 2 + y ** 2)

            if distance < 0.1:
                self.get_logger().info('Cél túl közel, nem vezérelünk.')
                return
            
            input_speed = self.constant_speed * self.cnt
            lookahead_distance = (self.lookahead_parameter * input_speed) - (input_speed - 1.0)
            alpha = math.atan2(y,x)
            curvature = (2*math.sin(alpha))/(lookahead_distance)
            steering_angle = math.atan(self.wheelbase*curvature)
            
            drive_msg.drive.speed = input_speed
            drive_msg.drive.steering_angle = steering_angle

            self.drive_pub.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitLocal()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
        
