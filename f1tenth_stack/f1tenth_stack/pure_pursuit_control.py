import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool
from math import atan2, sqrt


class PurePursuitLocal(Node):
    def __init__(self):
        super().__init__('pure_pursuit_local_node')

        self.lookahead_distance = 1.0
        self.constant_speed = 2.0

        self.target_point = None

        self.enabled = False
        self.subscription = self.create_subscription(Bool, '/autonomous_enable', self.enable_cb, 10)
        self.create_subscription(Point, '/target_point', self.target_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        self.create_timer(0.05, self.control_loop)

    def enable_cb(self, msg):
        self.enabled = msg.data

        if not self.enabled:
            self.stop_vehicle()

    def stop_vehicle(self):
        msg = AckermannDriveStamped()
        msg.drive.speed = 0.0
        self.drive_pub.publish(msg)

    def target_callback(self, msg):
        self.target_point = msg

    def control_loop(self):
        if self.target_point is None:
            return
        if self.enabled:
            x = self.target_point.x
            y = self.target_point.y
            distance = sqrt(x ** 2 + y ** 2)

            if distance < 0.1:
                self.get_logger().info('Cél túl közel, nem vezérelünk.')
                return

            # Pure Pursuit görbület alapú kormányzás
            curvature = (2 * y) / (distance ** 2)
            steering_angle = atan2(self.lookahead_distance * curvature, 1.0)

            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = self.constant_speed
            drive_msg.drive.steering_angle = steering_angle

            self.drive_pub.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitLocal()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
        
