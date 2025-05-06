import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from tf_transformations import euler_from_quaternion
from math import atan2, sqrt, pow, cos, sin


class PurePursuit(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')

        # Paraméterek
        self.lookahead_distance = 1.0
        self.constant_speed = 4.0  

        # Init állapot
        self.current_pose = None
        self.yaw = 0.0
        self.target_point = None

        # Feliratkozások
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Point, '/cone_gate', self.target_callback, 10)

        # Publikáló a vezérléshez
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        # Időzítő a vezérlési ciklushoz
        self.create_timer(0.05, self.control_loop)

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose.position

        orientation_q = msg.pose.pose.orientation
        _, _, self.yaw = euler_from_quaternion([
            orientation_q.x,
            orientation_q.y,
            orientation_q.z,
            orientation_q.w
        ])

    def target_callback(self, msg):
        self.target_point = msg

    def control_loop(self):
        if self.current_pose is None or self.target_point is None:
            return

        dx = self.target_point.x - self.current_pose.x
        dy = self.target_point.y - self.current_pose.y
        distance = sqrt(dx ** 2 + dy ** 2)

        if distance < 0.1:
            self.get_logger().info('Célpont elérve.')
            return

        # Célpont koordinátáinak transzformálása jármű koordinátarendszerébe
        x_local = dx * cos(-self.yaw) - dy * sin(-self.yaw)
        y_local = dx * sin(-self.yaw) + dy * cos(-self.yaw)

        # Kormányzási szög kiszámítása
        if x_local == 0:
            steering_angle = 0.0
        else:
            curvature = (2 * y_local) / (distance ** 2)
            steering_angle = atan2(self.lookahead_distance * curvature, 1.0)

        # Ackermann üzenet összeállítása
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = self.constant_speed
        drive_msg.drive.steering_angle = steering_angle

        self.drive_pub.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PurePursuit()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
