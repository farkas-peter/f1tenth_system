import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from ackermann_msgs.msg import AckermannDriveStamped

class AutonomousNode(Node):
    def __init__(self):
        super().__init__('autonomous_node')
        self.enabled = False
        self.subscription = self.create_subscription(Bool, '/autonomous_enable', self.enable_cb, 10)
        self.publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.timer = self.create_timer(0.1, self.timer_cb)

    def enable_cb(self, msg):
        self.enabled = msg.data

        if not self.enabled:
            self.stop_vehicle()

    def timer_cb(self):
        if self.enabled:
            msg = AckermannDriveStamped()
            msg.drive.speed = 4.0
            #autonomous_control
            self.publisher.publish(msg)

    def stop_vehicle(self):
        msg = AckermannDriveStamped()
        msg.drive.speed = 0.0
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = AutonomousNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
