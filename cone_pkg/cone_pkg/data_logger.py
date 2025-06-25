import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from vesc_msgs.msg import VescStateStamped
from geometry_msgs.msg import Point
from datetime import datetime
import math
import json
import os



class LoggerNode(Node):
    def __init__(self):
        super().__init__('data_logger')

        # Előállítjuk a log elérési útvonalát
        filename = f"log_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        log_path = os.path.join('/workspace/src/f1tenth_system/cone_pkg/logs', filename)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log_path = log_path

        # Belső adattároló (fő log dict)
        self.data_log = {}

        # Kezdő időpont a relatív időhöz
        self.start_time = self.get_clock().now()

        # Aktuális szenzoradatok
        self.odom_data = None
        self.ekf_data = None
        self.orientation = None
        self.input_speed_data = None
        self.input_steering_angle_data = None
        self.output_speed_data = None
        self.output_steering_angle_data = None
        self.pos = Point()
        self.target_point = Point()

        # Feliratkozások
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        self.create_subscription(Odometry, '/odom/ekf', self.ekf_cb, 10)
        self.create_subscription(Float64, '/commands/motor/speed', self.input_speed_cb, 10)
        self.create_subscription(Float64, '/commands/servo/position', self.input_steering_angle_cb, 10)
        self.create_subscription(VescStateStamped, '/sensors/core', self.output_speed_cb, 10)
        self.create_subscription(Float64, '/sensors/servo_position_command', self.output_steering_angle_cb, 10)
        self.create_subscription(Point, '/target_point', self.target_cb, 10)

        # Timer 10 Hz
        self.timer = self.create_timer(0.1, self.timer_cb)
        self.get_logger().info("Data logger node started.")

    def timer_cb(self):
        now = self.get_clock().now()
        elapsed = (now - self.start_time).nanoseconds / 1e9

        timestamp_key = f"{elapsed:.3f}"

        if self.odom_data:
            yaw = self.get_yaw_from_quaternion(self.orientation)

            x = self.pos.x + self.target_point.x * math.cos(yaw) - self.target_point.y * math.sin(yaw)
            y = self.pos.y + self.target_point.x * math.sin(yaw) + self.target_point.y * math.cos(yaw)
        else:
            x, y = None, None
        target_data = {
            'position':{
                'x':x,
                'y':y
            }
        }

        self.data_log[timestamp_key] = {
            'odom': self.odom_data,
            'ekf_odom': self.ekf_data,
            'input_speed': self.input_speed_data,
            'input_steering_angle': self.input_steering_angle_data,
            'output_speed': self.output_speed_data,
            'output_steering_angle': self.output_steering_angle_data,
            'target_point': target_data
        }

    def get_yaw_from_quaternion(self,q):
        # q: geometry_msgs.msg.Quaternion
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)


    def odom_cb(self, msg: Odometry):
        self.pos.x = msg.pose.pose.position.x
        self.pos.y = msg.pose.pose.position.y
        self.orientation = msg.pose.pose.orientation
        self.odom_data = {
            'position': {
                'x': msg.pose.pose.position.x,
                'y': msg.pose.pose.position.y,
                'z': msg.pose.pose.position.z
            },
            'orientation': {
                'x': msg.pose.pose.orientation.x,
                'y': msg.pose.pose.orientation.y,
                'z': msg.pose.pose.orientation.z,
                'w': msg.pose.pose.orientation.w
            },
            'twist': {
                'linear': {
                    'x': msg.twist.twist.linear.x,
                    'y': msg.twist.twist.linear.y,
                    'z': msg.twist.twist.linear.z
                },
                'angular': {
                    'x': msg.twist.twist.angular.x,
                    'y': msg.twist.twist.angular.y,
                    'z': msg.twist.twist.angular.z
                }
            }
        }

    def ekf_cb(self, msg: Odometry):
        self.ekf_data = {
            'position': {
                'x': msg.pose.pose.position.x,
                'y': msg.pose.pose.position.y,
                'z': msg.pose.pose.position.z
            },
            'orientation': {
                'x': msg.pose.pose.orientation.x,
                'y': msg.pose.pose.orientation.y,
                'z': msg.pose.pose.orientation.z,
                'w': msg.pose.pose.orientation.w
            },
            'twist': {
                'linear': {
                    'x': msg.twist.twist.linear.x,
                    'y': msg.twist.twist.linear.y,
                    'z': msg.twist.twist.linear.z
                },
                'angular': {
                    'x': msg.twist.twist.angular.x,
                    'y': msg.twist.twist.angular.y,
                    'z': msg.twist.twist.angular.z
                }
            }
        }

    def input_speed_cb(self, msg: Float64):
        self.input_speed_data = msg.data

    def input_steering_angle_cb(self, msg: Float64):
        self.input_steering_angle_data = msg.data

    def output_speed_cb(self, msg: VescStateStamped):
        self.output_speed_data = msg.state.speed

    def output_steering_angle_cb(self, msg: Float64):
        self.output_steering_angle_data = msg.data

    def target_cb(self, msg: Point):
        self.target_point.x = msg.x
        self.target_point.y = msg.y

    def destroy_node(self):
        # Kilépéskor fájlba mentés
        with open(self.log_path, 'w') as f:
            json.dump(self.data_log, f, indent=2)
        self.get_logger().info(f"Log file saved to: {self.log_path}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
