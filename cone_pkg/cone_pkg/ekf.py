import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Twist
import numpy as np

class EKF:
    def __init__(self, dt):
        self.dt = dt
        self.x = np.zeros((4, 1))  # [x, y, theta, v]
        self.P = np.eye(4) * 0.1
        self.Q = np.eye(4) * 0.01
        self.R = np.eye(2) * 0.05

    def predict(self, omega, a):
        theta = self.x[2, 0]
        v = self.x[3, 0]

        self.x[0, 0] += v * np.cos(theta) * self.dt
        self.x[1, 0] += v * np.sin(theta) * self.dt
        self.x[2, 0] += omega * self.dt
        self.x[3, 0] += a * self.dt

        F = np.eye(4)
        F[0, 2] = -v * np.sin(theta) * self.dt
        F[0, 3] = np.cos(theta) * self.dt
        F[1, 2] = v * np.cos(theta) * self.dt
        F[1, 3] = np.sin(theta) * self.dt
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        H = np.zeros((2, 4))
        H[0, 0] = 1
        H[1, 1] = 1
        y = z.reshape(2, 1) - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P

    def get_state(self):
        return self.x.flatten()

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')
        self.ekf = EKF(dt=0.02)
        self.imu_sub = self.create_subscription(Imu, '/camera/imu', self.imu_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.pub = self.create_publisher(Odometry, '/odom/ekf', 10)
        self.last_imu_time = self.get_clock().now()
        self.latest_omega = 0.0
        self.latest_accel = 0.0
        self.timer = self.create_timer(0.02, self.timer_callback)

    def imu_callback(self, msg):
        self.latest_omega = msg.angular_velocity.y
        self.latest_accel = msg.linear_acceleration.z

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.ekf.update(np.array([x, y]))

    def timer_callback(self):
        self.ekf.predict(self.latest_omega, self.latest_accel)
        x, y, theta, v = self.ekf.get_state()

        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'
        odom_msg.pose.pose.position.x = float(x)
        odom_msg.pose.pose.position.y = float(y)
        odom_msg.pose.pose.orientation.z = float(np.sin(theta / 2))
        odom_msg.pose.pose.orientation.w = float(np.cos(theta / 2))
        odom_msg.twist.twist.linear.x = float(v)
        odom_msg.twist.twist.angular.z = float(self.latest_omega)

        self.pub.publish(odom_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
