import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
from sensor_msgs.msg import Imu

class ImuNode(Node):
    def __init__(self):
        super().__init__('imu_node')

        self.imu_pub = self.create_publisher(Imu,'/imu',1)

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.accel)
        config.enable_stream(rs.stream.gyro)
        self.pipeline.start(config)

        self.timer = self.create_timer(0.01, self.timer_callback)

        self.get_logger().info('Imu node started.')

    def timer_callback(self):
        frames = self.pipeline.wait_for_frames()
        accel_frame = frames.first_or_default(rs.stream.accel)
        gyro_frame = frames.first_or_default(rs.stream.gyro)

        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = 'imu_link'
        
        imu_msg.orientation_covariance[0] = -1.0

        if accel_frame:
            accel = accel_frame.as_motion_frame().get_motion_data()
            imu_msg.linear_acceleration.x = accel.x
            imu_msg.linear_acceleration.y = accel.y
            imu_msg.linear_acceleration.z = accel.z

        if gyro_frame:
            gyro = gyro_frame.as_motion_frame().get_motion_data()
            imu_msg.angular_velocity.x = gyro.x
            imu_msg.angular_velocity.y = gyro.y
            imu_msg.angular_velocity.z = gyro.z

        self.imu_pub.publish(imu_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ImuNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
