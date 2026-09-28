import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge
import numpy as np
import h5py
from datetime import datetime
from pathlib import Path


class RealSenseHDF5Recorder(Node):

    def __init__(self):
        super().__init__('realsense_hdf5_recorder')

        self.bridge = CvBridge()

        # =========================================================
        # Configuration
        # =========================================================

        self.width = 848
        self.height = 480

        # RealSense D435i commonly uses 0.001 m/unit.
        # Change this if your camera uses a different depth scale.
        self.depth_unit = 0.001

        # =========================================================
        # Output file
        # =========================================================

        output_dir = Path("/workspace/LOG/recordings")
        output_dir.mkdir(parents=True, exist_ok=True)

        recording_name = datetime.now().strftime("recording_%Y%m%d_%H%M%S.h5")

        self.file_path = output_dir / recording_name

        self.h5_file = h5py.File(self.file_path, "w")

        # =========================================================
        # Metadata
        # =========================================================

        metadata = self.h5_file.create_group("metadata")

        metadata.attrs["width"] = self.width
        metadata.attrs["height"] = self.height
        metadata.attrs["depth_unit_m"] = self.depth_unit
        metadata.attrs["color_encoding"] = "bgr8"
        metadata.attrs["depth_encoding"] = "16UC1"

        # =========================================================
        # Color datasets
        # =========================================================

        color_group = self.h5_file.create_group("color")

        self.color_images = color_group.create_dataset(
            "images",
            shape=(0, self.height, self.width, 3),
            maxshape=(None, self.height, self.width, 3),
            dtype=np.uint8,
            chunks=(1, self.height, self.width, 3),
            compression="lzf"
        )

        self.color_timestamps = color_group.create_dataset(
            "timestamps",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=True
        )

        # =========================================================
        # Depth datasets
        # =========================================================

        depth_group = self.h5_file.create_group("depth")

        self.depth_images = depth_group.create_dataset(
            "images",
            shape=(0, self.height, self.width),
            maxshape=(None, self.height, self.width),
            dtype=np.uint16,
            chunks=(1, self.height, self.width),
            compression="lzf"
        )

        self.depth_timestamps = depth_group.create_dataset(
            "timestamps",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=True
        )

        # =========================================================
        # IMU datasets
        # =========================================================

        imu_group = self.h5_file.create_group("imu")

        self.imu_timestamps = imu_group.create_dataset(
            "timestamps",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=True
        )

        self.linear_acceleration = imu_group.create_dataset(
            "linear_acceleration",
            shape=(0, 3),
            maxshape=(None, 3),
            dtype=np.float64,
            chunks=True
        )

        self.angular_velocity = imu_group.create_dataset(
            "angular_velocity",
            shape=(0, 3),
            maxshape=(None, 3),
            dtype=np.float64,
            chunks=True
        )

        # =========================================================
        # Counters
        # =========================================================

        self.color_count = 0
        self.depth_count = 0
        self.imu_count = 0

        # =========================================================
        # ROS subscribers
        # =========================================================

        self.color_sub = self.create_subscription(Image, "/camera/image", self.color_callback, 10)
        self.depth_sub = self.create_subscription(Image, "/camera/depth", self.depth_callback, 10)
        self.imu_sub = self.create_subscription(Imu, "/camera/imu", self.imu_callback, 100)

        self.get_logger().info(f"Recording started: {self.file_path}")

    # =============================================================
    # Timestamp
    # =============================================================

    @staticmethod
    def timestamp_to_ns(stamp):
        return (int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec))

    # =============================================================
    # Color callback
    # =============================================================

    def color_callback(self, msg):

        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        except Exception as e:
            self.get_logger().error(f"Color image conversion failed: {e}")
            return

        timestamp = self.timestamp_to_ns(msg.header.stamp)

        index = self.color_count

        # Extend datasets
        self.color_images.resize(index + 1, axis=0)

        self.color_timestamps.resize(index + 1, axis=0)

        # Write data
        self.color_images[index] = image
        self.color_timestamps[index] = timestamp

        self.color_count += 1

    # =============================================================
    # Depth callback
    # =============================================================

    def depth_callback(self, msg):

        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        except Exception as e:
            self.get_logger().error(f"Depth image conversion failed: {e}")
            return

        if depth.dtype != np.uint16:
            self.get_logger().warning(f"Unexpected depth dtype: {depth.dtype}")
            return

        timestamp = self.timestamp_to_ns(msg.header.stamp)

        index = self.depth_count

        # Extend datasets
        self.depth_images.resize(index + 1, axis=0)

        self.depth_timestamps.resize(index + 1, axis=0)

        # Write data
        self.depth_images[index] = depth
        self.depth_timestamps[index] = timestamp

        self.depth_count += 1

    # =============================================================
    # IMU callback
    # =============================================================

    def imu_callback(self, msg):

        timestamp = self.timestamp_to_ns(msg.header.stamp)

        index = self.imu_count

        # Extend datasets
        self.imu_timestamps.resize(index + 1,axis=0)

        self.linear_acceleration.resize(index + 1, axis=0)

        self.angular_velocity.resize(index + 1, axis=0)

        # Write timestamp
        self.imu_timestamps[index] = timestamp

        # Write acceleration
        self.linear_acceleration[index] = [
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ]

        # Write gyro
        self.angular_velocity[index] = [
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ]

        self.imu_count += 1

    # =============================================================
    # Shutdown
    # =============================================================

    def shutdown(self):

        if self.h5_file:
            self.h5_file.flush()
            self.h5_file.close()

        self.get_logger().info("Recording stopped.")
        self.get_logger().info(f"Color frames: {self.color_count}")
        self.get_logger().info(f"Depth frames: {self.depth_count}")
        self.get_logger().info(f"IMU samples: {self.imu_count}")
        self.get_logger().info(f"Saved to: {self.file_path}")


def main(args=None):

    rclpy.init(args=args)
    node = RealSenseHDF5Recorder()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()