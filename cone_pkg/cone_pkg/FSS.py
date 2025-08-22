import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
import pyrealsense2 as rs
import numpy as np
import cv2
from cv_bridge import CvBridge
import time

class RealSenseNode(Node):
    def __init__(self):
        super().__init__('realsense_node')
        #Parameters
        self.res_width = 640
        self.res_height = 480
        self.clip_dist = 3.0
        self.upper_threshold = -0.1
        self.lower_threshold = -0.2

        self.bridge = CvBridge()
        
        # ROS 2 Publisher
        self.image_publish = self.create_publisher(Image, "/gray_image", 1)
        self.cloud_publish = self.create_publisher(PointCloud2, "/free_space", 1)

        #Stereo Camera

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        depth_sensor = device.first_depth_sensor()
        # Get depth scale of the device
        self.depth_scale =  depth_sensor.get_depth_scale()
        # Create an align object
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        config.enable_stream(rs.stream.depth, self.res_width, self.res_height, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, self.res_width, self.res_height, rs.format.bgr8, 30)
        
        # Start streaming
        self.pipeline.start(config)
        
        self.timer = self.create_timer(0.033, self.capture_frames)  # 30 FPS
        self.get_logger().info("FSS node started.")
    
    def capture_frames(self):
        
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return
        
        """
        color_image = np.asanyarray(color_frame.get_data())

        self.image_pub(color_image)
        
        start = time.time()
        
        end = time.time()
        elapsed = end - start
        self.get_logger().info(f"Run time: {elapsed:.3f}s")
        """

        points_xyz = self.depth2PointCloud(depth_frame)
        self.pointcloud_pub(points_xyz)

    def depth2PointCloud(self, depth):
    
        intrinsics = depth.profile.as_video_stream_profile().intrinsics
        depth = np.asanyarray(depth.get_data()) * self.depth_scale # 1000 mm => 0.001 meters
        rows,cols  = depth.shape

        c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
        r = r.astype(float)
        c = c.astype(float)

        valid = (depth > 0) & (depth < self.clip_dist) #remove from the depth image all values above a given value (meters).
        valid = np.ravel(valid)
        z = depth 
        x =  z * (c - intrinsics.ppx) / intrinsics.fx
        y =  z * (r - intrinsics.ppy) / intrinsics.fy
   
        z = np.ravel(z)[valid]
        x = np.ravel(x)[valid]
        y = np.ravel(y)[valid]
    
        pointsxyz = np.dstack((z, -x, -y))
        pointsxyz = pointsxyz.reshape(-1,3)

        return pointsxyz

    def image_pub(self, color_image):
        scaled_image = cv2.resize(color_image, (320, 180), interpolation=cv2.INTER_AREA)
        gray_scaled_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)
        self.image_publish.publish(self.bridge.cv2_to_imgmsg(gray_scaled_image, encoding="mono8"))

    def free_space_segmentaton(self, aligned_depth_frame):
        points = []

        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

        for x in range(848):
            for y in range(240,480):
                dist = aligned_depth_frame.get_distance(x, y)
                result = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], dist)
                z_map = round(-result[1],3)
                if self.lower_threshold < z_map < self.upper_threshold:
                    x_map = round(result[2],3)
                    y_map = round(-result[0],3) + 0.037

                    points.append([x_map,y_map,z_map])

                else:
                    continue
                
        return points
    
    def pointcloud_pub(self, points):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"

        cloud = point_cloud2.create_cloud_xyz32(header, points)

        self.cloud_publish.publish(cloud)

    
    def shutdown(self):
        self.pipeline.stop()
        self.get_logger().info("RealSense node stopped.")


def main(args=None):
    rclpy.init(args=args)
    node = RealSenseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.shutdown()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

