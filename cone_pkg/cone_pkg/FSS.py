import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker
from std_msgs.msg import Header
import pyrealsense2 as rs
import numpy as np
import cv2
from cv_bridge import CvBridge
import open3d as o3d
import time

class RealSenseNode(Node):
    def __init__(self):
        super().__init__('realsense_node')
        #Parameters
        self.width = 640
        self.height = 480
        self.clip_dist = 3.0
        self.zmax = 0.3
        self.zmin = -0.3
        #RANSAC parameters
        self.dist_threshold = 0.05
        self.ransac_n = 3
        self.num_iterations = 1000

        self.bridge = CvBridge()
        
        # ROS 2 Publisher
        self.image_publish = self.create_publisher(Image, "/gray_image", 1)
        self.cloud_publish = self.create_publisher(PointCloud2, "/free_space", 1)
        self.marker_publish= self.create_publisher(Marker, "/car",1)

        #Stereo Camera
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        depth_sensor = device.first_depth_sensor()

        #Auto-exposure setting
        depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
        xmin, xmax = int(0.05*self.width), int(0.95*self.width)
        ymin, ymax = int(0.5*self.height), int(0.95*self.height)
        roi_sensor = depth_sensor.as_roi_sensor()
        roi = rs.region_of_interest()
        roi.min_x, roi.min_y = xmin, ymin
        roi.max_x, roi.max_y = xmax, ymax
        roi_sensor.set_region_of_interest(roi)
        
        # Get depth scale of the device
        self.depth_scale =  depth_sensor.get_depth_scale()
        # Create an align object
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)
        
        # Start streaming
        self.pipeline.start(config)
        
        self.timer = self.create_timer(0.033, self.capture_frames)  # 30 FPS
        self.get_logger().info("FSS node started.")
    
    def capture_frames(self):
        
        frames = self.pipeline.wait_for_frames()

        #Filtering
        self.decimate = rs.decimation_filter()
        self.decimate.set_option(rs.option.filter_magnitude, 2)

        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        depth_frame = self.decimate.process(depth_frame)
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return
        
        
        #color_image = np.asanyarray(color_frame.get_data())
        #self.image_pub(color_image)

        """
        start = time.time()
        
        end = time.time()
        elapsed = end - start
        self.get_logger().info(f"Run time: {elapsed:.3f}s")
        """

        #Image points to 3D points
        points_xyz = self.depth2PointCloud(depth_frame)
        
        #Ground floor filtering
        #points_xyz = self.ground_filter(points_xyz)

        #Additional fltering for performance
        #points_xyz = self.random_subsample(points_xyz, 20000)

        #RANSAC segmentation
        _, points_xyz = self.RANSAC_segmentation(points_xyz)

        #Pointcloud publication
        self.pointcloud_pub(points_xyz)

        #Car marker publication
        self.marker_pub()

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
        y = np.ravel(y)[valid] - 0.2
    
        pointsxyz = np.dstack((z, -x, -y))
        pointsxyz = pointsxyz.reshape(-1,3)

        return pointsxyz

    def image_pub(self, color_image):
        scaled_image = cv2.resize(color_image, (320, 180), interpolation=cv2.INTER_AREA)
        gray_scaled_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)
        self.image_publish.publish(self.bridge.cv2_to_imgmsg(gray_scaled_image, encoding="mono8"))

    def pointcloud_pub(self, points):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"

        cloud = point_cloud2.create_cloud_xyz32(header, points)

        self.cloud_publish.publish(cloud)

    def ground_filter(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return points
        mask = (points[:,2] >= self.zmin) & (points[:,2] <= self.zmax)
        return points[mask]
    
    def RANSAC_segmentation(self, points):
        if len(points) == 0:
            return points
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        plane_model, inliers = pcd.segment_plane(self.dist_threshold, self.ransac_n, self.num_iterations)
        a, b, c, d = plane_model

        normal = np.array([a, b, c], dtype=np.float64)
        normal_norm = np.linalg.norm(normal) + 1e-12
        dist = np.abs(points @ normal + d) / normal_norm

        ground_mask = dist <= self.dist_threshold
        ground_points = points[ground_mask]
        object_points = points[~ground_mask]

        return ground_points, object_points

    def random_subsample(self, points: np.ndarray, max_points: int = 40000) -> np.ndarray:
        if len(points) <= max_points:
            return points
        idx = np.random.choice(len(points), size=max_points, replace=False)
        return points[idx]
    
    def marker_pub(self):
        robot_marker = Marker()

        #Car
        robot_marker.header.frame_id = "map"
        robot_marker.ns = "robot"
        robot_marker.id = 0
        robot_marker.type = Marker.CUBE
        robot_marker.action = Marker.ADD

        #Position
        robot_marker.pose.position.x = 0.0
        robot_marker.pose.position.y = 0.0
        robot_marker.pose.position.z = 0.075

        #Orientation
        robot_marker.pose.orientation.x = 0.0
        robot_marker.pose.orientation.y = 0.0
        robot_marker.pose.orientation.z = 0.0
        robot_marker.pose.orientation.w = 1.0 

        #Size
        robot_marker.scale.x = 0.3
        robot_marker.scale.y = 0.2
        robot_marker.scale.z = 0.15
            
        #Color
        robot_marker.color.a = 1.0
        robot_marker.color.r = 1.0
        robot_marker.color.g = 0.0
        robot_marker.color.b = 0.0

        robot_marker.lifetime = Duration(seconds=0.1).to_msg()

        self.marker_publish.publish(robot_marker)

    
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

