import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
import numpy as np
import cv2
from cv_bridge import CvBridge
import open3d as o3d

class PcdNode(Node):
    def __init__(self):
        super().__init__('pcd_node')
        #Parameters
        self.width = 640
        self.height = 480
        self.clip_dist = 3.0
        self.zmax = 0.2
        self.zmin = 0.05
        self.vehicle_width = 0.3

        #RANSAC parameters
        self.RANSAC_on = True
        self.dist_threshold = 0.05
        self.ransac_n = 3
        self.num_iterations = 1000
        self.tilt_tolerance = 10.0
        self.max_planes = 3
        
        # ROS 2 Publishers
        self.cloud_publish = self.create_publisher(PointCloud2, "/pointcloud", 1)

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

        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)

        # Start streaming
        self.pipeline.start(config)
        
        self.timer = self.create_timer(0.033, self.capture_frames)  # 30 FPS

        self.get_logger().info('Pcd node started.')

    def capture_frames(self):
        frames = self.pipeline.wait_for_frames()

        #Filtering
        self.decimate = rs.decimation_filter()
        self.decimate.set_option(rs.option.filter_magnitude, 2)

        depth_frame = frames.get_depth_frame()
        depth_frame = self.decimate.process(depth_frame)
        if not depth_frame:
            return
        
        #Image points to 3D points
        points_xyz = self.depth2PointCloud(depth_frame)

        #Additional fltering for performance
        points_xyz = self.random_subsample(points_xyz, 20000)

        if self.RANSAC_on:
            #RANSAC segmentation
            _, points_xyz = self.RANSAC_segmentation(points_xyz)
            points_xyz = self.ground_filter(points_xyz)

        #Pointcloud publication
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
        y = np.ravel(y)[valid] - 0.2
    
        pointsxyz = np.dstack((z, -x, -y))
        pointsxyz = pointsxyz.reshape(-1,3)

        return pointsxyz

    def random_subsample(self, points: np.ndarray, max_points: int = 40000) -> np.ndarray:
        if len(points) <= max_points:
            return points
        idx = np.random.choice(len(points), size=max_points, replace=False)
        return points[idx]
    
    def RANSAC_segmentation(self, points):
        if len(points) == 0:
            return points, points
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        z_axis = np.array([0.0, 0.0, 1.0])
        chosen_plane = None

        for i in range(self.max_planes):
            if len(np.asarray(pcd.points)) < 50:
                break

            plane_model, inliers = pcd.segment_plane(self.dist_threshold, self.ransac_n, self.num_iterations)
            a, b, c, d = plane_model
            normal = np.array([a, b, c], dtype=np.float64)
            normal /= np.linalg.norm(normal) + 1e-12

            angle = np.degrees(np.arccos(np.clip(abs(np.dot(normal, z_axis)), -1.0, 1.0)))

            if angle <= self.tilt_tolerance:
                chosen_plane = plane_model
                break

            pcd = pcd.select_by_index(inliers, invert=True)

        if chosen_plane is None:
            return points, points

        a, b, c, d = chosen_plane
        normal = np.array([a, b, c], dtype=np.float64)
        normal_norm = np.linalg.norm(normal) + 1e-12
        dist = np.abs(points @ normal + d) / normal_norm

        ground_mask = dist <= self.dist_threshold
        ground_points = points[ground_mask]
        object_points = points[~ground_mask]

        return ground_points, object_points
    
    def ground_filter(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return points
        mask = (points[:,2] >= self.zmin) & (points[:,2] <= self.zmax)
        return points[mask]
    
    def pointcloud_pub(self, points):
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "map"

        cloud = point_cloud2.create_cloud_xyz32(header, points)

        self.cloud_publish.publish(cloud)
        

def main(args=None):
    rclpy.init(args=args)
    node = PcdNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()
