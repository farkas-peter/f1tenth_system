import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Point
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
        self.clip_dist = 2.0
        self.zmax = 0.2
        self.zmin = 0.05
        self.vehicle_width = 0.3

        #RANSAC parameters
        self.dist_threshold = 0.05
        self.ransac_n = 3
        self.num_iterations = 1000
        self.tilt_tolerance = 10.0
        self.max_planes = 3

        #FGM parameters
        self.fov_deg = 87.0
        self.bin_deg = 0.5
        self.margin = 0.1
        self.min_lookahead = 1.0
        self.def_lookahead = 1.5
        self.max_lookahead = 2.0
        self.safe_dist = 2.0

        self.bridge = CvBridge()
        
        # ROS 2 Publishers
        self.image_publish = self.create_publisher(Image, "/gray_image", 1)
        self.cloud_publish = self.create_publisher(PointCloud2, "/pointcloud", 1)
        self.marker_publish= self.create_publisher(Marker, "/car",1)
        self.target_publish= self.create_publisher(Marker, "/target_vis",1)
        self.point_pub = self.create_publisher(Point,"/target_point",10)

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
        #config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)
        
        # Start streaming
        self.pipeline.start(config)
        
        self.timer = self.create_timer(0.033, self.capture_frames)  # 30 FPS
        self.get_logger().info("FSS node started.")
    
    def capture_frames(self):
        
        frames = self.pipeline.wait_for_frames()

        #Filtering
        self.decimate = rs.decimation_filter()
        self.decimate.set_option(rs.option.filter_magnitude, 2)

        depth_frame = frames.get_depth_frame()
        depth_frame = self.decimate.process(depth_frame)
        #color_frame = frames.get_color_frame()
        if not depth_frame:
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

        #Additional fltering for performance
        points_xyz = self.random_subsample(points_xyz, 20000)

        #RANSAC segmentation
        _, points_xyz = self.RANSAC_segmentation(points_xyz)
        points_xyz = self.ground_filter(points_xyz)

        #Follow-the-Gap-Method
        target = self.FGM(points_xyz)
        if target is None:
            #self.get_logger().info("Target not found!")
            pass
        else:
            tx, ty = float(target[0]), float(target[1])
            target_point = Point()
            target_point.x = tx
            target_point.y = ty
            target_point.z = 0.0
            self.point_pub.publish(target_point)
            self.target_pub(tx,ty)
            #self.get_logger().info(f"X: {tx:.3f}, Y: {ty:.3f}")
        
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
    
    def FGM(self, points):
        if points is None:
            return None
        
        if points.size == 0:
            return np.array([1.5,0.0])

        #Polar transformation
        pts2 = points[:, :2] if points.shape[1] >= 2 else points.copy()
        x, y = pts2[:, 0], pts2[:, 1]
        theta = np.arctan2(y, x)
        r = np.hypot(x, y)

        #FOV masking
        fov = np.deg2rad(self.fov_deg)
        half = fov / 2.0
        mask = (theta >= -half) & (theta <= +half)
        if not np.any(mask):
            #self.get_logger().info("Empty mask.")
            return np.array([1.5,0.0])
        theta = theta[mask]; r = r[mask]

        #Create of bins and point sorting
        bin_size = np.deg2rad(self.bin_deg)
        nbins = int(np.ceil(fov / bin_size))
        idx = np.clip(((theta + half) / bin_size).astype(int), 0, nbins - 1)

        rmin = np.full(nbins, np.inf, dtype=np.float32)
        np.minimum.at(rmin, idx, r)

        occupied = np.zeros(nbins, dtype=bool)
        occupied_r = np.full(nbins, np.nan, dtype=np.float32)
        for i, ri in enumerate(rmin):
            if np.isfinite(ri) and ri <=self.safe_dist:
                occupied[i] = True 
                occupied_r[i] = ri

        #Detect of gaps
        free = ~occupied
        if not np.any(free):
            #self.get_logger().info("There is no gap.")
            return None

        best_lo = best_hi = -1
        best_len = 0
        i = 0
        while i < nbins:
            if not free[i]:
                i += 1
                continue
            j = i
            while j < nbins and free[j]:
                j += 1
            if (j - i) > best_len:
                best_len = j - i
                best_lo, best_hi = i, j
            i = j

        if best_len <= 0:
            #self.get_logger().info("Best length is null.")
            return None
        
        #Dynamic lookahead distance
        left_index = best_lo - 1 if best_lo - 1 >= 0 else None
        rigth_index = best_hi if best_hi < nbins else None

        r_left = occupied_r[left_index] if left_index is not None else np.nan
        r_right = occupied_r[rigth_index] if rigth_index is not None else np.nan

        same_r = False
        if np.isnan(r_left):
            if np.isnan(r_right):
                same_r = True
            else:
                r_left = r_right

        if np.isnan(r_right):
            if np.isnan(r_left):
                same_r = True
            else:
                r_right = r_left

        r_mean = 0.0
        if (same_r):
            r_mean = self.def_lookahead
        else:
            r_mean = (r_left+r_right) / 2.0

        dyn_lookahead = float(np.clip(r_mean, self.min_lookahead, self.max_lookahead))

        #Gap width checking on lookahead distance
        ang_width = best_len * bin_size
        gap_width = 2.0 * dyn_lookahead * np.sin(ang_width / 2.0)
        
        if gap_width < (self.vehicle_width + 2.0 * self.margin):
            #self.get_logger().info(f"Gap: {gap_width:.3f}")
            return None
        #self.get_logger().info(f"Gap: {gap_width:.3f}")

        #Target point calculating in the middle of the gap
        center_idx = (best_lo + best_hi - 1) / 2.0
        theta_star = (center_idx + 0.5) * bin_size - half
        target = np.array([dyn_lookahead * np.cos(theta_star), dyn_lookahead * np.sin(theta_star)], dtype=np.float32)

        return target

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
        robot_marker.pose.position.x = -0.15
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

    def target_pub(self, x,y):
        point_marker = Marker()

        #Point
        point_marker.header.frame_id = "map"
        point_marker.ns = "target_point"
        point_marker.id = 1
        point_marker.type = Marker.SPHERE
        point_marker.action = Marker.ADD

        #Position
        point_marker.pose.position.x = x
        point_marker.pose.position.y = y
        point_marker.pose.position.z = 0.0

        #Orientation
        point_marker.pose.orientation.x = 0.0
        point_marker.pose.orientation.y = 0.0
        point_marker.pose.orientation.z = 0.0
        point_marker.pose.orientation.w = 1.0 

        #Size
        point_marker.scale.x = 0.1
        point_marker.scale.y = 0.1
        point_marker.scale.z = 0.1
            
        #Color
        point_marker.color.a = 1.0
        point_marker.color.r = 0.0
        point_marker.color.g = 1.0
        point_marker.color.b = 0.0

        point_marker.lifetime = Duration(seconds=0.1).to_msg()

        self.marker_publish.publish(point_marker)

    
    def shutdown(self):
        self.pipeline.stop()
        self.get_logger().info("FSS node stopped.")


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

