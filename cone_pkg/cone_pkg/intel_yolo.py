import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import Point
from visualization_msgs.msg import MarkerArray,Marker
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import copy
from cv_bridge import CvBridge
from ultralytics import YOLO

class RealSenseNode(Node):
    def __init__(self):
        super().__init__('realsense_node')
        #Object tracking parameters
        self.prev_dataset = None
        self.match_radius = 0.1 #object tracking radius
        self.min_frame = 0 #appearance protection
        self.max_frame = 5 #disapearance protection
        #New parameters
        self.max_distance = 0.8
        self.min_angle = 60
        self.max_angle = 120
        self.pre_point_dist = 0.4

        self.bridge = CvBridge()
        
        # ROS 2 Publisher
        self.det_image_pub = self.create_publisher(Image, "/ultralytics/detection/image", 1)
        self.point_pub = self.create_publisher(Point,"/target_point",10)
        self.rviz_pub = self.create_publisher(MarkerArray,"/rviz",1)
        self.imu_pub = self.create_publisher(Imu,'/camera/imu',1)

        #CLI:
        #yolo export model=Cone.pt format=engine imgsz=640#
        
        # YOLO model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "Cone.engine")
        #the model has to be in the same folder with the node
        self.trt_model = YOLO(model_path)

        
        #Stereo Camera
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.accel)
        config.enable_stream(rs.stream.gyro)

        self.pipeline.start(config)

        #color frame align to depth frame
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        
        self.timer = self.create_timer(0.033, self.capture_frames)  # 30 FPS
        self.get_logger().info("RealSense node started.")
    
    def capture_frames(self):
        current_dataset = []
        
        frames = self.pipeline.wait_for_frames()

        self.imu_publication(frames)    

        aligned_frames = self.align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not aligned_depth_frame or not color_frame:
            return
        
        color_image = np.asanyarray(color_frame.get_data())

        if self.det_image_pub.get_subscription_count() >= 0:
            det_annotated, center_points = self.object_identification(color_image, aligned_depth_frame)
            self.image_pub(det_annotated)
            current_dataset = self.pixel_to_point(center_points, current_dataset, aligned_depth_frame)
            
            if self.prev_dataset == None:
                self.prev_dataset = copy.deepcopy(current_dataset)

            else:
                self.prev_dataset = copy.deepcopy(self.update(current_dataset))

            #Appearance filtering
            filtered_dataset = [item for item in self.prev_dataset if item[4] >= self.min_frame]

            half_points = []
            if len(filtered_dataset) >= 2:
                xyz = [[x, y,z] for x, y, z, _ ,_,_ in filtered_dataset]
                half_points = self.gate_finding(xyz)
            else:
                half_points = [[0,0,0]]

            self.half_points_pub(half_points)
            self.marker_creator(filtered_dataset,half_points)

    def imu_publication(self,frames):
        accel_frame = frames.first_or_default(rs.stream.accel)
        gyro_frame = frames.first_or_default(rs.stream.gyro)

        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = 'camera_imu_frame'

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
        


    def object_identification(self, color_image, aligned_depth_frame):
        center_points = []
        #Object identification
        det_result = self.trt_model(color_image, conf=0.6, imgsz=640,verbose=False)
            
        det_annotated = det_result[0].plot(show=False)
            
        for r in det_result:
            if r.boxes.xywh.numel() == 0:
                #self.get_logger().info("No object detected!")
                continue
            
            for box in r.boxes.xywh:
                x, y, _, _ = box.tolist()
                x = round(x)
                y = round(y)
                dist = aligned_depth_frame.get_distance(x, y)
                center_points.append((x,y,round(dist,2)))

        return det_annotated, center_points
            
    def image_pub(self,det_annotated):
        scaled_image = cv2.resize(det_annotated, (320, 180), interpolation=cv2.INTER_AREA)
        gray_scaled_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)
        self.det_image_pub.publish(self.bridge.cv2_to_imgmsg(gray_scaled_image, encoding="mono8"))
            
    def pixel_to_point(self,center_points, current_dataset, aligned_depth_frame):
        #Pixels to 3D coordinates
        for index, xyd in enumerate(center_points):
            depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
            result = rs.rs2_deproject_pixel_to_point(depth_intrin, [xyd[0], xyd[1]], xyd[2])
            x_map = round(result[2],3)
            y_map = round(-result[0],3) + 0.037
            z_map = round(-result[1],3)
            current_dataset.append([x_map,y_map,z_map,index,0,0])

        return current_dataset
    
    def half_points_pub(self, half_points):
        #half point publication
        if half_points:
            if half_points[0][0] != 0 and half_points[0][1] != 0 and half_points[0][2] != 0:
                point_to_pub = Point()
                point_to_pub.x = float(half_points[0][0])
                point_to_pub.y = float(half_points[0][1])
                point_to_pub.z = float(half_points[0][2])
                self.point_pub.publish(point_to_pub)
            
    def gate_finding(self,cones):
        pairs = []
        used = set()

        for i in range(len(cones)):
            if i in used:
                continue
            for j in range(i + 1, len(cones)):
                if j in used:
                    continue

                c1, c2 = np.array(cones[i][:2]), np.array(cones[j][:2])
                dist = np.linalg.norm(c1 - c2)

                if dist > self.max_distance:
                    continue
                if not self.is_gate_direction(c1, c2):
                    continue

                #Perpendicular point
                x,y = self.compute_perpendicular_point(c1,c2,self.pre_point_dist)
                dist_p = np.sqrt(x**2+y**2)
                if dist_p > 0.4:
                    p = [x,y,cones[i][2]]
                    pairs.append(p)

                #Middle point
                c1 = [c1[0],c1[1],cones[i][2]]
                c2 = [c2[0],c2[1],cones[j][2]]
                gate = self.mean_point_calc(c1,c2)
                pairs.append(gate)

                used.add(i)
                used.add(j)
                break 

        if not pairs:
            pairs = [[0,0,0]]

        return pairs
    
    def is_gate_direction(self,c1, c2):
        v = c2 - c1
        direction = np.array([1.0, 0.0])
        angle = self.angle_between(v, direction)
        return self.min_angle <= angle <= self.max_angle
    
    def angle_between(self,v1, v2):
        unit_v1 = v1 / np.linalg.norm(v1)
        unit_v2 = v2 / np.linalg.norm(v2)
        dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        return np.degrees(angle_rad)
    
    def compute_perpendicular_point(self,c1, c2, x):
        #Midpoint
        mid = (c1 + c2) / 2.0

        #vector between the two cone
        v = c2 - c1
        v = v / np.linalg.norm(v)

        #Perpendicular vector in the both way
        perp1 = np.array([-v[1], v[0]])
        perp2 = -perp1

        #Two pre-point in the both way
        p1 = mid + perp1 * x
        p2 = mid + perp2 * x

        #Distances between the origo and the pre-points
        d1 = np.linalg.norm(p1)
        d2 = np.linalg.norm(p2)

        #Choosing the closer point
        p = p1 if d1 < d2 else p2

        x,y = p[0], p[1]
        return x,y

    def update(self, current):
        #Object tracking
        halmaz = set()
        if (len(current)>= len(self.prev_dataset)):
            for j in range(0,len(current)):
                halmaz.add(current[j][3])
                current[j][3] = np.nan
        else:
            for j in range(0,len(self.prev_dataset)):
                halmaz.add(self.prev_dataset[j][3])

        for i in range(0,len(current)):
            for j in range(0,len(self.prev_dataset)):
                dist = self.distance(self.prev_dataset[j],current[i])
                if(dist < self.match_radius):
                    mean_point = self.mean_point_calc(self.prev_dataset[j],current[i])
                    current[i][0] = mean_point[0]
                    current[i][1] = mean_point[1]
                    current[i][2] = mean_point[2]
                    current[i][3] = self.prev_dataset[j][3]
                    
                    if self.prev_dataset[j][4] < 100:
                        current[i][4] = self.prev_dataset[j][4] + 1
                    else:
                        current[i][4] = self.prev_dataset[j][4]
                    if self.prev_dataset[j][3] in halmaz:
                        halmaz.remove(self.prev_dataset[j][3])
                    break
        
        """
        #Debug
        s = "Current: {" + ",".join(str(row[3]) for row in current) + "}"
        prev_s = "Prev: {" + ",".join(str(row[3]) for row in self.prev_dataset) + "}"
        self.get_logger().info(s)
        self.get_logger().info(prev_s)
        self.get_logger().info("--------------")
        """

        for i in range(0,len(current)):
            if np.isnan(current[i][3]):
                current[i][3] = halmaz.pop()

        #Disappearance filtering
        new_ids = {row[3] for row in current}
        for row in self.prev_dataset:
            if row[3] not in new_ids and row[5] < self.max_frame:
                row[5] += 1
                current.append(row)

        return current
    

    def distance(self,p1,p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

    def mean_point_calc(self,p1,p2):
        dx = round((p1[0]+p2[0]) /2,3)
        dy = round((p1[1]+p2[1]) /2,3)
        dz = round((p1[2]+p2[2]) /2,3)
        return (dx,dy,dz)
    
    def marker_creator(self, cones,half_points):
        cone_array = MarkerArray()
        
        for point in cones:
            if point[0] == 0 and point[1] == 0.037 and point[2] == 0:
                continue

            cone = Marker()

            #Cones
            cone.header.frame_id = "map"
            cone.ns = "cone"
            cone.id = int(point[3])
            cone.type = Marker.SPHERE
            cone.action = Marker.ADD

            #Position
            cone.pose.position.x = point[0]
            cone.pose.position.y = point[1]
            cone.pose.position.z = point[2] + 0.1

            #Orientation
            cone.pose.orientation.x = 0.0
            cone.pose.orientation.y = 0.0
            cone.pose.orientation.z = 0.0
            cone.pose.orientation.w = 1.0 

            #Size
            cone.scale.x = 0.1
            cone.scale.y = 0.1
            cone.scale.z = 0.1
            
            #Color
            cone.color.a = 1.0
            cone.color.r = 255.0/255.0
            cone.color.g = 165.0/255.0
            cone.color.b = 0.0

            cone.lifetime = Duration(seconds=0.1).to_msg()

            cone_array.markers.append(cone)
        
        for point in cones:
            if point[0] == 0 and point[1] == 0.037 and point[2] == 0:
                continue

            text = Marker()

            #Text above the cones
            text.header.frame_id = "map"
            text.ns = "ID_Text"
            text.id = int(point[3]) + 100
            text.type = Marker.TEXT_VIEW_FACING
            text.text = f"#{point[3]}"

            text.action = Marker.ADD

            #Position
            text.pose.position.x = point[0]
            text.pose.position.y = point[1]
            text.pose.position.z = point[2] + 0.25

            #Orientation
            text.pose.orientation.x = 0.0
            text.pose.orientation.y = 0.0
            text.pose.orientation.z = 0.0
            text.pose.orientation.w = 1.0 

            #Size
            text.scale.x = 0.1
            text.scale.y = 0.1
            text.scale.z = 0.1
            
            #Color
            text.color.a = 1.0
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0

            text.lifetime = Duration(seconds=0.1).to_msg()

            cone_array.markers.append(text)

        for i,half_point in enumerate(half_points):
            if half_point[0] == 0 and half_point[1] == 0 and half_point[2] == 0:
                continue

            half_marker = Marker()

            #Mid points
            half_marker.header.frame_id = "map"
            half_marker.ns = "half_point"
            half_marker.id = 200+i
            half_marker.type = Marker.SPHERE
            half_marker.action = Marker.ADD

            #Position
            half_marker.pose.position.x = float(half_point[0])
            half_marker.pose.position.y = float(half_point[1])
            half_marker.pose.position.z = float(half_point[2]) + 0.1

            #Orientation
            half_marker.pose.orientation.x = 0.0
            half_marker.pose.orientation.y = 0.0
            half_marker.pose.orientation.z = 0.0
            half_marker.pose.orientation.w = 1.0 

            #Size
            half_marker.scale.x = 0.05
            half_marker.scale.y = 0.05
            half_marker.scale.z = 0.05
            
            #Color
            half_marker.color.a = 1.0
            half_marker.color.r = 0.0
            half_marker.color.g = 1.0
            half_marker.color.b = 0.0

            half_marker.lifetime = Duration(seconds=0.1).to_msg()

            cone_array.markers.append(half_marker)

        robot_marker = Marker()

        #Car
        robot_marker.header.frame_id = "map"
        robot_marker.ns = "robot"
        robot_marker.id = 99
        robot_marker.type = Marker.CUBE
        robot_marker.action = Marker.ADD

        #Position
        robot_marker.pose.position.x = 0.0
        robot_marker.pose.position.y = 0.0
        robot_marker.pose.position.z = 0.0

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

        cone_array.markers.append(robot_marker)

        self.rviz_pub.publish(cone_array)  
    
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

