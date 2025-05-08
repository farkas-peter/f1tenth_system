import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import MarkerArray,Marker
import pyrealsense2 as rs
import numpy as np
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO
import pygame

class RealSenseNode(Node):
    def __init__(self):
        super().__init__('realsense_node')
        #Object tracking parameters
        self.prev_dataset = None
        self.match_radius = 0.1 #object tracking radius
        self.min_frame = 0 #appearance protection
        self.max_frame = 5 #disapearance protection
        self.sensing_depth = 0.15 #distance between gates
        #New parameters
        self.max_distance = 0.6
        self.min_angle = 60
        self.max_angle = 120

        self.bridge = CvBridge()
        
        # ROS 2 Publisher
        self.det_image_pub = self.create_publisher(Image, "/ultralytics/detection/image", 10)
        self.point_pub = self.create_publisher(Point,"/target_point",10)
        self.rviz_pub = self.create_publisher(MarkerArray,"/rviz",10)

        #CLI:
        #yolo export model=Cone.pt format=engine imgsz=640#
        
        # YOLO model  
        self.trt_model = YOLO('/workspace/src/ros2_f1tenth/f1tenth/object_detection/Cone.engine')
        
        #pygame
        """
        pygame.init()
        self.colors = {}
        self.running = True
        self.screen_width, self.screen_height = 640, 480
        self.screen = pygame.display.set_mode((self.screen_width,self.screen_height))
        pygame.display.set_caption("Cone locations")  
        self.scale = 150
        self.offset_x, self.offset_y = 0,320
        self.robot_triangle_coordinates = [(self.screen_width//2-10, 0),(self.screen_width//2+10, 0),(self.screen_width//2, 20)]
        #----
        """
        #Stereo Camera
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        
        self.pipeline.start(config)

        #color frame align to depth frame
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        
        self.timer = self.create_timer(0.033, self.capture_frames)  # 30 FPS
        self.get_logger().info("RealSense node started.")
    
    def capture_frames(self):
        center_points = []
        current_dataset = []
        
        frames = self.pipeline.wait_for_frames()

        aligned_frames = self.align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not aligned_depth_frame or not color_frame:
            return
        
        color_image = np.asanyarray(color_frame.get_data())

        if self.det_image_pub.get_subscription_count() >= 0:
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

            scaled_image = cv2.resize(det_annotated, (640, 360), interpolation=cv2.INTER_AREA)
            self.det_image_pub.publish(self.bridge.cv2_to_imgmsg(scaled_image, encoding="bgr8"))
            
            #Pixels to 3D coordinates
            for index, xyd in enumerate(center_points):
                depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
                result = rs.rs2_deproject_pixel_to_point(depth_intrin, [xyd[0], xyd[1]], xyd[2])
                x_map = round(result[2],3)
                y_map = round(-result[0],3) + 0.037
                z_map = round(-result[1],3)
                current_dataset.append([x_map,y_map,z_map,index,0,0])

            if self.prev_dataset == None:
                self.prev_dataset = current_dataset
            else:
                self.prev_dataset = self.update(current_dataset)

            #Appearance filtering
            filtered_dataset = [item for item in self.prev_dataset if item[4] >= self.min_frame]

            half_points = []
            if len(filtered_dataset) >= 2:
                #half_point = self.half_point_calc(filtered_dataset)
                xyz = [[x, y,z] for x, y, z, _ ,_,_ in filtered_dataset]
                half_points = self.gate_finding(xyz)
            else:
                half_points = [[0,0,0]]

            #self.get_logger().info(str(half_points))

            #half point publication
            if half_points:
                if half_points[0][0] != 0 and half_points[0][1] != 0 and half_points[0][2] != 0:
                    point_to_pub = Point()
                    point_to_pub.x = float(half_points[0][0])
                    point_to_pub.y = float(half_points[0][1])
                    point_to_pub.z = float(half_points[0][2])
                    self.point_pub.publish(point_to_pub)

            self.marker_creator(filtered_dataset,half_points)
            """
            self.render_map(filtered_dataset, half_point)
            self.get_logger().info("Mapping:")
            self.get_logger().info(str(filtered_dataset))
            self.get_logger().info("Half point:")
            self.get_logger().info(str(half_point))
            """   
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

                #pairs.append([i, j,0])
                c1 = [c1[0],c1[1],cones[i][2]]
                c2 = [c2[0],c2[1],cones[j][2]]
                gate = self.mean_point_calc(c1,c2)
                pairs.append(gate)
                used.add(i)
                used.add(j)
                break  # csak egy párt keresünk egy bójához

        if not pairs:
            pairs = [[0,0,0]]

        return pairs
    
    def is_gate_direction(self,c1, c2):
        v = c2 - c1
        direction = np.array([1.0, 0.0])  # autó vízszintesen halad
        angle = self.angle_between(v, direction)
        return self.min_angle <= angle <= self.max_angle
    
    def angle_between(self,v1, v2):
        unit_v1 = v1 / np.linalg.norm(v1)
        unit_v2 = v2 / np.linalg.norm(v2)
        dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        return np.degrees(angle_rad)



    def half_point_calc(self, dataset):
        #Calculation of the first gate
        closest_point = None
        ref_dist = 10000
        del_index1 = 0
        for index, point in enumerate(dataset):
            x,y,z = point[0],point[1],point[2]
            dist = self.distance((x,y,z),(0,0,0))
            if ref_dist >= dist:
                ref_dist = dist
                closest_point = [x,y,z]
                del_index1 = index
                
        sec_closest_point = None
        ref_dist2 = 10000
        del_index2 = 0
        for index, point in enumerate(dataset):
            if index == del_index1:
                continue
            x,y,z = point[0],point[1],point[2]
            dist = self.distance((x,y,z),(0,0,0))
            if ref_dist2 >= dist:
                ref_dist2 = dist
                sec_closest_point = [x,y,z]
                del_index2 = index

        
        if (ref_dist+self.sensing_depth) >= ref_dist2 and (ref_dist-self.sensing_depth) <= ref_dist2:
            #Half point calculation between two cones
            half_point = self.mean_point_calc(closest_point,sec_closest_point)
        elif (ref_dist-self.sensing_depth) <= ref_dist2 and len(dataset) >= 3:
            #Calculation of the next gate if the vehicle reach a gate
            third_closest_point = None
            ref_dist3 = 10000
            for index, point in enumerate(dataset):
                if index == del_index1 or index == del_index2:
                    continue
                x,y,z = point[0],point[1],point[2]
                dist = self.distance((x,y,z),(0,0,0))
                if ref_dist3 >= dist:
                    ref_dist3 = dist
                    third_closest_point = [x,y,z]

            half_point = self.mean_point_calc(sec_closest_point,third_closest_point)
        else:
            half_point = [0,0,0]

        return half_point

    
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

            cone = Marker()

            #Pont letrehozasa
            cone.header.frame_id = "map"
            cone.ns = "cone"
            cone.id = int(point[3])
            cone.type = Marker.SPHERE
            cone.action = Marker.ADD

            #Pozicio
            cone.pose.position.x = point[0]
            cone.pose.position.y = point[1]
            cone.pose.position.z = point[2]

            cone.pose.orientation.x = 0.0
            cone.pose.orientation.y = 0.0
            cone.pose.orientation.z = 0.0
            cone.pose.orientation.w = 1.0 

            cone.scale.x = 0.1
            cone.scale.y = 0.1
            cone.scale.z = 0.1
            
            #szin
            cone.color.a = 1.0
            cone.color.r = 255.0/255.0
            cone.color.g = 165.0/255.0
            cone.color.b = 0.0

            cone.lifetime = Duration(seconds=0.5).to_msg()

            cone_array.markers.append(cone)

        for i,half_point in enumerate(half_points):
            half_marker = Marker()

            #Pont letrehozasa
            half_marker.header.frame_id = "map"
            half_marker.ns = "half_point"
            half_marker.id = 100+i
            half_marker.type = Marker.SPHERE
            half_marker.action = Marker.ADD

            #Pozicio
            half_marker.pose.position.x = float(half_point[0])
            half_marker.pose.position.y = float(half_point[1])
            half_marker.pose.position.z = float(half_point[2])

            half_marker.pose.orientation.x = 0.0
            half_marker.pose.orientation.y = 0.0
            half_marker.pose.orientation.z = 0.0
            half_marker.pose.orientation.w = 1.0 

            half_marker.scale.x = 0.05
            half_marker.scale.y = 0.05
            half_marker.scale.z = 0.05
            
            #szin
            half_marker.color.a = 1.0
            half_marker.color.r = 0.0
            half_marker.color.g = 1.0
            half_marker.color.b = 0.0

            half_marker.lifetime = Duration(seconds=0.5).to_msg()

            cone_array.markers.append(half_marker)

        robot_marker = Marker()

        #Pont letrehozasa
        robot_marker.header.frame_id = "map"
        robot_marker.ns = "robot"
        robot_marker.id = 99
        robot_marker.type = Marker.CUBE
        robot_marker.action = Marker.ADD

        #Pozicio
        robot_marker.pose.position.x = 0.0
        robot_marker.pose.position.y = 0.0
        robot_marker.pose.position.z = 0.0

        robot_marker.pose.orientation.x = 0.0
        robot_marker.pose.orientation.y = 0.0
        robot_marker.pose.orientation.z = 0.0
        robot_marker.pose.orientation.w = 1.0 

        robot_marker.scale.x = 0.2
        robot_marker.scale.y = 0.2
        robot_marker.scale.z = 0.3
            
        #szin
        robot_marker.color.a = 1.0
        robot_marker.color.r = 1.0
        robot_marker.color.g = 0.0
        robot_marker.color.b = 0.0

        robot_marker.lifetime = Duration(seconds=0.5).to_msg()

        cone_array.markers.append(robot_marker)


        self.rviz_pub.publish(cone_array)

    def render_map(self, dataset,half_point):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        self.screen.fill((255,255,255))
        font = pygame.font.Font(None, 20)

        #Vehicle and view angle drawing
        pygame.draw.polygon(self.screen, [255,0,0], self.robot_triangle_coordinates)
        pygame.draw.line(self.screen, [128,128,128], (self.screen_width//2, 20), (600, 400), 2)
        pygame.draw.line(self.screen, [128,128,128], (self.screen_width//2, 20), (40, 400), 2)

        #Half point drawing
        if half_point != [0,0,0]:
            green_x, green_y = self.transform_point(half_point[0], half_point[1])
            pygame.draw.circle(self.screen,[0,128,0], (green_y, green_x), 4)

        #Cones drawing
        for point in dataset:
            x, y, z, index = point[0], point[1], point[2], point[3]
            screen_x, screen_y = self.transform_point(x, y)
            pygame.draw.circle(self.screen,[255,165,0], (screen_y, screen_x), 5)
            if index == 0:
                point_label = font.render(f"x = {x*100:.2f}cm, y = {y*100:.2f}cm, z = {z*100:.2f}cm",[0,0,0], True)
                self.screen.blit(point_label,(5,460))
            point_id = font.render(f"#{index}",[0,0,0], True)
            self.screen.blit(point_id,(screen_y-10,screen_x+10))
       
        pygame.display.flip()   

    def transform_point(self,x, y):
        screen_x = int(x * self.scale + self.offset_x)
        screen_y = int(y * self.scale + self.offset_y)
        return screen_x, screen_y

    
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

