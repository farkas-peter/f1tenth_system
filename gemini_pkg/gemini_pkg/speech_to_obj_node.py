import cv2
import os
import time
import requests
import pyrealsense2 as rs
import numpy as np
import copy
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import base64
from io import BytesIO

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Point

from .webcam_audio_rec import record_webcam_audio


class SpeechToObject(Node):
    def __init__(self):
        super().__init__('speech_to_object_node')

        # Initialize variables
        self.bbox = None
        self.tracker = None
        self.busy = False
        self.prev_back_button = 0
        self.visualize  False

        self.thread_executor = ThreadPoolExecutor(max_workers=2)

        # Camera variables
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        # ROS2 variables
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.point_pub = self.create_publisher(Point, "/target_point", 10)
        self.timer = self.create_timer(0.033, self.camera_callback)  # 30 FPS

        self.get_logger().info("SpeechToObject node started.")

    def camera_callback(self):
        # Wait for frame and align the color and depth frames
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return

        color_image = np.asanyarray(color_frame.get_data())

        if self.bbox is not None and self.tracker is None:
            self.tracker = cv2.legacy.TrackerCSRT_create()
            initialized = self.tracker.init(color_image, self.bbox)

        if self.tracker is not None:
            ok, self.bbox = self.tracker.update(color_image)

            if ok:
                x, y, w, h = map(int, self.bbox)
                steering = self.compute_steering()
                if self.visualize:
                    cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(color_image, f"Steering: {steering:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                # Get world coordinates of the center of the bounding box
                center_x = x + w / 2
                center_y = y + h / 2
                x_world, y_world, z_world = self.pixel_to_world(center_x, center_y, depth_frame)
                self.publish_target_point((x_world, y_world, z_world))
                #self.get_logger().info(f"Object World Coordinates: x={round(x_world,2)} m, y={round(y_world,2)} m, z={z_world} m")
            else:
                self.get_logger().info("Tracking lost. Please provide a new bounding box.")
                self.tracker = None
                self.bbox = None

        # Visualize
        if self.visualize:
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', color_image)
            cv2.waitKey(1)

    def tracking(self, frame, bbox):
        tracker = cv2.legacy.TrackerCSRT_create()
        initialized = tracker.init(frame, bbox)  # bbox = (x, y, w, h)

        if initialized is not None and tracker is not None:
            ok, bbox = tracker.update(frame)

            if ok:
                # Bbox can be drawn
                pass
            else:
                # Tracking lost
                # Tracker should be initialized again
                pass

    def pixel_to_world(self, x, y, aligned_depth_frame):
        # Convert pixel coordinates to world coordinates using depth frame
        x = round(x)
        y = round(y)
        dist = aligned_depth_frame.get_distance(x, y)

        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        result = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], dist)
        x_world = round(result[2], 3)
        y_world = round(-result[0], 3) + 0.037
        z_world = round(-result[1], 3)

        return x_world, y_world, z_world
    
    def distance(self,p1,p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)
    
    def publish_target_point(self, point):
        # Calculate look-ahead distance
        ld = self.distance((0,0,0), point)

        # If the object is too close, do not publish target point, so the car stops
        if ld < 0.5:
            return

        if point != (0, 0, 0):
            point_to_pub = Point()
            point_to_pub.x = float(point[0])
            point_to_pub.y = float(point[1])
            point_to_pub.z = float(ld)
            self.point_pub.publish(point_to_pub)

    def joy_callback(self, msg):
        # For the Start button, start the speech to object detection process
        if msg.buttons[7] == 1 and not self.busy:
            self.busy = True
            self.get_logger().info("START: Starting speech to object pipeline.")
            self.thread_executor.submit(self.run_speech_to_obj_det)
        
        # For the back button, clean the tracker and bbox
        back_now = msg.buttons[6]
        if self.prev_back_button == 0 and back_now == 1:
            self.get_logger().info("BACK: Resetting tracker.")
            self.tracker = None
            self.bbox = None
        
        self.prev_back_button = back_now

    def run_speech_to_obj_det(self):
        try:
            # Record audio from webcam
            audio_path = record_webcam_audio(duration=3, filename="temp_audio.wav")

            # Capture image from camera
            color_image = self.capture_one_cam_img()
            base64_image = self.numpy_image_to_base64(color_image)

            # Call Gemini agent server to process audio and get bounding box
            desc, bbox = self.call_gemini_agent(audio_path, base64_image)
            self.get_logger().info(f"Answer from Gemini: {desc} | {bbox}")

            # Set variables
            self.bbox = copy.deepcopy(bbox)
            self.busy = False

        except Exception as e:
            self.get_logger().error(f'Error in audio worker: {e}')

        finally:
            self.get_logger().info('Speech to object detection task completed.')
            self.busy = False

    def call_gemini_agent(self, audio_path, base64_image):
        # Send audio file path to the Gemini server for processing
        response = requests.post(
            "http://127.0.0.1:8000/run_agent_pipeline",
            json={"audio_path": audio_path,
                  "base64_image": base64_image}
        )
        response = response.json()

        # Unpack the response
        desc = response.get("description", "")
        bbox = response.get("bb_list", [])
        bbox = self.gemini_bbox_to_csrt_bbox(bbox[0])

        return desc, bbox
    
    def capture_one_cam_img(self):
        # Capture an image from the camera
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        return color_image
    
    @staticmethod
    def numpy_image_to_base64(img_array: np.ndarray) -> str:
        # Convert a NumPy image array (H x W x C) into a Base64 PNG string.

        # Ensure array is uint8
        if img_array.dtype != np.uint8:
            img_array = img_array.astype(np.uint8)

        # Convert NumPy array to PIL image
        img = Image.fromarray(img_array)

        # Save to buffer as PNG
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        # Encode to base64
        img_bytes = buffer.getvalue()
        base64_str = base64.b64encode(img_bytes).decode("utf-8")

        return base64_str
    
    @staticmethod
    def gemini_bbox_to_csrt_bbox(bbox):
        # Converts [x1, y1, x2, y2] to [x1, y1, w, h]
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        return [x1, y1, w, h]
    
    def compute_steering(self, frame_width=848, Kp=0.5):
        x, y, w, h = map(int, self.bbox)
        bbox_center = x + w / 2
        x_center = frame_width / 2
        error = (bbox_center - x_center) / (frame_width / 2)
        steering = Kp * error

        return max(-1, min(1, steering))

    def shutdown(self):
        self.pipeline.stop()
        self.get_logger().info("ObjTrackerNode node stopped.")


def main(args=None):
    rclpy.init(args=args)
    node = SpeechToObject()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.shutdown()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()