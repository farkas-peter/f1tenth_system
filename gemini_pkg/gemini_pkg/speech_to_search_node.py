import os
import time
import cv2
import requests
import pyrealsense2 as rs
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import Joy
from ackermann_msgs.msg import AckermannDriveStamped

from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge

from .webcam_audio_rec import record_webcam_audio
from .mjpeg_streamer import MjpegStreamer


class SpeechToSearch(Node):
    def __init__(self):
        super().__init__('speech_to_search_node')

        # Parameters
        default_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo26s.pt")
        self.model_path = self.declare_parameter('model_path', default_model).value
        self.conf = self.declare_parameter('conf', 0.5).value
        self.search_speed = self.declare_parameter('search_speed', 0.5).value         # m/s
        self.search_steering = self.declare_parameter('search_steering', 0.3).value   # rad
        self.phase_duration = self.declare_parameter('phase_duration', 2.0).value     # s
        self.confirm_frames = self.declare_parameter('confirm_frames', 3).value
        self.publish_img = self.declare_parameter('publish_img', False).value
        self.stream = self.declare_parameter('stream', True).value                  # browser camera stream
        self.stream_port = self.declare_parameter('stream_port', 8080).value

        # Initialize variables
        self.state = "IDLE"  # IDLE -> SEARCHING -> FOUND
        self.target_class_id = None
        self.detection_count = 0
        self.phase = "BACKWARD"
        self.phase_start_time = time.monotonic()
        self.enabled = False
        self.last_joy_time = 0.0
        self.busy = False
        self.prev_back_button = 0

        self.thread_executor = ThreadPoolExecutor(max_workers=2)

        # YOLO model
        self.model = YOLO(self.model_path)
        self.get_logger().info(f"YOLO model loaded: {self.model_path}")

        # Camera variables
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)

        # ROS2 variables
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.enable_sub = self.create_subscription(Bool, '/autonomous_enable', self.enable_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.image_pub = self.create_publisher(ROSImage, '/search/detection/image', 1)
        self.timer = self.create_timer(0.033, self.camera_callback)  # 30 FPS
        self.bridge = CvBridge()

        # Browser camera stream
        self.streamer = None
        if self.stream:
            self.streamer = MjpegStreamer(port=self.stream_port)
            self.get_logger().info(f"Camera stream on http://<car-ip>:{self.stream_port}/")

        self.get_logger().info("SpeechToSearch node started.")
        self.get_logger().info(f"Available YOLO classes ({len(self.model.names)}): "
                               f"{', '.join(self.model.names.values())}")

    def camera_callback(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return

        color_image = np.asanyarray(color_frame.get_data())

        if self.state != "SEARCHING":
            self.publish_drive(0.0, 0.0)
            if self.streamer:
                self.streamer.update(color_image)
            return

        # Run YOLO only for the wanted class
        results = self.model(color_image, classes=[self.target_class_id], conf=self.conf, verbose=False)
        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            self.detection_count += 1
        else:
            self.detection_count = 0

        if self.detection_count >= self.confirm_frames:
            # Object found: stop the search movement
            self.state = "FOUND"
            self.publish_drive(0.0, 0.0)
            best = int(boxes.conf.argmax().item())
            x1, y1, x2, y2 = map(int, boxes.xyxy[best].cpu().numpy())
            self.get_logger().info(f"FOUND {self.model.names[self.target_class_id]} | "
                                   f"conf: {float(boxes.conf[best]):.2f} | bbox: {(x1, y1, x2, y2)}")
        else:
            speed, steering = self.search_command()
            self.publish_drive(speed, steering)

        annotated_image = results[0].plot()
        if self.streamer:
            self.streamer.update(annotated_image)

        if self.publish_img:
            small_image = cv2.resize(annotated_image, (424, 240), interpolation=cv2.INTER_AREA)
            ros_img = self.bridge.cv2_to_imgmsg(small_image, encoding='bgr8')
            self.image_pub.publish(ros_img)

    def search_command(self):
        # Back-and-forth movement: backward with steering one way, forward with steering the other way
        if time.monotonic() - self.phase_start_time >= self.phase_duration:
            self.phase = "FORWARD" if self.phase == "BACKWARD" else "BACKWARD"
            self.phase_start_time = time.monotonic()

        if self.phase == "BACKWARD":
            return -self.search_speed, self.search_steering
        return self.search_speed, -self.search_steering

    def publish_drive(self, speed, steering):
        # Only move while the autonomous enable button (X) is held and the joystick is alive
        if not self.enabled or time.monotonic() - self.last_joy_time > 0.5:
            speed, steering = 0.0, 0.0

        msg = AckermannDriveStamped()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering)
        self.drive_pub.publish(msg)

    def enable_callback(self, msg):
        # joy_teleop sends True when X is pressed and False when it is released
        self.enabled = msg.data

    def joy_callback(self, msg):
        self.last_joy_time = time.monotonic()

        # For the Start button, start the speech to search process
        if msg.buttons[7] == 1 and not self.busy:
            self.busy = True
            self.state = "IDLE"
            self.get_logger().info("START: Starting speech to search pipeline.")
            self.thread_executor.submit(self.run_speech_to_class)

        # For the back button, stop the search
        back_now = msg.buttons[6]
        if self.prev_back_button == 0 and back_now == 1:
            self.get_logger().info("BACK: Stopping search.")
            self.state = "IDLE"

        self.prev_back_button = back_now

    def run_speech_to_class(self):
        try:
            # Record audio from webcam
            audio_path = record_webcam_audio(duration=3, filename="temp_audio.wav")

            # Call Gemini agent server to get the wanted YOLO class
            target_class = self.call_gemini_agent(audio_path, list(self.model.names.values()))
            self.get_logger().info(f"Answer from Gemini: {target_class}")

            class_ids = [i for i, name in self.model.names.items() if name == target_class]
            if not class_ids:
                self.get_logger().warn(f"'{target_class}' is not a YOLO class, not searching.")
                return

            # Start searching
            self.target_class_id = class_ids[0]
            self.detection_count = 0
            self.phase = "BACKWARD"
            self.phase_start_time = time.monotonic()
            self.state = "SEARCHING"
            self.get_logger().info(f"SEARCHING: {target_class}")

        except Exception as e:
            self.get_logger().error(f'Error in audio worker: {e}')

        finally:
            self.busy = False

    def call_gemini_agent(self, audio_path, class_names):
        # Send audio file path and the YOLO class names to the Gemini server
        self.get_logger().info(f"Sending {audio_path} to Gemini...")
        response = requests.post(
            "http://127.0.0.1:8000/audio_to_class",
            json={"audio_path": audio_path,
                  "class_names": class_names}
        )
        return response.json().get("target_class", "none")

    def shutdown(self):
        self.pipeline.stop()
        if self.streamer:
            self.streamer.shutdown()
        self.get_logger().info("SpeechToSearch node stopped.")


def main(args=None):
    rclpy.init(args=args)
    node = SpeechToSearch()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.shutdown()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
