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
from nav_msgs.msg import Odometry
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
        self.search_speed = self.declare_parameter('search_speed', 0.7).value         # m/s
        self.search_steering = self.declare_parameter('search_steering', 0.3).value   # rad
        self.phase_duration = self.declare_parameter('phase_duration', 2.0).value     # s
        self.kick_speed = self.declare_parameter('kick_speed', 1.5).value             # m/s, burst to get moving
        self.kick_max_duration = self.declare_parameter('kick_max_duration', 1.0).value  # s, safety limit
        self.kick_until = self.declare_parameter('kick_until', 0.9).value             # fraction of search_speed to reach
        self.confirm_frames = self.declare_parameter('confirm_frames', 3).value
        self.center_tolerance = self.declare_parameter('center_tolerance', 0.15).value  # +- fraction of image width
        self.publish_img = self.declare_parameter('publish_img', False).value
        self.stream = self.declare_parameter('stream', True).value                  # browser camera stream
        self.stream_port = self.declare_parameter('stream_port', 8080).value

        # Initialize variables
        self.state = "IDLE"  # IDLE -> SEARCHING -> TRACKING
        self.status = "Press Start to give a voice command"  # shown on the camera image
        self.target_class_id = None
        self.detection_count = 0
        self.tracker = None
        self.phase = "BACKWARD"
        self.phase_start_time = time.monotonic()
        self.kick_start_time = time.monotonic()
        self.kicking = True
        self.odom_speed = 0.0
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
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
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

        if self.state == "TRACKING":
            self.publish_drive(0.0, 0.0)
            self.track(color_image)
            return

        if self.state != "SEARCHING":
            self.publish_drive(0.0, 0.0)
            self.publish_image(color_image.copy())
            return

        # Run YOLO only for the wanted class
        results = self.model(color_image, classes=[self.target_class_id], conf=self.conf, verbose=False)
        boxes = results[0].boxes

        # Only count the detection if the object is horizontally around the middle of the image (car faces it)
        width = color_image.shape[1]
        band = self.center_tolerance * width
        centered = False
        if boxes is not None and len(boxes) > 0:
            best = int(boxes.conf.argmax().item())
            x1, y1, x2, y2 = map(int, boxes.xyxy[best].cpu().numpy())
            centered = abs((x1 + x2) / 2 - width / 2) <= band

        if centered:
            self.detection_count += 1
        else:
            self.detection_count = 0

        if self.detection_count >= self.confirm_frames:
            # Object found: stop the search movement and keep following it with the CSRT tracker
            self.publish_drive(0.0, 0.0)
            self.get_logger().info(f"FOUND {self.model.names[self.target_class_id]} | "
                                   f"conf: {float(boxes.conf[best]):.2f} | bbox: {(x1, y1, x2, y2)}")
            self.tracker = cv2.legacy.TrackerCSRT_create()
            self.tracker.init(color_image, (x1, y1, x2 - x1, y2 - y1))  # bbox = (x, y, w, h)
            self.state = "TRACKING"
            self.status = f"Tracking: {self.model.names[self.target_class_id]}"
        else:
            speed, steering = self.search_command()
            self.publish_drive(speed, steering)

        # Draw the centre band on the image
        annotated_image = results[0].plot()
        for x in (int(width / 2 - band), int(width / 2 + band)):
            cv2.line(annotated_image, (x, 0), (x, annotated_image.shape[0]), (0, 255, 255), 2)
        self.publish_image(annotated_image)

    def track(self, color_image):
        ok, bbox = self.tracker.update(color_image)
        image = color_image.copy()

        if ok:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            target_class = self.model.names[self.target_class_id]
            self.get_logger().info(f"Tracking lost: {target_class}")
            self.tracker = None
            self.state = "IDLE"
            self.status = f"Lost {target_class}, press Start to search again"

        self.publish_image(image)

    def publish_image(self, image):
        # Write the current status in a dark bar at the top of the image
        cv2.rectangle(image, (0, 0), (image.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(image, self.status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        if self.streamer:
            self.streamer.update(image)

        if self.publish_img:
            small_image = cv2.resize(image, (424, 240), interpolation=cv2.INTER_AREA)
            ros_img = self.bridge.cv2_to_imgmsg(small_image, encoding='bgr8')
            self.image_pub.publish(ros_img)

    def search_command(self):
        # Back-and-forth movement: backward with steering one way, forward with steering the other way
        if time.monotonic() - self.phase_start_time >= self.phase_duration:
            self.phase = "FORWARD" if self.phase == "BACKWARD" else "BACKWARD"
            self.phase_start_time = time.monotonic()
            self.start_kick()

        # The motor needs extra power to start from standstill, so kick with a higher speed
        # until odom shows the car close to search speed in the wanted direction (or the safety limit is reached)
        direction = -1.0 if self.phase == "BACKWARD" else 1.0
        if self.kicking:
            moving = direction * self.odom_speed >= self.kick_until * self.search_speed
            timed_out = time.monotonic() - self.kick_start_time >= self.kick_max_duration
            if moving or timed_out:
                self.kicking = False
        speed = self.kick_speed if self.kicking else self.search_speed

        if self.phase == "BACKWARD":
            return -speed, self.search_steering
        return speed, -self.search_steering

    def start_kick(self):
        self.kicking = True
        self.kick_start_time = time.monotonic()

    def odom_callback(self, msg):
        self.odom_speed = msg.twist.twist.linear.x

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
        if msg.data and not self.enabled:
            # Car was stopped, so kick again when X is pressed
            self.start_kick()
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
            self.tracker = None
            self.status = "Stopped, press Start to give a voice command"

        self.prev_back_button = back_now

    def run_speech_to_class(self):
        try:
            # Record audio from webcam
            self.status = "Listening..."
            audio_path = record_webcam_audio(duration=3, filename="temp_audio.wav")

            # Call Gemini agent server to get the wanted YOLO class
            self.status = "Processing voice command..."
            target_class = self.call_gemini_agent(audio_path, list(self.model.names.values()))
            self.get_logger().info(f"Answer from Gemini: {target_class}")

            class_ids = [i for i, name in self.model.names.items() if name == target_class]
            if not class_ids:
                self.get_logger().warn(f"'{target_class}' is not a YOLO class, not searching.")
                self.status = f"Unknown object '{target_class}', press Start to try again"
                return

            # Start searching
            self.target_class_id = class_ids[0]
            self.detection_count = 0
            self.phase = "BACKWARD"
            self.phase_start_time = time.monotonic()
            self.start_kick()
            self.state = "SEARCHING"
            self.status = f"Searching for: {target_class}"
            self.get_logger().info(f"SEARCHING: {target_class}")

        except Exception as e:
            self.get_logger().error(f'Error in audio worker: {e}')
            self.status = "Error, press Start to try again"

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
