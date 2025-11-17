import cv2
import os
import time
import requests
import pyrealsense2 as rs
import numpy as np
import copy
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Joy

from webcam_audio_rec import record_webcam_audio

class SpeechToObject(Node):
    def __init__(self):
        super().__init__('speech_to_object_node')

        # Initialize variables
        self.bbox = None
        self.tracker = None
        self.busy = False

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
        # todo: rmv and False!!!
        if self.bbox is not None and self.tracker is None and False:
            self.tracker = cv2.legacy.TrackerCSRT_create()
            initialized = self.tracker.init(color_imagObjTrackerNodee, self.bbox)

        if self.tracker is not None:
            ok, self.bbox = self.tracker.update(color_frame)

            if ok:
                x, y, w, h = map(int, self.bbox)
                cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                self.get_logger().info("Tracking lost. Please provide a new bounding box.")
                self.tracker = None
                self.bbox = None

        # Visualize
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

    def joy_callback(self, msg):
        if msg.buttons[7] == 1 and not self.busy:
            self.busy = True
            self.get_logger().info("Start button pressed!")
            self.thread_executor.submit(self.run_speech_to_obj_det)


    def run_speech_to_obj_det(self):
        try:
            audio_path = record_webcam_audio(duration=3, filename="temp_audio.wav")
            self.get_logger().info(f"Recorded audio saved at: {audio_path}")


            bbox = self.call_gemini_agent(audio_path)
            self.get_logger().info(f"Bbox from Gemini agent: {bbox} {type(bbox[0])}")

            self.bbox = copy.deepcopy(bbox)
            self.busy = False

        except Exception as e:
            self.get_logger().error(f'Error in audio worker: {e}')

        finally:
            self.get_logger().info('Speech to object detection task completed.')
            self.busy = False

    def call_gemini_agent(self, audio_path):
        # Send audio file path to the Gemini server for processing
        response = requests.post(
            "http://127.0.0.1:8000/run_agent_pipeline",
            json={"path": audio_path}
        )

        bbox = response.json().get("bb_list", [])
        bbox = self.gemini_bbox_to_csrt_bbox(bbox[0])

        return bbox
    
    @staticmethod
    def gemini_bbox_to_csrt_bbox(bbox):
        # Converts [x1, y1, x2, y2] to [x1, y1, w, h]
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        return [x1, y1, w, h]

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