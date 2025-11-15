import sys
from google import genai
from google.genai import types
import cv2
import json
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import pyrealsense2 as rs
import sounddevice as sd
from scipy.io.wavfile import write
import time
from tqdm import tqdm

# Init
print(sys.version)
load_dotenv()

class Agent:
    def __init__(self):
        # Gemini
        self.client = genai.Client()
        self.config = types.GenerateContentConfig(response_mime_type="application/json")

        # Camera
        self.cam_pipeline = rs.pipeline()
        self.cam_config = rs.config()
        #self.cam_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.cam_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.cam_pipeline.start(self.cam_config)
    
    def detect_object(self, obj_to_detect, image):
        prompt = (
            f"Detect the {obj_to_detect} in the image. "
            f"The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000."
        )

        response = self.client.models.generate_content(model="gemini-2.5-flash",
                                                       contents=[image, prompt],
                                                       config=self.config
                                                       )
        
        width, height = image.size
        bounding_boxes = json.loads(response.text)

        converted_bounding_boxes = []
        for bounding_box in bounding_boxes:
            abs_y1 = int(bounding_box["box_2d"][0]/1000 * height)
            abs_x1 = int(bounding_box["box_2d"][1]/1000 * width)
            abs_y2 = int(bounding_box["box_2d"][2]/1000 * height)
            abs_x2 = int(bounding_box["box_2d"][3]/1000 * width)
            converted_bounding_boxes.append([abs_x1, abs_y1, abs_x2, abs_y2])

        print("Image size: ", width, height)
        print("Bounding boxes:", converted_bounding_boxes)

        self.visualize_detections(image, converted_bounding_boxes)

    def visualize_detections(self, image, converted_bounding_boxes):
        # Visualize the image with the bb
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        for box in converted_bounding_boxes:
            cv2.rectangle(cv_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.imshow("Detected Objects", cv_image)
        cv2.waitKey(0)

    def capture_image(self):
        frames = self.cam_pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        return color_image


if __name__ == "__main__":
    agent = Agent()

    image = Image.open("/workspace/src/f1tenth_system/agents_pkg/random_objects.png")

    image = agent.capture_image()
    image = Image.fromarray(image)

    agent.detect_object("controller", image)