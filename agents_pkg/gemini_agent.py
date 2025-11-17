import os
import json

import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

class GeminiAgent:
    def __init__(self):
        # Gemini
        self.gemini_model = "gemini-2.5-flash-lite"
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    def detect_object(self, obj_to_detect, image, visualize=False):
        # Detect object in image using Gemini
        prompt = (
            f"Detect the {obj_to_detect} in the image. "
            f"The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000."
        )

        config = types.GenerateContentConfig(response_mime_type="application/json")
        response = self.client.models.generate_content(model=self.gemini_model,
                                                       contents=[image, prompt],
                                                       config=config
                                                       )

        # Convert normalized bb to absolute pixel values
        width, height = image.size
        bounding_boxes = json.loads(response.text)
        bounding_boxes = bounding_boxes if isinstance(bounding_boxes, list) else [bounding_boxes]

        converted_bounding_boxes = []
        for bounding_box in bounding_boxes:
            abs_y1 = int(bounding_box["box_2d"][0]/1000 * height)
            abs_x1 = int(bounding_box["box_2d"][1]/1000 * width)
            abs_y2 = int(bounding_box["box_2d"][2]/1000 * height)
            abs_x2 = int(bounding_box["box_2d"][3]/1000 * width)
            converted_bounding_boxes.append([abs_x1, abs_y1, abs_x2, abs_y2])

        print("Image size: ", width, height)
        print("Bounding boxes:", converted_bounding_boxes)

        if visualize:
            self.visualize_detections(image, converted_bounding_boxes)

        return converted_bounding_boxes

    @staticmethod
    def visualize_detections(image, converted_bounding_boxes):
        # Visualize the image with the detection bounding boxes
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        for box in converted_bounding_boxes:
            cv2.rectangle(cv_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.imshow("Detected Objects", cv_image)
        cv2.waitKey(0)

    def get_obj_desc_from_audio(self, audio_filepath):
        # Get object description from an audio file using Gemini
        prompt = f"""
        You are an assistant that interprets spoken commands for visual search.
        Task: Process the provided audio file and extract the main object with all of its properties or spatial information.
        Ignore all filler words like 'go to', 'please', 'look for', etc.
        Return only a short clean phrase describing the search target.
        """

        audio_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), audio_filepath)
        with open(audio_filepath, 'rb') as f:
            audio_bytes = f.read()

        response = self.client.models.generate_content(
            model=self.gemini_model,
            contents=[prompt,
                      types.Part.from_bytes(data=audio_bytes, mime_type='audio/wav')
                      ]
        )

        print("Object description from audio:", response.text)

        return response.text

    def run_pipeline(self, audio_filepath: str, image: Image.Image):
        # Run the full pipeline: audio -> object description -> camera image -> object detection
        obj_description = self.get_obj_desc_from_audio(audio_filepath)
        bounding_box = self.detect_object(obj_description, image)

        return bounding_box


if __name__ == "__main__":
    agent = GeminiAgent()

    # Test image
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "random_objects.png")
    img = Image.open(img_path)

    #obj = agent.get_obj_desc_from_audio("temp_audio.wav")
    #bbs = agent.detect_object("object above the pizza", img, visualize=True)

    agent.run_pipeline("temp_audio.wav", img)