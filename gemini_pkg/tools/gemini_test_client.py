import os
import cv2
import numpy as np
import requests
from PIL import Image


def visualize_result(image, bounding_boxes):
    # Visualize the image with the detection bounding boxes
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for box in bounding_boxes:
        cv2.rectangle(cv_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    cv2.imshow("Detected Objects", cv_image)
    cv2.waitKey(0)



if __name__ == "__main__":
    # Get a hello from the server
    response = requests.get("http://127.0.0.1:8000/hello")
    print("Server response:", response.text)

    # Send audio file path to the server for processing
    # todo: pass base64 image!!!
    response = requests.post(
        "http://127.0.0.1:8000/run_agent_pipeline",
        json={"audio_path": "temp_audio.wav",
              "base64_image": ""}
    )
    response = response.json()
    print("Server response:", response)

    # Load the image and visualize the bb
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "random_objects.png")
    img = Image.open(img_path)

    bbs = response.get("bb_list", [])
    visualize_result(img, bbs)




