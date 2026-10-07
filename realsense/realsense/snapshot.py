"""
RealSense single color image snapshot.

Starts the RealSense camera with only the color stream enabled,
captures a single frame, and saves it to /workspace/LOG/pictures
with a timestamp-based filename.
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime


# --- Configuration ---
WIDTH = 848
HEIGHT = 480
FPS = 30
SAVE_DIR = "/workspace/LOG/pictures"


def main():
    # Ensure the output directory exists
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Configure pipeline – color stream only
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    pipeline.start(config)

    try:
        # Let auto-exposure settle (skip a few frames)
        for _ in range(30):
            pipeline.wait_for_frames()

        # Capture a single frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            print("ERROR: Could not capture a color frame.")
            return

        color_image = np.asanyarray(color_frame.get_data())

        # Build a timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}.png"
        filepath = os.path.join(SAVE_DIR, filename)

        cv2.imwrite(filepath, color_image)
        print(f"Image saved to: {filepath}")

    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
