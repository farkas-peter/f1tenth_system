import json

import rclpy
import requests
from rclpy.node import Node
from std_msgs.msg import String

from webcam_audio_rec import record_webcam_audio


class SpeechToObjDetectionNode(Node):
    def __init__(self):
        super().__init__('speech_to_obj_detection_node')
        self.publisher = self.create_publisher(String, 'detected_bbox', 10)

    def run(self):
        while rclpy.ok():
            user_input = input("Enter duration in seconds (or stop): ").strip()
            if user_input.lower() in ("q", "stop", "exit"):
                break

            audio_path = record_webcam_audio(duration=float(user_input))

            bbox = self.call_gemini_agent(audio_path)

            msg = String()
            msg.data = json.dumps({"bbox": bbox})
            self.publisher.publish(msg)

    @staticmethod
    def call_gemini_agent(audio_path):
        # Send audio file path to the Gemini server for processing
        response = requests.post(
            "http://127.0.0.1:8000/run_agent_pipeline",
            json={"path": audio_path}
        )
        bbox = response.json().get("bb_list", [])

        return bbox

    def shutdown(self):
        self.get_logger().info("SpeechToObjDetectionNode stopped.")

def main(args=None):
    rclpy.init(args=args)
    node = SpeechToObjDetectionNode()

    try:
        node.run()
    except KeyboardInterrupt:
        node.shutdown()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
