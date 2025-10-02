import cv2
import sounddevice as sd
import io
import base64
from scipy.io.wavfile import write
import google.generativeai as genai
from PIL import Image
import json
import sys
 
# ---- Gemini API Setup ----
API_KEY = ""
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")
 
# ---- Settings ----
VERIFY_DETECTION = True  
 
 
# ---- Audio recording ----
def record_audio(duration, samplerate=44100):
    device_info = sd.query_devices(sd.default.device[0], 'input')
    channels = 2 if device_info['max_input_channels'] >= 2 else 1
    print(f"[INFO] Recording {duration}s... (channels: {channels})")
    print("[INFO] Press ESC to cancel recording.")
 
    audio = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=channels,
        dtype='int16'
    )
 
    for _ in range(duration * 10):
        if cv2.waitKey(100) & 0xFF == 27:
            sd.stop()
            print("[INFO] Recording cancelled by user (ESC).")
            sys.exit(0)
 
    sd.wait()
    buffer = io.BytesIO()
    write(buffer, samplerate, audio)
    buffer.seek(0)
    return buffer.read()
 
 
def transcribe_audio(audio_bytes):
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    response = model.generate_content([
        {"mime_type": "audio/wav", "data": audio_b64},
        "Transcribe exactly what you hear in the audio, in English."
    ])
    if hasattr(response, "text") and response.text:
        return response.text.strip()
 
    output = []
    for cand in response.candidates:
        for part in cand.content.parts:
            if hasattr(part, "text") and part.text:
                output.append(part.text)
    return "\n".join(output).strip() if output else None
 
 
def extract_search_target(command_text):
    prompt = f"""
    You are an assistant that interprets spoken commands for visual search.
    The user said: "{command_text}"
 
    Task: Extract the main object and any spatial/positional qualifier.
    Ignore filler words like 'go to', 'please', 'look for'.
    Return only a short clean phrase describing the search target.
    """
    response = model.generate_content(prompt)
    return response.text.strip()
 
 
def detect_object_gemini(frame, search_text):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    prompt = (
        f"Detect '{search_text}' in this image. "
        f"Return box_2d as [ymin, xmin, ymax, xmax] normalized to 0-1000."
    )
    response = model.generate_content([image, prompt])
    response_text = response.candidates[0].content.parts[0].text.strip()
 
    if response_text.startswith("```json"):
        response_text = response_text.replace("```json", "", 1)
    if response_text.endswith("```"):
        response_text = response_text.rsplit("```", 1)[0]
    response_text = response_text.strip()
 
    boxes = json.loads(response_text)
    if not boxes:
        raise ValueError(f"'{search_text}' not found.")
 
    h, w, _ = frame.shape
    box = boxes[0]["box_2d"]
    y1, x1, y2, x2 = [
        int(box[i] / 1000 * dim) for i, dim in zip([0, 1, 2, 3], [h, w, h, w])
    ]
    return (x1, y1, x2 - x1, y2 - y1)
 
 
def verify_detection(frame, bbox, search_text):
    if not VERIFY_DETECTION:
        return True
 
    x, y, w, h = bbox
    frame_copy = frame.copy()
    cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 0, 255), 3)
 
    image = Image.fromarray(cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB))
    prompt = f"Verify if the red rectangle in this image correctly surrounds '{search_text}'. Answer only with 'yes' or 'no'."
 
    response = model.generate_content([image, prompt])
    answer = response.text.strip().lower()
    return answer.startswith("y")
 
 
def compute_steering(frame_width, bbox, Kp=0.5):
    x, y, w, h = bbox
    bbox_center = x + w / 2
    x_center = frame_width / 2
    error = (bbox_center - x_center) / (frame_width / 2)
    steering = Kp * error
    return max(-1, min(1, steering))
 
 
def track_with_webcam(search_text, duration):
    cap = cv2.VideoCapture(0)
    tracker = None
    init_bbox = None
    detected = False
    pending_verification = False  
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
 
        if not detected and not pending_verification:
            cv2.putText(
                frame,
                "SPACE=detect | H=re-record | ESC=quit",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            cv2.imshow("Webcam", frame)
            key = cv2.waitKey(1) & 0xFF
 
            if key == 32:  
                try:
                    init_bbox = detect_object_gemini(frame, search_text)
                    if verify_detection(frame, init_bbox, search_text):
                        tracker = cv2.legacy.TrackerCSRT_create()
                        tracker.init(frame, init_bbox)
                        detected = True
                        print(f"[INFO] Object '{search_text}' verified and tracking started.")
                    else:
                        print(f"[WARNING] Verification failed for '{search_text}'.")
                        pending_verification = True  
                except Exception as e:
                    print(f"[ERROR] {e}")
 
            elif key == ord("h"):  
                print("[INFO] Re-recording audio...")
                audio_bytes = record_audio(duration)
                transcript = transcribe_audio(audio_bytes)
                if transcript:
                    search_text = extract_search_target(transcript)
                    print(f"[INFO] New search target: {search_text}")
                else:
                    print("[ERROR] Could not recognize speech.")
 
            elif key == 27:  
                print("[INFO] ESC pressed, exiting.")
                break
 
        elif pending_verification:
            cv2.putText(
                frame,
                "Verification failed! Press V=retry | H=re-record | ESC=quit",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )
            cv2.imshow("Webcam", frame)
            key = cv2.waitKey(1) & 0xFF
 
            if key == ord("v"):  # újraellenőrzés
                if verify_detection(frame, init_bbox, search_text):
                    tracker = cv2.legacy.TrackerCSRT_create()
                    tracker.init(frame, init_bbox)
                    detected = True
                    pending_verification = False
                    print(f"[INFO] Verification succeeded on retry. Tracking started.")
                else:
                    print(f"[WARNING] Verification failed again.")
                    #
 
            elif key == ord("h"):
                print("[INFO] Re-recording audio...")
                audio_bytes = record_audio(duration)
                transcript = transcribe_audio(audio_bytes)
                if transcript:
                    search_text = extract_search_target(transcript)
                    pending_verification = False
                    print(f"[INFO] New search target: {search_text}")
                else:
                    print("[ERROR] Could not recognize speech.")
 
            elif key == 27:
                print("[INFO] ESC pressed, exiting.")
                break
 
        else:
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                cv2.putText(
                    frame,
                    search_text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
                steering = compute_steering(frame.shape[1], bbox, Kp=0.7)
                cv2.putText(
                    frame,
                    f"Steering: {steering:.2f}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    frame,
                    "Lost tracking! R=redetect | H=re-record | ESC=quit",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )
 
            cv2.imshow("Webcam", frame)
            key = cv2.waitKey(1) & 0xFF
 
            if key == ord("r"):
                try:
                    init_bbox = detect_object_gemini(frame, search_text)
                    if verify_detection(frame, init_bbox, search_text):
                        tracker = cv2.legacy.TrackerCSRT_create()
                        tracker.init(frame, init_bbox)
                        print(f"[INFO] Object '{search_text}' re-detected.")
                    else:
                        print("[WARNING] Verification failed on re-detection.")
                        detected = False
                        pending_verification = True
                except Exception as e:
                    print(f"[ERROR] Redetection failed: {e}")
                    detected = False
                    pending_verification = False
 
            elif key == ord("h"):
                print("[INFO] Re-recording audio...")
                audio_bytes = record_audio(duration)
                transcript = transcribe_audio(audio_bytes)
                if transcript:
                    search_text = extract_search_target(transcript)
                    detected = False
                    print(f"[INFO] New search target: {search_text}")
                else:
                    print("[ERROR] Could not recognize speech.")
 
            elif key == 27 or key == ord("q"):
                print("[INFO] Program stopped by user.")
                break
 
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    try:
        duration = int(input("How many seconds should I record? "))
    except KeyboardInterrupt:
        print("\n[INFO] Program stopped by user.")
        sys.exit(0)
 
    audio_bytes = record_audio(duration)
    transcript = transcribe_audio(audio_bytes)
    if not transcript:
        print("[ERROR] No recognizable text found in audio.")
        sys.exit(0)
 
    search_target = extract_search_target(transcript)
    print(f"[INFO] Command: {transcript}")
    print(f"[INFO] Target: {search_target}")
 
    track_with_webcam(search_target, duration)