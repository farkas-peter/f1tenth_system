"""
Object tracking from webcam with mouse-selected ROI.

Controls:
  - When the first frame appears, drag a box to select the object, then press ENTER or SPACE.
  - Press 'r' to reselect the ROI at any time.
  - Press 'q' or ESC to quit.

Requires: opencv-contrib-python (for CSRT/KCF/MOSSE trackers)
Install:  pip install opencv-contrib-python
"""

import cv2
import sys
import time

def select_roi(frame):
    # Returns (x, y, w, h) or None if user cancels
    roi = cv2.selectROI("Tracking", frame, fromCenter=False, showCrosshair=True)
    if roi == (0, 0, 0, 0):
        return None
    return roi

def put_status(img, text, y=30, color=(0, 255, 0)):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(2)
    time.sleep(0.5) # Warm up camera

    if not cap.isOpened():
        print("ERROR: Cannot open webcam.", file=sys.stderr)
        sys.exit(1)

    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)

    # Grab an initial frame to select ROI
    ok, frame = cap.read()
    if not ok:
        print("ERROR: Cannot read from webcam.", file=sys.stderr)
        sys.exit(1)

    put_status(frame, "Select ROI, then press ENTER/SPACE. (Press 'q' to exit)", y=30)
    cv2.imshow("Tracking", frame)
    bbox = select_roi(frame)
    if bbox is None:
        print("No ROI selected. Exiting.")
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

    tracker = cv2.legacy.TrackerCSRT_create()
    initialized = tracker.init(frame, bbox)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if initialized and tracker is not None:
            ok, bbox = tracker.update(frame)

            if ok:
                # Draw tracked bbox
                x, y, w, h = map(int, bbox)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                put_status(frame, f"CSRT tracking...", y=30, color=(0, 255, 0))
            else:
                put_status(frame, "Tracking lost! Press 'r' to reselect ROI.", y=30, color=(0, 0, 255))
        else:
            put_status(frame, "Tracker not initialized. Press 'r' to select ROI.", y=30, color=(0, 255, 255))

        put_status(frame, "Keys: 'r' reselect | 'q'/ESC quit", y=60, color=(255, 255, 255))
        cv2.imshow("Tracking", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC or 'q'
            break
        elif key == ord('r'):
            # Reselect ROI on current frame
            temp = frame.copy()
            put_status(temp, "Select ROI, then press ENTER/SPACE. (ESC to cancel)", y=30)
            cv2.imshow("Tracking", temp)
            new_bbox = select_roi(frame)
            if new_bbox is not None:
                bbox = new_bbox
                tracker = cv2.legacy.TrackerCSRT_create()
                initialized = tracker.init(frame, bbox)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
 