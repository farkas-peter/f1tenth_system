# gemini_pkg

Voice-commanded object following: you say which object to go to, Gemini finds it
on the RealSense colour image, the node tracks it and publishes it as
`/target_point` for the pure pursuit controller (`cone_pkg`). The car stops
within 1 m of the object.

The system is split in two, because ROS 2 Foxy runs on Python 3.8 while
`google-genai` needs Python >= 3.10:

| Part | Python | What it does |
|---|---|---|
| `gemini_pkg/gemini_server.py` | venv, 3.12 | FastAPI server on `127.0.0.1:8000`, calls Gemini (`gemini_agent.py`) |
| `gemini_pkg/speech_to_obj_node.py` | system, 3.8 (ROS) | records audio, grabs RealSense image, calls the server, tracks the object, publishes `/target_point` |

All commands below run **inside the container**.

## Setup

Steps 1, 3 and 4 below are done by one script, safe to
re-run after every container restart:

```bash
bash /workspace/src/f1tenth_system/gemini_pkg/setup_env.sh
```

### 1. Gemini server venv (Python 3.12)

`uv` downloads Python 3.12 on its own (the container only has 3.8 and 3.9).
Both the venv and the downloaded Python live in the mounted `gemini_pkg/`
folder, so they survive a container rebuild. Both are gitignored.

```bash
cd /workspace/src/f1tenth_system/gemini_pkg
export UV_PYTHON_INSTALL_DIR=$PWD/.uv-python
uv venv --python 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -r requirements.txt
touch .venv/COLCON_IGNORE .uv-python/COLCON_IGNORE   # keep colcon out of them
```

`uv` comes from the Ultralytics Jetson base image, it is not installed by our
Dockerfile.

### 2. Gemini API key

Create `gemini_pkg/gemini_pkg/.env` (gitignored) with your key from
https://aistudio.google.com/apikey :

```
GEMINI_API_KEY=your-key-here
```

### 3. System Python (ROS node side)

The ROS node needs a few extras in the container's Python 3.8:

```bash
sudo apt-get install -y libportaudio2         # for sounddevice
pip3 install --user sounddevice
```

These are not persistent: redo them after the container is recreated.

### 4. OpenCV with GUI + CSRT tracker

The container ships `opencv-python-headless` (no windows, no CSRT). The node and
the tools use `cv2.legacy.TrackerCSRT_create()` and `cv2.imshow()`, so install the
contrib build for the user (takes priority over the system one):

```bash
pip3 install --user opencv-contrib-python==4.13.0.92
# silence Qt "Cannot find font directory" warnings
ln -sfn /usr/share/fonts/truetype/dejavu ~/.local/lib/python3.8/site-packages/cv2/qt/fonts
```

Keep the version equal to the headless one (`pip3 list | grep opencv`).
Undo with `pip3 uninstall opencv-contrib-python`. Not persistent either.

Windows open on the Jetson's own screen: run GUI programs with `DISPLAY=:1`.

## Running

1. Gemini server:
   ```bash
   /workspace/src/f1tenth_system/gemini_pkg/.venv/bin/python /workspace/src/f1tenth_system/gemini_pkg/gemini_pkg/gemini_server.py
   ```
2. Everything else (joy, VESC, mux, pure pursuit, speech_to_obj_node):
   ```bash
   ros2 launch f1tenth_stack gemini_launch.py
   ```

Or, without the launch file:
```bash
ros2 run joy joy_node
/bin/python3 /workspace/src/f1tenth_system/gemini_pkg/gemini_pkg/speech_to_obj_node.py
```

## Voice search (`speech_to_search_node`)

Say e.g. "let's find the mobile phone": Gemini turns the audio into one YOLO (COCO)
class name (`cell phone`), then YOLO (`yolo26s.pt`, only that class) runs on the
RealSense image while the car searches with a back-and-forth movement (backward with
steering one way, forward with steering the other way). When the object is seen
around the horizontal middle of the image (within `center_tolerance`, the yellow lines
on the stream) in `confirm_frames` consecutive frames, the car stops: it faces the object.

1. ```bash
   ros2 launch f1tenth_stack gemini_search_launch.py
   ```
   This also starts the Gemini server from
   `/workspace/src/f1tenth_system/gemini_pkg/.venv` (override with
   `gemini_pkg_dir:=<path>`), so don't start it separately (port 8000 would be taken).
3. Press **Start**, speak (3 s), then **hold X** (deadman, `/autonomous_enable`):
   the car only moves while X is held. **Back** stops the search.

Parameters: `model_path`, `conf`, `search_speed` (m/s), `search_steering` (rad),
`phase_duration` (s), `confirm_frames`, `center_tolerance` (+- fraction of image width,
default `0.15`), `publish_img` (annotated image on
`/search/detection/image`).

### Camera stream in the browser

The node serves what the camera sees as an MJPEG stream (424x240, YOLO boxes drawn
while searching): open `http://<car-ip>:8080/` in a browser on the same network.
Parameters: `stream` (default `true`), `stream_port` (default `8080`).

## Tools

Standalone helper scripts in `tools/` (not installed by colcon, run them directly):

| Script | What it is |
|---|---|
| `cv2_csrt_tracker.py` | CSRT tracker test: select an object with the mouse, it gets tracked. `DISPLAY=:1 python3 tools/cv2_csrt_tracker.py` |
| `gemini_test_client.py` | Sends a test request to the running Gemini server |
| `pc_version.py` | All-in-one version for a laptop (mic + webcam + Gemini, no ROS). Uses the old `google.generativeai` library |

## Camera devices

The RealSense D435i shows up as several video devices; on this Jetson:

| Device | Stream |
|---|---|
| `/dev/video0` | depth |
| `/dev/video2` | infrared |
| `/dev/video4` | colour (RGB) |

`cv2.VideoCapture(4)` is the colour camera. The numbers can shift if other
cameras are plugged in; check with `v4l2-ctl -d /dev/videoN --list-formats`
(colour is `YUYV`).
