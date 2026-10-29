#!/usr/bin/env bash
# One-shot setup for gemini_pkg inside the container (see README.md "Setup").
# Safe to re-run: run it after every container restart / recreation.
#   bash gemini_pkg/setup_env.sh

set -e

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_DISTRO="${ROS_DISTRO:-foxy}"
SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

step() { echo -e "\n\033[1;34m==> $1\033[0m"; }
warn() { echo -e "\033[1;33mWARNING: $1\033[0m"; }

# 1. System packages (not persistent)
step "apt packages"
$SUDO apt-get update -qq
$SUDO apt-get install -y libportaudio2
$SUDO apt-get install -y "ros-${ROS_DISTRO}-web-video-server" \
    || warn "ros-${ROS_DISTRO}-web-video-server not installed (only needed for the browser camera stream)"

# 2. System Python 3.8 extras for the ROS nodes (not persistent)
step "pip3 --user packages (ROS node side)"
pip3 install --user sounddevice   # scipy, tqdm, requests come with the image

# OpenCV contrib (CSRT tracker + imshow), same version as the headless build
CV_VERSION="$(pip3 show opencv-python-headless 2>/dev/null | awk '/^Version:/{print $2}')"
CV_VERSION="${CV_VERSION:-4.13.0.92}"
pip3 install --user "opencv-contrib-python==${CV_VERSION}"
CV2_QT_DIR="$HOME/.local/lib/python3.8/site-packages/cv2/qt"
if [ -d "$CV2_QT_DIR" ]; then
    ln -sfn /usr/share/fonts/truetype/dejavu "$CV2_QT_DIR/fonts"
fi

# 3. Gemini server venv, Python 3.12 (persistent, lives in the mounted gemini_pkg/)
step "Gemini server venv"
export UV_PYTHON_INSTALL_DIR="$PKG_DIR/.uv-python"
if [ ! -x "$PKG_DIR/.venv/bin/python" ]; then
    uv venv --python 3.12 "$PKG_DIR/.venv"
else
    echo "venv already exists: $PKG_DIR/.venv"
fi
VIRTUAL_ENV="$PKG_DIR/.venv" uv pip install -r "$PKG_DIR/requirements.txt"
touch "$PKG_DIR/.venv/COLCON_IGNORE" "$PKG_DIR/.uv-python/COLCON_IGNORE"

# 4. Checks
step "Checks"
[ -f "$PKG_DIR/gemini_pkg/.env" ] || warn "missing $PKG_DIR/gemini_pkg/.env (GEMINI_API_KEY=...)"
[ -f "$PKG_DIR/gemini_pkg/yolo26s.pt" ] || warn "missing $PKG_DIR/gemini_pkg/yolo26s.pt (YOLO model)"
python3 -c "import cv2, sounddevice, scipy, tqdm, requests, pyrealsense2, ultralytics; cv2.legacy.TrackerCSRT_create" \
    && echo "System Python imports OK" || warn "some system Python imports failed (see above)"
"$PKG_DIR/.venv/bin/python" -c "import google.genai, dotenv, fastapi, uvicorn, pydantic, PIL, cv2, numpy" \
    && echo "venv imports OK" || warn "some venv imports failed (see above)"

echo -e "\n\033[1;32mSetup done.\033[0m Don't forget: colcon build + source install/setup.bash"
