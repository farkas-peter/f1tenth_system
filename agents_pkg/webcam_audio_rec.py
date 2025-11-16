"""
pip install sounddevice scipy
sudo apt-get install libportaudio2
sudo apt-get install -y alsa-utils
"""

import sounddevice as sd
from scipy.io.wavfile import write
import time
from tqdm import tqdm
 
def record_from_webcam(filename="webcam_mic.wav", duration=5, samplerate=48000):
    # Find the webcam mic
    devices = sd.query_devices()
    mic_index = None
    for i, d in enumerate(devices):
        if "C270" in d['name'] and d['max_input_channels'] > 0:
            mic_index = i
            print(f"Using device {i}: {d['name']}")
            break
    if mic_index is None:
        raise RuntimeError("Webcam mic not found!")
 
    # Record
    audio = sd.rec(int(duration * samplerate),
                   samplerate=samplerate,
                   channels=1,
                   dtype='int16',
                   device=mic_index)
 
    for _ in tqdm(range(duration), desc="Recording...", unit="s", colour="green"):
        time.sleep(1)
 
    sd.wait()
    write(filename, samplerate, audio)
    print(f"Saved recording to {filename}")
 
record_from_webcam()