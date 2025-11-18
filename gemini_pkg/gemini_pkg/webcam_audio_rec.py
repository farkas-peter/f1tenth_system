"""
pip install sounddevice scipy
sudo apt-get install libportaudio2
sudo apt-get install -y alsa-utils
"""

import sounddevice as sd
from scipy.io.wavfile import write
import time
from tqdm import tqdm
import os
 
def record_webcam_audio(filename="temp_audio.wav", duration=5, samplerate=48000):
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
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    write(filepath, samplerate, audio)
    print(f"Saved recording to {filename}")

    return filename

if __name__ == '__main__':
    audio_file = record_webcam_audio()
    # check if the file is created
    
    print(os.path.exists(audio_file))