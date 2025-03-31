# OculoClap

## Overview
A real-time drowsiness and clap detection system using OpenCV, dlib, and pyaudio.  It detects a user's facial landmarks in real time to detect drowsiness and yawning and also provides the functionality to mute an alarm by clapping a specified number of times.

## Features
- **Drowsiness Detection:** Uses the Eye Aspect Ratio (EAR) to detect prolonged eye closure.
- **Yawn Detection:** Measures the lip distance to identify excessive yawning.
- **Clap Detection:** Listens for claps to stop an alarm.
- **Alarm System:** Plays an alarm sound when drowsiness or yawning is detected.
- **Background Noise Calibration:** Dynamically adjusts the clap detection threshold based on environmental noise.

## Dependencies
Ensure you have the following installed:

```sh
pip install opencv-python dlib numpy playsound pyaudio
```

## File Structure
```
|-- facial_landmarks.py  # Defines facial landmark indices and helper functions
|-- landmarks.py         # Main script for drowsiness and clap detection
|-- shape_predictor_68_face_landmarks.dat  # Pre-trained model for facial landmark detection
|-- alarm_audio.wav      # Alarm sound file
```

## Usage
### 1. Run the Detection System
```sh
python landmarks.py
```
### 2. Controls
- Press **'q'** to exit.
- Clap **N** times (default: 1) to stop the alarm.

## How It Works
### Drowsiness Detection
- Extracts **eye landmarks** using dlib’s `shape_predictor_68_face_landmarks.dat` model.
- Computes **EAR (Eye Aspect Ratio)** to detect eye closure.
- Triggers an alarm if EAR falls below **0.25** for more than **30 frames**.

### Yawn Detection
- Extracts **mouth landmarks**.
- Measures the vertical distance between upper and lower lips.
- Triggers an alarm if the yawn threshold exceeds **23 pixels**.

### Clap Detection
- Uses **pyaudio** to monitor real-time audio.
- Measures sound energy levels and compares them against a dynamically set threshold.
- If **N claps** are detected within a short period, the alarm is dismissed.

## Customization
You can modify the sensitivity of the system by changing these constants in `landmarks.py`:

```python
EAR_THRESHOLD = 0.25  # Adjust for eye closure sensitivity
EAR_CONSEC_FRAMES = 30  # Number of consecutive frames before triggering alarm
YAWN_THRESHOLD = 23  # Adjust for yawning detection sensitivity
CLAP_THRESHOLD = 1  # Number of claps needed to dismiss alarm
CLAP_COOLDOWN = 10  # Frames to wait between clap detections
```

## Notes
- Make sure your microphone is functioning properly for clap detection.
- The facial landmark model `shape_predictor_68_face_landmarks.dat` must be downloaded separately.
- Adjust the camera angle for better facial detection accuracy.

