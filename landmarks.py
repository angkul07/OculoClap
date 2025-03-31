import cv2
import dlib
import numpy as np
from threading import Thread
import playsound
import pyaudio
import time
import audioop
from facial_landmarks import calc_ear, lip_distance

# Constants
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 30
YAWN_THRESHOLD = 23
CLAP_THRESHOLD = 1  
CLAP_COOLDOWN = 10  # Frames to wait between clap detections

# Clap detection parameters - more sensitive defaults
CHUNK = 1024  # Smaller chunk size for better temporal resolution
RATE = 16000  # Lower sample rate focusing on audible range
FORMAT = pyaudio.paInt16
CHANNELS = 1
CLAP_ENERGY_THRESHOLD = 0.08  # Starting threshold - it will be calibrated
CLAP_DYNAMIC_ADJUSTMENT = 1.5  # Used to multiply background noise level

alarm_status = False
alarm_status2 = False
COUNTER = 0
clap_count = 0
clap_cooldown = 0
background_noise = 0

def continuous_alarm(path="alarm_audio.wav"):
    """Function to play alarm continuously until stopped"""
    global alarm_status, alarm_status2
    while alarm_status or alarm_status2:
        playsound.playsound(path)
        time.sleep(0.1)

def single_alarm(path="alarm_audio.wav"):
    playsound.playsound(path)

# Set up audio for clap detection
def setup_audio():
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    return p, stream

def calibrate_background_noise(stream, samples=50):
    """Measure the background noise level to dynamically set clap threshold"""
    print("Calibrating background noise... Please be quiet.")
    
    # Collect multiple samples
    noise_levels = []
    for _ in range(samples):
        data = stream.read(CHUNK, exception_on_overflow=False)
        rms = audioop.rms(data, 2)  # Get RMS value
        normalized_rms = rms / 32767.0  # Normalize
        noise_levels.append(normalized_rms)
        time.sleep(0.02)  # Short delay between measurements
        
    # Calculate average noise level
    avg_noise = sum(noise_levels) / len(noise_levels)
    threshold = avg_noise * CLAP_DYNAMIC_ADJUSTMENT
    
    print(f"Background noise: {avg_noise:.6f}")
    print(f"Clap threshold set to: {threshold:.6f}")
    
    return avg_noise, threshold

def detect_clap(stream, threshold, prev_energy=0):
    """Improved clap detection using energy delta"""
    data = stream.read(CHUNK, exception_on_overflow=False)
    
    # Use audioop module for more efficient RMS calculation
    rms = audioop.rms(data, 2)
    
    # Normalize
    current_energy = rms / 32767.0
    
    # Look for a rapid increase in energy followed by decrease (typical of claps)
    energy_delta = abs(current_energy - prev_energy)
    
    # Check if energy exceeds threshold and we have a significant change
    if current_energy > threshold and energy_delta > threshold * 0.5:
        return True, current_energy
    
    return False, current_energy

# Initialize
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize audio for clap detection
p, audio_stream = setup_audio()

# Calibrate the microphone for background noise
background_noise, clap_threshold = calibrate_background_noise(audio_stream)

cap = cv2.VideoCapture(0)

# Variable to store previous energy level for delta calculation
prev_energy = 0

# Flag for continuous alarm thread
alarm_thread_running = False

print(f"Clap detection ready. Threshold: {clap_threshold}")
print("Starting capture...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    
    # Check for claps if alarm is active
    if alarm_status or alarm_status2:
        # Start continuous alarm thread if not already running
        if not alarm_thread_running:
            t = Thread(target=continuous_alarm)
            t.daemon = True
            t.start()
            alarm_thread_running = True
            
        if clap_cooldown <= 0:
            clap_detected, prev_energy = detect_clap(audio_stream, clap_threshold, prev_energy)
            if clap_detected:
                clap_count += 1
                clap_cooldown = CLAP_COOLDOWN
                print(f"Clap detected! Count: {clap_count}/{CLAP_THRESHOLD}")
        else:
            # Just update the previous energy value
            _, prev_energy = detect_clap(audio_stream, clap_threshold, prev_energy)

        # If enough claps, dismiss the alarm
        if clap_count >= CLAP_THRESHOLD:
            alarm_status = False
            alarm_status2 = False
            COUNTER = 0
            clap_count = 0
            alarm_thread_running = False
            print("Alarm dismissed by claps!")
    else:
        # Reset clap count when no alarm is active
        clap_count = 0
        alarm_thread_running = False
        
        # Still update the energy level for better detection when alarm activates
        _, prev_energy = detect_clap(audio_stream, clap_threshold, prev_energy)
    
    # Decrease cooldown counter
    if clap_cooldown > 0:
        clap_cooldown -= 1

    for rect in rects:
        # Get the bounding box coordinates
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
            
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        # Predict facial landmarks
        landmarks = predictor(gray, rect)
        landmarks = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])

        # Calculate eye aspect ratio
        eye = calc_ear(landmarks)
        ear = eye[0]
        leftEye = eye[1]
        rightEye = eye[2]

        # Draw eye contours
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        distance = lip_distance(landmarks)
        lip = landmarks[48: 60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        # Drowsiness detection (EAR)
        if ear < EAR_THRESHOLD:
            COUNTER += 1
            if COUNTER >= EAR_CONSEC_FRAMES:
                if not alarm_status:
                    alarm_status = True
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            if not alarm_status2:  # Only reset counter if yawn alarm isn't active
                COUNTER = 0
                alarm_status = False

        # Yawn detection
        if distance > YAWN_THRESHOLD:
            cv2.putText(frame, "Yawn Alert", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not alarm_status2:
                alarm_status2 = True
        else:
            if not alarm_status: 
                alarm_status2 = False

        # Display metrics
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display clap count if alarms are active
        if alarm_status or alarm_status2:
            cv2.putText(frame, f"Claps: {clap_count}/{CLAP_THRESHOLD}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Always show current audio level for debugging
        audio_level = prev_energy / clap_threshold  # Normalized to threshold
        level_width = int(100 * audio_level)
        cv2.rectangle(frame, (10, 120), (10 + level_width, 130), (0, 255, 255), -1)
        cv2.putText(frame, f"Audio: {prev_energy:.3f}/{clap_threshold:.3f}", (120, 130),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow("Drowsiness Detector", frame)
            
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            
# Clean up
cap.release()
audio_stream.stop_stream()
audio_stream.close()
p.terminate()
cv2.destroyAllWindows()