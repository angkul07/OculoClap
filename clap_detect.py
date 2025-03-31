import playsound
import sounddevice as sd
import numpy as np

EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 30
YAWN_THRESHOLD = 23
CLAP_THRESHOLD = 80
CLAP_COUNT_THRESHOLD = 3
TIME_WINDOW = 3

def audio_callback(indata, frames, time, status):
    global clap_count, ALARM_ON, noise_baseline, samples_to_calibrate, last_clap_time
    
    # Calibrate noise baseline first
    if samples_to_calibrate > 0:
        noise_baseline += np.sqrt(np.mean(indata**2))
        samples_to_calibrate -= 1
        if samples_to_calibrate == 0:
            noise_baseline /= 10
            print(f"Noise baseline: {noise_baseline:.4f}")
        return
    
    # Compute volume (RMS)
    volume = np.sqrt(np.mean(indata**2))
    
    # Detect clap (only if significantly louder than baseline)
    current_time = time.time()
    if volume > noise_baseline + 0.2 and (current_time - last_clap_time) > 0.2:
        clap_count += 1
        last_clap_time = current_time
        print(f"Clap {clap_count} (Volume: {volume:.4f})")
        
        if clap_count >= CLAP_COUNT_THRESHOLD and ALARM_ON:
            ALARM_ON = False
            clap_count = 0
            print("Alarm stopped!")

def start_audio_stream():
    with sd.InputStream(callback=audio_callback):
        while True:
            sd.sleep(2000)

def alarm(path="CV/lazy_detect/alarm_audio.mp3"):
    global ALARM_ON
    while ALARM_ON:
        playsound.playsound(path)
        if not ALARM_ON:
            break