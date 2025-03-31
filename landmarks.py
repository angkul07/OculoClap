import cv2
import dlib
import numpy as np
from threading import Thread, Lock
# from clap_detect import start_audio_stream, alarm
import sounddevice as sd
import playsound
from facial_landmarks import calc_ear, lip_distance

# Constants
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 30
YAWN_THRESHOLD = 23
CLAP_THRESHOLD = 0.5
CLAP_COUNT_THRESHOLD = 2
TIME_WINDOW = 3

# Shared variables with thread lock
alarm_lock = Lock()
clap_count = 0
ALARM_ON = False
COUNTER = 0
saying = False

def audio_callback(indata, frames, time, status):
    global clap_count, ALARM_ON
    volume = np.linalg.norm(indata) * 10
    
    if volume > CLAP_THRESHOLD:
        with alarm_lock:
            clap_count += 1
            print(f"Clap detected! Count: {clap_count}")
            
            if clap_count >= CLAP_COUNT_THRESHOLD and ALARM_ON:
                ALARM_ON = False
                clap_count = 0
                print("Alarm stopped!")

def start_audio_stream():
    """Start the audio stream in a separate thread"""
    with sd.InputStream(callback=audio_callback):
        while True:
            sd.sleep(1000) 

def alarm(path="CV/lazy_detect/alarm_audio.mp3"):
    global ALARM_ON
    while ALARM_ON:
        playsound.playsound(path)
        if not ALARM_ON:
            break

# Initialize dlib's face detector and landmark predictor


def main():
    global ALARM_ON, COUNTER, clap_count

    audio_thread = Thread(target=start_audio_stream)
    audio_thread.daemon = True
    audio_thread.start() 

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("CV/lazy_detect/shape_predictor_68_face_landmarks.dat")

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

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

            # Drowsiness detection logic
            if ear < EAR_THRESHOLD:
                COUNTER += 1

                if COUNTER >= EAR_CONSEC_FRAMES:
                    with alarm_lock:
                        if not ALARM_ON:
                            ALARM_ON = True
                            t = Thread(target=alarm)
                            t.daemon = True
                            t.start()

                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    COUNTER = 0
                    # ALARM_ON = False

                if (distance > YAWN_THRESHOLD):
                        cv2.putText(frame, "Yawn Alert", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        if not ALARM_ON and not saying:
                            ALARM_ON = True
                            t = Thread(target=alarm)
                            t.deamon = True
                            t.start()
                # else:
                #     ALARM_ON = False

            # Display EAR on frame
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        with alarm_lock:
            if clap_count > 0 and not ALARM_ON:
                clap_count = 0

        cv2.imshow("Drowsiness Detector", frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()