import cv2
import dlib
import numpy as np
from threading import Thread
import playsound
# from facial_landmarks import calc_ear, lip_distance
from collections import OrderedDict

# Constants
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 30
YAWN_THRESHOLD = 23
alarm_status = False
alarm_status2 = False
COUNTER = 0
saying = False

def alarm(path="CV/lazy_detect/alarm_audio.mp3"):
    playsound.playsound(path)

FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])

def eye_aspect_ratio(eye):
    # Calculate vertical distances
    vertical1_dist = np.linalg.norm(eye[1] - eye[5])
    vertical2_dist = np.linalg.norm(eye[2] - eye[4])

    # Calculate horizontal distance
    horizontal_dist = np.linalg.norm(eye[0] - eye[3])

    # Compute eye aspect ratio
    ear = (vertical1_dist + vertical2_dist) / (2.0 * horizontal_dist)
    return ear

def calc_ear(landmarks):
    (lstart, lend) = FACIAL_LANDMARKS_IDXS["left_eye"]
    (rstart, rend) = FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = landmarks[lstart:lend]
    rightEye = landmarks[rstart:rend]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    
    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(landmarks):
    top_lip = landmarks[50:53]
    top_lip = np.concatenate((top_lip, landmarks[61:64]))

    low_lip = landmarks[56:59]
    low_lip = np.concatenate((low_lip, landmarks[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

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

        # Drowsiness detection (EAR)
        if ear < EAR_THRESHOLD:
            COUNTER += 1
            if COUNTER >= EAR_CONSEC_FRAMES:
                if not alarm_status:
                    alarm_status = True
                    t = Thread(target=alarm)
                    t.daemon = True
                    t.start()
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            alarm_status = False

        # Yawn detection (independent of EAR)
        if distance > YAWN_THRESHOLD:
            cv2.putText(frame, "Yawn Alert", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not alarm_status2:
                alarm_status2 = True
                t = Thread(target=alarm)
                t.daemon = True
                t.start()
        else:
            alarm_status2 = False

        # Display metrics
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Drowsiness Detector", frame)
            
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            
cap.release()
cv2.destroyAllWindows()
