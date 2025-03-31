from collections import OrderedDict
import numpy as np

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