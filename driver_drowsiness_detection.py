import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
import threading
from scipy.spatial.transform import Rotation as R
import csv

# Initialize Pygame for alert sound
pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")  # Replace with a valid alarm sound file

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Landmark indices for eyes and mouth
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [78, 81, 13, 311, 308, 402]

# Function to calculate EAR (Eye Aspect Ratio)
def calculate_ear(eye_landmarks):
    P1, P2, P3, P4, P5, P6 = eye_landmarks
    vertical_1 = np.linalg.norm(P2 - P6)
    vertical_2 = np.linalg.norm(P3 - P5)
    horizontal = np.linalg.norm(P1 - P4)
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

# Function to calculate MAR (Mouth Aspect Ratio) for yawning detection
def calculate_mar(mouth_landmarks):
    P1, P2, P3, P4, P5, P6 = mouth_landmarks
    vertical_1 = np.linalg.norm(P2 - P6)
    vertical_2 = np.linalg.norm(P3 - P5)
    horizontal = np.linalg.norm(P1 - P4)
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

# Function to play alert sound
def play_alert():
    pygame.mixer.music.play()

# Function to log events
def log_event(event_type, ear, mar):
    with open("drowsiness_log.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), event_type, ear, mar])

# Function to calculate head pose
def get_head_pose(face_landmarks, frame_shape):
    image_points = np.array([
        [face_landmarks.landmark[1].x * frame_shape[1], face_landmarks.landmark[1].y * frame_shape[0]],  # Nose tip
        [face_landmarks.landmark[33].x * frame_shape[1], face_landmarks.landmark[33].y * frame_shape[0]],  # Chin
        [face_landmarks.landmark[61].x * frame_shape[1], face_landmarks.landmark[61].y * frame_shape[0]],  # Left eye corner
        [face_landmarks.landmark[291].x * frame_shape[1], face_landmarks.landmark[291].y * frame_shape[0]],  # Right eye corner
        [face_landmarks.landmark[199].x * frame_shape[1], face_landmarks.landmark[199].y * frame_shape[0]],  # Left mouth corner
        [face_landmarks.landmark[425].x * frame_shape[1], face_landmarks.landmark[425].y * frame_shape[0]]   # Right mouth corner
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),      # Chin
        (-225.0, 170.0, -135.0),   # Left eye corner
        (225.0, 170.0, -135.0),    # Right eye corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0)    # Right mouth corner
    ])

    focal_length = frame_shape[1]
    center = (frame_shape[1] / 2, frame_shape[0] / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")

    _, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, None)
    return rotation_vector, translation_vector

# Capture video
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera opened successfully.")

# Calibration phase
calibration_time = 5
start_time = time.time()
ear_values = []
mar_values = []

while time.time() - start_time < calibration_time:
    ret, frame = cap.read()
    if not ret:
        continue
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            left_eye_points = np.array([[face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h] for i in LEFT_EYE])
            right_eye_points = np.array([[face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h] for i in RIGHT_EYE])
            mouth_points = np.array([[face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h] for i in MOUTH])
            left_ear = calculate_ear(left_eye_points)
            right_ear = calculate_ear(right_eye_points)
            avg_ear = (left_ear + right_ear) / 2.0
            mar = calculate_mar(mouth_points)
            ear_values.append(avg_ear)
            mar_values.append(mar)

ear_threshold = np.mean(ear_values) * 0.8  # Adjust threshold based on calibration
mar_threshold = np.mean(mar_values) * 1.2  # Adjust threshold based on calibration

# Drowsiness parameters
drowsy_time = 3  # Seconds before drowsiness alert
ear_below_threshold_start = None
mar_above_threshold_start = None
blink_threshold = 0.2
blink_counter = 0
blink_start_time = None
ear_history = []
mar_history = []
history_length = 10
frame_counter = 0
skip_frames = 2

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame_counter += 1
    if frame_counter % skip_frames != 0:
        continue

    # Convert frame to RGB (for MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            # Extract landmarks for eyes and mouth
            left_eye_points = np.array([[face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h] for i in LEFT_EYE])
            right_eye_points = np.array([[face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h] for i in RIGHT_EYE])
            mouth_points = np.array([[face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h] for i in MOUTH])

            # Calculate EAR for both eyes
            left_ear = calculate_ear(left_eye_points)
            right_ear = calculate_ear(right_eye_points)
            avg_ear = (left_ear + right_ear) / 2.0

            # Calculate MAR for yawning detection
            mar = calculate_mar(mouth_points)

            # Smooth EAR and MAR values
            ear_history.append(avg_ear)
            mar_history.append(mar)
            if len(ear_history) > history_length:
                ear_history.pop(0)
                mar_history.pop(0)
            smoothed_ear = np.mean(ear_history)
            smoothed_mar = np.mean(mar_history)

            # Display EAR and MAR values
            cv2.putText(frame, f'EAR: {smoothed_ear:.2f}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f'MAR: {smoothed_mar:.2f}', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Drowsiness detection based on EAR
            if smoothed_ear < ear_threshold:
                if ear_below_threshold_start is None:
                    ear_below_threshold_start = time.time()
                elif time.time() - ear_below_threshold_start >= drowsy_time:
                    cv2.putText(frame, "DROWSINESS ALERT!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    threading.Thread(target=play_alert, daemon=True).start()
                    log_event("Drowsiness", smoothed_ear, smoothed_mar)
            else:
                ear_below_threshold_start = None

            # Yawning detection based on MAR
            if smoothed_mar > mar_threshold:
                if mar_above_threshold_start is None:
                    mar_above_threshold_start = time.time()
                elif time.time() - mar_above_threshold_start >= 2:  # Detect yawning for at least 2 seconds
                    cv2.putText(frame, "YAWNING ALERT!", (100, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    threading.Thread(target=play_alert, daemon=True).start()
                    log_event("Yawning", smoothed_ear, smoothed_mar)
            else:
                mar_above_threshold_start = None

            # Blink detection
            if smoothed_ear < blink_threshold:
                if blink_start_time is None:
                    blink_start_time = time.time()
                elif time.time() - blink_start_time < 0.5:  # Blink duration < 0.5 seconds
                    blink_counter += 1
                    blink_start_time = None
            else:
                blink_start_time = None

            if blink_counter >= 3:  # Detect drowsiness if 3 blinks occur in a short time
                cv2.putText(frame, "BLINK ALERT!", (100, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                threading.Thread(target=play_alert, daemon=True).start()
                log_event("Blink", smoothed_ear, smoothed_mar)
                blink_counter = 0

            # Head pose estimation
            rotation_vector, translation_vector = get_head_pose(face_landmarks, (h, w))
            r = R.from_rotvec(rotation_vector.ravel())
            angles = r.as_euler('xyz', degrees=True)
            cv2.putText(frame, f'Head Pose: {angles[0]:.1f}, {angles[1]:.1f}, {angles[2]:.1f}', (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display frame
    cv2.imshow("Driver Drowsiness Detection", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()