import cv2
import numpy as np
import os
import time
import pygame
from collections import deque
from synthetic_data_generator import SyntheticSmartwatchDataGenerator
import pandas as pd
import random
import tensorflow.lite as tflite  # Direct import for TFLite

# Initialize Pygame for alert sound
pygame.mixer.init()
try:
    pygame.mixer.music.load("alarm.wav")
except:
    print("[WARNING] Could not load alarm sound file")

# Initialize synthetic data generator
smartwatch_generator = SyntheticSmartwatchDataGenerator()

# Constants
IMG_SIZE = (128, 128)
MIN_FRAMES_FOR_ALERT = 10  # Minimum consecutive drowsy frames for alert
ALERT_COOLDOWN = 5  # Seconds between alerts
PHYSIO_CHECK_INTERVAL = 2.0  # Seconds between physiological checks

# Haar cascades for fallback detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

# TFLite model initialization
interpreter = None
input_details = None
output_details = None

def load_model(model_path):
    global interpreter, input_details, output_details
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file '{model_path}' not found.")
        
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("[INFO] TFLite model loaded successfully.")
        return True
    except Exception as e:
        print(f"[ERROR] Could not load TFLite model: {e}")
        return False

# Try multiple possible model locations
model_locations = [
    "drowsiness_model.tflite",
    os.path.join(os.path.dirname(__file__), "drowsiness_model.tflite"),
    os.path.join(os.path.dirname(__file__), "models", "drowsiness_model.tflite"),
    os.path.join(os.getcwd(), "drowsiness_model.tflite")  # Current working directory
]

model_loaded = False
for model_path in model_locations:
    if load_model(model_path):
        model_loaded = True
        break

if not model_loaded:
    print("[WARNING] Continuing without TFLite model - using fallback methods only")

def predict_drowsiness(frame):
    """Predict drowsiness using TFLite model"""
    if interpreter is None:
        return "Uncertain (No Model)"
    
    try:
        # Preprocess frame
        img = cv2.resize(frame, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0  # Combine operations
        img = np.expand_dims(img, axis=0)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        # Apply confidence threshold
        confidence = output[0][0]
        if confidence > 0.75:
            return "Drowsy (High Confidence)", confidence
        elif confidence > 0.4:
            return "Drowsy (Low Confidence)", confidence
        else:
            return "Not Drowsy", confidence
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        return "Uncertain (Error)", 0.0

def detect_drowsiness_fallback(frame):
    """Fallback detection using Haar cascades"""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return "Uncertain (No Face)", 0.0

        # Process largest face only
        (x, y, w, h) = max(faces, key=lambda f: f[2]*f[3])
        face_roi = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_roi, 1.1, 3)

        if len(eyes) < 1:
            return "Uncertain (No Eyes)", 0.0

        # Eye closure analysis
        closed_eyes = 0
        for (ex, ey, ew, eh) in eyes:
            eye_region = face_roi[ey:ey+eh, ex:ex+ew]
            
            # Detect if eye is closed using intensity analysis
            _, thresh_eye = cv2.threshold(eye_region, 45, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                closed_eyes += 1
            else:
                largest_contour = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(largest_contour)
                if contour_area < 50:  # Threshold for closed eye
                    closed_eyes += 1

        # Determine drowsiness based on closed eyes
        confidence = min(1.0, closed_eyes / max(1, len(eyes)))
        if confidence > 0.75:
            return "Drowsy (Fallback)", confidence
        elif confidence > 0.5:
            return "Uncertain (Fallback)", confidence
        else:
            return "Not Drowsy (Fallback)", confidence
    except Exception as e:
        print(f"[ERROR] Fallback detection failed: {e}")
        return "Uncertain (Error)", 0.0

def confirm_with_physiological_data():
    """Get current physiological state"""
    try:
        # Generate faster physiological data (0.05 minutes = 3 seconds)
        data = smartwatch_generator.generate_data(duration_minutes=0.05, frequency_hz=2)
        return data.iloc[-1]['state'] == 'drowsy'
    except Exception as e:
        print(f"[ERROR] Physiological data generation failed: {e}")
        return False

def play_alert():
    """Play alert sound with error handling"""
    try:
        pygame.mixer.music.play()
    except:
        print("[WARNING] Could not play alert sound")

# Main detection loop
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return
    
    print("[INFO] Starting drowsiness detection system")
    state_history = deque(maxlen=15)  # Track last 15 states
    last_physio_check = 0
    last_alert_time = 0
    physio_state = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame.")
            continue
        
        # Get prediction from appropriate method
        if model_loaded:
            label, confidence = predict_drowsiness(frame)
        else:
            label, confidence = detect_drowsiness_fallback(frame)
        
        # Handle uncertain cases with physiological data
        current_time = time.time()
        if "Uncertain" in label and (current_time - last_physio_check) > PHYSIO_CHECK_INTERVAL:
            physio_state = confirm_with_physiological_data()
            last_physio_check = current_time
        
        # Final classification
        if "Uncertain" in label:
            final_label = "Drowsy (Physio)" if physio_state else "Not Drowsy (Physio)"
        else:
            final_label = label
        
        # Update state history (1 = drowsy, 0 = not drowsy)
        state_history.append(1 if "Drowsy" in final_label else 0)
        
        # Trigger alert if conditions met
        current_time = time.time()
        alert_active = False
        
        if (sum(state_history) >= MIN_FRAMES_FOR_ALERT and 
            (current_time - last_alert_time) > ALERT_COOLDOWN):
            play_alert()
            last_alert_time = current_time
            alert_active = True
        
        # Display information
        color = (0, 0, 255) if "Drowsy" in final_label else (0, 255, 0)
        cv2.putText(frame, f"Status: {final_label}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Show alert if triggered
        if alert_active:
            cv2.putText(frame, "DROWSINESS ALERT!", (50, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        # Show frame
        cv2.imshow("Drowsiness Detection", frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
