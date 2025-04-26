import cv2
import numpy as np
import os
import time
import pygame
from collections import deque
from synthetic_data_generator import SyntheticSmartwatchDataGenerator
import pandas as pd
import random

# Initialize Pygame for alert sound
pygame.mixer.init()
try:
    pygame.mixer.music.load("alarm.wav")  # Make sure this file exists
except:
    print("[WARNING] Could not load alarm sound file")

# Initialize synthetic data generator
smartwatch_generator = SyntheticSmartwatchDataGenerator()

# Try loading TFLite model with absolute path handling
interpreter = None
input_details = None
output_details = None

def load_model(model_path):
    global interpreter, input_details, output_details
    try:
        import tensorflow.lite as tflite
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
    os.path.join(os.path.dirname(__file__), "models", "drowsiness_model.tflite")
]

model_loaded = False
for model_path in model_locations:
    if load_model(model_path):
        model_loaded = True
        break

if not model_loaded:
    print("[WARNING] Continuing without TFLite model - using fallback methods only")

IMG_SIZE = (128, 128)

def predict_drowsiness(frame):
    if interpreter is None:
        # Fallback to traditional methods if model not loaded
        return "Unknown (No Model)"
    
    try:
        # Preprocess frame
        img = cv2.resize(frame, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = np.expand_dims(img, axis=0).astype(np.float32)
        
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        return "Drowsy" if output[0][0] > 0.5 else "Not Drowsy"
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        return "Unknown (Error)"

def confirm_with_physiological_data():
    """Get current physiological state (only called when visual detection is uncertain)"""
    try:
        data = smartwatch_generator.generate_data(duration_minutes=0.1, frequency_hz=1).iloc[-1]
        return data['state'] == 'drowsy'
    except Exception as e:
        print(f"[ERROR] Physiological data generation failed: {e}")
        return False

def play_alert():
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
    
    print("[INFO] Webcam started.")
    frame_counter = 0
    skip_frames = 2
    drowsy_frames = 0
    required_drowsy_frames = 5  # Number of consecutive drowsy frames to trigger alert
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame.")
            break
        
        frame_counter += 1
        if frame_counter % skip_frames != 0:
            continue
        
        # Get prediction from model
        label = predict_drowsiness(frame)
        
        # Check physiological data if model is uncertain
        if label == "Unknown (No Model)":
            if confirm_with_physiological_data():
                label = "Drowsy (Physio)"
            else:
                label = "Not Drowsy (Physio)"
        
        # Simple state tracking
        if "Drowsy" in label:
            drowsy_frames += 1
            if drowsy_frames >= required_drowsy_frames:
                play_alert()
                cv2.putText(frame, "DROWSINESS ALERT!", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            drowsy_frames = 0
        
        # Display output
        color = (0, 0, 255) if "Drowsy" in label else (0, 255, 0)
        cv2.putText(frame, f"Status: {label}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Show frame
        cv2.imshow("Drowsiness Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()