import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
import threading
from scipy.spatial.transform import Rotation as R
from collections import deque
from typing import Tuple, Optional, List, Dict, Any
import os
import logging
from dataclasses import dataclass
import pandas as pd
from synthetic_data_generator import SyntheticSmartwatchDataGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_ALERT_SOUND = "alarm.wav"
CALIBRATION_FRAMES = 50
EAR_ALERT_DURATION = 2.0  # seconds
MAR_ALERT_DURATION = 2.0  # seconds
UNCERTAINTY_THRESHOLD = 5
VISIBILITY_THRESHOLD = 0.5
EAR_THRESHOLD_MULTIPLIER = 0.8
MAR_THRESHOLD_MULTIPLIER = 1.2
SMOOTHING_WINDOW_SIZE = 5
PHYSIO_CONFIRMATION_DURATION = 10  # seconds of physiological data to analyze
PHYSIO_SAMPLING_RATE = 1  # Hz

@dataclass
class DetectionResult:
    ear: float
    mar: float
    is_drowsy: bool
    is_yawning: bool
    visibility: float
    needs_physiological_confirmation: bool

class PhysiologicalMonitor:
    def __init__(self):
        self.generator = SyntheticSmartwatchDataGenerator()
        self.data_buffer = deque(maxlen=int(PHYSIO_CONFIRMATION_DURATION * PHYSIO_SAMPLING_RATE))
        self.last_sample_time = 0
        self.sampling_interval = 1.0 / PHYSIO_SAMPLING_RATE
    
    def collect_data(self) -> Optional[pd.DataFrame]:
        """Collect physiological data if sampling interval has passed."""
        current_time = time.time()
        if current_time - self.last_sample_time >= self.sampling_interval:
            try:
                # Generate 1 second of data at a time
                new_data = self.generator.generate_data(duration_minutes=1/60, frequency_hz=PHYSIO_SAMPLING_RATE)
                self.data_buffer.extend(new_data.to_dict('records'))
                self.last_sample_time = current_time
                return new_data
            except Exception as e:
                logger.error(f"Error generating physiological data: {e}")
        return None
    
    def analyze_drowsiness(self) -> Tuple[bool, Dict[str, Any]]:
        """Analyze collected physiological data for drowsiness indicators."""
        if len(self.data_buffer) < 5:  # Minimum samples needed
            return False, {}
        
        df = pd.DataFrame(self.data_buffer)
        analysis = {
            'heart_rate_avg': df['heart_rate'].mean(),
            'heart_rate_std': df['heart_rate'].std(),
            'gsr_avg': df['gsr'].mean(),
            'movement_avg': df['movement'].mean(),
            'drowsy_samples': sum(df['state'] == 'drowsy'),
            'total_samples': len(df)
        }
        
        # Simple drowsiness detection logic
        is_drowsy = (
            (analysis['heart_rate_avg'] < 60) or  # Low heart rate
            (analysis['heart_rate_std'] < 2) or   # Low heart rate variability
            (analysis['gsr_avg'] < 0.2) or       # Low skin conductance
            (analysis['movement_avg'] < 0.1) or   # Low movement
            (analysis['drowsy_samples'] / analysis['total_samples'] > 0.7)  # Majority drowsy
        )
        
        return is_drowsy, analysis

class AlertSystem:
    def __init__(self, sound_file: str = DEFAULT_ALERT_SOUND):
        pygame.mixer.init()
        self.sound_file = sound_file
        self.is_playing = False
        self._load_sound()
        
    def _load_sound(self):
        """Load the alert sound file with fallback options."""
        try:
            if os.path.exists(self.sound_file):
                pygame.mixer.music.load(self.sound_file)
            else:
                logger.warning(f"Alert sound file not found: {self.sound_file}")
                self._generate_fallback_beep()
        except Exception as e:
            logger.error(f"Error loading sound file: {e}")
            self._generate_fallback_beep()
    
    def _generate_fallback_beep(self):
        """Generate a simple beep sound as fallback."""
        try:
            sample_rate = 44100
            duration = 0.5
            frequency = 800
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            beep = np.sin(2 * np.pi * frequency * t) * 0.5
            beep = np.c_[beep, beep].astype(np.float32)
            sound = pygame.sndarray.make_sound(beep)
            pygame.mixer.Sound.play(sound)
        except Exception as e:
            logger.error(f"Failed to generate fallback beep: {e}")
    
    def play(self):
        """Play the alert sound if not already playing."""
        if not self.is_playing:
            try:
                pygame.mixer.music.play()
                self.is_playing = True
            except Exception as e:
                logger.error(f"Error playing alert sound: {e}")
    
    def stop(self):
        """Stop the alert sound."""
        pygame.mixer.music.stop()
        self.is_playing = False

class FaceMeshDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmark indices
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.MOUTH = [78, 81, 13, 311, 308, 402]
        self.VISIBILITY_POINTS = [33, 133, 362, 263]
        
        # Smoothing buffers
        self.ear_buffer = deque(maxlen=SMOOTHING_WINDOW_SIZE)
        self.mar_buffer = deque(maxlen=SMOOTHING_WINDOW_SIZE)
    
    def calculate_ear(self, eye_points: np.ndarray) -> float:
        """Calculate Eye Aspect Ratio (EAR) for given eye points."""
        P1, P2, P3, P4, P5, P6 = eye_points
        return (np.linalg.norm(P2 - P6) + np.linalg.norm(P3 - P5)) / (2.0 * np.linalg.norm(P1 - P4))
    
    def calculate_mar(self, mouth_points: np.ndarray) -> float:
        """Calculate Mouth Aspect Ratio (MAR) for given mouth points."""
        P1, P2, P3, P4, P5, P6 = mouth_points
        return (np.linalg.norm(P2 - P6) + np.linalg.norm(P3 - P5)) / (2.0 * np.linalg.norm(P1 - P4))
    
    def process_frame(self, frame: np.ndarray) -> Optional[DetectionResult]:
        """Process a frame and return detection results."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return None
            
        landmarks = results.multi_face_landmarks[0]
        h, w = frame.shape[:2]
        
        # Extract eye and mouth landmarks
        leye = np.array([[landmarks.landmark[i].x * w, landmarks.landmark[i].y * h] for i in self.LEFT_EYE])
        reye = np.array([[landmarks.landmark[i].x * w, landmarks.landmark[i].y * h] for i in self.RIGHT_EYE])
        mouth = np.array([[landmarks.landmark[i].x * w, landmarks.landmark[i].y * h] for i in self.MOUTH])
        
        # Calculate metrics
        left_ear = self.calculate_ear(leye)
        right_ear = self.calculate_ear(reye)
        mar = self.calculate_mar(mouth)
        visibility = np.mean([landmarks.landmark[i].visibility for i in self.VISIBILITY_POINTS])
        
        # Apply smoothing
        self.ear_buffer.append((left_ear + right_ear) / 2.0)
        self.mar_buffer.append(mar)
        
        smoothed_ear = np.mean(self.ear_buffer) if self.ear_buffer else 0
        smoothed_mar = np.mean(self.mar_buffer) if self.mar_buffer else 0
        
        # Determine if we need physiological confirmation
        needs_confirmation = (
            abs(smoothed_ear - self.ear_threshold) < 0.05 or 
            visibility < VISIBILITY_THRESHOLD
        ) if hasattr(self, 'ear_threshold') else False
        
        return DetectionResult(
            ear=smoothed_ear,
            mar=smoothed_mar,
            is_drowsy=smoothed_ear < getattr(self, 'ear_threshold', 0),
            is_yawning=smoothed_mar > getattr(self, 'mar_threshold', 0),
            visibility=visibility,
            needs_physiological_confirmation=needs_confirmation
        )
    
    def calibrate(self, cap: cv2.VideoCapture, calibration_frames: int = CALIBRATION_FRAMES) -> Tuple[float, float]:
        """Calibrate EAR and MAR thresholds based on user's normal state."""
        ear_values, mar_values = [], []
        
        logger.info("Starting calibration...")
        for _ in range(calibration_frames):
            ret, frame = cap.read()
            if not ret:
                continue
                
            result = self.process_frame(frame)
            if result:
                ear_values.append(result.ear)
                mar_values.append(result.mar)
        
        if not ear_values or not mar_values:
            raise RuntimeError("Calibration failed - no face detected")
        
        self.ear_threshold = np.mean(ear_values) * EAR_THRESHOLD_MULTIPLIER
        self.mar_threshold = np.mean(mar_values) * MAR_THRESHOLD_MULTIPLIER
        
        logger.info(f"Calibration complete. EAR threshold: {self.ear_threshold:.2f}, MAR threshold: {self.mar_threshold:.2f}")
        return self.ear_threshold, self.mar_threshold

class DrowsinessDetector:
    def __init__(self):
        self.alert_system = AlertSystem()
        self.detector = FaceMeshDetector()
        self.physio_monitor = PhysiologicalMonitor()
        self.cap = self._initialize_camera()
        
        # State tracking
        self.drowsy_start_time = None
        self.mar_start_time = None
        self.uncertainty_counter = 0
        self.is_running = False
        self.physio_confirmation_active = False
        self.last_physio_analysis = {}
    
    def _initialize_camera(self) -> cv2.VideoCapture:
        """Initialize and return the camera capture object."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not initialize camera")
        
        # Set preferred camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap
    
    def run(self):
        """Main execution loop for drowsiness detection."""
        try:
            self.is_running = True
            self.detector.calibrate(self.cap)
            
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    break
                
                # Collect physiological data in background
                self.physio_monitor.collect_data()
                
                result = self.detector.process_frame(frame)
                if result:
                    self._update_state(result)
                    self._draw_ui(frame, result)
                
                cv2.imshow("Drowsiness Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
        finally:
            self.cleanup()
    
    def _update_state(self, result: DetectionResult):
        """Update the internal state based on detection results."""
        # EAR-based drowsiness detection
        if result.is_drowsy:
            if self.drowsy_start_time is None:
                self.drowsy_start_time = time.time()
            elif time.time() - self.drowsy_start_time > EAR_ALERT_DURATION:
                self.alert_system.play()
        else:
            self.drowsy_start_time = None
            self.alert_system.stop()
        
        # MAR-based yawning detection
        if result.is_yawning:
            if self.mar_start_time is None:
                self.mar_start_time = time.time()
            elif time.time() - self.mar_start_time > MAR_ALERT_DURATION:
                self.alert_system.play()
        else:
            self.mar_start_time = None
        
        # Handle uncertain cases with physiological confirmation
        if result.needs_physiological_confirmation:
            self.uncertainty_counter += 1
            if self.uncertainty_counter > UNCERTAINTY_THRESHOLD:
                self.physio_confirmation_active = True
                is_drowsy, analysis = self.physio_monitor.analyze_drowsiness()
                self.last_physio_analysis = analysis
                
                if is_drowsy:
                    self.alert_system.play()
                self.uncertainty_counter = 0
        else:
            self.physio_confirmation_active = False
            self.uncertainty_counter = 0
    
    def _draw_ui(self, frame: np.ndarray, result: DetectionResult):
        """Draw the user interface on the frame."""
        # Display metrics
        cv2.putText(frame, f'EAR: {result.ear:.2f} (Thresh: {self.detector.ear_threshold:.2f})', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 255, 0) if result.ear > self.detector.ear_threshold else (0, 0, 255), 2)
        cv2.putText(frame, f'MAR: {result.mar:.2f} (Thresh: {self.detector.mar_threshold:.2f})', 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 255, 0) if result.mar < self.detector.mar_threshold else (0, 0, 255), 2)
        cv2.putText(frame, f'Visibility: {result.visibility:.2f}', 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 255, 0) if result.visibility > VISIBILITY_THRESHOLD else (0, 0, 255), 2)
        
        # Display alerts if needed
        if result.is_drowsy and self.drowsy_start_time and (time.time() - self.drowsy_start_time > EAR_ALERT_DURATION):
            cv2.putText(frame, "DROWSINESS ALERT!", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if result.is_yawning and self.mar_start_time and (time.time() - self.mar_start_time > MAR_ALERT_DURATION):
            cv2.putText(frame, "YAWNING ALERT!", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if self.physio_confirmation_active:
            self._draw_physio_analysis(frame)
    
    def _draw_physio_analysis(self, frame: np.ndarray):
        """Draw physiological data analysis on the frame."""
        y_pos = 220
        cv2.putText(frame, "PHYSIOLOGICAL CONFIRMATION:", (50, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        y_pos += 30
        
        if not self.last_physio_analysis:
            cv2.putText(frame, "Collecting data...", (70, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
            return
        
        for i, (k, v) in enumerate(self.last_physio_analysis.items()):
            if i > 4:  # Limit displayed metrics
                break
            cv2.putText(frame, f"{k.replace('_', ' ').title()}: {v:.2f}", 
                       (70, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (0, 200, 255) if "drowsy" in k and v > 0.5 else (0, 165, 255), 1)
            y_pos += 25
    
    def cleanup(self):
        """Clean up resources."""
        self.is_running = False
        self.alert_system.stop()
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("System shutdown complete")

if __name__ == "__main__":
    try:
        detector = DrowsinessDetector()
        detector.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
