import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
from scipy.signal import savgol_filter
import json
import os

class SyntheticSmartwatchDataGenerator:
    def __init__(self, config_file=None):
        """
        Initialize the data generator with default or custom configuration
        """
        # Default configuration
        self.config = {
            "base_values": {
                "alert": {
                    "heart_rate": {"mean": 75, "std": 5},
                    "acceleration": {"mean": 0.5, "std": 0.1},
                    "temperature": {"mean": 36.5, "std": 0.3},
                    "gsr": {"mean": 0.8, "std": 0.05}
                },
                "drowsy": {
                    "heart_rate": {"mean": 60, "std": 5},
                    "acceleration": {"mean": 0.1, "std": 0.05},
                    "temperature": {"mean": 36.0, "std": 0.3},
                    "gsr": {"mean": 0.5, "std": 0.1}
                }
            },
            "state_transition": {
                "min_duration": 60,  # seconds
                "max_duration": 300  # seconds
            },
            "noise": {
                "heart_rate": 0.5,
                "acceleration": 0.05,
                "temperature": 0.1,
                "gsr": 0.02
            }
        }
        
        # Load custom config if provided
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                custom_config = json.load(f)
                self._update_config(custom_config)
    
    def _update_config(self, custom_config):
        """Update configuration with custom values"""
        for section in custom_config:
            if section in self.config:
                for key in custom_config[section]:
                    self.config[section][key] = custom_config[section][key]
    
    def _generate_transition_curve(self, start_val, end_val, duration_samples):
        """Generate smooth transition between two states"""
        x = np.linspace(0, 1, duration_samples)
        # Sigmoid function for smooth transition
        transition = 1 / (1 + np.exp(-12*(x-0.5)))
        return start_val + (end_val - start_val) * transition
    
    def _add_transition_effects(self, data, transition_start, transition_end, prev_state, current_state):
        """Add smooth transitions between states"""
        transition_duration = transition_end - transition_start
        
        # Get base values for previous and current states
        prev_hr = self.config['base_values'][prev_state]['heart_rate']['mean']
        curr_hr = self.config['base_values'][current_state]['heart_rate']['mean']
        
        prev_acc = self.config['base_values'][prev_state]['acceleration']['mean']
        curr_acc = self.config['base_values'][current_state]['acceleration']['mean']
        
        prev_temp = self.config['base_values'][prev_state]['temperature']['mean']
        curr_temp = self.config['base_values'][current_state]['temperature']['mean']
        
        prev_gsr = self.config['base_values'][prev_state]['gsr']['mean']
        curr_gsr = self.config['base_values'][current_state]['gsr']['mean']
        
        # Generate transition curves
        hr_transition = self._generate_transition_curve(prev_hr, curr_hr, transition_duration)
        acc_transition = self._generate_transition_curve(prev_acc, curr_acc, transition_duration)
        temp_transition = self._generate_transition_curve(prev_temp, curr_temp, transition_duration)
        gsr_transition = self._generate_transition_curve(prev_gsr, curr_gsr, transition_duration)
        
        # Apply transitions to data
        for i in range(transition_duration):
            idx = transition_start + i
            data[idx][1] = hr_transition[i]  # heart rate
            data[idx][2] = acc_transition[i]  # acc_x
            data[idx][3] = acc_transition[i]  # acc_y
            data[idx][4] = acc_transition[i]  # acc_z
            data[idx][5] = temp_transition[i]  # temperature
            data[idx][6] = gsr_transition[i]  # gsr
    
    def _add_realistic_noise(self, data):
        """Add realistic noise patterns to the data"""
        n_samples = len(data)
        
        # Generate base noise
        hr_noise = np.random.normal(0, self.config['noise']['heart_rate'], n_samples)
        acc_noise = np.random.normal(0, self.config['noise']['acceleration'], (n_samples, 3))
        temp_noise = np.random.normal(0, self.config['noise']['temperature'], n_samples)
        gsr_noise = np.random.normal(0, self.config['noise']['gsr'], n_samples)
        
        # Add some periodic patterns (like breathing for HR)
        t = np.arange(n_samples)
        breathing_effect = 0.3 * np.sin(2 * np.pi * t / 30)  # ~30 samples per breath cycle
        
        # Apply noise and effects
        for i in range(n_samples):
            data[i][1] += hr_noise[i] + breathing_effect[i]  # heart rate
            data[i][2] += acc_noise[i][0]  # acc_x
            data[i][3] += acc_noise[i][1]  # acc_y
            data[i][4] += acc_noise[i][2]  # acc_z
            data[i][5] += temp_noise[i]  # temperature
            data[i][6] += gsr_noise[i]  # gsr
    
    def _post_process_data(self, data):
        """Apply smoothing and final adjustments"""
        n_samples = len(data)
        
        # Extract columns for processing
        hr = np.array([x[1] for x in data])
        acc_x = np.array([x[2] for x in data])
        acc_y = np.array([x[3] for x in data])
        acc_z = np.array([x[4] for x in data])
        temp = np.array([x[5] for x in data])
        gsr = np.array([x[6] for x in data])
        
        # Apply smoothing (Savitzky-Golay filter)
        window_size = min(15, n_samples // 2)  # Ensure window size is appropriate
        if window_size % 2 == 0:  # Must be odd
            window_size -= 1
        
        if window_size > 1:
            hr = savgol_filter(hr, window_size, 3)
            acc_x = savgol_filter(acc_x, window_size, 3)
            acc_y = savgol_filter(acc_y, window_size, 3)
            acc_z = savgol_filter(acc_z, window_size, 3)
            temp = savgol_filter(temp, window_size, 3)
            gsr = savgol_filter(gsr, window_size, 3)
        
        # Update data with smoothed values
        for i in range(n_samples):
            data[i][1] = round(hr[i], 2)
            data[i][2] = round(acc_x[i], 3)
            data[i][3] = round(acc_y[i], 3)
            data[i][4] = round(acc_z[i], 3)
            data[i][5] = round(temp[i], 2)
            data[i][6] = round(gsr[i], 3)
    
    def generate_data(self, duration_minutes=30, frequency_hz=1, initial_state=None):
        """
        Generate synthetic smartwatch data for drowsiness detection
        
        Parameters:
        - duration_minutes: Total duration in minutes
        - frequency_hz: Sampling frequency in Hz
        - initial_state: Starting state ('alert' or 'drowsy'), random if None
        
        Returns:
        - Pandas DataFrame with synthetic data
        """
        total_samples = duration_minutes * 60 * frequency_hz
        timestamps = [datetime.now() + timedelta(seconds=i/frequency_hz) for i in range(total_samples)]
        
        data = []
        current_state = initial_state if initial_state else random.choice(["alert", "drowsy"])
        switch_interval = random.randint(
            self.config['state_transition']['min_duration'] * frequency_hz,
            self.config['state_transition']['max_duration'] * frequency_hz
        )
        next_switch = switch_interval
        transition_start = 0
        
        for i in range(total_samples):
            if i == next_switch:
                # Store transition points
                transition_start = i
                prev_state = current_state
                current_state = "drowsy" if current_state == "alert" else "alert"
                switch_interval = random.randint(
                    self.config['state_transition']['min_duration'] * frequency_hz,
                    self.config['state_transition']['max_duration'] * frequency_hz
                )
                next_switch = i + switch_interval
            
            # Get base values for current state
            state_config = self.config['base_values'][current_state]
            
            hr = np.random.normal(state_config['heart_rate']['mean'], state_config['heart_rate']['std'])
            acc = np.random.normal(state_config['acceleration']['mean'], state_config['acceleration']['std'], 3)
            temp = np.random.normal(state_config['temperature']['mean'], state_config['temperature']['std'])
            gsr = np.random.normal(state_config['gsr']['mean'], state_config['gsr']['std'])
            
            data.append([
                timestamps[i], hr,
                acc[0], acc[1], acc[2],
                temp, gsr,
                current_state
            ])
            
            # If we just passed a transition point, apply smooth transition
            if i == next_switch - 1 and transition_start > 0:
                self._add_transition_effects(data, transition_start, i, prev_state, current_state)
        
        # Add realistic noise patterns
        self._add_realistic_noise(data)
        
        # Apply post-processing and smoothing
        self._post_process_data(data)
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=[
            "timestamp", "heart_rate", 
            "acc_x", "acc_y", "acc_z", 
            "temperature", "gsr", 
            "state"
        ])
        
        return df

def save_data(df, filename, output_dir="data"):
    """Save generated data to file with proper directory structure"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"Synthetic data saved to '{filepath}'")

if __name__ == "__main__":
    # Example usage
    generator = SyntheticSmartwatchDataGenerator()
    
    # Generate 1 hour of data at 2Hz starting in alert state
    df = generator.generate_data(duration_minutes=60, frequency_hz=2, initial_state="alert")
    
    # Save the data
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_data(df, f"synthetic_smartwatch_data_{timestamp_str}.csv")