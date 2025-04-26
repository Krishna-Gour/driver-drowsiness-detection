import random
import time
import pandas as pd
from typing import Dict, List, Any
import numpy as np

class SyntheticSmartwatchDataGenerator:
    def __init__(self):
        self.states = ['awake', 'drowsy']
        self.current_state = 'awake'
        self.last_state_change = time.time()
        
    def generate_data(self, duration_minutes: float = 1.0, frequency_hz: float = 1.0) -> pd.DataFrame:
        """Generate synthetic physiological data."""
        num_samples = int(duration_minutes * 60 * frequency_hz)
        data = []
        
        # Randomly change state occasionally
        if random.random() < 0.05:  # 5% chance to change state
            self.current_state = random.choice(self.states)
            self.last_state_change = time.time()
        
        for _ in range(num_samples):
            if self.current_state == 'awake':
                heart_rate = random.normalvariate(75, 5)
                gsr = random.normalvariate(0.5, 0.1)
                movement = random.normalvariate(0.7, 0.2)
            else:  # drowsy
                heart_rate = random.normalvariate(65, 3)
                gsr = random.normalvariate(0.3, 0.05)
                movement = random.normalvariate(0.2, 0.1)
            
            # Ensure values stay within reasonable bounds
            heart_rate = max(40, min(120, heart_rate))
            gsr = max(0.1, min(1.0, gsr))
            movement = max(0.0, min(1.0, movement))
            
            data.append({
                'timestamp': time.time(),
                'heart_rate': heart_rate,
                'gsr': gsr,  # Galvanic skin response
                'movement': movement,
                'state': self.current_state
            })
            time.sleep(1.0 / frequency_hz if frequency_hz > 0 else 0)
        
        return pd.DataFrame(data)
