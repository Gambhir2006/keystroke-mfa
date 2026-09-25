"""
Keystroke Data Collection Module
This module provides utilities for collecting and processing keystroke timing data.
"""

import json
import sqlite3
from datetime import datetime
import os

# Get the directory of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(SCRIPT_DIR, 'keystroke_mfa.db')

def extract_features(keystroke_data):
    """
    Extract features from keystroke timing data
    
    Args:
        keystroke_data: Dictionary containing keyEvents array with timing information
        
    Returns:
        Dictionary of extracted features or None if data is invalid
    """
    key_events = keystroke_data.get('keyEvents', [])
    
    if not key_events:
        return None
    
    hold_times = []
    flight_times = []
    
    for i, event in enumerate(key_events):
        if event['type'] == 'hold':
            hold_times.append(event['duration'])
        elif event['type'] == 'flight' and i > 0:
            flight_times.append(event['duration'])
    
    if not hold_times:
        return None
    
    import numpy as np
    
    features = {
        'mean_hold_time': float(np.mean(hold_times)),
        'hold_time_std': float(np.std(hold_times)) if len(hold_times) > 1 else 0.0,
        'mean_flight_time': float(np.mean(flight_times)) if flight_times else 0.0,
        'flight_time_std': float(np.std(flight_times)) if len(flight_times) > 1 else 0.0,
        'total_time': float(keystroke_data.get('totalTime', 0)),
        'typing_speed': float(len(key_events) / max(keystroke_data.get('totalTime', 1), 0.001)),
        'char_count': int(keystroke_data.get('charCount', 0))
    }
    
    return features

def save_typing_sample(user_id, features):
    """
    Save a typing sample to the database
    
    Args:
        user_id: ID of the user
        features: Dictionary of extracted features
    """
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    cursor.execute(
        'INSERT INTO typing_samples (user_id, features_json) VALUES (?, ?)',
        (user_id, json.dumps(features))
    )
    
    conn.commit()
    conn.close()

def get_user_samples(user_id):
    """
    Retrieve all typing samples for a user
    
    Args:
        user_id: ID of the user
        
    Returns:
        List of feature dictionaries
    """
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    cursor.execute('SELECT features_json FROM typing_samples WHERE user_id = ?', (user_id,))
    samples = cursor.fetchall()
    
    conn.close()
    
    return [json.loads(sample[0]) for sample in samples]

def validate_keystroke_data(keystroke_data):
    """
    Validate keystroke data structure
    
    Args:
        keystroke_data: Dictionary to validate
        
    Returns:
        Boolean indicating validity
    """
    required_fields = ['keyEvents', 'totalTime', 'charCount']
    
    if not all(field in keystroke_data for field in required_fields):
        return False
    
    if not isinstance(keystroke_data['keyEvents'], list):
        return False
    
    if len(keystroke_data['keyEvents']) == 0:
        return False
    
    return True

if __name__ == '__main__':
    # Test the module
    test_data = {
        'keyEvents': [
            {'type': 'hold', 'duration': 100},
            {'type': 'flight', 'duration': 50},
            {'type': 'hold', 'duration': 120}
        ],
        'totalTime': 270,
        'charCount': 2
    }
    
    features = extract_features(test_data)
    print("Extracted features:", features)
    print("Validation:", validate_keystroke_data(test_data))