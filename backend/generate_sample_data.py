"""
Synthetic Data Generator for Keystroke MFA
This module generates synthetic keystroke data for testing and demonstration purposes.
"""

import argparse
import sqlite3
import json
import random
import numpy as np
from datetime import datetime
import os

# Get the directory of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(SCRIPT_DIR, 'keystroke_mfa.db')

def generate_user_features(user_id, num_samples=30):
    """
    Generate synthetic keystroke features for a user
    
    Args:
        user_id: ID of the user
        num_samples: Number of samples to generate
        
    Returns:
        List of feature dictionaries
    """
    # Each user has a unique "typing signature"
    base_hold_time = random.uniform(80, 150)  # Base hold time in ms
    base_flight_time = random.uniform(40, 100)  # Base flight time in ms
    typing_speed = random.uniform(3, 8)  # Characters per second
    
    samples = []
    for _ in range(num_samples):
        # Add natural variation to each sample
        hold_variation = random.uniform(0.7, 1.3)
        flight_variation = random.uniform(0.6, 1.4)
        
        hold_times = [base_hold_time * hold_variation + random.uniform(-20, 20) for _ in range(8)]
        flight_times = [base_flight_time * flight_variation + random.uniform(-15, 15) for _ in range(7)]
        
        features = {
            'mean_hold_time': float(np.mean(hold_times)),
            'hold_time_std': float(np.std(hold_times)),
            'mean_flight_time': float(np.mean(flight_times)),
            'flight_time_std': float(np.std(flight_times)),
            'total_time': float(sum(hold_times) + sum(flight_times)),
            'typing_speed': typing_speed * random.uniform(0.8, 1.2),
            'char_count': 8
        }
        
        samples.append(features)
    
    return samples

def generate_sample_users(num_users=3, samples_per_user=30):
    """
    Generate synthetic users and their keystroke data
    
    Args:
        num_users: Number of users to create
        samples_per_user: Number of samples per user
    """
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Check if users table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
    if not cursor.fetchone():
        print("Database not initialized. Run init_db.py first.")
        conn.close()
        return
    
    for i in range(1, num_users + 1):
        username = f"testuser{i}"
        password = f"password{i}"
        
        # Check if user already exists
        cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
        existing = cursor.fetchone()
        
        if existing:
            user_id = existing[0]
            print(f"User {username} already exists (ID: {user_id})")
        else:
            # Create new user
            from werkzeug.security import generate_password_hash
            password_hash = generate_password_hash(password)
            
            cursor.execute(
                'INSERT INTO users (username, password_hash) VALUES (?, ?)',
                (username, password_hash)
            )
            conn.commit()
            user_id = cursor.lastrowid
            print(f"Created user {username} (ID: {user_id})")
        
        # Generate and save typing samples
        samples = generate_user_features(user_id, samples_per_user)
        
        for sample in samples:
            cursor.execute(
                'INSERT INTO typing_samples (user_id, features_json) VALUES (?, ?)',
                (user_id, json.dumps(sample))
            )
        
        conn.commit()
        print(f"Generated {len(samples)} typing samples for {username}")
    
    conn.close()
    print(f"\nSuccessfully generated synthetic data for {num_users} users.")

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic keystroke data')
    parser.add_argument('--users', type=int, default=3, help='Number of users to create')
    parser.add_argument('--samples', type=int, default=30, help='Samples per user')
    
    args = parser.parse_args()
    
    print(f"Generating synthetic data for {args.users} users with {args.samples} samples each...")
    generate_sample_users(args.users, args.samples)

if __name__ == '__main__':
    main()