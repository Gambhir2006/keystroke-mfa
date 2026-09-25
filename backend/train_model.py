"""
Model Training Module for Keystroke MFA
This module trains Isolation Forest models for user authentication based on keystroke dynamics.
"""

import sqlite3
import json
import joblib
import numpy as np
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Get the directory of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(SCRIPT_DIR, 'keystroke_mfa.db')
MODEL_DIR = os.path.join(SCRIPT_DIR, 'model')

def get_user_samples(user_id):
    """
    Retrieve all typing samples for a user from the database
    
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

def prepare_training_data(samples):
    """
    Convert feature dictionaries to numpy array for training
    
    Args:
        samples: List of feature dictionaries
        
    Returns:
        numpy array of features
    """
    X = []
    for sample in samples:
        features = [
            sample['mean_hold_time'],
            sample['hold_time_std'],
            sample['mean_flight_time'],
            sample['flight_time_std'],
            sample['total_time'],
            sample['typing_speed'],
            sample['char_count']
        ]
        X.append(features)
    
    return np.array(X)

def train_user_model(user_id, contamination=0.1):
    """
    Train an Isolation Forest model for a specific user
    
    Args:
        user_id: ID of the user
        contamination: Expected proportion of outliers in the dataset
        
    Returns:
        Trained IsolationForest model and scaler
    """
    samples = get_user_samples(user_id)
    
    if len(samples) < 10:
        raise ValueError(f"Insufficient samples for user {user_id}. Need at least 10, got {len(samples)}")
    
    X = prepare_training_data(samples)
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest
    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100
    )
    model.fit(X_scaled)
    
    return model, scaler

def save_user_model(user_id, model, scaler):
    """
    Save the trained model and scaler for a user
    
    Args:
        user_id: ID of the user
        model: Trained IsolationForest model
        scaler: Fitted StandardScaler
    """
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    model_path = os.path.join(MODEL_DIR, f'user_{user_id}_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, f'user_{user_id}_scaler.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"Saved model for user {user_id} to {model_path}")

def load_user_model(user_id):
    """
    Load a trained model and scaler for a user
    
    Args:
        user_id: ID of the user
        
    Returns:
        Tuple of (model, scaler) or (None, None) if not found
    """
    model_path = os.path.join(MODEL_DIR, f'user_{user_id}_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, f'user_{user_id}_scaler.pkl')
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    
    return None, None

def train_all_users():
    """
    Train models for all users in the database who have sufficient samples
    """
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, username FROM users')
    users = cursor.fetchall()
    
    conn.close()
    
    trained_count = 0
    skipped_count = 0
    
    for user_id, username in users:
        try:
            model, scaler = train_user_model(user_id)
            save_user_model(user_id, model, scaler)
            print(f"✓ Trained model for {username} (ID: {user_id})")
            trained_count += 1
        except ValueError as e:
            print(f"✗ Skipped {username} (ID: {user_id}): {e}")
            skipped_count += 1
    
    print(f"\nTraining complete: {trained_count} models trained, {skipped_count} skipped")

def evaluate_model(user_id, test_features):
    """
    Evaluate a sample against a user's trained model
    
    Args:
        user_id: ID of the user
        test_features: Dictionary of features to evaluate
        
    Returns:
        Tuple of (prediction, anomaly_score)
        prediction: 1 for normal, -1 for anomaly
        anomaly_score: Lower values indicate more anomalous
    """
    model, scaler = load_user_model(user_id)
    
    if model is None or scaler is None:
        raise ValueError(f"No model found for user {user_id}")
    
    X = [[
        test_features['mean_hold_time'],
        test_features['hold_time_std'],
        test_features['mean_flight_time'],
        test_features['flight_time_std'],
        test_features['total_time'],
        test_features['typing_speed'],
        test_features['char_count']
    ]]
    
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    anomaly_score = model.score_samples(X_scaled)[0]
    
    return prediction, anomaly_score

if __name__ == '__main__':
    print("Training keystroke dynamics models for all users...")
    train_all_users()