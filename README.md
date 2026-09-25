# Keystroke Dynamic Multifactor Authentication

A comprehensive multifactor authentication system that combines traditional password authentication with behavioral biometrics based on keystroke dynamics, now with separated frontend and backend architecture.

## Project Structure

```
keystroke-mfa/
│
├── backend/                    # Python Flask backend
│   ├── app.py                  # Main Flask application
│   ├── init_db.py              # Database initialization
│   ├── collect_data.py         # Keystroke data collection utilities
│   ├── generate_sample_data.py # Synthetic data generator
│   ├── train_model.py          # ML model training module
│   ├── requirements.txt        # Python dependencies
│   ├── .env.example            # Environment variables template
│   ├── data/                   # Data directory
│   └── model/                  # Trained models directory
│
└── frontend/                   # HTML/CSS/JS frontend
    ├── index.html             # Landing page
    ├── register.html          # User registration
    ├── enroll.html            # Keystroke enrollment
    ├── login.html             # Login page
    ├── dashboard.html         # User dashboard
    ├── base.html              # Base template
    ├── css/
    │   └── style.css          # Application styles
    └── js/
        └── keystroke.js        # Keystroke capture JavaScript
```

## Features

- **Password Authentication**: Secure password hashing using Werkzeug
- **Keystroke Dynamics**: Captures and analyzes typing patterns including:
  - Key hold times (duration between keydown and keyup)
  - Key flight times (duration between consecutive keys)
  - Typing speed and rhythm
- **Machine Learning**: Uses Isolation Forest for anomaly detection
- **User Enrollment**: Collects 15+ samples to build user behavioral profiles
- **Real-time Authentication**: Verifies both password and typing pattern during login
- **Email/OTP Verification**: Optional two-factor authentication with email verification
- **Dashboard**: User interface for enrollment status and authentication history
- **Separated Architecture**: Clean separation between frontend and backend

## Installation

1. **Navigate to the project directory**
   ```bash
   cd keystroke-mfa
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate virtual environment**
   - Windows: `.venv\Scripts\activate`
   - Mac/Linux: `source .venv/bin/activate`

4. **Install dependencies**
   ```bash
   pip install -r backend/requirements.txt
   ```

5. **Initialize database**
   ```bash
   cd backend
   python init_db.py
   cd ..
   ```

6. **Optional: Generate synthetic data for testing**
   ```bash
   cd backend
   python generate_sample_data.py --users 3 --samples 30
   cd ..
   ```

7. **Train models**
   ```bash
   cd backend
   python train_model.py
   cd ..
   ```

8. **Run the application**
   ```bash
   cd backend
   python app.py
   ```

9. **Open in browser**
   ```
   http://127.0.0.1:5000
   ```

## Authentication Flow

### Registration
1. User provides username and password
2. Password is hashed and stored in SQLite database
3. User is redirected to enrollment page

### Enrollment
1. User types their password 15 times
2. JavaScript captures keydown/keyup timing
3. Features are extracted:
   - Mean hold time
   - Hold-time deviation
   - Mean flight time
   - Flight-time deviation
   - Total typing time
   - Typing speed
   - Character count
4. Isolation Forest model is trained on user's samples
5. User's behavioral profile is created

### Login
1. User provides username and password
2. System verifies password hash
3. If email is verified, OTP code is required
4. Keystroke dynamics are captured during password entry
5. Features are extracted and compared to user's profile
6. Authentication succeeds only if password, OTP (if required), and typing pattern match

## Security Features

- **Password Hashing**: Uses Werkzeug's secure password hashing
- **No Plaintext Storage**: Raw keystrokes and passwords are never stored
- **Behavioral Biometrics**: Typing patterns are difficult to replicate
- **Anomaly Detection**: ML model identifies unusual typing patterns
- **Session Management**: Secure session handling with Flask
- **Email Verification**: Optional 2FA with OTP codes

## Database Schema

### Users Table
- `id`: Primary key
- `username`: Unique username
- `password_hash`: Hashed password
- `email`: User email (optional)
- `email_verified`: Email verification status
- `created_at`: Account creation timestamp

### Typing Samples Table
- `id`: Primary key
- `user_id`: Foreign key to users
- `features_json`: JSON string of extracted features
- `created_at`: Sample collection timestamp

### OTP Codes Table
- `id`: Primary key
- `user_id`: Foreign key to users
- `code`: OTP code
- `expires_at`: Expiration timestamp
- `used`: Whether code has been used
- `created_at`: Code generation timestamp

## Configuration

Copy `backend/.env.example` to `backend/.env` and configure:

```env
SECRET_KEY=your-secret-key-here
FLASK_ENV=development
FLASK_DEBUG=1

# Email Configuration (Optional - for OTP verification)
EMAIL_ENABLED=false
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
FROM_EMAIL=your-email@gmail.com
```

## Email/OTP Verification

The system includes optional email verification for enhanced security:

### Setup Email Verification

1. **Configure environment variables** in `backend/.env`:
   ```env
   EMAIL_ENABLED=true
   SMTP_SERVER=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USERNAME=your-email@gmail.com
   SMTP_PASSWORD=your-app-password
   FROM_EMAIL=your-email@gmail.com
   ```

2. **For Gmail users**: Generate an app password:
   - Go to Google Account settings
   - Enable 2-factor authentication
   - Generate an app password for "Mail"
   - Use the app password in `SMTP_PASSWORD`

3. **Restart the application** after configuring email settings

### How Email Verification Works

1. **Email Verification**: Users can verify their email from the dashboard
2. **OTP Login**: When logging in, users with verified emails must enter a 6-digit OTP code
3. **Code Generation**: OTP codes are generated using `pyotp` and sent via email
4. **Code Expiration**: OTP codes expire after 10 minutes for security
5. **Single Use**: Each OTP code can only be used once

### Testing Without Email Server

For testing purposes, you can run the system without email configuration:
- Set `EMAIL_ENABLED=false` in `backend/.env`
- OTP codes will be displayed in the response for testing
- This is useful for development and testing environments

## Usage

### Registering a New User
1. Navigate to the registration page
2. Enter username and password
3. Complete the enrollment process by typing password 15 times

### Logging In
1. Enter username and password
2. Type naturally - the system analyzes your typing pattern
3. If email is verified, enter the OTP code sent to your email
4. Access granted if all authentication factors match

### Testing with Synthetic Data
Generate test users and data:
```bash
cd backend
python generate_sample_data.py --users 5 --samples 30
python train_model.py
```

## Machine Learning Model

The system uses **Isolation Forest** for anomaly detection:
- Contamination parameter: 0.1 (10% expected outliers)
- Features: 7 keystroke timing metrics
- Training: Requires 15+ samples per user
- Prediction: Returns 1 (normal) or -1 (anomaly)

## Development

### Adding New Features
1. Modify feature extraction in `backend/collect_data.py`
2. Update model training in `backend/train_model.py`
3. Adjust frontend capture in `frontend/js/keystroke.js`
4. Update UI in relevant HTML files in `frontend/`

### Extending Authentication
- Add email verification: Already implemented with OTP sending
- Add 2FA: Integrated with email verification
- Add device fingerprinting: Include additional biometrics

## Troubleshooting

**Database locked error**: Ensure only one instance of the app is running
**Model not found**: Run `backend/train_model.py` after adding samples
**Enrollment fails**: Check browser console for JavaScript errors
**Path issues**: Ensure you're running commands from the correct directory

## Architecture Benefits

The separated frontend/backend architecture provides:
- **Clear separation of concerns**: UI vs business logic
- **Easier maintenance**: Independent updates to frontend and backend
- **Better scalability**: Can deploy frontend and backend separately
- **Team collaboration**: Frontend and backend developers can work independently
- **Technology flexibility**: Can change frontend technology without affecting backend

## License

This project is for educational and demonstration purposes.