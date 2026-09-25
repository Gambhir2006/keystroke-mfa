from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session,
    redirect,
    url_for
)

from werkzeug.security import generate_password_hash, check_password_hash

import json
import os
import numpy as np
from datetime import datetime, timedelta
import secrets
import smtplib

from dotenv import load_dotenv

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from sklearn.ensemble import IsolationForest

# Import persistence modules
from db import (
    init_db,
    get_user_by_username,
    get_user_by_id,
    create_user,
    get_typing_sample_count,
    get_typing_samples,
    create_typing_sample,
    get_last_sample_time,
    store_otp_code,
    verify_otp_code,
    update_email_verified
)

from model_storage import (
    save_model,
    load_model,
    delete_model,
    model_exists,
    get_storage_info
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load environment variables from .env if present
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

FRONTEND_DIR = os.path.join(SCRIPT_DIR, "frontend")


# ============================================================
# FLASK CONFIGURATION
# ============================================================

app = Flask(
    __name__,
    template_folder=FRONTEND_DIR,
    static_folder=FRONTEND_DIR,
    static_url_path=""
)

app.secret_key = os.environ.get(
    "SECRET_KEY",
    secrets.token_hex(32)
)


# ============================================================
# EMAIL / OTP CONFIGURATION
# ============================================================

EMAIL_ENABLED = (
    os.environ.get("EMAIL_ENABLED", "false").lower() == "true"
)

SMTP_SERVER = os.environ.get(
    "SMTP_SERVER",
    "smtp.gmail.com"
)

SMTP_PORT = int(
    os.environ.get("SMTP_PORT", "587")
)

SMTP_USERNAME = os.environ.get(
    "SMTP_USERNAME",
    ""
)

SMTP_PASSWORD = os.environ.get(
    "SMTP_PASSWORD",
    ""
)

FROM_EMAIL = os.environ.get(
    "FROM_EMAIL",
    SMTP_USERNAME
)


# ============================================================
# CONSTANTS
# ============================================================

REQUIRED_SAMPLES = 15

FEATURE_NAMES = [
    "mean_hold_time",
    "hold_time_std",
    "mean_flight_time",
    "flight_time_std",
    "total_time",
    "typing_speed",
    "char_count"
]


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(keystroke_data):

    """
    Supports the project's keyEvents format:

    {
        "keyEvents": [
            {
                "type": "hold",
                "duration": 120
            },
            {
                "type": "flight",
                "duration": 80
            }
        ],
        "totalTime": 1200,
        "charCount": 10
    }

    It also supports raw browser events:

    {
        "events": [
            {
                "key": "a",
                "down": 100,
                "up": 200
            }
        ]
    }
    """

    if not isinstance(keystroke_data, dict):
        return None

    # --------------------------------------------------------
    # FORMAT 1: keyEvents
    # --------------------------------------------------------

    key_events = keystroke_data.get(
        "keyEvents",
        []
    )

    if key_events:

        hold_times = []
        flight_times = []

        for event in key_events:

            if not isinstance(event, dict):
                continue

            event_type = event.get("type")

            try:
                duration = float(
                    event.get("duration", 0)
                )
            except (
                TypeError,
                ValueError
            ):
                continue

            if duration < 0:
                continue

            if event_type == "hold":

                hold_times.append(duration)

            elif event_type == "flight":

                flight_times.append(duration)

        if not hold_times:
            return None

        total_time = float(
            keystroke_data.get(
                "totalTime",
                0
            )
        )

        char_count = int(
            keystroke_data.get(
                "charCount",
                len(hold_times)
            )
        )

        if total_time <= 0:
            total_time = (
                sum(hold_times)
                + sum(flight_times)
            )

        if total_time <= 0:
            total_time = 1

        # If totalTime is milliseconds,
        # convert to seconds for typing speed.
        typing_speed = (
            char_count
            / (total_time / 1000.0)
        )

        features = {
            "mean_hold_time": float(
                np.mean(hold_times)
            ),

            "hold_time_std": float(
                np.std(hold_times)
                if len(hold_times) > 1
                else 0
            ),

            "mean_flight_time": float(
                np.mean(flight_times)
                if flight_times
                else 0
            ),

            "flight_time_std": float(
                np.std(flight_times)
                if len(flight_times) > 1
                else 0
            ),

            "total_time": float(
                total_time
            ),

            "typing_speed": float(
                typing_speed
            ),

            "char_count": int(
                char_count
            )
        }

        return features

    # --------------------------------------------------------
    # FORMAT 2: raw events
    # --------------------------------------------------------

    events = keystroke_data.get(
        "events",
        []
    )

    if not events or len(events) < 2:
        return None

    holds = []
    flights = []

    clean_events = []

    for event in events:

        if not isinstance(event, dict):
            continue

        try:
            down = float(event["down"])
            up = float(event["up"])
        except (
            KeyError,
            TypeError,
            ValueError
        ):
            continue

        if up <= down:
            continue

        clean_events.append(
            {
                "down": down,
                "up": up
            }
        )

    if len(clean_events) < 2:
        return None

    clean_events.sort(
        key=lambda x: x["down"]
    )

    # Hold times
    for event in clean_events:

        hold = (
            event["up"]
            - event["down"]
        )

        holds.append(
            max(0, hold)
        )

    # Flight times
    for i in range(
        1,
        len(clean_events)
    ):

        flight = (
            clean_events[i]["down"]
            - clean_events[i - 1]["up"]
        )

        flights.append(
            max(0, flight)
        )

    total_time = (
        clean_events[-1]["up"]
        - clean_events[0]["down"]
    )

    total_time = max(
        total_time,
        1
    )

    char_count = len(
        clean_events
    )

    typing_speed = (
        char_count
        / (total_time / 1000.0)
    )

    return {
        "mean_hold_time": float(
            np.mean(holds)
        ),

        "hold_time_std": float(
            np.std(holds)
        ),

        "mean_flight_time": float(
            np.mean(flights)
            if flights
            else 0
        ),

        "flight_time_std": float(
            np.std(flights)
            if len(flights) > 1
            else 0
        ),

        "total_time": float(
            total_time
        ),

        "typing_speed": float(
            typing_speed
        ),

        "char_count": int(
            char_count
        )
    }


# ============================================================
# CONVERT FEATURES TO ML ARRAY
# ============================================================

def features_to_vector(features):

    return [
        features["mean_hold_time"],
        features["hold_time_std"],
        features["mean_flight_time"],
        features["flight_time_std"],
        features["total_time"],
        features["typing_speed"],
        features["char_count"]
    ]


# ============================================================
# LOAD USER MODEL
# ============================================================

def load_user_model(user_id):

    return load_model(user_id)


# ============================================================
# TRAIN USER MODEL
# ============================================================

def train_user_model(user_id):

    rows = get_typing_samples(user_id)

    if len(rows) < REQUIRED_SAMPLES:

        return None

    X = []

    for row in rows:

        features = json.loads(
            row["features_json"]
        )

        X.append(
            features_to_vector(
                features
            )
        )

    X = np.asarray(
        X,
        dtype=float
    )

    # Train Isolation Forest
    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=42
    )

    model.fit(X)

    # ---------------------------------------------------------
    # PERSONALIZED THRESHOLD
    # ---------------------------------------------------------
    training_scores = model.score_samples(X)

    # Allow a small amount of natural variation.
    # The lowest 10% of the user's own training scores
    # becomes the acceptance boundary.
    threshold = float(
        np.percentile(training_scores, 10)
    )

    model_data = {
        "model": model,
        "threshold": threshold
    }

    # Save using model_storage module
    save_model(user_id, model_data)

    return model_data


# ============================================================
# OTP FUNCTIONS
# ============================================================

def generate_otp_code():

    return str(
        secrets.randbelow(
            1000000
        )
    ).zfill(6)


def send_otp_email(
    email,
    otp_code
):

    if not EMAIL_ENABLED:

        return (
            False,
            "Email sending is disabled"
        )

    if not SMTP_USERNAME or not SMTP_PASSWORD:

        return (
            False,
            "SMTP credentials are not configured"
        )

    try:

        msg = MIMEMultipart()

        msg["From"] = FROM_EMAIL
        msg["To"] = email
        msg["Subject"] = (
            "Keystroke MFA Verification Code"
        )

        body = f"""
Your Keystroke MFA verification code is:

{otp_code}

This code will expire in 10 minutes.

If you did not request this code,
please ignore this email.
"""

        msg.attach(
            MIMEText(
                body,
                "plain"
            )
        )

        server = smtplib.SMTP(
            SMTP_SERVER,
            SMTP_PORT
        )

        server.starttls()

        server.login(
            SMTP_USERNAME,
            SMTP_PASSWORD
        )

        server.send_message(
            msg
        )

        server.quit()

        return (
            True,
            "Verification email sent"
        )

    except Exception as e:

        return (
            False,
            str(e)
        )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/api/health")
def health():
    storage_info = get_storage_info()
    return jsonify({
        "success": True,
        "service": "Keystroke MFA",
        "environment": "vercel" if os.environ.get("VERCEL") else "local",
        "storage": storage_info
    })


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    if "user_id" in session:

        return redirect(
            url_for("dashboard")
        )

    return render_template(
        "index.html"
    )


# ============================================================
# REGISTER
# ============================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if request.method == "GET":

        return render_template(
            "register.html"
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # Accept BOTH JSON and normal HTML form data.
    # This fixes the "Username and password required" issue.
    # --------------------------------------------------------

    if request.is_json:

        data = (
            request.get_json(
                silent=True
            )
            or {}
        )

        username = data.get(
            "username",
            ""
        ).strip()

        password = data.get(
            "password",
            ""
        )

        email = data.get(
            "email",
            ""
        ).strip()

    else:

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        email = request.form.get(
            "email",
            ""
        ).strip()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not username or not password:

        return jsonify({
            "error":
            "Username and password required"
        }), 400

    if len(username) < 3:

        return jsonify({
            "error":
            "Username must contain at least 3 characters"
        }), 400

    if len(password) < 8:

        return jsonify({
            "error":
            "Password must contain at least 8 characters"
        }), 400

    # --------------------------------------------------------
    # CREATE USER
    # --------------------------------------------------------

    password_hash = generate_password_hash(
        password
    )

    try:

        user_id = create_user(username, password_hash, email)

    except Exception:

        return jsonify({
            "error":
            "Username already exists"
        }), 400

    # --------------------------------------------------------
    # CREATE SESSION
    # --------------------------------------------------------

    session["user_id"] = user_id
    session["username"] = username

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return jsonify({
        "success": True,
        "user_id": user_id,
        "username": username,
        "message":
            "Account created successfully",
        "redirect":
            url_for("enroll")
    })


# ============================================================
# ENROLLMENT
# ============================================================

@app.route(
    "/enroll",
    methods=["GET", "POST"]
)
def enroll():

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )

    user_id = session["user_id"]

    # --------------------------------------------------------
    # GET
    # --------------------------------------------------------

    if request.method == "GET":

        return render_template(
            "enroll.html"
        )

    # --------------------------------------------------------
    # POST
    # --------------------------------------------------------

    data = (
        request.get_json(
            silent=True
        )
        or {}
    )

    password = data.get(
        "password"
    )

    if not password:

        return jsonify({
            "error":
            "Password required"
        }), 400

    # --------------------------------------------------------
    # VERIFY PASSWORD
    # --------------------------------------------------------

    user = get_user_by_id(user_id)

    if not user:

        return jsonify({
            "error":
            "User not found"
        }), 404

    if not check_password_hash(
        user["password_hash"],
        password
    ):

        return jsonify({
            "error":
            "Invalid password for enrollment"
        }), 400

    # --------------------------------------------------------
    # EXTRACT FEATURES
    # --------------------------------------------------------

    features = extract_features(
        data
    )

    if not features:

        return jsonify({
            "error":
            "Invalid keystroke data"
        }), 400

    # --------------------------------------------------------
    # SAVE SAMPLE
    # --------------------------------------------------------

    create_typing_sample(user_id, json.dumps(features))

    # Get sample count
    sample_count = get_typing_sample_count(user_id)

    # --------------------------------------------------------
    # TRAIN MODEL AFTER 15 SAMPLES
    # --------------------------------------------------------

    enrollment_complete = (
        sample_count
        >= REQUIRED_SAMPLES
    )

    if enrollment_complete:

        try:

            train_user_model(
                user_id
            )

        except Exception as e:

            return jsonify({
                "error":
                    "Model training failed",
                "details":
                    str(e)
            }), 500

    return jsonify({
        "success": True,
        "samples_collected":
            sample_count,
        "required_samples":
            REQUIRED_SAMPLES,
        "enrollment_complete":
            enrollment_complete
    })


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
@app.route(
    "/api/login",
    methods=["POST"]
)
def login():

    if request.method == "GET":

        return render_template(
            "login.html"
        )

    # --------------------------------------------------------
    # Accept JSON
    # --------------------------------------------------------

    data = (
        request.get_json(
            silent=True
        )
        or {}
    )

    username = data.get(
        "username",
        ""
    ).strip()

    password = data.get(
        "password",
        ""
    )

    # Accept both the original keyEvents payload and the newer
    # raw browser events payload used by the standalone frontend.
    keystroke_data = data.get("keystrokeData")

    if not keystroke_data and data.get("events"):
        keystroke_data = {
            "events": data.get("events", [])
        }

    otp_code = data.get(
        "otpCode"
    )

    # --------------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------------

    if not username or not password:

        return jsonify({
            "error":
            "Username and password required"
        }), 400

    # --------------------------------------------------------
    # FIND USER
    # --------------------------------------------------------

    user = get_user_by_username(username)

    if not user:

        return jsonify({
            "error":
            "Invalid credentials"
        }), 401

    # --------------------------------------------------------
    # PASSWORD FACTOR
    # --------------------------------------------------------

    if not check_password_hash(
        user["password_hash"],
        password
    ):

        return jsonify({
            "error":
            "Invalid credentials"
        }), 401

    user_id = user["id"]

    # --------------------------------------------------------
    # OTP FACTOR IF EMAIL VERIFIED
    # --------------------------------------------------------

    if (
        user["email"]
        and user["email_verified"]
    ):

        # ----------------------------------------------------
        # NO OTP PROVIDED: GENERATE AND SEND ONE
        # ----------------------------------------------------

        if not otp_code:

            generated_otp = generate_otp_code()

            store_otp_code(
                user_id,
                generated_otp
            )

            if EMAIL_ENABLED:

                success, message = send_otp_email(
                    user["email"],
                    generated_otp
                )

                if not success:

                    return jsonify({
                        "error": message
                    }), 500

                return jsonify({
                    "require_otp": True,
                    "message":
                        "OTP sent to your email."
                })

            # Local development mode. The OTP is returned so
            # the project can be tested without SMTP.
            print(
                f"[LOGIN OTP] User={username} OTP={generated_otp}"
            )

            return jsonify({
                "require_otp": True,
                "message":
                    "OTP generated. Use the development OTP.",
                "code": generated_otp
            })

        # ----------------------------------------------------
        # OTP PROVIDED: VERIFY IT
        # ----------------------------------------------------

        success, message = verify_otp_code(
            user_id,
            str(otp_code).strip()
        )

        if not success:

            return jsonify({
                "error": message
            }), 400

    # --------------------------------------------------------
    # CHECK ENROLLMENT
    # --------------------------------------------------------

    sample_count = get_typing_sample_count(user_id)

    if sample_count < REQUIRED_SAMPLES:

        return jsonify({
            "error":
                "User not fully enrolled. "
                "Please complete enrollment.",
            "redirect":
                url_for("enroll")
        }), 400

    # --------------------------------------------------------
    # CHECK KEYSTROKE DATA
    # --------------------------------------------------------

    if not keystroke_data:

        return jsonify({
            "error":
                "Keystroke data required"
        }), 400

    features = extract_features(
        keystroke_data
    )

    if not features:

        return jsonify({
            "error":
                "Invalid keystroke data"
        }), 400

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model = load_user_model(
        user_id
    )

    if model is None:

        try:

            model = train_user_model(
                user_id
            )

        except Exception as e:

            return jsonify({
                "error":
                    "Could not load/train model",
                "details":
                    str(e)
            }), 500

    if model is None:

        return jsonify({
            "error":
                "Keystroke model not available"
        }), 500

    # --------------------------------------------------------
    # ML PREDICTION
    # --------------------------------------------------------

    X = np.asarray(
        [
            features_to_vector(
                features
            )
        ],
        dtype=float
    )

    try:

        # Support the new personalized model format
        if isinstance(model, dict):
            isolation_model = model["model"]
            threshold = float(model["threshold"])
        else:
            # Backward compatibility with old .pkl files
            isolation_model = model
            threshold = None

        anomaly_score = float(
            isolation_model.score_samples(X)[0]
        )

        if threshold is not None:
            prediction = 1 if anomaly_score >= threshold else -1
        else:
            prediction = int(
                isolation_model.predict(X)[0]
            )

    except Exception as e:

        return jsonify({
            "error":
                "Keystroke model prediction failed",
            "details":
                str(e)
        }), 500

    # --------------------------------------------------------
    # AUTHENTICATION RESULT
    # --------------------------------------------------------

    if prediction == 1:

        session["user_id"] = user_id
        session["username"] = (
            user["username"]
        )

        return jsonify({
            "success": True,
            "anomaly_score":
                anomaly_score,
            "message":
                "Authentication successful",
            "redirect":
                url_for("dashboard")
        })

    return jsonify({
        "error":
            "Typing pattern does not match",
        "anomaly_score":
            anomaly_score
    }), 401


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )

    user_id = session[
        "user_id"
    ]

    sample_count = get_typing_sample_count(user_id)

    last_sample_time = get_last_sample_time(user_id)

    user_info = get_user_by_id(user_id)

    return render_template(
        "dashboard.html",
        username=session[
            "username"
        ],
        sample_count=sample_count,
        required_samples=REQUIRED_SAMPLES,
        last_sample=last_sample_time,
        enrolled=(
            sample_count
            >= REQUIRED_SAMPLES
        ),
        email=(
            user_info["email"]
            if user_info
            else None
        ),
        email_verified=(
            bool(
                user_info[
                    "email_verified"
                ]
            )
            if user_info
            else False
        )
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("index")
    )


# ============================================================
# CHECK ENROLLMENT
# ============================================================

@app.route(
    "/api/check_enrollment",
    methods=["GET"]
)
def check_enrollment():

    if "user_id" not in session:

        return jsonify({
            "enrolled": False,
            "sample_count": 0
        })

    user_id = session[
        "user_id"
    ]

    sample_count = get_typing_sample_count(user_id)

    return jsonify({
        "enrolled":
            sample_count
            >= REQUIRED_SAMPLES,
        "sample_count":
            sample_count,
        "required_samples":
            REQUIRED_SAMPLES
    })


# ============================================================
# SEND EMAIL VERIFICATION
# ============================================================

@app.route(
    "/send-verification",
    methods=["POST"]
)
def send_verification():

    if "user_id" not in session:

        return jsonify({
            "error":
                "Not logged in"
        }), 401

    user_id = session[
        "user_id"
    ]

    user = get_user_by_id(user_id)

    if not user:

        return jsonify({
            "error":
                "User not found"
        }), 404

    if not user["email"]:

        return jsonify({
            "error":
                "No email address on file"
        }), 400

    if user["email_verified"]:

        return jsonify({
            "message":
                "Email already verified"
        })

    otp_code = generate_otp_code()

    store_otp_code(
        user_id,
        otp_code
    )

    # --------------------------------------------------------
    # EMAIL ENABLED
    # --------------------------------------------------------

    if EMAIL_ENABLED:

        success, message = (
            send_otp_email(
                user["email"],
                otp_code
            )
        )

        if success:

            return jsonify({
                "success": True,
                "message":
                    "Verification email sent"
            })

        return jsonify({
            "error": message
        }), 500

    # --------------------------------------------------------
    # DEVELOPMENT MODE
    # --------------------------------------------------------

    return jsonify({
        "success": True,
        "message":
            "Verification code generated",
        "code":
            otp_code,
        "note":
            "Email sending is disabled. "
            "Use this code for local testing."
    })


# ============================================================
# VERIFY EMAIL
# ============================================================

@app.route(
    "/verify-email",
    methods=["POST"]
)
def verify_email():

    if "user_id" not in session:

        return jsonify({
            "error":
                "Not logged in"
        }), 401

    data = (
        request.get_json(
            silent=True
        )
        or {}
    )

    otp_code = data.get(
        "otp_code",
        ""
    ).strip()

    if not otp_code:

        return jsonify({
            "error":
                "OTP code required"
        }), 400

    success, message = (
        verify_otp_code(
            session["user_id"],
            otp_code
        )
    )

    if not success:

        return jsonify({
            "error": message
        }), 400

    update_email_verified(session["user_id"])

    return jsonify({
        "success": True,
        "message":
            "Email verified successfully"
    })


# ============================================================
# REQUIRE OTP
# ============================================================

@app.route(
    "/require-otp",
    methods=["POST"]
)
def require_otp():

    data = (
        request.get_json(
            silent=True
        )
        or {}
    )

    username = data.get(
        "username",
        ""
    ).strip()

    if not username:

        return jsonify({
            "error":
                "Username required"
        }), 400

    user = get_user_by_username(username)

    if not user:

        return jsonify({
            "error":
                "User not found"
        }), 404

    if (
        not user["email"]
        or not user["email_verified"]
    ):

        return jsonify({
            "require_otp": False,
            "message":
                "OTP not required - "
                "email not verified"
        })

    otp_code = generate_otp_code()

    store_otp_code(
        user["id"],
        otp_code
    )

    if EMAIL_ENABLED:

        success, message = (
            send_otp_email(
                user["email"],
                otp_code
            )
        )

        if success:

            return jsonify({
                "require_otp": True,
                "message":
                    "OTP sent to email"
            })

        return jsonify({
            "error": message
        }), 500

    # Local testing
    return jsonify({
        "require_otp": True,
        "message":
            "OTP generated",
        "code":
            otp_code,
        "note":
            "Email sending is disabled"
    })


# ============================================================
# APPLICATION START
# ============================================================

# Initialize database at module level for Vercel compatibility
init_db()

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )