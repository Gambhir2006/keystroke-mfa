"""
Database Module - Supports both SQLite (local) and PostgreSQL (Vercel/Neon)
"""

import os
import sqlite3
from typing import Optional
from contextlib import contextmanager

# Try to import PostgreSQL driver, fallback to SQLite
try:
    import psycopg
    from psycopg.rows import dict_row
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

# Database configuration
DATABASE_URL = os.environ.get("DATABASE_URL")
IS_VERCEL = bool(os.environ.get("VERCEL"))

# Local SQLite path (fallback)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get(
    "DATA_DIR",
    "/tmp/keystroke_mfa" if IS_VERCEL else SCRIPT_DIR
)
os.makedirs(DATA_DIR, exist_ok=True)
SQLITE_DB_PATH = os.path.join(DATA_DIR, "keystroke_mfa.db")

# Determine database type
USE_POSTGRES = bool(DATABASE_URL) and POSTGRES_AVAILABLE


@contextmanager
def get_db():
    """
    Context manager for database connections.
    Automatically handles connection pooling and cleanup.
    """
    if USE_POSTGRES:
        with psycopg.connect(DATABASE_URL) as conn:
            conn.row_factory = dict_row
            yield conn
    else:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()


def init_db():
    """
    Initialize database tables.
    Creates tables if they don't exist.
    """
    with get_db() as conn:
        cursor = conn.cursor()

        # --------------------------------------------------------
        # USERS TABLE
        # --------------------------------------------------------
        if USE_POSTGRES:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(255) UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    email VARCHAR(255),
                    email_verified BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        else:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    email TEXT,
                    email_verified INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

        # --------------------------------------------------------
        # TYPING SAMPLES TABLE
        # --------------------------------------------------------
        if USE_POSTGRES:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS typing_samples (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    features_json JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        else:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS typing_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    features_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id)
                    REFERENCES users(id)
                    ON DELETE CASCADE
                )
            """)

        # --------------------------------------------------------
        # OTP TABLE
        # --------------------------------------------------------
        if USE_POSTGRES:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS otp_codes (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    code VARCHAR(10) NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    verified BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        else:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS otp_codes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    code TEXT NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    used INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id)
                    REFERENCES users(id)
                    ON DELETE CASCADE
                )
            """)

        conn.commit()

    db_type = "PostgreSQL" if USE_POSTGRES else "SQLite"
    db_path = DATABASE_URL if USE_POSTGRES else SQLITE_DB_PATH
    print(f"Database '{db_path}' ({db_type}) initialized successfully.")


def get_user_by_username(username: str) -> Optional[dict]:
    """Get user by username."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, username, password_hash, email, email_verified
            FROM users
            WHERE username = %s
            """ if USE_POSTGRES else
            """
            SELECT id, username, password_hash, email, email_verified
            FROM users
            WHERE username = ?
            """,
            (username,)
        )
        return cursor.fetchone()


def get_user_by_id(user_id: int) -> Optional[dict]:
    """Get user by ID."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, username, password_hash, email, email_verified
            FROM users
            WHERE id = %s
            """ if USE_POSTGRES else
            """
            SELECT id, username, password_hash, email, email_verified
            FROM users
            WHERE id = ?
            """,
            (user_id,)
        )
        return cursor.fetchone()


def create_user(username: str, password_hash: str, email: str = None) -> int:
    """Create a new user and return the user ID."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO users (username, password_hash, email)
            VALUES (%s, %s, %s)
            RETURNING id
            """ if USE_POSTGRES else
            """
            INSERT INTO users (username, password_hash, email)
            VALUES (?, ?, ?)
            """,
            (username, password_hash, email)
        )
        
        if USE_POSTGRES:
            user_id = cursor.fetchone()["id"]
        else:
            user_id = cursor.lastrowid
        
        conn.commit()
        return user_id


def get_typing_sample_count(user_id: int) -> int:
    """Get the number of typing samples for a user."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) AS count
            FROM typing_samples
            WHERE user_id = %s
            """ if USE_POSTGRES else
            """
            SELECT COUNT(*) AS count
            FROM typing_samples
            WHERE user_id = ?
            """,
            (user_id,)
        )
        result = cursor.fetchone()
        return result["count"] if result else 0


def get_typing_samples(user_id: int) -> list:
    """Get all typing samples for a user."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT features_json
            FROM typing_samples
            WHERE user_id = %s
            ORDER BY id ASC
            """ if USE_POSTGRES else
            """
            SELECT features_json
            FROM typing_samples
            WHERE user_id = ?
            ORDER BY id ASC
            """,
            (user_id,)
        )
        return cursor.fetchall()


def create_typing_sample(user_id: int, features_json: str) -> int:
    """Create a typing sample and return the sample ID."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO typing_samples (user_id, features_json)
            VALUES (%s, %s)
            RETURNING id
            """ if USE_POSTGRES else
            """
            INSERT INTO typing_samples (user_id, features_json)
            VALUES (?, ?)
            """,
            (user_id, features_json)
        )
        
        if USE_POSTGRES:
            sample_id = cursor.fetchone()["id"]
        else:
            sample_id = cursor.lastrowid
        
        conn.commit()
        return sample_id


def get_last_sample_time(user_id: int) -> Optional[str]:
    """Get the timestamp of the last typing sample for a user."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT created_at
            FROM typing_samples
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """ if USE_POSTGRES else
            """
            SELECT created_at
            FROM typing_samples
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (user_id,)
        )
        result = cursor.fetchone()
        return result["created_at"] if result else None


def store_otp_code(user_id: int, otp_code: str, expires_at):
    """Store an OTP code for a user."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO otp_codes (user_id, code, expires_at)
            VALUES (%s, %s, %s)
            """ if USE_POSTGRES else
            """
            INSERT INTO otp_codes (user_id, code, expires_at)
            VALUES (?, ?, ?)
            """,
            (user_id, otp_code, expires_at)
        )
        conn.commit()


def verify_otp_code(user_id, otp_code):
    """Verify an OTP code for a user."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, expires_at, verified
            FROM otp_codes
            WHERE user_id = %s AND code = %s
            ORDER BY created_at DESC
            LIMIT 1
            """ if USE_POSTGRES else
            """
            SELECT id, expires_at, used
            FROM otp_codes
            WHERE user_id = ? AND code = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (user_id, otp_code)
        )
        record = cursor.fetchone()

        if not record:
            return False, "Invalid OTP code"

        # Check if already used/verified
        is_used = record.get("verified") if USE_POSTGRES else record.get("used")
        if is_used:
            return False, "OTP code already used"

        # Check expiration
        from datetime import datetime
        try:
            expires_at = record["expires_at"]
            if isinstance(expires_at, str):
                expires_at = datetime.strptime(expires_at, "%Y-%m-%d %H:%M:%S")
            
            if datetime.now() > expires_at:
                return False, "OTP code expired"
        except (ValueError, TypeError):
            return False, "Invalid OTP expiry"

        # Mark as used
        cursor.execute(
            """
            UPDATE otp_codes
            SET verified = TRUE
            WHERE id = %s
            """ if USE_POSTGRES else
            """
            UPDATE otp_codes
            SET used = 1
            WHERE id = ?
            """,
            (record["id"],)
        )
        conn.commit()

        return True, "OTP verified successfully"


def update_email_verified(user_id: int):
    """Mark a user's email as verified."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE users
            SET email_verified = TRUE
            WHERE id = %s
            """ if USE_POSTGRES else
            """
            UPDATE users
            SET email_verified = 1
            WHERE id = ?
            """,
            (user_id,)
        )
        conn.commit()
