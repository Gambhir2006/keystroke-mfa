@echo off
echo Starting Keystroke MFA Application...
echo.

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Installing dependencies...
pip install -q -r backend\requirements.txt

cd backend

if not exist "keystroke_mfa.db" (
    echo Initializing database...
    python init_db.py
)

if not exist "model" mkdir model

echo.
echo Starting Flask server...
echo http://127.0.0.1:5000
echo.
python app.py

cd ..
deactivate