@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo Starting server...
echo Please leave this window open while using the application.
uvicorn main:app --reload --host 0.0.0.0 --port 8000

pause
