@echo off
REM Launch the dev server. Activate your venv first if you use one.
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
