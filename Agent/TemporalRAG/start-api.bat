@echo off
REM TemporalRAG - Backend API Only
echo Starting TemporalRAG API on port 8002...
if not exist .env copy .env.example .env

if not "%CONDA_DEFAULT_ENV%"=="" goto RUN
if not "%VIRTUAL_ENV%"=="" goto RUN
if exist .venv call .venv\Scripts\activate.bat

:RUN
pip install -r requirements.txt -q
set PORT=8002
python main.py
