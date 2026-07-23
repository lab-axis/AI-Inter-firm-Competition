@echo off
REM TemporalRAG - Windows One-Click Start Script
REM Starts FastAPI backend (port 8002) and React UI static server (port 3001)

REM Change directory to the folder where this batch file is located
cd /d "%~dp0"

echo.
echo  ============================================
echo   TemporalRAG Web Application
echo   Backend : http://localhost:8002
echo   Frontend: http://localhost:3001
echo  ============================================
echo.

REM Check if .env file exists
if not exist .env (
    echo [WARNING] .env file not found!
    echo Copying .env.example to .env...
    copy .env.example .env
    echo Please edit .env with your Neo4j and vLLM settings before proceeding.
    echo.
)

REM Detect Python executable
set PYTHON_EXE=python
if exist "C:\Users\Admin\miniconda3\envs\ai_Company\python.exe" (
    set PYTHON_EXE=C:\Users\Admin\miniconda3\envs\ai_Company\python.exe
    echo [INFO] Using conda ai_Company environment
)

REM Install backend dependencies
echo [INFO] Installing backend dependencies...
"%PYTHON_EXE%" -m pip install -r requirements.txt -q
echo [INFO] Backend dependencies ready.

REM Install frontend dependencies and build if needed
cd ui
if not exist node_modules (
    echo [INFO] Installing frontend dependencies...
    call npm install --silent
) else (
    echo [INFO] Frontend dependencies already installed.
)

if not exist build (
    echo [INFO] Building React UI...
    call npm run build
) else (
    echo [INFO] React UI build already exists.
)
cd ..
echo [INFO] Frontend assets ready.

REM Start backend
echo [INFO] Starting TemporalRAG API on port 8002...
start cmd /k "title TemporalRAG API && set "PYTHONUTF8=1" && "%PYTHON_EXE%" main.py"

REM Wait for backend to start
echo [INFO] Waiting for backend to initialize (8s)...
timeout /t 8 /nobreak > nul

REM Start frontend with serve (static file server)
echo [INFO] Starting TemporalRAG UI on port 3001...
cd ui
start cmd /k "title TemporalRAG UI && npx serve -s build -l 3001"
cd ..

echo.
echo  ============================================
echo   TemporalRAG is starting up!
echo.
echo   API Docs : http://localhost:8002/docs
echo   Web UI   : http://localhost:3001
echo.
echo   No login required - access directly!
echo  ============================================
echo.
echo  Close the terminal windows to stop the services.
