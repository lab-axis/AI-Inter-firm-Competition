@echo off
setlocal
cd /d "%~dp0"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONUTF8=1"
set "SMOKE_LAUNCHER="
if not "%~1"=="" set "SMOKE_PYTHON=%~1"
if defined SMOKE_PYTHON goto run
if exist "%~dp0.venv\Scripts\python.exe" (
  set "SMOKE_PYTHON=%~dp0.venv\Scripts\python.exe"
) else (
  where py >nul 2>nul
  if errorlevel 1 (
    set "SMOKE_PYTHON=python"
  ) else (
    set "SMOKE_PYTHON=py"
    set "SMOKE_LAUNCHER=-3"
  )
)
:run
set "SMOKE_SELECTION=--all"
if not "%~2"=="" if /I not "%~2"=="all" set "SMOKE_SELECTION=--run %~2"
echo Offline smoke test: no training, API calls or rendered figures.
echo Python: "%SMOKE_PYTHON%"
"%SMOKE_PYTHON%" %SMOKE_LAUNCHER% -B "%~dp0smoke\scripts\smoke_test.py" %SMOKE_SELECTION%
set "SMOKE_EXIT=%ERRORLEVEL%"
echo.
if "%SMOKE_EXIT%"=="0" (
  echo Offline smoke completed. See the printed summary path.
) else (
  echo Smoke failed. Read the error above; no packages were installed.
  echo To select an environment: RUN_SMOKE.bat "C:\path\to\python.exe"
)
if not defined SMOKE_NO_PAUSE pause
exit /b %SMOKE_EXIT%
