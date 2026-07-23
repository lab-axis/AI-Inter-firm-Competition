@echo off
REM TemporalRAG - Frontend UI Only
echo Starting TemporalRAG UI on port 3001...
cd ui
call npm install --silent
set PORT=3001
set REACT_APP_API_URL=http://localhost:8002
npx cross-env PORT=3001 npm start
