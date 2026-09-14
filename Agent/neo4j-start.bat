@echo off
setlocal
if not defined NEO4J_HOME (
  echo Set NEO4J_HOME to your Neo4j installation directory first.
  exit /b 1
)
if not exist "%NEO4J_HOME%\bin\neo4j.bat" (
  echo Neo4j executable not found under NEO4J_HOME.
  exit /b 1
)
call "%NEO4J_HOME%\bin\neo4j.bat" console
