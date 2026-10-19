@echo off
rem voice-lab: double-click opens the window (scripts\ui_server.py, a local page in an Edge app window).
rem With arguments, or before the tools env exists, it runs the text launcher (scripts\launch.ps1) instead:
rem   -Menu  the text menu    -Check  -Setup  -Live  -Snapshot  -Restore  -GameDefault ...  (see launch.ps1)
set "PYW=%~dp0envs\tools\Scripts\pythonw.exe"
if "%~1"=="" if exist "%PYW%" (
  start "" "%PYW%" "%~dp0scripts\ui_server.py"
  exit /b 0
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\launch.ps1" %*
if errorlevel 1 pause