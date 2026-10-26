@echo off
title Autonomous Driving Threat Lab Web Simulator
cd /d "%~dp0"
echo ====================================================================
echo  Launching Interactive Autonomous Driving Threat Simulator...
echo  Opening browser at http://127.0.0.1:8501
echo ====================================================================
python web_app.py
pause
