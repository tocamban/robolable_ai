@echo off
chcp 65001 > nul
title RoboLabel AI - Tat Server
cd /d "%~dp0"

echo =====================================================================
echo       DANG DUNG SERVER ROBOLABEL AI (PORT 5050)...
echo =====================================================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop_server.ps1"

echo.
echo =====================================================================
echo Nhan phim bat ky hoac dong cua so nay de hoan tat...
pause > nul
