@echo off
title RoboLabel AI Studio - Port 5050
cd /d "%~dp0"

echo =====================================================================
echo    ROBOLABEL AI - CONG CU TU DONG GAN NHAN DAU TAY (YOLO / SAM 2)
echo =====================================================================
echo.

set PYTHON_EXE=venv\Scripts\python.exe
if not exist "%PYTHON_EXE%" (
    echo [!] LOI: Khong tim thay moi truong Python ao tai: %PYTHON_EXE%
    echo [!] Vui long nhap dup file 'setup.bat' de cai dat truoc.
    echo.
    pause
    exit /b
)

:: Buoc 1: Giai phong port 5050 neu co tien trinh cu
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $c = Get-NetTCPConnection -LocalPort 5050 -ErrorAction SilentlyContinue; if ($c) { Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue; Write-Host '[*] Da giai phong port 5050 bi ket.' } } catch {}"

:: Buoc 2: Tu dong mo trinh duyet sau 2 giay o che do background
start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 2; Start-Process 'http://127.0.0.1:5050'"

echo [*] Dang khoi dong Web Server tai: http://127.0.0.1:5050 ...
echo [*] Trinh duyet se tu dong mo sau 2 giay.
echo.
echo ---------------------------------------------------------------------
echo [LUU Y QUAN TRONG]:
echo  - CUA SO CMD NAY SE LUON HIEN de hien thi toan bo log AI va Server.
echo  - DE TAT SERVER:
echo      + Cach 1: Dong truc tiep cua so CMD nay (nhan nut X).
echo      + Cach 2: Nhan to hop phim Ctrl + C trong cua so nay.
echo      + Cach 3: Nhap dup vao file 'stop_server.bat'.
echo ---------------------------------------------------------------------
echo.

:: Buoc 3: Chay server truc tiep o foreground (cmd luon luon mo)
"%PYTHON_EXE%" app.py

echo.
echo =====================================================================
echo [*] Server RoboLabel da dung.
echo =====================================================================
pause
