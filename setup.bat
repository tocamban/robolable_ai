@echo off
chcp 65001 > nul
title RoboLabel AI - Cai dat moi truong tu dong
cd /d "%~dp0"

echo =========================================================
echo    ROBOLABEL AI - CAI DAT MOI TRUONG VA KICH HOAT GPU
echo =========================================================
echo.

:: 1. Kiem tra Python
where python >nul 2>nul
if errorlevel 1 goto :no_python

echo [*] Da tim thay Python:
python --version
echo.

:: 2. Khoi tao moi truong ao (venv)
if exist "venv\Scripts\python.exe" goto :venv_exists
echo [*] Dang khoi tao moi truong ao (venv)...
python -m venv venv
if errorlevel 1 goto :venv_fail
echo [+] Da tao venv thanh cong.
goto :venv_ready

:venv_exists
echo [*] Thu muc moi truong ao venv da ton tai.

:venv_ready
echo.

:: 3. Kiem tra GPU va Cai dat PyTorch
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $c = Get-NetTCPConnection -LocalPort 5050 -ErrorAction SilentlyContinue; if ($c) { Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue; Write-Host '[*] Da tam dung server port 5050 de cap nhat thu vien an toan.' } } catch {}"
echo [*] Dang cau hinh va kiem tra GPU NVIDIA...
where nvidia-smi >nul 2>nul
if errorlevel 1 goto :install_cpu_fresh

echo [*] Phat hien may co GPU NVIDIA! Dang kiem tra PyTorch CUDA trong venv...
venv\Scripts\python.exe -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>nul
if not errorlevel 1 goto :cuda_ready

echo [*] Hien tai venv chua co PyTorch CUDA. Dang kiem tra goi CUDA cu126 (tuong thich Python 3.10 - 3.14)...
venv\Scripts\pip.exe install --dry-run --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu126 >nul 2>nul
if errorlevel 1 goto :try_cu124

echo [*] Dang go bo PyTorch CPU va tai PyTorch CUDA cu126 (~2.45 GB, khong luu cache o C:)...
venv\Scripts\pip.exe uninstall -y torch torchvision >nul 2>nul
venv\Scripts\pip.exe install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126
if not errorlevel 1 goto :after_torch
echo [!] Tai cu126 bi gian doan, thu khoi phuc ban PyTorch tieu chuan...
goto :install_cpu_fresh

:try_cu124
echo [*] Thu du phong PyTorch CUDA cu124...
venv\Scripts\pip.exe install --dry-run --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu124 >nul 2>nul
if errorlevel 1 goto :install_cpu_fallback
venv\Scripts\pip.exe uninstall -y torch torchvision >nul 2>nul
venv\Scripts\pip.exe install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124
if not errorlevel 1 goto :after_torch
goto :install_cpu_fresh

:cuda_ready
echo [+] PyTorch CUDA da duoc cai dat san va san sang!
goto :after_torch

:install_cpu_fallback
echo [!] Khong tai duoc ban CUDA phu hop, kiem tra ban PyTorch CPU...

:install_cpu_fresh
venv\Scripts\python.exe -c "import torch" >nul 2>nul
if not errorlevel 1 goto :after_torch
echo [*] Dang cai dat PyTorch ban CPU...
venv\Scripts\pip.exe install --no-cache-dir torch torchvision
goto :after_torch

:after_torch
echo.

:: 4. Cai dat cac thu vien bo tro
echo [*] Dang kiem tra va cai dat cac thu vien tu requirements.txt...
venv\Scripts\pip.exe install -r requirements.txt
echo.

:: 5. Kiem tra va tai tu dong model AI (SAM 2.1 Tiny & CLIP) neu chua co
if not exist "weights\clip" mkdir "weights\clip"
if exist "weights\sam2.1_t.pt" goto :check_clip_weight
echo [*] Dang tai model SAM 2.1 Tiny (~75 MB) vao thu muc weights\...
venv\Scripts\python.exe -c "from ultralytics.utils.downloads import attempt_download_asset; attempt_download_asset('weights/sam2.1_t.pt')"

:check_clip_weight
if exist "weights\clip\ViT-B-32.pt" goto :weights_ready
echo [*] Dang tai model CLIP ViT-B/32 (~338 MB) vao thu muc weights\clip\...
venv\Scripts\python.exe -c "import clip; clip.load('ViT-B/32', device='cpu', download_root='weights/clip')"

:weights_ready
echo.

echo [*] Trang thai PyTorch sau cai dat:
venv\Scripts\python.exe -c "import torch; print('    PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
echo.

echo =========================================================
echo    CAI DAT HOAN TAT! BAN DA CO THE CHAY PROJECT!
echo =========================================================
echo.
echo Ban co the khoi dong ngay bang cach nhap dup vao: run_app.bat
echo.
pause
exit /b 0

:no_python
echo [!] KHONG TIM THAY PYTHON TREN MAY NAY!
echo.
echo Vui long tai va cai dat Python (ho tro Python 3.10 - 3.14)
echo Link tai: https://www.python.org/downloads/
echo LUU Y QUAN TRONG: Hay tich chon vao o [Add python.exe to PATH] khi cai dat!
echo.
pause
exit /b 1

:venv_fail
echo [!] Loi khoi tao venv. Vui long kiem tra quyen ghi thu muc.
pause
exit /b 1
