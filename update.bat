@echo off
chcp 65001 > nul
title RoboLabel AI - Cap nhat tu GitHub
cd /d "%~dp0"

echo =========================================================
echo    ROBOLABEL AI - TU DONG CAP NHAT PHIEN BAN MOI (GITHUB)
echo =========================================================
echo.

:: 1. Tim kiem Git trong PATH hoac thu muc cai dat mac dinh
where git >nul 2>nul
if not errorlevel 1 goto :git_found

if exist "C:\Program Files\Git\cmd\git.exe" set "PATH=C:\Program Files\Git\cmd;%PATH%" & goto :git_found
if exist "C:\Program Files\Git\bin\git.exe" set "PATH=C:\Program Files\Git\bin;%PATH%" & goto :git_found
if exist "C:\Program Files (x86)\Git\cmd\git.exe" set "PATH=C:\Program Files (x86)\Git\cmd;%PATH%" & goto :git_found
if exist "%LOCALAPPDATA%\Programs\Git\cmd\git.exe" set "PATH=%LOCALAPPDATA%\Programs\Git\cmd;%PATH%" & goto :git_found
goto :no_git

:git_found
echo [*] Da tim thay Git:
git --version
echo.

:: 2. Kiem tra day co phai thu muc clone tu Git khong
if not exist ".git" goto :no_git_repo

git remote get-url origin >nul 2>nul
if errorlevel 1 goto :no_remote

:: 3. Tam dung server cong 5050 neu dang chay de tranh khoa file khi cap nhat
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $c = Get-NetTCPConnection -LocalPort 5050 -ErrorAction SilentlyContinue; if ($c) { Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue; Write-Host '[*] Da tam dung server port 5050 de cap nhat an toan.' } } catch {}"

:: 4. Xac dinh nhanh hien tai va ma commit truoc khi cap nhat
set "CUR_BRANCH=main"
for /f "delims=" %%b in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "CUR_BRANCH=%%b"
if "%CUR_BRANCH%"=="" set "CUR_BRANCH=main"
if "%CUR_BRANCH%"=="HEAD" set "CUR_BRANCH=main"

set "OLD_HEAD="
for /f "delims=" %%h in ('git rev-parse HEAD 2^>nul') do set "OLD_HEAD=%%h"

:: 5. Sao luu tam cac thay doi cuc bo (neu co) de tranh loi xung dot khi git pull
set "DID_STASH=0"
git diff --quiet >nul 2>nul
if errorlevel 1 goto :stash_local
git diff --cached --quiet >nul 2>nul
if not errorlevel 1 goto :do_pull

:stash_local
echo [*] Phat hien co thay doi cuc bo, dang sao luu tam truoc khi tai ban moi...
git -c user.name="RoboLabel Updater" -c user.email="updater@local" stash push -m "RoboLabel auto-backup before update" >nul 2>nul
if not errorlevel 1 set "DID_STASH=1"

:do_pull
echo [*] Dang kiem tra va tai ban cap nhat moi nhat tu GitHub (nhanh %CUR_BRANCH%)...
git -c user.name="RoboLabel Updater" -c user.email="updater@local" pull origin %CUR_BRANCH%
if errorlevel 1 goto :pull_fail

:: Khoi phuc thay doi cuc bo neu truoc do da sao luu (khong gay hong file neu xung dot)
if "%DID_STASH%"=="0" goto :check_diff
git stash apply >nul 2>nul
if not errorlevel 1 goto :drop_stash
git checkout -- . >nul 2>nul
echo [!] Luu y: Thay doi cuc bo truoc do co xung dot voi ban moi nen da duoc giu lai an toan trong "git stash".
goto :check_diff

:drop_stash
git stash drop >nul 2>nul

:check_diff
set "NEW_HEAD="
for /f "delims=" %%h in ('git rev-parse HEAD 2^>nul') do set "NEW_HEAD=%%h"

echo.
if "%OLD_HEAD%"=="%NEW_HEAD%" goto :already_latest

echo [+] Da cap nhat thanh cong len phien ban moi (%NEW_HEAD:~0,7%)!
echo [*] Cac thay doi vua tai ve:
git log --oneline %OLD_HEAD%..%NEW_HEAD%
echo.

:: 6. Neu requirements.txt thay doi hoac chua co venv, tu dong cap nhat thu vien
if not exist "venv\Scripts\pip.exe" goto :suggest_setup

git diff --name-only %OLD_HEAD% %NEW_HEAD% | findstr /i /c:"requirements.txt" >nul 2>nul
if errorlevel 1 goto :check_weights
echo [*] Phat hien requirements.txt co thu vien moi. Dang cai dat bo sung...
venv\Scripts\pip.exe install -r requirements.txt
echo.

:check_weights
if not exist "weights\clip" mkdir "weights\clip"
if exist "weights\sam2.1_t.pt" goto :check_clip_w
echo [*] Dang tai model SAM 2.1 Tiny (~75 MB)...
venv\Scripts\python.exe -c "from ultralytics.utils.downloads import attempt_download_asset; attempt_download_asset('weights/sam2.1_t.pt')"

:check_clip_w
if exist "weights\clip\ViT-B-32.pt" goto :done_ok
echo [*] Dang tai model CLIP ViT-B/32 (~338 MB)...
venv\Scripts\python.exe -c "import clip; clip.load('ViT-B/32', device='cpu', download_root='weights/clip')"
goto :done_ok

:suggest_setup
echo [!] May nay chua tao moi truong ao (venv). Hay chay file setup.bat de cai dat!
goto :done_ok

:already_latest
echo [=] May ban dang o phien ban moi nhat roi (Commit: %NEW_HEAD:~0,7%)!
if not exist "venv\Scripts\python.exe" echo [!] Luu y: Chua thay thu muc venv, hay chay setup.bat neu chua cai dat moi truong.

:done_ok
echo.
echo =========================================================
echo    HOAN TAT CAP NHAT! BAN CO THE CHAY: run_app.bat
echo =========================================================
echo.
pause
exit /b 0

:no_git
echo [!] MAY TINH CHUA CAI DAT GIT!
echo.
echo Hay cai Git bang 1 trong 2 cach sau roi chay lai update.bat:
echo   Cach 1 (PowerShell): winget install --id Git.Git -e --source winget
echo   Cach 2 (Tai thu cong): https://git-scm.com/download/win
echo.
pause
exit /b 1

:no_git_repo
echo [!] Thu muc nay khong chua lich su Git (.git)!
echo     De dung chuc nang tu cap nhat 1-click, hay tai du an bang lenh:
echo     git clone https://github.com/tocamban/robolable_ai.git
echo.
pause
exit /b 1

:no_remote
echo [!] Chua gan dia chi GitHub Repository (origin).
echo     Hay chay: git remote add origin https://github.com/tocamban/robolable_ai.git
echo.
pause
exit /b 1

:pull_fail
echo.
echo [!] Khong the tai ban cap nhat tu GitHub.
echo     Vui long kiem tra ket noi mang hoac dang nhap GitHub neu la repository rieng tu.
if "%DID_STASH%"=="1" (
    git stash apply >nul 2>nul
    if not errorlevel 1 git stash drop >nul 2>nul
)
echo.
pause
exit /b 1
