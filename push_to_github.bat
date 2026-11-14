@echo off
chcp 65001 > nul
title RoboLabel AI - Dong goi & Day len GitHub
cd /d "%~dp0"

echo =========================================================
echo    ROBOLABEL AI - DONG GOI VA DAY DU AN LEN GITHUB
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

:: 2. Kiem tra .gitignore de tranh push nham venv/ hoac weights/*.pt >100MB
if not exist ".gitignore" (
    echo [!] Canh bao: Khong thay file .gitignore. Vui long kiem tra lai truoc khi push!
    pause
    exit /b 1
)

:: 3. Khoi tao Git repository neu chua co
if exist ".git" goto :git_inited
echo [*] Khoi tao Git repository (nhanh main)...
git init -b main
if errorlevel 1 goto :git_fail

:git_inited
:: 4. Kiem tra thong tin tac gia Git (user.name / user.email)
set "GIT_USER="
for /f "delims=" %%a in ('git config user.name 2^>nul') do set "GIT_USER=%%a"
if not "%GIT_USER%"=="" goto :check_email

:ask_user
set /p GIT_USER="Nhap ten hien thi Git (vi du: Nguyen Van A): "
if "%GIT_USER%"=="" goto :ask_user
git config --global user.name "%GIT_USER%"

:check_email
set "GIT_EMAIL="
for /f "delims=" %%a in ('git config user.email 2^>nul') do set "GIT_EMAIL=%%a"
if not "%GIT_EMAIL%"=="" goto :stage_files

:ask_email
set /p GIT_EMAIL="Nhap Email tai khoan GitHub: "
if "%GIT_EMAIL%"=="" goto :ask_email
git config --global user.email "%GIT_EMAIL%"

:stage_files
echo.
echo [*] Dang chuan bi danh sach file se dong goi (da loai tru venv/ va weights/*.pt)...
git add .
if errorlevel 1 goto :git_fail

git status -s
echo.

:: 5. Tao commit neu co thay doi
git diff --cached --quiet
if not errorlevel 1 goto :check_remote

set /p COMMIT_MSG="Nhap noi dung commit (Enter de dung mac dinh): "
if "%COMMIT_MSG%"=="" set "COMMIT_MSG=Initial commit: RoboLabel AI Studio (SAM 2.1 + CLIP + YOLO Project Manager)"
git commit -m "%COMMIT_MSG%"
if errorlevel 1 goto :git_fail

:check_remote
echo.
git remote get-url origin >nul 2>nul
if not errorlevel 1 goto :do_push

echo [*] Chua co dia chi GitHub Repository (remote origin).
echo     Hay tao 1 repository trong tren https://github.com/new roi dan link .git vao day.
echo     Vi du: https://github.com/tocamban/robolable_ai.git
echo.
set /p REPO_URL="Dan URL GitHub Repository (.git): "
if "%REPO_URL%"=="" (
    echo [!] Ban chua nhap URL repository. Da luu commit cuc bo, ban co the chay lai file nay sau khi co link.
    pause
    exit /b 1
)
git remote add origin "%REPO_URL%"
if errorlevel 1 goto :git_fail

:do_push
echo.
echo [*] Dang day ma nguon len GitHub (nhanh main)...
git branch -M main
git push -u origin main
if errorlevel 1 goto :push_fail

echo.
echo =========================================================
echo    HOAN TAT! DU AN DA DUOC DAY LEN GITHUB THANH CONG!
echo =========================================================
echo.
pause
exit /b 0

:no_git
echo [!] MAY TINH CHUA CAI DAT GIT!
echo.
echo Ban co 3 cach cai nhanh:
echo   Cach 1 (PowerShell/CMD): winget install --id Git.Git -e --source winget
echo   Cach 2 (Tai thu cong):   https://git-scm.com/download/win
echo   Cach 3 (Giao dien):      Tai GitHub Desktop (https://desktop.github.com)
echo.
echo Sau khi cai xong, hay nhap dup lai file push_to_github.bat nay!
echo.
pause
exit /b 1

:git_fail
echo.
echo [!] Co loi trong qua trinh dong goi Git. Vui long kiem tra thong bao o tren.
pause
exit /b 1

:push_fail
echo.
echo [!] Khong the day len GitHub. Nguyen nhan thuong gap:
echo   1. Chua dang nhap tai khoan GitHub (hoac sai URL repository).
echo   2. Repository tren GitHub da co san README/License (neu vay hay chay: git pull origin main --allow-unrelated-histories).
echo.
pause
exit /b 1
