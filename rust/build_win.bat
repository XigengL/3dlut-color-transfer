@echo off
REM 在 Windows 上构建 lutfit.exe
setlocal
cd /d "%~dp0"
cargo build --release
if not exist dist mkdir dist
copy /Y target\release\lutfit.exe dist\lutfit.exe
echo.
echo 完成: dist\lutfit.exe
endlocal
