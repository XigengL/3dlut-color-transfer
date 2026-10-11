@echo off
REM 在 Windows 上构建 lut3d.exe
setlocal
cd /d "%~dp0"
cargo build --release
if not exist dist mkdir dist
copy /Y target\release\lut3d.exe dist\lut3d.exe
echo.
echo 完成: dist\lut3d.exe
endlocal
