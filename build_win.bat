@echo off
REM 在 Windows 上构建 3DLUT.exe —— PyInstaller 不能交叉编译，Win 版必须在 Windows 上构建
setlocal
cd /d "%~dp0"

python -m venv .venv-build
call .venv-build\Scripts\activate.bat
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt pyinstaller

if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

pyinstaller --noconfirm --clean --windowed --onedir ^
  --name "3DLUT" ^
  --collect-submodules lut3d ^
  app.py

echo.
echo 完成：dist\3DLUT\3DLUT.exe
endlocal
