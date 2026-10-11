#!/usr/bin/env bash
# 在 macOS 上构建 3DLUT.app   —— PyInstaller 不能交叉编译，Mac 版必须在 Mac 上构建
set -e
cd "$(dirname "$0")"

python3 -m venv .venv-build 2>/dev/null || true
source .venv-build/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt pyinstaller

rm -rf build dist
pyinstaller --noconfirm --clean --windowed --onedir \
  --name "3DLUT-standalone" \
  --osx-bundle-identifier com.local.lutfit \
  --collect-submodules lutfit \
  app.py

echo
echo "完成：dist/3DLUT-standalone.app"
echo "首次打开若被 Gatekeeper 拦：右键 -> 打开，或执行"
echo "  xattr -dr com.apple.quarantine dist/3DLUT-standalone.app"
