#!/usr/bin/env bash
#
# 手工打一个 macOS .app —— 不需要 PyInstaller。
#
# 为什么需要它：
#   直接 `python3 app.py` 启动的进程不属于任何 app bundle，macOS 不会把它
#   当成「当前活动应用」。结果是窗口虽然显示在最前面，系统仍按「后台窗口」
#   处理：每次点击都要先被吞掉一次用于激活，表现就是「每个控件都要点两次」，
#   或者要先点一下标题栏。
#
#   （试过用 ctypes 调 NSApp.activateIgnoringOtherApps: —— 那个 API 在
#    macOS 14+ 已废弃且被系统忽略，实测调用后 isActive 仍为 0。）
#
#   打成 .app 之后，双击启动就是一个正常应用，问题消失。
#
# 用法：  ./make_mac_app.sh
# 产物：  dist/3DLUT.app        （双击即可）
#
# 注意：产物内嵌了本机的 Python 路径，只能在本机运行。
#       要分发给别人，用 build_mac.sh（PyInstaller，会把解释器和依赖一起打包）。
set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
APP="$HERE/dist/3DLUT.app"   # 与 build_mac.sh 的 3DLUT-standalone.app 区分开
PY="$(command -v python3)"

if [ -z "$PY" ]; then
  echo "找不到 python3" >&2
  exit 1
fi

echo "项目目录 : $HERE"
echo "Python   : $PY"
"$PY" -c "import numpy, PIL, tkinter" 2>/dev/null || {
  echo "这个 python3 缺少 numpy / Pillow / tkinter，请换一个：" >&2
  echo "  python3 -c 'import numpy, PIL, tkinter'" >&2
  exit 1
}

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>                  <string>3DLUT</string>
    <key>CFBundleDisplayName</key>           <string>3D LUT 拟合工具</string>
    <key>CFBundleIdentifier</key>            <string>local.lutfit.app</string>
    <key>CFBundleExecutable</key>            <string>3DLUT</string>
    <key>CFBundlePackageType</key>           <string>APPL</string>
    <key>CFBundleSignature</key>             <string>????</string>
    <key>CFBundleShortVersionString</key>    <string>1.0</string>
    <key>CFBundleVersion</key>               <string>1</string>
    <key>LSMinimumSystemVersion</key>        <string>10.13</string>
    <key>NSHighResolutionCapable</key>       <true/>
    <key>LSApplicationCategoryType</key>     <string>public.app-category.photography</string>
</dict>
</plist>
PLIST

cat > "$APP/Contents/MacOS/3DLUT" <<LAUNCH
#!/bin/bash
# 由 make_mac_app.sh 生成
# 从 Finder 启动时没有终端，把输出写到日志，出问题可以查
LOG="$HERE/dist/3DLUT.log"
cd "$HERE" || exit 1
{
  echo "===== \$(date) ====="
  echo "python: $PY"
  echo "cwd   : \$(pwd)"
} >> "\$LOG" 2>&1
exec "$PY" "$HERE/app.py" "\$@" >> "\$LOG" 2>&1
LAUNCH
chmod +x "$APP/Contents/MacOS/3DLUT"

# 让 Finder 立刻认出新 bundle
touch "$APP"

echo
echo "完成： $APP"
echo
echo "双击运行即可。首次打开若被 Gatekeeper 拦，右键 -> 打开，或执行："
echo "  xattr -dr com.apple.quarantine \"$APP\""
