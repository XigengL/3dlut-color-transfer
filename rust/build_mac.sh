#!/usr/bin/env bash
# 在 macOS 上构建 lutfit.app
set -e
cd "$(dirname "$0")"
export RUSTUP_HOME="${RUSTUP_HOME:-$HOME/.rustup}"
export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
export PATH="$CARGO_HOME/bin:$PATH"

cargo build --release

APP="dist/lutfit.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp target/release/lutfit "$APP/Contents/MacOS/lutfit"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>lutfit</string>
  <key>CFBundleDisplayName</key><string>3D LUT 拟合工具</string>
  <key>CFBundleIdentifier</key><string>local.lutfit.app</string>
  <key>CFBundleExecutable</key><string>lutfit</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST

echo "完成: $APP  ($(du -sh "$APP" | cut -f1))"
echo "首次打开若被拦：右键 -> 打开，或 xattr -dr com.apple.quarantine \"$APP\""
