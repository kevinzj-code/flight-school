#!/bin/sh
# Build Flight School as a real .app (debug, fast to build), sign it with this
# Mac's development certificate, and open it — `npm run app`.
#
# Screen Recording and Input Monitoring attach to an app bundle's identity
# (mode.airoplane.flightschool). The bare dev binary `npm run dev` starts from a
# terminal has no bundle, and macOS may credit the permission to the terminal
# instead, so recording is tried in this build.
#
# The app is only re-bundled and re-signed when the build actually changed.
# Rewriting and re-signing an unchanged app every launch was the likeliest
# reason a freshly granted Input Monitoring permission stopped applying.
set -e
cd "$(dirname "$0")/.."
npx tauri build --debug --no-bundle
bin="target/debug/airoplane-flight-school"
app="target/debug/bundle/macos/Airoplane Mode - Flight School.app"
stamp="$app/Contents/Resources/.built-from"
# The stamp covers the binary, the sidecars, and everything the bundle carries
# that the binary does not: the config, the capabilities, the web pages, and the
# Python the app runs (a change there alone used to leave the old copy in place).
now="$(shasum -a 256 "$bin" binaries/fs-recorder-aarch64-apple-darwin binaries/llama-server-aarch64-apple-darwin tauri.conf.json capabilities/*.json $(find ../src -type f \( -name '*.js' -o -name '*.css' -o -name '*.html' \) | sort) $(find python -type f -name '*.py' -not -path '*/__pycache__/*' | sort) | shasum -a 256 | cut -d' ' -f1)"
if [ -d "$app" ] && [ "$(cat "$stamp" 2>/dev/null)" = "$now" ]; then
  echo "[app] nothing changed — opening the app as it was built and signed"
else
  npx tauri bundle --debug --bundles app
  echo "$now" > "$stamp"
  id="${FS_SIGN_IDENTITY:-$(security find-identity -v -p codesigning 2>/dev/null | sed -n 's/^ *1) \([0-9A-F]*\) .*/\1/p')}"
  if [ -n "$id" ]; then
    codesign --force --deep --sign "$id" "$app"
  else
    echo "no signing identity on this Mac — the app runs unsigned, and macOS may forget its permissions after a rebuild" >&2
  fi
fi
[ "$1" = "--no-open" ] || open "$app"
