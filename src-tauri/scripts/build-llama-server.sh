#!/bin/bash
# ============================================================================
# Flight School — llama-server sidecar build
#
# Builds llama-server as ONE static binary and drops it where Tauri expects:
# src-tauri/binaries/llama-server-aarch64-apple-darwin. The committed binary was
# built from the same tag with the same flags.
#
# -DBUILD_SHARED_LIBS=OFF: one file, no dylibs to lose at launch.
# -DGGML_METAL_EMBED_LIBRARY=ON: the Metal shader source is baked in, so the
#   binary never looks for ggml-metal.metal on disk.
# -DLLAMA_OPENSSL=OFF: the server only ever listens on 127.0.0.1, and an
#   OpenSSL link would tie the binary to Homebrew's /opt/homebrew/opt/openssl@3.
# -ffile-prefix-map: build paths are rewritten to "." so the binary does not
#   carry the folder it was built in.
#
# Needs (one-time): Xcode Command Line Tools + cmake
#   xcode-select --install
#   brew install cmake
#
# Run from anywhere:  sh src-tauri/scripts/build-llama-server.sh
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# Tauri resolves sidecars by target triple. This exact name is not optional.
TRIPLE="aarch64-apple-darwin"
DEST="$ROOT/binaries/llama-server-$TRIPLE"

# Pinned so every rebuild gets the same engine; the vendored converter in
# python/vendor/ is from the same tag. Override for a one-off:
#   LLAMA_TAG=b1234 sh src-tauri/scripts/build-llama-server.sh
LLAMA_TAG="${LLAMA_TAG:-b9967}"

command -v cmake >/dev/null 2>&1 || { echo "cmake not found. Run: brew install cmake"; exit 1; }
command -v git   >/dev/null 2>&1 || { echo "git not found. Run: xcode-select --install"; exit 1; }

WORK="$ROOT/target/llama-server-build"
mkdir -p "$WORK"

if [ ! -d "$WORK/llama.cpp" ]; then
  echo "==> Cloning llama.cpp $LLAMA_TAG (shallow)…"
  git clone --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp "$WORK/llama.cpp"
else
  echo "==> Reusing existing clone at $WORK/llama.cpp"
  echo "    (delete that folder to force a fresh pull)"
fi

cd "$WORK/llama.cpp"
echo "==> llama.cpp commit: $(git rev-parse --short HEAD)"

MAP="-ffile-prefix-map=$WORK/llama.cpp=."
echo "==> Configuring static Metal build…"
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON \
  -DLLAMA_CURL=OFF \
  -DLLAMA_OPENSSL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=ON \
  -DCMAKE_C_FLAGS="$MAP" \
  -DCMAKE_CXX_FLAGS="$MAP" \
  -DCMAKE_OBJC_FLAGS="$MAP"

echo "==> Building llama-server…"
cmake --build build --target llama-server -j "$(sysctl -n hw.ncpu)"

BIN="build/bin/llama-server"
[ -f "$BIN" ] || { echo "Build finished but $BIN is missing — scroll up for the real error."; exit 1; }

echo "==> Verifying it links only system frameworks…"
if otool -L "$BIN" | grep -Ei 'libllama|libggml|openssl|/opt/homebrew'; then
  echo "FAIL: llama-server links a library outside the system. Not copying it."
  exit 1
fi

echo "==> Smoke test (--version)…"
"$BIN" --version || { echo "FAIL: the binary won't even print its version."; exit 1; }

mkdir -p "$ROOT/binaries"
cp "$BIN" "$DEST"
chmod +x "$DEST"

echo
echo "OK  $DEST"
echo "    $(du -h "$DEST" | cut -f1) — one file, no dylibs, Metal embedded."
