#!/usr/bin/env bash
# Build whisper.cpp (CPU) and fetch the Whisper Tiny ggml model. Voice is optional (AGENT.md 1.4, 13).
set -euo pipefail

TAG="${WHISPER_CPP_TAG:-v1.9.4}"
DEST="${WHISPER_CPP_DIR:-$HOME/.bayanalytics/whisper.cpp}"
MODEL="${WHISPER_MODEL:-tiny}"
MODELS_DIR="${WHISPER_MODELS_DIR:-models/whisper}"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

if [ ! -d "$DEST/.git" ]; then
  git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/whisper.cpp "$DEST"
else
  git -C "$DEST" fetch --depth 1 origin "refs/tags/$TAG:refs/tags/$TAG"
  git -C "$DEST" checkout -q "$TAG"
fi
cmake -S "$DEST" -B "$DEST/build" -DCMAKE_BUILD_TYPE=Release -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=ON
cmake --build "$DEST/build" --config Release -j "$JOBS" --target whisper-cli

mkdir -p "$MODELS_DIR"
bash "$DEST/models/download-ggml-model.sh" "$MODEL" "$MODELS_DIR"

echo
echo "Set BAY_WHISPER_MODE=cli"
echo "Set BAY_WHISPER_BIN=$DEST/build/bin/whisper-cli"
echo "Set BAY_WHISPER_MODEL_PATH=$MODELS_DIR/ggml-$MODEL.bin"
