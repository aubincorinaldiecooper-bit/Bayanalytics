#!/usr/bin/env bash
# Build llama.cpp (CPU only) at a pinned tag and record the version in models/spark.lock.json.
# Spark X2.5 architecture support requires b10828 or later (AGENT.md section 1.3).
set -euo pipefail

TAG="${LLAMA_CPP_TAG:-b10828}"
DEST="${LLAMA_CPP_DIR:-$HOME/.bayanalytics/llama.cpp}"
LOCK="${SPARK_LOCKFILE:-models/spark.lock.json}"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

if [ ! -d "$DEST/.git" ]; then
  git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp "$DEST"
else
  git -C "$DEST" fetch --depth 1 origin "refs/tags/$TAG:refs/tags/$TAG"
  git -C "$DEST" checkout -q "$TAG"
fi

cmake -S "$DEST" -B "$DEST/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DGGML_METAL=OFF \
  -DGGML_CUDA=OFF \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=ON
cmake --build "$DEST/build" --config Release -j "$JOBS" --target llama-server

BIN="$DEST/build/bin/llama-server"
COMMIT="$(git -C "$DEST" rev-parse HEAD)"
VERSION="$("$BIN" --version 2>&1 | head -1 || true)"

python3 - "$LOCK" "$BIN" "$TAG" "$COMMIT" "$VERSION" <<'PY'
import json, sys, pathlib
lock, binary, tag, commit, version = sys.argv[1:6]
path = pathlib.Path(lock)
data = json.loads(path.read_text()) if path.exists() else {}
data.update({"llama_cpp_tag": tag, "llama_cpp_commit": commit, "llama_cpp_version": version or tag, "llama_server_bin": binary})
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(data, indent=2) + "\n")
print(json.dumps(data, indent=2))
PY

echo
echo "Set BAY_SPARK_LLAMA_SERVER_BIN=$BIN"
