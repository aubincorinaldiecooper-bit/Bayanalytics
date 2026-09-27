#!/usr/bin/env bash
# Install the pinned @receptron/laya@0.1.2 worker dependencies and warm the ONNX bundle cache.
# The worker is a compatibility boundary only (AGENT.md section 36 "Laya worker boundary").
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
WORKER="$HERE/src/bayanalytics/laya/worker"
# One cache directory for the script and the backend: BAY_LAYA_CACHE_DIR wins, then LAYA_CACHE.
export LAYA_CACHE="${BAY_LAYA_CACHE_DIR:-${LAYA_CACHE:-$HOME/.cache/receptron-laya}}"

cd "$WORKER"
npm ci
node --input-type=module -e '
import { Laya } from "@receptron/laya";
const t0 = Date.now();
const laya = await Laya.load({
  modelDir: process.env.BAY_LAYA_MODEL_DIR || undefined,
  cacheDir: process.env.LAYA_CACHE,
});
console.error(`laya loaded in ${Date.now() - t0} ms, rss ${(process.memoryUsage().rss/1048576).toFixed(0)} MB, bundle ${laya.modelDir}`);
await laya.close();
'
echo
echo "Laya bundle cached under $LAYA_CACHE. Set BAY_LAYA_CACHE_DIR=$LAYA_CACHE (or BAY_LAYA_MODEL_DIR to the bundle directory printed above) in .env."
