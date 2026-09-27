#!/usr/bin/env bash
# Install the pinned @receptron/laya@0.1.2 worker dependencies and warm the ONNX bundle cache.
# The worker is a compatibility boundary only (AGENT.md section 36 "Laya worker boundary").
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
WORKER="$HERE/src/bayanalytics/laya/worker"
export LAYA_CACHE="${LAYA_CACHE:-$HOME/.cache/receptron-laya}"

cd "$WORKER"
npm ci
node --input-type=module -e '
import { Laya } from "@receptron/laya";
const t0 = Date.now();
const laya = await Laya.load({ modelDir: process.env.BAY_LAYA_MODEL_DIR || undefined });
console.error(`laya loaded in ${Date.now() - t0} ms, rss ${(process.memoryUsage().rss/1048576).toFixed(0)} MB`);
await laya.close();
'
echo
echo "Laya bundle cached under $LAYA_CACHE (set BAY_LAYA_CACHE_DIR to relocate)."
