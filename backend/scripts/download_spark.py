#!/usr/bin/env python3
"""Download the official Spark X2.5 1.7B Q4_K_M GGUF and write the reproducibility lockfile.

Authoritative artifact (AGENT.md section 1.3): repo XHToken/Spark-X2.5-1.7B-GGUF, quantization
Q4_K_M, runtime llama.cpp >= b10828. This script never substitutes another repack.

Usage:
    python scripts/download_spark.py [--dest models/spark] [--revision <commit>]
Requires: pip install huggingface_hub
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

HF_REPO = "XHToken/Spark-X2.5-1.7B-GGUF"
QUANT = "Q4_K_M"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="models/spark")
    parser.add_argument("--revision", default=None, help="pin a specific HF commit hash")
    parser.add_argument("--lockfile", default="models/spark.lock.json")
    args = parser.parse_args()
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        print("pip install huggingface_hub first", file=sys.stderr)
        return 2

    api = HfApi()
    info = api.model_info(HF_REPO, revision=args.revision, files_metadata=True)
    candidates = [
        s.rfilename
        for s in info.siblings or []
        if s.rfilename.lower().endswith(".gguf") and QUANT.lower() in s.rfilename.lower()
    ]
    if len(candidates) != 1:
        print(f"expected exactly one {QUANT} GGUF in {HF_REPO}, found: {candidates}", file=sys.stderr)
        return 1
    filename = candidates[0]
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    local = Path(
        hf_hub_download(HF_REPO, filename, revision=info.sha, local_dir=str(dest))
    )
    digest = sha256_of(local)
    lock_path = Path(args.lockfile)
    existing = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    existing.update(
        {
            "hf_repo": HF_REPO,
            "hf_revision": info.sha,
            "gguf_quantization": QUANT,
            "gguf_file": str(local),
            "gguf_sha256": digest,
            "chat_template_source": "embedded in GGUF metadata (tokenizer.chat_template); "
            "llama-server runs with --jinja",
            "downloaded_at": datetime.now(tz=UTC).isoformat(),
        }
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(existing, indent=2) + "\n")
    print(json.dumps(existing, indent=2))
    print(f"\nSet BAY_SPARK_MODEL_PATH={local} and BAY_SPARK_LOCKFILE={lock_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
