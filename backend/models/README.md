# Model artifacts (not committed)

This directory holds downloaded model files and the reproducibility lockfile. Weights are
git-ignored. Run the scripts in `../scripts` to populate it:

| Component | Script | Produces |
| --- | --- | --- |
| Spark X2.5 1.7B Q4_K_M | `scripts/download_spark.py` | `spark/<file>.gguf`, `spark.lock.json` |
| llama.cpp (b10828+) | `scripts/setup_llama_cpp.sh` | `llama-server` binary path recorded in `spark.lock.json` |
| Laya (`@receptron/laya@0.1.2`) | `scripts/install_laya_worker.sh` | ONNX bundle cached under `$LAYA_CACHE` |
| Whisper Tiny (whisper.cpp) | `scripts/setup_whisper.sh` | `whisper/ggml-tiny.bin`, `whisper-cli` |

`spark.lock.json` records `hf_repo`, `hf_revision`, `gguf_quantization`, `gguf_file`,
`gguf_sha256`, `llama_cpp_version` and `chat_template_source`. Use the same artifact and runtime
revision for every local memory / latency / quality measurement.
