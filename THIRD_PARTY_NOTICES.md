# Third-party notices

## Code adapted from GNSIS (MIT)

`backend/src/bayanalytics/research/searxng.py` adapts the SearXNG discovery call from the
`internetSearchSpec` tool in GNSIS `desktop/src/tools/registry.ts`. Nothing else from GNSIS is
imported or depended upon.

```
MIT License

Copyright (c) 2026 Aubin Cooper / Sine Studios

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Runtimes and models used at run time (not vendored)

| Component | Licence | Where it is declared |
| --- | --- | --- |
| `@receptron/laya` 0.1.2 (package) | MIT | `backend/src/bayanalytics/laya/worker/package.json` |
| Laya ONNX bundle (weights) | Apache-2.0 per the package README | downloaded by `scripts/install_laya_worker.sh` |
| `@huggingface/tokenizers` | Apache-2.0 | worker `package.json` |
| `onnxruntime-node` | MIT | dependency of `@receptron/laya` |
| llama.cpp (`llama-server`, tag b10828) | MIT | built by `scripts/setup_llama_cpp.sh` |
| whisper.cpp (`whisper-cli`, tag v1.9.4) | MIT | built by `scripts/setup_whisper.sh` |
| Spark X2.5 1.7B Q4_K_M GGUF (`XHToken/Spark-X2.5-1.7B-GGUF`) | see the model card on Hugging Face; record it in `models/spark.lock.json` when downloading | `scripts/download_spark.py` |
| Whisper Tiny ggml weights | MIT (OpenAI Whisper) | `scripts/setup_whisper.sh` |

Data comes only from web search: the search service in `search/` (SearXNG) and the pages its
searches return, each under its publisher's own terms. `SourceRecord.redistribution` and
`terms_note` carry the per-source note into every result, and third-party excerpts are sent to
clients only when those terms allow redistribution.
