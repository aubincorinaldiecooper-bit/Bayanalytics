# BayAnalytics backend image. Railway detects it at the root of the service's source directory
# (/backend); the service settings are in README.md, "Deploying on Railway".
#
# The image holds everything the managed runtime needs, all inside the image rather than a
# build-only interpreter or $HOME cache:
#   - the Python package, installed editable so the Laya worker directory resolves to /app/src
#     (where its node_modules live), not to a copy in site-packages;
#   - Node 22 and the Laya worker's node_modules;
#   - llama-server from llama.cpp b10828, built for a portable x86-64 target;
#   - the Spark GGUF and the Laya bundle under /app/models (skip with --build-arg SKIP_MODELS=1).
# The last RUN asserts each piece, so a broken image fails the build instead of the deploy.
#
#   docker build -t bayanalytics-backend backend
#   docker build --build-arg SKIP_MODELS=1 -t bayanalytics-backend:no-models backend

ARG LLAMA_CPP_TAG=b10828

# ---------- llama-server ----------
FROM debian:bookworm-slim AS llama
ARG LLAMA_CPP_TAG
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch "${LLAMA_CPP_TAG}" https://github.com/ggml-org/llama.cpp /src/llama.cpp
# GGML_NATIVE=ON would target the build host's CPU; Railway builds and runs on different hosts,
# so pin a portable x86-64 target (AVX2/FMA/F16C, available on every current cloud x86 CPU).
# Static libraries let the binary move to the runtime stage without an rpath; no libcurl needed.
RUN cmake -S /src/llama.cpp -B /src/llama.cpp/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
        -DGGML_METAL=OFF -DGGML_CUDA=OFF \
        -DLLAMA_CURL=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON \
    && cmake --build /src/llama.cpp/build --config Release -j "$(nproc)" --target llama-server \
    && install -D /src/llama.cpp/build/bin/llama-server /opt/llama/bin/llama-server \
    && commit="$(git -C /src/llama.cpp rev-parse HEAD)" \
    && version="$(/opt/llama/bin/llama-server --version 2>&1 | head -1)" \
    && printf '{\n  "llama_cpp_tag": "%s",\n  "llama_cpp_commit": "%s",\n  "llama_cpp_version": "%s",\n  "llama_server_bin": "/opt/llama/bin/llama-server"\n}\n' \
        "${LLAMA_CPP_TAG}" "${commit}" "${version}" > /opt/llama/spark.lock.json \
    && cat /opt/llama/spark.lock.json

# ---------- Node runtime for the Laya worker ----------
FROM node:22-bookworm-slim AS node

# ---------- application ----------
FROM python:3.12-slim-bookworm AS app
ARG SKIP_MODELS=0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    BAY_SPARK_LLAMA_SERVER_BIN=/opt/llama/bin/llama-server \
    BAY_SPARK_MODEL_PATH=/app/models/spark.gguf \
    BAY_SPARK_LOCKFILE=/app/models/spark.lock.json \
    BAY_LAYA_CACHE_DIR=/app/models/laya

# libgomp1: llama-server is built with OpenMP.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libstdc++6 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && node --version && npm --version

COPY --from=llama /opt/llama /opt/llama

RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
RUN pip install -e . huggingface_hub \
    && mkdir -p /app/models \
    && cp /opt/llama/spark.lock.json /app/models/spark.lock.json \
    && chown -R app:app /app

USER app

# onnxruntime-node needs its install scripts: never --ignore-scripts here.
RUN cd /app/src/bayanalytics/laya/worker && npm ci --omit=dev && npm cache clean --force

# The Spark GGUF (recorded in the lockfile with its sha256) and the Laya bundle, fetched from
# Hugging Face at build time and loaded from /app/models at run time.
RUN if [ "${SKIP_MODELS}" = "1" ]; then \
        echo "SKIP_MODELS=1: no Spark GGUF or Laya bundle in this image"; \
    else \
        set -eux; \
        python scripts/download_spark.py --dest /app/models/spark --lockfile /app/models/spark.lock.json; \
        gguf="$(find /app/models/spark -name '*.gguf' -type f | head -1)"; \
        test -n "$gguf"; \
        ln -sf "$gguf" /app/models/spark.gguf; \
        cd /app/src/bayanalytics/laya/worker; \
        node --input-type=module -e 'import { Laya } from "@receptron/laya"; const t0 = Date.now(); const laya = await Laya.load({ cacheDir: process.env.BAY_LAYA_CACHE_DIR }); console.error(`laya bundle ready in ${Date.now() - t0} ms: ${laya.modelDir}`); await laya.close();'; \
        rm -rf /home/app/.cache/huggingface; \
    fi

# Fail the build, not the deploy, when any piece is missing.
RUN set -eux; \
    python -c "import bayanalytics.main, bayanalytics.cli"; \
    bayanalytics --version; \
    python -c "from bayanalytics.config import Settings; s = Settings.from_env(); w = s.laya_worker_dir; assert (w / 'node_modules' / '@receptron' / 'laya').is_dir(), w; print('laya worker', w)"; \
    /opt/llama/bin/llama-server --version; \
    cd /app/src/bayanalytics/laya/worker && node --input-type=module -e 'await import("@receptron/laya"); await import("@huggingface/tokenizers"); console.log("laya worker modules ok")'; \
    if [ "${SKIP_MODELS}" != "1" ]; then \
        test -s "$(readlink -f "${BAY_SPARK_MODEL_PATH}")"; \
        test -n "$(ls -A "${BAY_LAYA_CACHE_DIR}")"; \
        python -c "import json; d = json.load(open('/app/models/spark.lock.json')); assert d['gguf_sha256'] and d['llama_cpp_commit'], d; print(d['hf_repo'], d['hf_revision'])"; \
    fi

EXPOSE 8000
# A shell so ${PORT} expands (Railway sets PORT); exec so uvicorn is PID 1 and gets SIGTERM.
# "::" makes `bayanalytics serve` open one dual-stack socket: IPv4 for Railway's healthcheck and
# the public edge, IPv6 for the private network. A non-loopback host requires BAY_API_KEY (the
# backend refuses to start without it).
CMD ["sh", "-c", "exec bayanalytics serve --host :: --port \"${PORT:-8000}\""]
