#!/usr/bin/env node
/**
 * BayAnalytics Laya worker (AGENT.md section 36, "Laya worker boundary").
 *
 * A very small persistent process that keeps one `@receptron/laya` instance resident and exposes
 * exactly four operations to the Python backend over newline-delimited JSON:
 *
 *   stdin  : {"id": string, "op": "load" | "system_one" | "health" | "close", "params": object}
 *   stdout : {"id": string, "ok": true, "result": object}
 *          | {"id": string | null, "ok": false, "error": {"code": string, "message": string}}
 *
 * Rules:
 *   - stdout carries protocol lines only; every log line goes to stderr;
 *   - exactly one response per request, in arrival order (requests are processed sequentially);
 *   - no finance logic, no retrieval, no Spark, no orchestration lives here;
 *   - the process only exits on `close`, when stdin ends (parent gone) or on a real crash.
 *
 * The Laya implementation module is `@receptron/laya` unless the LAYA_MODULE environment variable
 * names another module (tests inject `stub_laya.mjs` this way).
 */

import { readFile } from "node:fs/promises";
import path from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_MODULE = "@receptron/laya";
const MODULE = process.env.LAYA_MODULE || DEFAULT_MODULE;
// Limits of the English checkpoint shipped with @receptron/laya 0.1.2; overridden by the loaded
// bundle's laya_config.json when the implementation exposes it.
const MAX_LEN = 512;
const HEAD_MAX_LEN = 192;
const MAX_ERROR_CHARS = 2000;

const startedAt = Date.now();
let laya = null;
let loadInfo = null;
let closing = false;

// ---- plumbing -----------------------------------------------------------------------------

function log(...parts) {
  process.stderr.write(`[laya-worker ${process.pid}] ${parts.join(" ")}\n`);
}

function send(obj, cb) {
  process.stdout.write(JSON.stringify(obj) + "\n", cb);
}

function ok(id, result) {
  send({ id, ok: true, result });
}

function fail(id, code, message) {
  send({ id, ok: false, error: { code, message: String(message ?? "").slice(0, MAX_ERROR_CHARS) } });
}

class WorkerError extends Error {
  constructor(code, message) {
    super(message);
    this.code = code;
  }
}

const rssMb = () => process.memoryUsage().rss / 1048576;
const round1 = (x) => Math.round(x * 10) / 10;

function moduleSpecifier() {
  // A path (absolute, or relative to this directory) becomes a file URL; anything else is a package.
  if (MODULE.startsWith(".") || path.isAbsolute(MODULE)) {
    return pathToFileURL(path.resolve(HERE, MODULE)).href;
  }
  return MODULE;
}

async function packageVersion() {
  // `@receptron/laya` restricts its exports, so read the file directly instead of resolving it.
  try {
    const raw = await readFile(path.join(HERE, "node_modules", "@receptron", "laya", "package.json"), "utf8");
    return JSON.parse(raw).version ?? null;
  } catch {
    return null;
  }
}

// ---- operations ---------------------------------------------------------------------------

async function opLoad(params = {}) {
  if (laya) {
    return { ...loadInfo, loaded: true, already_loaded: true };
  }
  const t0 = performance.now();
  let mod;
  try {
    mod = await import(moduleSpecifier());
  } catch (err) {
    throw new WorkerError("LOAD_FAILED", `cannot import ${MODULE}: ${err?.message ?? err}`);
  }
  const Laya = mod.Laya ?? mod.default?.Laya;
  if (!Laya || typeof Laya.load !== "function") {
    throw new WorkerError("LOAD_FAILED", `${MODULE} does not export a Laya class with a static load()`);
  }
  const opts = { executionProviders: ["cpu"] };
  if (params.modelDir) opts.modelDir = String(params.modelDir);
  if (params.cacheDir) opts.cacheDir = String(params.cacheDir);
  if (params.revision) opts.revision = String(params.revision);
  if (params.threads) opts.sessionOptions = { intraOpNumThreads: Number(params.threads) };
  log(`loading ${MODULE}` + (opts.modelDir ? ` from ${opts.modelDir}` : " (cache / download)"));
  try {
    laya = await Laya.load(opts);
  } catch (err) {
    laya = null;
    throw new WorkerError("LOAD_FAILED", err?.message ?? String(err));
  }
  const load_ms = round1(performance.now() - t0);
  loadInfo = {
    load_ms,
    package_version: await packageVersion(),
    rss_mb: round1(rssMb()),
    max_len: laya.config?.max_len ?? MAX_LEN,
    head_max_len: laya.config?.head_max_len ?? HEAD_MAX_LEN,
    model_dir: laya.modelDir ?? opts.modelDir ?? null,
  };
  log(`loaded in ${load_ms} ms, rss ${loadInfo.rss_mb} MB`);
  return { ...loadInfo, loaded: true, already_loaded: false };
}

async function opSystemOne(params) {
  if (!laya) {
    throw new WorkerError("NOT_LOADED", "system_one called before load");
  }
  if (!params || typeof params !== "object" || !("state" in params) || !params.questions || typeof params.questions !== "object") {
    throw new WorkerError("BAD_REQUEST", "system_one needs params.state and params.questions");
  }
  const t0 = performance.now();
  let res;
  try {
    res = await laya.systemOne(params.state, params.questions);
  } catch (err) {
    throw new WorkerError("LAYA_ERROR", err?.message ?? String(err));
  }
  return {
    answers: res.answers,
    usage: res.usage ?? {},
    latency_ms: round1(performance.now() - t0),
  };
}

function opHealth() {
  return {
    loaded: laya !== null,
    rss_mb: round1(rssMb()),
    pid: process.pid,
    uptime_ms: Date.now() - startedAt,
  };
}

async function shutdown(code) {
  if (closing) return;
  closing = true;
  try {
    if (laya) await laya.close();
  } catch (err) {
    log(`close error: ${err?.message ?? err}`);
  }
  laya = null;
  process.exit(code);
}

// ---- request loop -------------------------------------------------------------------------

async function handle(line) {
  const text = line.trim();
  if (text === "") return;
  let req;
  try {
    req = JSON.parse(text);
  } catch (err) {
    fail(null, "BAD_REQUEST", `malformed JSON: ${err?.message ?? err}`);
    return;
  }
  if (!req || typeof req !== "object" || Array.isArray(req)) {
    fail(null, "BAD_REQUEST", "request must be a JSON object");
    return;
  }
  const id = typeof req.id === "string" || typeof req.id === "number" ? req.id : null;
  if (id === null) {
    fail(null, "BAD_REQUEST", "request needs a string id");
    return;
  }
  const params = req.params && typeof req.params === "object" ? req.params : {};
  try {
    switch (req.op) {
      case "load":
        ok(id, await opLoad(params));
        break;
      case "system_one":
        ok(id, await opSystemOne(params));
        break;
      case "health":
        ok(id, opHealth());
        break;
      case "close":
        await new Promise((resolve) => ok(id, { closed: true }, resolve));
        await shutdown(0);
        break;
      default:
        fail(id, "UNKNOWN_OP", `unknown op ${JSON.stringify(req.op)}`);
    }
  } catch (err) {
    if (err instanceof WorkerError) {
      fail(id, err.code, err.message);
    } else {
      log(`unexpected error in ${req.op}: ${err?.stack ?? err}`);
      fail(id, "INTERNAL", err?.message ?? String(err));
    }
  }
}

let chain = Promise.resolve();
const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });
rl.on("line", (line) => {
  chain = chain.then(() => handle(line)).catch((err) => log(`handler failure: ${err?.stack ?? err}`));
});
rl.on("close", () => {
  // stdin ended: the parent is gone or finished with us. Drain queued requests, then leave.
  chain.finally(() => {
    log("stdin closed, exiting");
    shutdown(0);
  });
});

process.stdout.on("error", (err) => {
  log(`stdout error (${err?.code ?? err}), exiting`);
  process.exit(0);
});
process.on("uncaughtException", (err) => {
  log(`uncaught exception: ${err?.stack ?? err}`);
});
process.on("unhandledRejection", (err) => {
  log(`unhandled rejection: ${err?.stack ?? err}`);
});

log(`ready (node ${process.version}, module ${MODULE})`);
