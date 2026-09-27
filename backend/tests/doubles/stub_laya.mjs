/**
 * TEST-ONLY stand-in for `@receptron/laya`.
 *
 * Loaded by worker.mjs when LAYA_MODULE points at this file, so the Python client and the NDJSON
 * protocol can be exercised without the 1.7 GB ONNX bundle. It mirrors the public surface used by
 * the worker (`Laya.load`, `laya.systemOne`, `laya.close`, `laya.config`, `laya.modelDir`) and
 * returns deterministic answers:
 *
 *   choice -> the first criteria key with probability 0.6, the rest sharing 0.4 evenly
 *   score  -> (criteria.length - 1) / 2, with a flat distribution
 *   noul   -> 0.5
 *   usage.input_tokens -> the sum over questions of the sequence length Laya would build
 *                         (frame + head + state, capped at max_len), measured with `encode`
 *
 * `encode(text)` is the deterministic stand-in for the bundle tokenizer: one id per word or
 * punctuation mark (`/\w+|[^\w\s]/g`). The worker's `count_tokens` op uses it, so every token
 * figure that leaves this module is a measurement of that rule, never a characters-per-token
 * estimate.
 *
 * Special instruction markers drive failure paths in the tests:
 *   "__crash__" -> process.exit(3)   (simulates a worker death mid-request)
 *   "__throw__" -> throws Error("stub failure")
 *   "__slow__"  -> waits 3000 ms before answering (request-timeout tests)
 *
 * Never use this module outside tests: it has no model and no judgement.
 */

const round4 = (x) => Math.round(x * 1e4) / 1e4;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const SLOW_MS = 3000;
// [CLS], [SEP] after the instructions, [SEP] after the options, final [SEP].
const SEQUENCE_SPECIALS = 4;

export class Laya {
  constructor(opts) {
    this.modelDir = opts.modelDir ?? "stub";
    this.config = { max_len: 512, head_max_len: 192 };
  }

  static async load(opts = {}) {
    return new Laya(opts);
  }

  /** Deterministic stand-in for the bundle tokenizer: one id per word or punctuation mark. */
  encode(text) {
    return (String(text).match(/\w+|[^\w\s]/g) ?? []).map((_, i) => i + 1);
  }

  /** Head size as `buildSequence` renders it: instructions plus one [MASK] per option text. */
  headTokens(q) {
    const ins = typeof q.instructions === "string" ? q.instructions : JSON.stringify(q.instructions ?? "");
    let options;
    if (q.type === "choice") {
      const c = q.criteria ?? {};
      options = Array.isArray(c) ? c.map(String) : Object.entries(c).map(([k, v]) => (v ? `${k}: ${v}` : String(k)));
    } else if (q.type === "score") {
      options = (Array.isArray(q.criteria) ? q.criteria : []).map((c, i) => `level ${i}: ${c}`);
    } else {
      options = ["false: no, the statement does not hold", "true: yes, the statement holds"];
    }
    const scrub = (t) => String(t).replaceAll("[MASK]", " ");
    let n = this.encode(scrub(`${q.type} question: ${ins}`)).length;
    for (const opt of options) n += 1 + this.encode(" " + scrub(opt)).length;
    return n;
  }

  async systemOne(state, questions) {
    const qids = Object.keys(questions ?? {});
    if (qids.length === 0) {
      throw new Error("systemOne: at least one question is required");
    }
    const instructions = qids.map((qid) => {
      const ins = questions[qid]?.instructions;
      return typeof ins === "string" ? ins : JSON.stringify(ins ?? "");
    });
    if (instructions.some((s) => s.includes("__crash__"))) {
      process.exit(3);
    }
    if (instructions.some((s) => s.includes("__throw__"))) {
      throw new Error("stub failure");
    }
    if (instructions.some((s) => s.includes("__slow__"))) {
      await sleep(SLOW_MS);
    }

    const answers = {};
    for (const qid of qids) {
      const q = questions[qid];
      if (q.type === "choice") {
        const keys = Array.isArray(q.criteria) ? q.criteria : Object.keys(q.criteria ?? {});
        if (keys.length === 0) throw new Error(`question ${JSON.stringify(qid)}: choice needs options`);
        const rest = keys.length > 1 ? 0.4 / (keys.length - 1) : 0;
        const probabilities = Object.fromEntries(keys.map((k, i) => [k, round4(i === 0 ? (keys.length > 1 ? 0.6 : 1) : rest)]));
        answers[qid] = { type: "choice", choice: keys[0], probabilities, confidence: 0.5, rl_agent: { act_probability: 0.5 } };
      } else if (q.type === "score") {
        const levels = Array.isArray(q.criteria) ? q.criteria : [];
        if (levels.length === 0) throw new Error(`question ${JSON.stringify(qid)}: score needs levels`);
        const flat = round4(1 / levels.length);
        answers[qid] = {
          type: "score",
          score: (levels.length - 1) / 2,
          legend: Object.fromEntries(levels.map((c, i) => [String(i), c])),
          probabilities: Object.fromEntries(levels.map((_, i) => [String(i), flat])),
          confidence: 0,
          rl_agent: { act_probability: 0.5 },
        };
      } else if (q.type === "noul") {
        answers[qid] = { type: "noul", noul: 0.5, rl_agent: { act_probability: 0.5 } };
      } else {
        throw new Error(`question ${JSON.stringify(qid)}: unknown type ${JSON.stringify(q.type)}`);
      }
    }
    const serialized = typeof state === "string" ? state : JSON.stringify(state ?? null);
    const stateTokens = this.encode(serialized.replaceAll("[MASK]", " ")).length;
    let inputTokens = 0;
    for (const qid of qids) {
      inputTokens += Math.min(this.config.max_len, SEQUENCE_SPECIALS + this.headTokens(questions[qid]) + stateTokens);
    }
    return {
      model: "laya-stub",
      answers,
      usage: { input_tokens: inputTokens, output_tokens: 0 },
    };
  }

  async close() {}
}
