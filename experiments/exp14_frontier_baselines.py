"""exp14 — frontier-LLM baselines (prompted) vs the fine-tuned KhmerGrader.

Evaluates closed/frontier models (GPT, Claude, Gemini, DeepSeek) via API on the same
Khmer test split, prompted ZERO-SHOT, in two variants each: a bare-integer answer and a
reasoning-on answer. The KhmerGrader champion (fine-tuned) is the contribution this
table is compared against (read tab:champs / tab:llmfamily).

Design notes (see the thesis methodology + limitations):
  * One unified OpenAI-compatible client for all providers (per-provider base_url +
    model id + env-var key). Determinism: temperature=0.
  * Reproducibility: every raw API response is CACHED to disk
    (results_frontier/<dataset>/<provider>_<mode>.jsonl), so re-runs are free AND the
    dated snapshot is archived even if a provider later revises/retires the version.
    Closed-API numbers are a dated snapshot; only the cache makes them reproducible.
  * Robust parsing reuses the marker-based integer parser; on an unparseable response
    the item is EXCLUDED from metrics and the unparse rate is reported (no imputation).

Usage (needs API keys in the environment; never hardcode them):
  export OPENAI_API_KEY=... ANTHROPIC_API_KEY=... GEMINI_API_KEY=... DEEPSEEK_API_KEY=...
  python experiments/exp14_frontier_baselines.py --dataset no10c
  python experiments/exp14_frontier_baselines.py --providers openai deepseek --modes bare

Output: results_frontier/<dataset>/frontier_metrics.csv  (one row per provider x mode).

Cost (latest flagships, June 2026; ~136 no10c test answers x 4 models x {bare, reasoning}):
  GPT-5.5 $5/$30, Claude Opus 4.8 $5/$25, Gemini 3.5 Flash $1.50/$9, DeepSeek V4 $0.435/$0.87 per 1M tok.
  Total ~$5-11 (central ~$6), paid once (responses cached). GPT-5.5 + Claude dominate; DeepSeek ~$0.1.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import date

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C            # noqa: E402
import data                   # noqa: E402
from evaluate import metrics  # noqa: E402

# ── Frontier model registry. Pin the exact model id + the eval date for reproducibility.
#    Update `model` to the snapshot you actually call; `EVAL_DATE` is stamped into the cache.
EVAL_DATE = date.today().isoformat()
# Latest flagship per provider as of June 2026. Pin the exact dated snapshot
# (e.g. claude-opus-4-8-YYYYMMDD) for reproducibility before running.
FRONTIER = {
    "openai":   {"base_url": "https://api.openai.com/v1",
                 "model": "gpt-5.5", "key_env": "OPENAI_API_KEY"},
    "anthropic":{"base_url": "https://api.anthropic.com/v1",
                 "model": "claude-opus-4-8", "key_env": "ANTHROPIC_API_KEY"},
    "gemini":   {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                 "model": "gemini-3.5-flash", "key_env": "GEMINI_API_KEY"},
    "deepseek": {"base_url": "https://api.deepseek.com", "model": "deepseek-chat",
                 "key_env": "DEEPSEEK_API_KEY"},
}

# OpenRouter gateway: one OpenAI-compatible endpoint + one key + one prepaid balance for
# all four providers (load e.g. $12 of credits, ~5.5% top-up fee). Namespaced model ids;
# verify the exact ids on https://openrouter.ai/models before running.
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
# Date-pinned flagship slugs, verified against the live OpenRouter model list on 2026-06-23.
# These dated ids ARE the reproducibility pin (a closed model behind a bare alias can change
# silently). Re-verify on https://openrouter.ai/models if a 404 appears.
OPENROUTER_MODELS = {
    "openai":   "openai/gpt-5.5-20260423",
    "anthropic":"anthropic/claude-opus-4.8-20260528",
    "gemini":   "google/gemini-3.5-flash-20260519",
    "deepseek": "deepseek/deepseek-v4-flash-20260423",
}

MODES = ["bare", "reasoning"]   # bare = direct integer; reasoning = allow CoT, parse final int

DATASET_CSV = {"full": "dataset.csv", "no10c": "dataset_no_10c_biology.csv"}

_ANSWER_MARKERS = ("</think>", "<|channel|>final", "final answer", "Final answer",
                   "answer:", "Answer:", "score:", "Score:")


def grading_prompt(question, reference, answer, max_score):
    """The same grading prompt used to fine-tune the KhmerGrader (fair comparison)."""
    return (
        f"Below is a Khmer short-answer grading task. Score the student's answer "
        f"on a scale from 0 to {int(max_score)}.\n\n"
        f"Question: {question}\n\n"
        f"Reference answer: {reference}\n\n"
        f"Student answer: {answer}\n\n"
        f"The score (integer from 0 to {int(max_score)}):"
    )


def parse_int(text, max_score):
    """Marker-aware integer parser. Returns None if no integer is found (excluded)."""
    parsed = text or ""
    for marker in _ANSWER_MARKERS:
        if marker in parsed:
            parsed = parsed.split(marker)[-1]
    m = re.search(r"\d+", parsed)
    if not m:
        return None
    return max(0, min(int(m.group()), int(max_score)))


def call_api(base_url, model, key, prompt, mode, max_retries=5):
    """One OpenAI-compatible /chat/completions call with exponential backoff.
    `reasoning` mode allows a longer answer (model may emit CoT); `bare` asks for the
    integer directly with a tight token budget."""
    import requests
    max_tokens = 1024 if mode == "reasoning" else 16
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, "max_tokens": max_tokens, "stream": False}
    delay = 2.0
    for attempt in range(max_retries):
        try:
            r = requests.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=body, timeout=120,
            )
            if r.status_code in (429, 500, 502, 503, 504):
                ra = r.headers.get("Retry-After")
                time.sleep(float(ra) if ra else delay); delay *= 2; continue
            r.raise_for_status()
            return (r.json()["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay); delay *= 2
    return ""


def _cache_load(path):
    cache = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    cache[rec["answer_id"]] = rec
                except Exception:
                    continue
    return cache


def score_model(provider, mode, test_df, out_dir, gateway="direct"):
    """Score every test answer with one provider+mode, caching raw responses. Returns
    (pred_scores, true_labels, max_scores, true_raw, n_unparsed) over PARSED items only.

    gateway="openrouter" routes ALL providers through OpenRouter (one OPENROUTER_API_KEY,
    one prepaid balance) using its namespaced model ids; "direct" uses each provider's own
    endpoint + key."""
    if gateway == "openrouter":
        base_url, model = OPENROUTER_BASE, OPENROUTER_MODELS[provider]
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        key_name = "OPENROUTER_API_KEY"
    else:
        cfg = FRONTIER[provider]
        base_url, model = cfg["base_url"], cfg["model"]
        key = os.environ.get(cfg["key_env"], "").strip()
        key_name = cfg["key_env"]
    cache_path = os.path.join(out_dir, f"{provider}_{mode}.jsonl")
    cache = _cache_load(cache_path)
    if not key and not cache:
        print(f"  [skip] {provider}/{mode}: ${key_name} not set and no cache")
        return None

    preds, labels, maxes, raws, n_unparsed = [], [], [], [], 0
    fcache = open(cache_path, "a", encoding="utf-8")
    for i, row in test_df.iterrows():
        aid = str(row.get("AnswerID", i))
        max_score = int(row["Max Score"])
        if aid in cache:
            raw_text = cache[aid]["raw_response"]
        else:
            if not key:
                continue
            prompt = grading_prompt(row["Question"], row["Reference"], row["Answer"], max_score)
            try:
                raw_text = call_api(base_url, model, key, prompt, mode)
            except Exception as e:
                print(f"    {provider}/{mode} answer {aid}: API error {type(e).__name__}")
                raw_text = ""
            rec = {"answer_id": aid, "provider": provider, "mode": mode,
                   "model_version": model, "eval_date": EVAL_DATE,
                   "raw_response": raw_text}
            fcache.write(json.dumps(rec, ensure_ascii=False) + "\n"); fcache.flush()
        parsed = parse_int(raw_text, max_score)
        if parsed is None:
            n_unparsed += 1; continue
        preds.append(parsed / max(max_score, 1))
        labels.append(int(row["score_label"]))
        maxes.append(max_score)
        raws.append(int(row["Student Score"]))
    fcache.close()
    if not preds:
        print(f"  [skip] {provider}/{mode}: no parsed predictions")
        return None
    import numpy as np
    m = metrics(np.array(preds), np.array(labels),
                max_scores=np.array(maxes), true_raw=np.array(raws))
    return m, n_unparsed, len(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="no10c", choices=list(DATASET_CSV))
    ap.add_argument("--providers", nargs="+", default=list(FRONTIER),
                    choices=list(FRONTIER))
    ap.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    ap.add_argument("--gateway", default="direct", choices=["direct", "openrouter"],
                    help="direct = each provider's own endpoint+key; openrouter = route all four "
                         "through one OPENROUTER_API_KEY and one prepaid balance")
    args = ap.parse_args()

    C.RAW_CSV = os.path.join(C.PROJECT_ROOT, "data", DATASET_CSV[args.dataset])
    C.DROP_SCORE_ZERO = False
    C.SPLIT_MODE = "random"
    df = data.load_dataframe(C.RAW_CSV)
    _, _, test_df = data.split_dataframe(df)
    print(f"[exp14] {args.dataset}: {len(test_df)} test answers; eval date {EVAL_DATE}; "
          f"gateway={args.gateway}")

    out_dir = os.path.join(C.PROJECT_ROOT, "results_frontier", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    cols = ["provider", "model_version", "mode", "n", "n_unparsed",
            "qwk", "accuracy", "f1_macro", "raw_exact", "raw_within1"]
    rows = []
    for provider in args.providers:
        for mode in args.modes:
            res = score_model(provider, mode, test_df, out_dir, gateway=args.gateway)
            if res is None:
                continue
            m, n_unparsed, n = res
            model_ver = (OPENROUTER_MODELS[provider] if args.gateway == "openrouter"
                         else FRONTIER[provider]["model"])
            row = {"provider": provider, "model_version": model_ver,
                   "mode": mode, "n": n, "n_unparsed": n_unparsed,
                   "qwk": round(float(m["qwk"]), 3), "accuracy": round(float(m["accuracy"]), 3),
                   "f1_macro": round(float(m["f1_macro"]), 3),
                   "raw_exact": round(float(m["raw_exact"]), 3),
                   "raw_within1": round(float(m["raw_within1"]), 3)}
            rows.append(row)
            print(f"  {provider:9s} {mode:9s} QWK={row['qwk']} exact={row['raw_exact']} "
                  f"(n={n}, unparsed={n_unparsed})")

    path = os.path.join(out_dir, "frontier_metrics.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"\n[exp14] wrote {path}")


if __name__ == "__main__":
    main()
