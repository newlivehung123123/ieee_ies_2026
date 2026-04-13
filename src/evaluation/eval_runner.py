"""
eval_runner.py
==============
Evaluation runner for the GAID v2 geographic-bias benchmark.

Paper: "Benchmarking Open-Weight Foundation Models for Global AI Technical Governance"
Author: Jason Hung

Design requirements from §III-E:
  - temperature = 0 for all model calls (deterministic / reproducible)
  - 5 % of observations queried TWICE (within-run consistency check)
  - Three query variants per observation; default run = variant 1 (direct numeric)
  - Full run = all three variants

Usage
-----
  # Minimum viable run (variant 1 only):
  python eval_runner.py --variant 1 --model llama4

  # Full run (all three variants), 5 parallel workers:
  python eval_runner.py --variant all --model llama4 --workers 5

  # Resume interrupted run (auto-skips completed rows):
  python eval_runner.py --variant all --model llama4 --workers 5

API provider
------------
  Providers:
    - Groq            (Llama 4 Maverick)   https://api.groq.com/openai/v1
    - Together AI     (Qwen3, DeepSeek)    https://api.together.xyz/v1
    - Mistral AI      (Mistral Large 3)    https://api.mistral.ai/v1

  Set your API keys in the environment before running:
      export OPENROUTER_API_KEY="your-key-here"        # for Llama 4
      export TOGETHER_API_KEY="your-key-here"    # for Qwen3, DeepSeek
      export MISTRAL_API_KEY="your-key-here"     # for Mistral Large 3
"""

import os
import sys
import time
import random
import argparse
import logging
import threading
import pandas as pd
from pathlib import Path
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these before running
# ─────────────────────────────────────────────────────────────────────────────

QUERIES_CSV    = "GAID_queries_all_variants.csv"   # query file (same folder)
RESULTS_DIR    = "results"                          # folder for output CSVs
REPEAT_SEED    = 42                                 # for reproducible 5 % sample
REPEAT_FRACTION = 0.05                             # 5 % double-queried
REQUEST_DELAY  = 0.1                               # seconds between requests (parallel mode)
MAX_RETRIES    = 5                                 # on rate-limit / server errors
RETRY_BACKOFF  = 2.0                               # exponential backoff multiplier

# Model registry — Together AI model strings
# Check current availability at: https://api.together.xyz/models
MODEL_REGISTRY = {
    "llama4": {
        "display":  "Llama 4 Maverick",
        "provider": "openrouter",                                        # ← switched from Together AI (FP8 moved to dedicated-only)
        "model_id": "meta-llama/llama-4-maverick",
    },
    "mistral": {
        "display":  "Mistral Large 3",
        "provider": "mistral",                   # uses Mistral AI endpoint
        "model_id": "mistral-large-latest",
    },
    "qwen3": {
        "display":  "Qwen3-235B-A22B",
        "provider": "together",
        "model_id": "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
    },
    "deepseek": {
        "display":  "DeepSeek-V3-0324",
        "provider": "together",
        "model_id": "deepseek-ai/DeepSeek-V3",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_client(provider: str) -> OpenAI:
    """Return an OpenAI-compatible client for the specified provider."""
    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY not set.  Run: export OPENROUTER_API_KEY='your-key'"
            )
        return OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
    elif provider == "together":
        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "TOGETHER_API_KEY not set.  Run: export TOGETHER_API_KEY='your-key'"
            )
        return OpenAI(
            api_key=api_key,
            base_url="https://api.together.xyz/v1",
        )
    elif provider == "mistral":
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "MISTRAL_API_KEY not set.  Run: export MISTRAL_API_KEY='your-key'"
            )
        return OpenAI(
            api_key=api_key,
            base_url="https://api.mistral.ai/v1",
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


def query_model(client: OpenAI, model_id: str, prompt: str) -> str:
    """Send a single prompt with temperature=0.  Retries on transient errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,          # §III-E: deterministic outputs
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()

        except Exception as exc:
            error_str = str(exc).lower()
            is_rate_limit = "rate" in error_str or "429" in error_str
            is_server     = "500" in error_str or "503" in error_str or "timeout" in error_str

            if (is_rate_limit or is_server) and attempt < MAX_RETRIES:
                wait = REQUEST_DELAY * (RETRY_BACKOFF ** attempt)
                log.warning(f"  Attempt {attempt} failed ({exc}). Retrying in {wait:.1f}s …")
                time.sleep(wait)
            else:
                log.error(f"  Failed after {attempt} attempts: {exc}")
                return f"ERROR: {exc}"

    return "ERROR: max retries exceeded"


def select_repeat_obs(obs_ids: list, fraction: float = REPEAT_FRACTION,
                      seed: int = REPEAT_SEED) -> set:
    """Randomly select 5 % of obs_ids for double-querying (§III-E)."""
    rng = random.Random(seed)
    k = max(1, round(len(obs_ids) * fraction))
    return set(rng.sample(obs_ids, k))


def load_completed(results_path: Path) -> set:
    """Return set of (obs_id, variant, run_id) already saved successfully (non-ERROR), for resumption."""
    if not results_path.exists():
        return set()
    # on_bad_lines='skip' handles any partially written row from an abrupt shutdown
    done = pd.read_csv(results_path, usecols=["obs_id", "variant", "run_id", "response"],
                       on_bad_lines='skip')
    done = done[~done["response"].astype(str).str.startswith("ERROR:")]
    return set(zip(done["obs_id"], done["variant"], done["run_id"]))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(model_key: str, variants: list[int], num_workers: int = 1):
    cfg = MODEL_REGISTRY[model_key]
    display   = cfg["display"]
    provider  = cfg["provider"]
    model_id  = cfg["model_id"]

    log.info(f"=== Evaluation: {display} ({model_id}) ===")
    log.info(f"    Variants to run: {variants}")
    log.info(f"    Parallel workers: {num_workers}")

    # Load queries
    queries = pd.read_csv(QUERIES_CSV)
    queries = queries[queries["variant"].isin(variants)].copy()
    log.info(f"    Loaded {len(queries)} query rows ({queries['obs_id'].nunique()} unique obs)")

    # Select 5 % repeat observations
    all_obs_ids = queries["obs_id"].unique().tolist()
    repeat_obs  = select_repeat_obs(all_obs_ids)
    log.info(f"    Repeat obs (5 %): {len(repeat_obs)} observations will be queried twice")

    # Output file
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    out_file  = Path(RESULTS_DIR) / f"results_{model_key}.csv"
    completed = load_completed(out_file)
    log.info(f"    Already completed: {len(completed)} rows (resuming)")

    # Thread-safety primitives
    csv_lock       = threading.Lock()
    completed_lock = threading.Lock()

    # Build work list
    work_items = []
    for _, row in queries.iterrows():
        obs_id  = int(row["obs_id"])
        variant = int(row["variant"])
        run_ids = [1, 2] if obs_id in repeat_obs else [1]
        for run_id in run_ids:
            key = (obs_id, variant, run_id)
            if key not in completed:
                work_items.append((key, row))

    log.info(f"    Pending queries: {len(work_items)}")

    # Counters
    total_sent = 0
    total_skip = len(completed)

    def process_one(args):
        key, row = args
        obs_id, variant, run_id = key
        client = get_client(provider)   # each thread gets its own client

        log.info(f"  obs={obs_id:5d} variant={variant} run={run_id} "
                 f"[{row['code']:15s}] {row['country']} {row['year']}")

        response_text = query_model(client, model_id, row["query"])

        if response_text.startswith("ERROR:"):
            log.warning(f"  Skipping save for obs={obs_id} variant={variant} run={run_id} due to error response")
            return None

        result_row = {
            "obs_id":         obs_id,
            "code":           row["code"],
            "irai_theme":     row["irai_theme"],
            "country":        row["country"],
            "iso3":           row["iso3"],
            "year":           int(row["year"]),
            "verified_value": row["verified_value"],
            "indicator_name": row["indicator_name"],
            "indicator_type": row["indicator_type"],
            "model":          display,
            "model_key":      model_key,
            "variant":        variant,
            "variant_type":   row["variant_type"],
            "run_id":         run_id,
            "is_repeat_obs":  obs_id in repeat_obs,
            "query":          row["query"],
            "response":       response_text,
            "classification": "",
        }

        # Thread-safe CSV append — force flush to disk so abrupt shutdown can't lose rows
        with csv_lock:
            write_header = not out_file.exists() or out_file.stat().st_size == 0
            with open(out_file, "a", newline="", encoding="utf-8") as f:
                pd.DataFrame([result_row]).to_csv(f, header=write_header, index=False)
                f.flush()
                os.fsync(f.fileno())  # force OS to write buffer to physical disk

        # Thread-safe completed set update
        with completed_lock:
            completed.add(key)

        return key

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_one, item): item for item in work_items}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                total_sent += 1

    log.info(f"\n=== Done: {total_sent} queries sent, {total_skip} skipped (already done) ===")
    log.info(f"    Results saved to: {out_file}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAID v2 LLM evaluation runner")
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model key to evaluate (llama4 | mistral | qwen3 | deepseek)",
    )
    parser.add_argument(
        "--variant",
        default="1",
        help="Query variant(s) to run: 1, 2, 3, or 'all' (default: 1)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Number of parallel workers (default: 5).",
    )
    args = parser.parse_args()

    if args.variant == "all":
        variants_to_run = [1, 2, 3]
    else:
        variants_to_run = [int(v.strip()) for v in args.variant.split(",")]

    run_evaluation(args.model, variants_to_run, num_workers=args.workers)
