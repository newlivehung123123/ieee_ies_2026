"""
classifier.py
=============
Rule-based 5-category response classifier for the GAID v2 benchmark.

Paper: §III-F — "Each model response is classified into one of five mutually
exclusive categories using a rule-based automated classifier."

Categories
----------
  VF  Verified Accurate   — numeric response within ±10 % of verified GAID v2 value
  HF  Confident Fabr.     — numeric response OUTSIDE ±10 % (fabricated figure)
  HR  Honest Refusal      — explicitly acknowledges no specific knowledge
  QH  Qualitative Hedging — directional/qualitative description, no numeric commit
  MF  Misattribution      — responds about a different country, year, or indicator

Design notes
------------
  • Purely rule-based; no LLM judge.
  • ±10 % is relative: |extracted - verified| / |verified| ≤ 0.10
    Edge cases:
      – verified = 0 → exact match (or |extracted| ≤ 0.5 absolute tolerance)
      – verified ≤ 1  (0–1 scale: Coursera, WB_GTMI) → also try rescaling
        model's percentage answer by /100 before comparing
  • Unit normalisation: "1.5 million" → 1 500 000; "2.3 billion" → 2 300 000 000;
    "1.2e+15" → 1.2 × 10^15.
  • Binary indicators (Nat_AI_Strat, verified ∈ {0, 1}): "yes"/1 → 1; "no"/0 → 0.

Usage
-----
  # Classify a single model's raw results:
  python classifier.py --input results/results_llama4.csv

  # Classify all model results at once:
  python classifier.py --input results/results_*.csv

  # Output written to:  results/classified_results.csv
"""

import re
import sys
import argparse
import logging
import pandas as pd
from pathlib import Path
from glob import glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS STORED ON 0–1 SCALE (model likely responds in %)
# ─────────────────────────────────────────────────────────────────────────────

ZERO_TO_ONE_INDICATORS = {"Coursera_Biz", "Coursera_Tech", "WB_GTMI", "FWCI"}

# ─────────────────────────────────────────────────────────────────────────────
# HR — HONEST REFUSAL KEYWORDS
# ─────────────────────────────────────────────────────────────────────────────

HR_PATTERNS = re.compile(
    r"(i don'?t have|i do not have|don'?t know|no specific|not available|"
    r"not accessible|cannot provide|can'?t provide|no data|lack (?:the )?(?:specific |"
    r"reliable )?(?:data|information)|unavailable|not in my (?:training|knowledge)|"
    r"my knowledge does not|outside my knowledge|i'?m unable to provide a specific|"
    r"i cannot (?:confirm|give|provide) a specific|no reliable figure|"
    r"not aware of (?:the )?(?:specific|exact)|i have no (?:specific|reliable)|"
    r"insufficient (?:data|information)|this information (?:is )?not available)",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────────────────
# QH — QUALITATIVE HEDGING KEYWORDS
# ─────────────────────────────────────────────────────────────────────────────

QH_PATTERNS = re.compile(
    r"(relatively (high|low|moderate)|higher than|lower than|among the (highest|lowest)|"
    r"ranked (high|low|near the top|near the bottom)|performs? (well|poorly)|"
    r"significant(ly)?|modest(ly)?|limited data|broadly (similar|comparable)|"
    r"generally (low|high|moderate)|tend(s)? to be|may be|could be|likely|"
    r"probably|approximately|around|roughly|in the range of|between .* and .*)",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────────────────
# NUMERIC EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

# Multiplier words
_MULT = {
    "thousand": 1e3, "thousands": 1e3,
    "million":  1e6, "millions":  1e6,
    "billion":  1e9, "billions":  1e9,
    "trillion": 1e12, "trillions": 1e12,
    "quadrillion": 1e15,
    "exaflop":  1e18, "exaflops":  1e18,
    "petaflop": 1e15, "petaflops": 1e15,
    "teraflop": 1e12, "teraflops": 1e12,
    "gigaflop": 1e9,  "gigaflops": 1e9,
    "k": 1e3, "m": 1e6, "b": 1e9,
}

_NUM_RE = re.compile(
    r"""
    (?:                          # optional negative sign
        -\s*
    )?
    \d{1,3}(?:[,\s]\d{3})*      # integer with optional thousands separators
    (?:\.\d+)?                   # optional decimal part
    (?:[eE][+\-]?\d+)?           # optional scientific notation
    |
    \d+\.\d+(?:[eE][+\-]?\d+)?  # decimal first, then optional sci notation
    """,
    re.VERBOSE,
)


def _clean_num_str(s: str) -> str:
    """Remove thousands separators."""
    return s.replace(",", "").replace(" ", "")


def extract_numbers(text: str) -> list[float]:
    """
    Extract all numeric values from text, handling:
      - thousands separators  (1,234,567)
      - scientific notation   (1.5e+15)
      - word multipliers      (1.5 billion → 1 500 000 000)
      - percentage sign       (15% → 15.0)
    Returns list of floats, largest-absolute-value first.
    """
    results = []
    # Work token by token for multiplier detection
    # First pass: find all raw numeric tokens
    for m in _NUM_RE.finditer(text):
        raw = m.group(0)
        try:
            val = float(_clean_num_str(raw))
        except ValueError:
            continue

        # Look ahead up to 3 words for a multiplier
        after = text[m.end(): m.end() + 20].strip().lower()
        for word, mult in _MULT.items():
            if after.startswith(word):
                val *= mult
                break

        results.append(val)

    # Sort so the most informative (often largest) values come first
    results.sort(key=abs, reverse=True)
    return results


def is_numeric_response(text: str) -> bool:
    """True if the response contains at least one extractable number."""
    return len(extract_numbers(text)) > 0


# ─────────────────────────────────────────────────────────────────────────────
# ±10 % COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

RELATIVE_TOLERANCE = 0.10   # §III-F: ±10 %
ABSOLUTE_TOLERANCE_ZERO = 0.5   # for verified_value = 0


def within_10pct(extracted: float, verified: float) -> bool:
    """True if extracted is within ±10 % of verified (relative)."""
    if verified == 0:
        return abs(extracted) <= ABSOLUTE_TOLERANCE_ZERO
    return abs(extracted - verified) / abs(verified) <= RELATIVE_TOLERANCE


def best_numeric_match(numbers: list[float], verified: float,
                        code: str) -> tuple[float | None, bool]:
    """
    Given candidate numbers from the response, return (best_value, is_within_10pct).
    For 0–1 scale indicators, also try /100 rescaling of model's answer.
    """
    is_zero_one = code in ZERO_TO_ONE_INDICATORS

    for n in numbers:
        if within_10pct(n, verified):
            return n, True
        # Rescale: model said e.g. "56 %" for a 0–1 indicator stored as 0.56
        if is_zero_one and verified <= 1.0 and n > 1.0:
            rescaled = n / 100.0
            if within_10pct(rescaled, verified):
                return rescaled, True

    # No match → return the first (most prominent) number as the fabricated value
    return (numbers[0] if numbers else None), False


# ─────────────────────────────────────────────────────────────────────────────
# BINARY INDICATOR HANDLING
# ─────────────────────────────────────────────────────────────────────────────

_YES_RE = re.compile(r"\byes\b|\b1\b|strategy (was |has been )?published|released a strategy",
                     re.IGNORECASE)
_NO_RE  = re.compile(r"\bno\b|\b0\b|no strategy|has not published|had not published|"
                     r"no national ai strategy",
                     re.IGNORECASE)


def classify_binary(response: str, verified: float) -> str:
    """Classify a binary-indicator response (Nat_AI_Strat)."""
    has_yes = bool(_YES_RE.search(response))
    has_no  = bool(_NO_RE.search(response))

    if has_yes and not has_no:
        predicted = 1
    elif has_no and not has_yes:
        predicted = 0
    else:
        # Ambiguous — treat as QH
        return "QH"

    return "VF" if predicted == verified else "HF"


# ─────────────────────────────────────────────────────────────────────────────
# MF — MISATTRIBUTION DETECTION  (heuristic)
# ─────────────────────────────────────────────────────────────────────────────

def check_misattribution(response: str, queried_country: str) -> bool:
    """
    Heuristic: flag as MF if the response explicitly names a DIFFERENT country.
    Conservative by design — the 5 % manual review (§III-F) will catch harder
    cases.  We only flag if the response contains a clear country-name pattern
    like 'for [Country]', 'in [Country]', or '[Country]'s', AND that country
    differs from the queried one.
    """
    country_lower = queried_country.lower()
    resp_lower    = response.lower()
    if country_lower in resp_lower:
        return False
    # Look for explicit geographic references to another country
    geo_patterns = re.findall(
        r"(?:for|in|of|about)\s+([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)|"
        r"([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)'s",
        response,
        re.IGNORECASE,
    )
    # Exclude generic words that are not country names
    STOPWORDS = {
        "the", "this", "that", "these", "those", "which", "there", "here",
        "please", "according", "based", "note", "however", "while", "since",
        "with", "without", "between", "within", "across", "into", "onto",
        "from", "through", "during", "before", "after", "above", "below",
        "index", "score", "report", "data", "value", "survey", "stanford",
        "coursera", "epoch", "wipo", "oecd", "ipsos", "scopus", "world",
        "bank", "govtech", "global", "south", "north", "east", "west",
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "monday", "tuesday", "wednesday", "thursday", "friday",
    }
    for groups in geo_patterns:
        for name in groups:
            if not name:
                continue
            name_lower = name.lower().strip()
            if name_lower in STOPWORDS:
                continue
            if name_lower != country_lower:
                return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def classify_response(response: str, verified_value: float,
                      code: str, country: str,
                      indicator_type: str) -> tuple[str, float | None]:
    """
    Classify a single model response.

    Returns
    -------
    (classification_code, extracted_value)
        classification_code : one of VF, HF, HR, QH, MF
        extracted_value     : the numeric value extracted (or None)
    """
    # Handle API errors
    if response.startswith("ERROR:"):
        return "ERROR", None

    # ── Binary indicators ────────────────────────────────────────────────────
    if indicator_type == "binary":
        code_result = classify_binary(response, verified_value)
        return code_result, None

    # ── Extract numbers ──────────────────────────────────────────────────────
    numbers = extract_numbers(response)

    if numbers:
        best_val, is_match = best_numeric_match(numbers, verified_value, code)
        if is_match:
            return "VF", best_val
        # Check MF before declaring HF
        if check_misattribution(response, country):
            return "MF", best_val
        return "HF", best_val

    # ── No numeric content ───────────────────────────────────────────────────
    if HR_PATTERNS.search(response):
        return "HR", None

    if check_misattribution(response, country):
        return "MF", None

    # Default: qualitative hedging
    return "QH", None


# ─────────────────────────────────────────────────────────────────────────────
# BATCH CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def classify_file(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    log.info(f"Classifying {len(df)} rows from {input_path} …")

    classifications = []
    extracted_vals  = []

    for _, row in df.iterrows():
        code, val = classify_response(
            response        = str(row["response"]),
            verified_value  = float(row["verified_value"]),
            code            = str(row["code"]),
            country         = str(row["country"]),
            indicator_type  = str(row["indicator_type"]),
        )
        classifications.append(code)
        extracted_vals.append(val)

    df["classification"]   = classifications
    df["extracted_value"]  = extracted_vals
    return df


def summarise(df: pd.DataFrame):
    log.info("\n─── Classification summary ───")
    counts = df["classification"].value_counts()
    total  = len(df)
    for cat, n in counts.items():
        log.info(f"  {cat:6s}: {n:5d}  ({100*n/total:.1f} %)")
    log.info(f"  {'TOTAL':6s}: {total:5d}")

    log.info("\n─── By model ───")
    if "model" in df.columns:
        print(df.groupby("model")["classification"].value_counts().unstack(fill_value=0))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAID v2 response classifier")
    parser.add_argument(
        "--input", nargs="+", required=True,
        help="Input CSV file(s) from eval_runner.py  (e.g. results/results_llama4.csv)",
    )
    parser.add_argument(
        "--output", default="results/classified_results.csv",
        help="Output CSV path (default: results/classified_results.csv)",
    )
    args = parser.parse_args()

    # Expand globs
    files = []
    for pattern in args.input:
        files.extend(glob(pattern))
    files = sorted(set(files))

    if not files:
        log.error("No input files found."); sys.exit(1)

    dfs = [classify_file(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)

    Path(args.output).parent.mkdir(exist_ok=True)
    combined.to_csv(args.output, index=False)
    log.info(f"\nClassified results saved to: {args.output}")
    summarise(combined)


# ─────────────────────────────────────────────────────────────────────────────
# UNIT TESTS  (run with: python classifier.py --test)
# ─────────────────────────────────────────────────────────────────────────────

def _run_tests():
    cases = [
        # (response, verified, code, country, ind_type, expected_cat)
        ("The value was 1,234 publications.", 1200, "AI_Pubs", "India", "continuous", "VF"),
        ("Approximately 1,234 publications.", 1200, "AI_Pubs", "India", "continuous", "VF"),
        ("It had about 5,000 publications.", 1200, "AI_Pubs", "India", "continuous", "HF"),
        ("I don't have specific data for that country.", 100, "AI_Pubs", "Chad", "continuous", "HR"),
        ("This country has relatively low AI publication output.", 50, "AI_Pubs", "Bolivia", "continuous", "QH"),
        ("Yes, Argentina published a national AI strategy by 2019.", 1, "Nat_AI_Strat", "Argentina", "binary", "VF"),
        ("No, Bolivia had not published a national AI strategy.", 0, "Nat_AI_Strat", "Bolivia", "binary", "VF"),
        ("Yes it did.", 0, "Nat_AI_Strat", "Bolivia", "binary", "HF"),
        ("The score was 56%", 0.56, "Coursera_Biz", "Brazil", "continuous", "VF"),   # rescale test
        ("The value was 0.57", 0.56, "Coursera_Biz", "Brazil", "continuous", "VF"),
        ("Training compute was 1.5 billion exaflops.", 1.5e18, "Train_Compute", "USA", "continuous", "VF"),
        ("ERROR: rate limit", 100, "AI_Pubs", "India", "continuous", "ERROR"),
    ]
    passed = failed = 0
    for resp, ver, code, country, itype, expected in cases:
        cat, _ = classify_response(resp, ver, code, country, itype)
        ok = (cat == expected)
        status = "PASS" if ok else "FAIL"
        if not ok:
            print(f"  {status}  expected={expected}, got={cat}  →  '{resp[:60]}'")
            failed += 1
        else:
            passed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(cases)} tests.")


if "--test" in sys.argv:
    _run_tests()
    sys.exit(0)
