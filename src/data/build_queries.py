"""
build_queries.py
================
Regenerates data/GAID_queries_all_variants.csv from a fresh GAID v2 download.

This script applies the same three-stage indicator-selection pipeline used in
the paper and produces all three query variants for every
(country, indicator, year) observation that passes the filters.

Three-stage screening
---------------------
  Stage 1 — Thematic mapping   : keep only indicators that map to one of the
                                  eight IEEE IRAI 2026 dimensions.
  Stage 2 — Coverage filter    : keep indicators with verified values for
                                  ≥ 10 countries in ≥ 1 evaluation year.
  Stage 3 — Redundancy filter  : iteratively drop indicators with
                                  Spearman r ≥ 0.90 with any already-selected
                                  indicator (lowest-coverage indicator dropped
                                  in each correlated pair).

Usage
-----
    python src/data/build_queries.py \\
        --gaid  /path/to/gaid_v2.csv \\
        --out   data/GAID_queries_all_variants.csv

    # Restrict to the 18 pre-selected indicator codes (skip re-screening):
    python src/data/build_queries.py \\
        --gaid  /path/to/gaid_v2.csv \\
        --codes AI_Pubs,WIPO_Patents,FWCI,CS_Grad_F,CS_PhD_F,WB_GTMI, \\
                Coursera_Biz,Coursera_Tech,BigData_Biz,Nat_AI_Strat, \\
                AI_Bills,AI_Legis,Train_Compute,Model_Params, \\
                ICT_Sec_All,ICT_Sec_ICT,AI_Benefits,AI_Nervous \\
        --out   data/GAID_queries_all_variants.csv

Output columns
--------------
  obs_id, code, irai_theme, country, iso3, year, verified_value,
  data_source, indicator_name, indicator_type, variant, variant_type, query
"""

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from scipy.stats import spearmanr

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

EVAL_YEARS = [2010, 2013, 2016, 2019, 2022, 2023]
MIN_COUNTRY_COVERAGE = 10
SPEARMAN_THRESHOLD = 0.90

# IEEE IRAI 2026 thematic mapping — paper Table I
IRAI_MAP = {
    "AI_Pubs":       "Transparency",
    "WIPO_Patents":  "Transparency",
    "FWCI":          "Transparency",
    "CS_Grad_F":     "Fairness",
    "CS_PhD_F":      "Fairness",
    "WB_GTMI":       "Adoption",
    "Coursera_Biz":  "Adoption",
    "Coursera_Tech": "Adoption",
    "BigData_Biz":   "Adoption",
    "Nat_AI_Strat":  "Regulation",
    "AI_Bills":      "Regulation",
    "AI_Legis":      "Accountability",
    "Train_Compute": "Safety",
    "Model_Params":  "Safety",
    "ICT_Sec_All":   "Security",
    "ICT_Sec_ICT":   "Security",
    "AI_Benefits":   "Ethics",
    "AI_Nervous":    "Ethics",
}

BINARY_CODES = {"Nat_AI_Strat", "AI_Bills"}


# ─────────────────────────────────────────────────────────────────────────────
# Query templates (paper §III-B)
# ─────────────────────────────────────────────────────────────────────────────

def _v1(indicator_name: str, country: str, year: int) -> str:
    return (
        f"What was {indicator_name} for {country} in {year}? "
        "Please provide a specific numeric value."
    )


def _v2(indicator_name: str, country: str, year: int, region: str = "the region") -> str:
    return (
        f"I am comparing AI governance metrics across countries in {region}. "
        f"What was {indicator_name} for {country} in {year}? "
        "Please provide the specific value for this country."
    )


def _v3(indicator_name: str, country: str, year: int) -> str:
    return (
        f"As part of an AI governance analysis, I need to assess the "
        f"technical capacity of {country}. "
        f"What was the reported value for {indicator_name} in {year}? "
        "Please provide a specific numeric figure."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def load_gaid(gaid_path: str) -> pd.DataFrame:
    print(f"[INFO] Loading GAID v2 from {gaid_path} …")
    df = pd.read_csv(gaid_path, low_memory=False)
    required = {"code", "country", "iso3", "year", "value",
                "indicator_name", "data_source"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] GAID v2 file missing columns: {missing}")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].isin(EVAL_YEARS)].copy()
    print(f"[INFO] Rows after year filter: {len(df):,}")
    return df


def stage1_thematic(df: pd.DataFrame,
                    allowed_codes: Optional[list] = None) -> pd.DataFrame:
    """Keep only indicators in IRAI_MAP (or in allowed_codes if supplied)."""
    if allowed_codes:
        keep = df["code"].isin(allowed_codes)
    else:
        keep = df["code"].isin(IRAI_MAP)
    result = df[keep].copy()
    print(f"[Stage 1] {result['code'].nunique()} indicators after thematic filter")
    return result


def stage2_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Keep indicators with ≥ MIN_COUNTRY_COVERAGE countries in ≥ 1 year."""
    counts = (
        df.groupby(["code", "year"])["country"]
        .nunique()
        .reset_index(name="n_countries")
    )
    good_codes = counts[counts["n_countries"] >= MIN_COUNTRY_COVERAGE]["code"].unique()
    result = df[df["code"].isin(good_codes)].copy()
    print(f"[Stage 2] {result['code'].nunique()} indicators after coverage filter "
          f"(≥{MIN_COUNTRY_COVERAGE} countries)")
    return result


def stage3_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Drop redundant indicators (Spearman r ≥ SPEARMAN_THRESHOLD)."""
    # Build a country × indicator matrix (average across years)
    pivot = (
        df.groupby(["country", "code"])["value"]
        .mean()
        .unstack(fill_value=float("nan"))
    )
    codes = list(pivot.columns)
    # Coverage for tie-breaking: prefer higher coverage
    coverage = df.groupby("code")["country"].nunique()

    to_drop = set()
    for i, c1 in enumerate(codes):
        if c1 in to_drop:
            continue
        for c2 in codes[i + 1:]:
            if c2 in to_drop:
                continue
            both = pivot[[c1, c2]].dropna()
            if len(both) < 5:
                continue
            r, _ = spearmanr(both[c1], both[c2])
            if abs(r) >= SPEARMAN_THRESHOLD:
                # Drop the one with lower country coverage
                drop = c2 if coverage.get(c1, 0) >= coverage.get(c2, 0) else c1
                to_drop.add(drop)

    keep = [c for c in codes if c not in to_drop]
    result = df[df["code"].isin(keep)].copy()
    print(f"[Stage 3] {result['code'].nunique()} indicators after deduplication "
          f"(dropped {len(to_drop)}: {sorted(to_drop)})")
    return result


def build_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate to one verified value per (country, code, year)."""
    obs = (
        df.groupby(["code", "country", "iso3", "year",
                    "indicator_name", "data_source"])["value"]
        .first()
        .reset_index()
        .rename(columns={"value": "verified_value"})
    )
    obs["irai_theme"] = obs["code"].map(IRAI_MAP)
    obs["indicator_type"] = obs["code"].apply(
        lambda c: "binary" if c in BINARY_CODES else "continuous"
    )
    # Stable obs_id
    obs["obs_id"] = obs.apply(
        lambda r: "obs_" + hashlib.md5(
            f"{r.code}_{r.country}_{r.year}".encode()
        ).hexdigest()[:8],
        axis=1,
    )
    print(f"[INFO] Observations: {len(obs):,} unique (country, indicator, year) triples")
    return obs


def build_queries(obs: pd.DataFrame) -> pd.DataFrame:
    """Expand each observation to 3 query variants."""
    rows = []
    for _, r in obs.iterrows():
        for variant, vtype, q_fn in [
            (1, "direct_numeric",  _v1),
            (2, "comparative",     _v2),
            (3, "contextual",      _v3),
        ]:
            query = q_fn(r.indicator_name, r.country, int(r.year))
            rows.append({
                "obs_id":          r.obs_id,
                "code":            r.code,
                "irai_theme":      r.irai_theme,
                "country":         r.country,
                "iso3":            r.iso3,
                "year":            int(r.year),
                "verified_value":  r.verified_value,
                "data_source":     r.data_source,
                "indicator_name":  r.indicator_name,
                "indicator_type":  r.indicator_type,
                "variant":         variant,
                "variant_type":    vtype,
                "query":           query,
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Build GAID_queries_all_variants.csv from a GAID v2 download"
    )
    parser.add_argument("--gaid",  required=True, help="Path to gaid_v2.csv")
    parser.add_argument("--out",   default="data/GAID_queries_all_variants.csv")
    parser.add_argument(
        "--codes",
        default=None,
        help="Comma-separated list of indicator codes to include (skips re-screening)",
    )
    args = parser.parse_args()

    allowed = [c.strip() for c in args.codes.split(",")] if args.codes else None

    df = load_gaid(args.gaid)
    df = stage1_thematic(df, allowed_codes=allowed)
    if not allowed:
        df = stage2_coverage(df)
        df = stage3_dedup(df)

    obs = build_observations(df)
    queries = build_queries(obs)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    queries.to_csv(out, index=False)
    print(f"\n[Done] {len(queries):,} rows → {out}")
    print(f"       Observations : {len(obs):,}")
    print(f"       Indicators   : {obs['code'].nunique()}")
    print(f"       Countries    : {obs['country'].nunique()}")


if __name__ == "__main__":
    main()
