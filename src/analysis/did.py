"""
did.py
======
Difference-in-differences (DiD) estimation for the GAID v2 benchmark.

Paper §III-F / §V-D:
    The geopolitically balanced 2×2 design (Developer Origin × Geography) enables
    a DiD estimator of whether the Global South accuracy advantage differs across
    Western-origin vs. Chinese-origin developer groups.

DiD design
----------
    Treatment group: Chinese-developer models  (Qwen3-235B-A22B, DeepSeek-V3-0324)
    Control group:   Western-developer models  (Llama 4 Maverick, Mistral Large 3)
    Pre  / Group A:  Global North countries
    Post / Group B:  Global South countries

    DiD = (HF_Chinese_GN − HF_Chinese_GS) − (HF_Western_GN − HF_Western_GS)
        = [advantage for Chinese in GS] − [advantage for Western in GS]

    A positive DiD means Chinese models have a larger GS accuracy advantage
    than Western models (i.e., Chinese models fabricate proportionally less
    for GS relative to GN, compared to Western models).

Paper reported values (§V-D):
    Western:  GN=78.8%, GS=68.8%  →  spread =  −10.0 pp
    Chinese:  GN=77.2%, GS=65.2%  →  spread =  −12.0 pp
    Raw DiD = +2.07 pp  (OR=1.090, SE=0.049, z=1.776, p=0.076)

Usage
-----
    python did.py --results results/classified_results.csv
"""

import argparse
import logging
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL → DEVELOPER ORIGIN MAPPING
# ─────────────────────────────────────────────────────────────────────────────

DEVELOPER_ORIGIN = {
    "Llama 4 Maverick":    "Western",
    "Mistral Large 3":     "Western",
    "Qwen3-235B-A22B":     "Chinese",
    "DeepSeek-V3-0324":    "Chinese",
}

PRIMARY_YEARS = {2010, 2013, 2016, 2019}


# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prepare(results_path: str,
                     region_path: str = "data/region_lookup.csv") -> pd.DataFrame:
    log.info(f"Loading results from {results_path}")
    df = pd.read_csv(results_path, low_memory=False)

    # Primary window
    df = df[df["year"].isin(PRIMARY_YEARS) & (df["variant"] == 1)].copy()
    log.info(f"  {len(df):,} rows (primary window, variant 1)")

    # Binary outcome
    df["hf"] = (df["classification"] == "HF").astype(int)

    # Region
    regions = pd.read_csv(region_path)
    df = df.merge(regions[["country", "global_north_south"]], on="country", how="left")
    df = df.dropna(subset=["global_north_south"])

    # Developer origin
    df["developer_origin"] = df["model"].map(DEVELOPER_ORIGIN)
    df = df.dropna(subset=["developer_origin"])

    return df


# ─────────────────────────────────────────────────────────────────────────────
# RAW 2×2 HF RATES
# ─────────────────────────────────────────────────────────────────────────────

def compute_2x2_hf_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 2×2 table of HF rates:
        rows   = developer origin (Western, Chinese)
        columns = geography (Global North, Global South)
    """
    pivot = (df.groupby(["developer_origin", "global_north_south"])["hf"]
               .agg(["mean", "count"])
               .reset_index())
    pivot.columns = ["developer_origin", "geography", "hf_rate", "n"]
    pivot["hf_pct"] = pivot["hf_rate"] * 100
    return pivot


def print_2x2_table(pivot: pd.DataFrame):
    print("\n" + "═" * 60)
    print("2×2 HF RATES BY DEVELOPER ORIGIN × GEOGRAPHY  (Paper §V-D)")
    print("═" * 60)

    for origin in ["Western", "Chinese"]:
        sub = pivot[pivot["developer_origin"] == origin]
        gn = sub[sub["geography"] == "Global North"]
        gs = sub[sub["geography"] == "Global South"]
        if gn.empty or gs.empty:
            continue
        gn_pct = gn["hf_pct"].iloc[0]
        gs_pct = gs["hf_pct"].iloc[0]
        spread = gs_pct - gn_pct
        print(f"\n  {origin} models")
        print(f"    Global North: {gn_pct:.1f}%  (n={int(gn['n'].iloc[0]):,})")
        print(f"    Global South: {gs_pct:.1f}%  (n={int(gs['n'].iloc[0]):,})")
        print(f"    GS − GN:      {spread:+.1f} pp")


# ─────────────────────────────────────────────────────────────────────────────
# RAW DiD CALCULATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_raw_did(pivot: pd.DataFrame) -> dict:
    """
    Compute raw DiD in percentage points.

    DiD = (C_GN − C_GS) − (W_GN − W_GS)
        = [(Chinese GN spread) − (Western GN spread)]

    A positive value means Chinese models have a LARGER GS advantage
    (i.e., Chinese models fabricate relatively less in GS vs GN).
    """
    def rate(origin, geo):
        return (pivot[(pivot["developer_origin"] == origin) &
                      (pivot["geography"] == geo)]["hf_pct"].iloc[0])

    w_gn = rate("Western", "Global North")
    w_gs = rate("Western", "Global South")
    c_gn = rate("Chinese", "Global North")
    c_gs = rate("Chinese", "Global South")

    western_spread = w_gn - w_gs   # how much more Western fabricates in GN vs GS
    chinese_spread = c_gn - c_gs   # how much more Chinese fabricates in GN vs GS
    did = chinese_spread - western_spread

    return {
        "W_GN": w_gn, "W_GS": w_gs, "W_spread": western_spread,
        "C_GN": c_gn, "C_GS": c_gs, "C_spread": chinese_spread,
        "raw_did_pp": did,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DiD via LOGISTIC REGRESSION (for OR, SE, z, p)
# ─────────────────────────────────────────────────────────────────────────────

def run_did_regression(df: pd.DataFrame) -> dict:
    """
    Estimate DiD via logistic regression:
        hf ~ developer_origin * geography

    The interaction term (developer_origin × geography) IS the DiD estimator.

    Paper §V-D reports:
        OR=1.090, SE=0.049, z=1.776, p=0.076
    """
    log.info("Running DiD logistic regression …")

    # Recode: Chinese=1, Western=0; Global South=1, Global North=0
    df = df.copy()
    df["is_chinese"] = (df["developer_origin"] == "Chinese").astype(int)
    df["is_gs"]      = (df["global_north_south"] == "Global South").astype(int)

    formula = "hf ~ is_chinese + is_gs + is_chinese:is_gs"
    model  = smf.logit(formula, data=df)
    result = model.fit(method="bfgs", maxiter=500, disp=False)

    interaction_row = result.summary2().tables[1].loc["is_chinese:is_gs"]
    coef = interaction_row["Coef."]
    se   = interaction_row["Std.Err."]
    z    = interaction_row["z"]
    p    = interaction_row["P>|z|"]
    ci_lo = interaction_row["[0.025"]
    ci_hi = interaction_row["0.975]"]

    output = {
        "interaction_coef": coef,
        "OR":               np.exp(coef),
        "SE":               se,
        "z":                z,
        "p_value":          p,
        "ci_or_lower":      np.exp(ci_lo),
        "ci_or_upper":      np.exp(ci_hi),
        "developer_origin_main_effect": {
            "OR": np.exp(result.params["is_chinese"]),
            "p":  result.pvalues["is_chinese"],
        },
    }

    return output, result


# ─────────────────────────────────────────────────────────────────────────────
# FULL DiD REPORT
# ─────────────────────────────────────────────────────────────────────────────

def print_did_report(pivot: pd.DataFrame, raw: dict, reg: dict):
    print("\n" + "═" * 60)
    print("DIFFERENCE-IN-DIFFERENCES RESULTS  (Paper §V-D)")
    print("═" * 60)
    print(f"\n  Raw DiD (percentage points): {raw['raw_did_pp']:+.2f} pp")
    print(f"\n  DiD via logistic regression (interaction term):")
    print(f"    OR   = {reg['OR']:.3f}")
    print(f"    SE   = {reg['SE']:.3f}")
    print(f"    z    = {reg['z']:.3f}")
    print(f"    p    = {reg['p_value']:.3f}")
    print(f"    95% CI (OR): [{reg['ci_or_lower']:.3f}, {reg['ci_or_upper']:.3f}]")

    main = reg["developer_origin_main_effect"]
    print(f"\n  Developer-origin main effect:")
    print(f"    OR = {main['OR']:.3f},  p = {main['p']:.3f}")
    print(f"\n  Interpretation:")
    if reg['p_value'] < 0.05:
        print("    ✓ Developer origin SIGNIFICANTLY moderates the GS accuracy advantage.")
    else:
        print("    ~ Developer origin provides only MARGINAL evidence of moderation")
        print(f"      (p={reg['p_value']:.3f} — does not survive α=0.05 threshold).")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DiD estimation for GAID v2 developer-origin × geography analysis"
    )
    parser.add_argument("--results", default="results/classified_results.csv")
    parser.add_argument("--regions", default="data/region_lookup.csv")
    parser.add_argument("--output",  default="results/did_results.csv")
    args = parser.parse_args()

    df    = load_and_prepare(args.results, args.regions)
    pivot = compute_2x2_hf_rates(df)
    raw   = compute_raw_did(pivot)
    reg, model_result = run_did_regression(df)

    print_2x2_table(pivot)
    print_did_report(pivot, raw, reg)

    # Save 2×2 table
    pivot.to_csv(args.output, index=False)
    log.info(f"2×2 table saved to {args.output}")
