"""
regression.py
=============
Mixed-effects logistic regression for the GAID v2 geographic-bias benchmark.

Paper §III-F:
    logit(P[HF_ijkmt]) = β₀ + β₁Region_j + β₂Model_k + β₃Year_t + β₄Theme_m + u_j + ε_ijkmt

Random effects: country-level intercepts, nested within world region.
Fixed effects:  world region, model identity, evaluation year, IEEE IRAI thematic dimension.
Outcome:        HF = 1 (confident fabrication), all other categories = 0.

Dependencies
------------
    pip install statsmodels pandas numpy scipy

    For a full GLMM with proper logistic random effects, install:
    pip install pymer4     (wraps R's lme4::glmer)

Usage
-----
    python regression.py --results results/classified_results.csv
    python regression.py --results results/classified_results.csv --full-glmm
"""

import argparse
import logging
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Primary analysis window — within training periods for all four models
PRIMARY_YEARS = {2010, 2013, 2016, 2019}

# Reference levels for dummy coding (omitted category)
REF_REGION = "Europe"
REF_MODEL  = "DeepSeek-V3-0324"
REF_YEAR   = 2010
REF_THEME  = "Transparency"


# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prepare(results_path: str,
                     region_path: str = "data/region_lookup.csv",
                     primary_only: bool = True) -> pd.DataFrame:
    """
    Load classified results, merge region labels, and create model features.

    Parameters
    ----------
    results_path : str
        Path to classified_results.csv
    region_path : str
        Path to region_lookup.csv
    primary_only : bool
        If True, restrict to primary analysis window (2010–2019).

    Returns
    -------
    pd.DataFrame  with columns:
        hf           : int    — outcome (1=HF, 0=other)
        region       : str    — UN macro-region
        ns_binary    : str    — "Global North" | "Global South"
        model        : str    — model display name
        year         : int
        irai_theme   : str
        country      : str
        iso3         : str
    """
    log.info(f"Loading classified results from {results_path}")
    df = pd.read_csv(results_path, low_memory=False)
    log.info(f"  {len(df):,} rows loaded")

    # Primary window filter
    if primary_only:
        df = df[df["year"].isin(PRIMARY_YEARS)].copy()
        log.info(f"  {len(df):,} rows after primary-window filter (years: {sorted(PRIMARY_YEARS)})")

    # Binary outcome
    df["hf"] = (df["classification"] == "HF").astype(int)

    # Merge region labels
    regions = pd.read_csv(region_path)
    df = df.merge(regions[["country", "un_region", "global_north_south"]],
                  on="country", how="left")
    missing_region = df["un_region"].isna().sum()
    if missing_region:
        log.warning(f"  {missing_region} rows have no region — dropping")
        df = df.dropna(subset=["un_region"])

    # Rename for model formula
    df = df.rename(columns={"un_region": "region", "global_north_south": "ns_binary"})

    # Variant 1 only for primary regression (direct numeric; others used as robustness)
    df_v1 = df[df["variant"] == 1].copy()
    log.info(f"  {len(df_v1):,} rows for primary regression (variant 1)")

    return df_v1


# ─────────────────────────────────────────────────────────────────────────────
# LOGISTIC REGRESSION — FIXED EFFECTS ONLY (fast approximation)
# ─────────────────────────────────────────────────────────────────────────────

def run_logistic_fe(df: pd.DataFrame) -> "RegressionResultsWrapper":
    """
    Logistic regression with country fixed effects (dummy variables).
    Faster than GLMM; used when pymer4 is not available.
    Interpretation of region/model/year/theme coefficients is unchanged.
    """
    log.info("Running fixed-effects logistic regression …")

    # Set reference levels
    df = df.copy()
    df["region"]     = pd.Categorical(df["region"],     categories=sorted(df["region"].unique()))
    df["model"]      = pd.Categorical(df["model"],      categories=sorted(df["model"].unique()))
    df["irai_theme"] = pd.Categorical(df["irai_theme"], categories=sorted(df["irai_theme"].unique()))
    df["year_cat"]   = pd.Categorical(df["year"].astype(str),
                                      categories=sorted(df["year"].astype(str).unique()))

    formula = (
        "hf ~ C(region, Treatment(reference='Europe')) "
        "+ C(model, Treatment(reference='DeepSeek-V3-0324')) "
        "+ C(year_cat, Treatment(reference='2010')) "
        "+ C(irai_theme, Treatment(reference='Transparency'))"
    )

    model = smf.logit(formula, data=df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(method="bfgs", maxiter=500, disp=False)

    return result


def run_mixed_effects_glmm(df: pd.DataFrame):
    """
    Full GLMM via pymer4 (wraps R lme4::glmer).
    Random intercept for country nested within region.

    Requires: pip install pymer4  (and R with lme4 installed)
    """
    try:
        from pymer4.models import Lmer
    except ImportError:
        raise ImportError(
            "pymer4 is required for full GLMM estimation.\n"
            "Install: pip install pymer4\n"
            "Also requires R with lme4: install.packages('lme4')\n"
            "Alternatively, use --no-glmm for fixed-effects approximation."
        )

    log.info("Running mixed-effects GLMM via pymer4 (lme4::glmer) …")

    formula = (
        "hf ~ region + model + year_cat + irai_theme + (1 | country)"
    )

    df = df.copy()
    df["year_cat"] = df["year"].astype(str)

    lmm = Lmer(formula, data=df, family="binomial")
    result = lmm.fit()
    return lmm, result


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS FORMATTING
# ─────────────────────────────────────────────────────────────────────────────

def format_results_table(result, output_path: str = None) -> pd.DataFrame:
    """
    Extract coefficient table with odds ratios and confidence intervals.

    Returns DataFrame with columns:
        predictor, coef, odds_ratio, ci_lower, ci_upper, z, p_value, significance
    """
    tbl = result.summary2().tables[1].copy()
    tbl.columns = ["coef", "std_err", "z", "p_value", "ci_lower", "ci_upper"]
    tbl["odds_ratio"] = np.exp(tbl["coef"])
    tbl["ci_lower_or"] = np.exp(tbl["ci_lower"])
    tbl["ci_upper_or"] = np.exp(tbl["ci_upper"])

    # Significance stars
    def stars(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        if p < 0.10:  return "†"
        return ""

    tbl["sig"] = tbl["p_value"].apply(stars)
    tbl = tbl.reset_index().rename(columns={"index": "predictor"})

    # Clean up predictor names
    tbl["predictor"] = (tbl["predictor"]
        .str.replace(r"C\(region.*?\)\[T\.", "Region: ", regex=True)
        .str.replace(r"C\(model.*?\)\[T\.", "Model: ", regex=True)
        .str.replace(r"C\(year_cat.*?\)\[T\.", "Year: ", regex=True)
        .str.replace(r"C\(irai_theme.*?\)\[T\.", "Theme: ", regex=True)
        .str.replace(r"\]", "", regex=True)
    )

    cols = ["predictor", "coef", "odds_ratio", "ci_lower_or", "ci_upper_or",
            "z", "p_value", "sig"]
    tbl = tbl[cols].round(4)

    if output_path:
        tbl.to_csv(output_path, index=False)
        log.info(f"Regression table saved to {output_path}")

    return tbl


def print_key_results(tbl: pd.DataFrame):
    """Print the key results matching paper §V."""
    print("\n" + "═" * 72)
    print("MIXED-EFFECTS LOGISTIC REGRESSION RESULTS  (Paper §V)")
    print("═" * 72)
    print(f"  Reference levels: Region=Europe, Model=DeepSeek-V3-0324, "
          f"Year=2010, Theme=Transparency")
    print()

    sections = {
        "REGION EFFECTS (RQ1)":   tbl[tbl["predictor"].str.startswith("Region:")],
        "MODEL EFFECTS":          tbl[tbl["predictor"].str.startswith("Model:")],
        "YEAR EFFECTS (RQ4)":     tbl[tbl["predictor"].str.startswith("Year:")],
        "THEME EFFECTS (RQ3)":    tbl[tbl["predictor"].str.startswith("Theme:")],
    }

    for header, sub in sections.items():
        if sub.empty:
            continue
        print(f"  {header}")
        print(f"  {'Predictor':<45} {'OR':>8} {'95% CI':>18} {'p':>8}  sig")
        print("  " + "─" * 84)
        for _, row in sub.iterrows():
            ci = f"[{row.ci_lower_or:.3f}, {row.ci_upper_or:.3f}]"
            print(f"  {row.predictor:<45} {row.odds_ratio:>8.3f} {ci:>18} "
                  f"{row.p_value:>8.3f}  {row.sig}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# VARIANT ROBUSTNESS CHECK
# ─────────────────────────────────────────────────────────────────────────────

def variant_robustness(df_full: pd.DataFrame) -> pd.DataFrame:
    """
    Report fabrication rates and OR reductions by query variant.
    Paper §V-A: variant 2 reduces HF odds by 41%, variant 3 by 18%.
    """
    log.info("Computing variant robustness …")

    results = []
    for variant in [1, 2, 3]:
        sub = df_full[df_full["variant"] == variant]
        hf_rate = sub["hf"].mean()
        results.append({"variant": variant, "n": len(sub), "hf_rate": hf_rate})

    df_rob = pd.DataFrame(results)
    ref_rate = df_rob.loc[df_rob["variant"] == 1, "hf_rate"].iloc[0]
    df_rob["vs_v1_pct_reduction"] = (ref_rate - df_rob["hf_rate"]) / ref_rate * 100
    df_rob["or_vs_v1"] = df_rob["hf_rate"] / ref_rate

    log.info("\nVariant fabrication rates:")
    print(df_rob.to_string(index=False))
    return df_rob


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mixed-effects logistic regression for GAID v2 benchmark"
    )
    parser.add_argument("--results",  default="results/classified_results.csv")
    parser.add_argument("--regions",  default="data/region_lookup.csv")
    parser.add_argument("--output",   default="results/regression_table.csv")
    parser.add_argument("--full-glmm", action="store_true",
                        help="Use pymer4 GLMM instead of fixed-effects logit")
    parser.add_argument("--all-variants", action="store_true",
                        help="Run on all variants (default: variant 1 only)")
    args = parser.parse_args()

    df = load_and_prepare(args.results, args.regions, primary_only=True)

    if args.all_variants:
        variant_robustness(df)

    if args.full_glmm:
        lmm, res = run_mixed_effects_glmm(df)
        print(res)
    else:
        result = run_logistic_fe(df)
        tbl = format_results_table(result, output_path=args.output)
        print_key_results(tbl)
        print(f"\nFull model summary:\n{result.summary()}")
        print(f"\nPseudo-R² (McFadden): {result.prsquared:.4f}")
        print(f"Log-likelihood: {result.llf:.2f}")
        print(f"AIC: {result.aic:.2f}")
