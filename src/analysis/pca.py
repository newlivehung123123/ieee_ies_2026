"""
pca.py
======
Principal Component Analysis of the country × indicator VF-rate matrix.

Paper §V-F:
    PCA is applied to the 177-country × 18-indicator VF rate matrix to assess
    the dimensionality of geographic bias.

    PC1: 11.3% variance explained (contrasts Regulation+Safety vs Ethics+Adoption)
    PC2: 10.4% variance explained (mixed Accountability, Security, Transparency, Ethics)
    n90: 15 components required to explain 90% of variance  (cumulative: 92.9% at PC15)
    No single component separates Global North from Global South.

Key loadings (PC1):
    AI_Bills:      +0.373  (Regulation)
    Train_Compute: +0.348  (Safety)
    AI_Nervous:    −0.370  (Ethics)
    WB_GTMI:       −0.328  (Adoption)

Key loadings (PC2):
    AI_Legis:      +0.379  (Accountability)

Usage
-----
    python pca.py --results results/classified_results.csv
    python pca.py --results results/classified_results.csv --biplot figures/fig5_pca_biplot.png
"""

import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

PRIMARY_YEARS = {2010, 2013, 2016, 2019}

# IRAI theme mapping for axis labels and colour coding
THEME_MAP = {
    "AI_Pubs":        "Transparency",
    "WIPO_Patents":   "Transparency",
    "FWCI":           "Transparency",
    "CS_Grad_F":      "Fairness",
    "CS_PhD_F":       "Fairness",
    "WB_GTMI":        "Adoption",
    "Coursera_Biz":   "Adoption",
    "Coursera_Tech":  "Adoption",
    "BigData_Biz":    "Adoption",
    "Nat_AI_Strat":   "Regulation",
    "AI_Bills":       "Regulation",
    "AI_Legis":       "Accountability",
    "Train_Compute":  "Safety",
    "Model_Params":   "Safety",
    "ICT_Sec_All":    "Security",
    "ICT_Sec_ICT":    "Security",
    "AI_Benefits":    "Ethics",
    "AI_Nervous":     "Ethics",
}

THEME_COLORS = {
    "Transparency":   "#1f77b4",
    "Fairness":       "#ff7f0e",
    "Adoption":       "#2ca02c",
    "Regulation":     "#d62728",
    "Accountability": "#9467bd",
    "Safety":         "#8c564b",
    "Security":       "#e377c2",
    "Ethics":         "#7f7f7f",
}


# ─────────────────────────────────────────────────────────────────────────────
# BUILD COUNTRY × INDICATOR VF-RATE MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def build_vf_matrix(results_path: str,
                    region_path: str = "data/region_lookup.csv",
                    min_obs: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the (n_countries × n_indicators) matrix of mean VF rates.

    Parameters
    ----------
    results_path : str
    region_path  : str
    min_obs      : int   minimum observations per cell; cells with fewer are NaN

    Returns
    -------
    (matrix_df, region_df)
        matrix_df  : DataFrame [country × indicator], VF rates, NaN for sparse cells
        region_df  : DataFrame [country, iso3, un_region, global_north_south]
    """
    log.info(f"Loading results …")
    df = pd.read_csv(results_path, low_memory=False)
    df = df[df["year"].isin(PRIMARY_YEARS) & (df["variant"] == 1)].copy()
    df["vf"] = (df["classification"] == "VF").astype(int)

    # Average across models and years per (country, indicator)
    cell = df.groupby(["country", "code"]).agg(
        vf_rate=("vf", "mean"),
        n=("vf", "count"),
    ).reset_index()

    # Drop sparse cells
    cell.loc[cell["n"] < min_obs, "vf_rate"] = np.nan

    # Pivot to wide format
    matrix = cell.pivot(index="country", columns="code", values="vf_rate")

    # Drop countries with too many missing indicators (keep ≥ 50% coverage)
    min_indicators = len(matrix.columns) * 0.5
    matrix = matrix.dropna(thresh=int(min_indicators))

    # Fill remaining NaN with column median (conservative imputation)
    matrix = matrix.fillna(matrix.median())

    log.info(f"  VF matrix shape: {matrix.shape}  "
             f"({matrix.shape[0]} countries × {matrix.shape[1]} indicators)")

    # Load region labels
    regions = pd.read_csv(region_path).set_index("country")
    region_df = regions.loc[regions.index.isin(matrix.index)].copy()

    return matrix, region_df


# ─────────────────────────────────────────────────────────────────────────────
# RUN PCA
# ─────────────────────────────────────────────────────────────────────────────

def run_pca(matrix: pd.DataFrame, n_components: int = None) -> dict:
    """
    Run PCA on the country × indicator VF-rate matrix.

    Returns dict with:
        pca          : fitted sklearn PCA object
        scores       : DataFrame [country × PC1..PCk]
        loadings     : DataFrame [indicator × PC1..PCk]
        variance     : Series of explained variance ratios
        cumulative   : Series of cumulative explained variance
        n90          : number of components to reach 90% variance
    """
    n_components = n_components or min(matrix.shape)

    # Standardise columns (zero mean, unit variance)
    scaler = StandardScaler()
    X = scaler.fit_transform(matrix.values)

    pca = PCA(n_components=n_components, random_state=42)
    scores_array = pca.fit_transform(X)

    pc_labels = [f"PC{i+1}" for i in range(n_components)]
    scores   = pd.DataFrame(scores_array, index=matrix.index, columns=pc_labels)
    loadings = pd.DataFrame(pca.components_.T,
                            index=matrix.columns,
                            columns=pc_labels)

    variance   = pd.Series(pca.explained_variance_ratio_ * 100, index=pc_labels)
    cumulative = variance.cumsum()
    n90        = int((cumulative >= 90).idxmax().replace("PC", ""))

    log.info(f"  PC1: {variance['PC1']:.1f}%  PC2: {variance['PC2']:.1f}%  "
             f"n90={n90}  (cumulative at PC{n90}: {cumulative[f'PC{n90}']:.1f}%)")

    return {
        "pca":        pca,
        "scores":     scores,
        "loadings":   loadings,
        "variance":   variance,
        "cumulative": cumulative,
        "n90":        n90,
        "scaler":     scaler,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRINT KEY LOADINGS
# ─────────────────────────────────────────────────────────────────────────────

def print_key_loadings(result: dict, n_pcs: int = 3, top_n: int = 5):
    loadings = result["loadings"]
    variance = result["variance"]

    print("\n" + "═" * 62)
    print("PCA LOADINGS  (Paper §V-F)")
    print("═" * 62)

    for i in range(1, n_pcs + 1):
        pc = f"PC{i}"
        pct = variance[pc]
        col = loadings[pc].sort_values(ascending=False)

        print(f"\n  {pc}  ({pct:.1f}% variance explained)")
        print(f"  {'Indicator':<20} {'Theme':<16} {'Loading':>8}")
        print("  " + "─" * 46)

        top    = col.head(top_n)
        bottom = col.tail(top_n)

        for ind, val in top.items():
            theme = THEME_MAP.get(ind, "—")
            print(f"  {ind:<20} {theme:<16} {val:>+8.3f}  ← positive pole")

        print("  ···")
        for ind, val in bottom.items():
            theme = THEME_MAP.get(ind, "—")
            print(f"  {ind:<20} {theme:<16} {val:>+8.3f}  ← negative pole")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def plot_scree(result: dict, output_path: str = "figures/fig4_pca_scree.png"):
    """Reproduce Fig. 4 — PCA Scree Plot."""
    variance   = result["variance"]
    cumulative = result["cumulative"]
    n90        = result["n90"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(1, len(variance) + 1)

    ax.bar(x, variance.values, color="#4C72B0", alpha=0.7, label="Individual variance")
    ax.plot(x, cumulative.values, "k-o", markersize=5, label="Cumulative variance")
    ax.axhline(90, color="gray", linestyle="--", linewidth=1, label="90% threshold")
    ax.annotate(f"PC{n90}\n({cumulative[f'PC{n90}']:.0f}%)",
                xy=(n90, cumulative[f"PC{n90}"]),
                xytext=(n90 + 0.5, cumulative[f"PC{n90}"] - 7),
                fontsize=9, color="dimgray")

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_xlim(0.5, min(len(variance), 11) + 0.5)
    ax.set_ylim(0, 105)
    ax.legend(framealpha=0.8, fontsize=9)
    ax.set_title("Fig. 4. PCA Scree Plot — Country × Indicator VF Matrix (2010–2023)")
    plt.tight_layout()
    Path(output_path).parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Scree plot saved to {output_path}")


def plot_biplot(result: dict, region_df: pd.DataFrame,
                output_path: str = "figures/fig5_pca_biplot.png"):
    """Reproduce Fig. 5 — Country PCA Scores PC1 vs PC2, coloured by GN/GS."""
    scores  = result["scores"]
    var     = result["variance"]

    merged = scores[["PC1", "PC2"]].join(region_df[["global_north_south"]], how="inner")
    gn = merged[merged["global_north_south"] == "Global North"]
    gs = merged[merged["global_north_south"] == "Global South"]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(gn["PC1"], gn["PC2"], c="#2196F3", s=40, alpha=0.75,
               label="Global North", marker="o")
    ax.scatter(gs["PC1"], gs["PC2"], c="#FF5722", s=40, alpha=0.75,
               label="Global South", marker="^")

    # Label selected countries for orientation
    label_countries = {"United States", "China", "Germany", "India",
                       "Nigeria", "Brazil", "Australia", "Japan",
                       "Russia", "South Korea", "France", "Canada",
                       "Sudan", "Spain", "Italy", "Netherlands",
                       "Czechia", "Estonia", "Turkey"}
    for country, row in merged.iterrows():
        if country in label_countries:
            ax.annotate(country, (row["PC1"], row["PC2"]),
                        fontsize=6.5, alpha=0.8,
                        xytext=(3, 3), textcoords="offset points")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_xlabel(f"PC1 ({var['PC1']:.1f}% variance explained) — General Accuracy Factor",
                  fontsize=10)
    ax.set_ylabel(f"PC2 ({var['PC2']:.1f}% variance explained)", fontsize=10)
    ax.set_title("Fig. 5. Country PCA Scores — PC1 vs PC2 (2010–2023)\n"
                 "PCA of Country Accuracy Profiles (avg across 4 models, primary window)\n"
                 "Colored by Global North/South", fontsize=9)
    ax.legend(fontsize=9, framealpha=0.8)
    plt.tight_layout()
    Path(output_path).parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Biplot saved to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SUPPLEMENTARY: GN vs GS SEPARATION TEST
# ─────────────────────────────────────────────────────────────────────────────

def test_gn_gs_separation(result: dict, region_df: pd.DataFrame):
    """
    Paper §V-F: 'the biplot shows no cluster separating Global North from Global South.'
    Test: Mann-Whitney U test on PC1 scores between GN and GS.
    """
    from scipy import stats

    scores = result["scores"]
    merged = scores[["PC1", "PC2"]].join(region_df[["global_north_south"]], how="inner")

    gn_scores = merged[merged["global_north_south"] == "Global North"]["PC1"].values
    gs_scores = merged[merged["global_north_south"] == "Global South"]["PC1"].values

    stat, p = stats.mannwhitneyu(gn_scores, gs_scores, alternative="two-sided")

    print("\n  GN vs GS SEPARATION TEST (PC1):")
    print(f"    Global North: n={len(gn_scores)}, mean={np.mean(gn_scores):.3f}")
    print(f"    Global South: n={len(gs_scores)}, mean={np.mean(gs_scores):.3f}")
    print(f"    Mann-Whitney U: {stat:.1f},  p={p:.4f}")
    if p < 0.05:
        print("    ✓ Significant GN/GS separation on PC1.")
    else:
        print("    ~ No significant GN/GS separation on PC1  "
              "(consistent with paper findings).")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PCA of country × indicator VF-rate matrix (GAID v2 benchmark)"
    )
    parser.add_argument("--results",    default="results/classified_results.csv")
    parser.add_argument("--regions",    default="data/region_lookup.csv")
    parser.add_argument("--scree-out",  default="figures/fig4_pca_scree.png")
    parser.add_argument("--biplot-out", default="figures/fig5_pca_biplot.png")
    parser.add_argument("--loadings-out", default="results/pca_loadings.csv")
    parser.add_argument("--scores-out",   default="results/pca_scores.csv")
    args = parser.parse_args()

    matrix, region_df = build_vf_matrix(args.results, args.regions)
    result = run_pca(matrix)

    print_key_loadings(result)
    test_gn_gs_separation(result, region_df)

    plot_scree(result, output_path=args.scree_out)
    plot_biplot(result, region_df, output_path=args.biplot_out)

    result["loadings"].round(4).to_csv(args.loadings_out)
    result["scores"].round(4).to_csv(args.scores_out)
    log.info(f"Loadings → {args.loadings_out}")
    log.info(f"Scores   → {args.scores_out}")
