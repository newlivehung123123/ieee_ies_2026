"""
figures.py
==========
Reproduces all five paper figures from classified_results.csv.

Paper figures
-------------
  Fig. 1  VF Rate by Model and Global North/South (2010–2023)
  Fig. 2  DiD: Developer Origin × Country Income  (2010–2023)
  Fig. 3  VF Rate (%) by IEEE IRAI Theme and Model (2010–2023)
  Fig. 4  PCA Scree Plot — Country × Indicator VF Matrix
  Fig. 5  Country PCA Scores — PC1 vs PC2

Usage
-----
    # Reproduce all five figures:
    python figures.py --results results/classified_results.csv --out figures/

    # Single figure:
    python figures.py --results results/classified_results.csv --fig 1
"""

import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

PRIMARY_YEARS   = {2010, 2013, 2016, 2019}
SECONDARY_YEARS = {2022, 2023}
ALL_YEARS       = PRIMARY_YEARS | SECONDARY_YEARS

MODEL_ORDER = ["DeepSeek-V3-0324", "Llama 4 Maverick", "Mistral Large 3", "Qwen3-235B-A22B"]
MODEL_SHORT = {
    "DeepSeek-V3-0324": "DeepSeek\nV3-0324",
    "Llama 4 Maverick":  "Llama 4\nMaverick",
    "Mistral Large 3":   "Mistral\nLarge 3",
    "Qwen3-235B-A22B":   "Qwen3\n235B-A22B",
}

DEVELOPER_ORIGIN = {
    "Llama 4 Maverick":  "Western",
    "Mistral Large 3":   "Western",
    "Qwen3-235B-A22B":   "Chinese",
    "DeepSeek-V3-0324":  "Chinese",
}

THEME_ORDER = [
    "Regulation", "Accountability", "Ethics", "Transparency",
    "Fairness", "Adoption", "Security", "Safety",
]

GN_COLOR = "#1f77b4"   # solid blue  — Global North
GS_COLOR = "#aec7e8"   # hatched     — Global South
GS_HATCH = "///"


# ─────────────────────────────────────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load(results_path: str, region_path: str,
          years=ALL_YEARS, variant: int = 1) -> pd.DataFrame:
    df = pd.read_csv(results_path, low_memory=False)
    df = df[df["year"].isin(years) & (df["variant"] == variant)].copy()
    df["vf"] = (df["classification"] == "VF").astype(float)
    df["hf"] = (df["classification"] == "HF").astype(float)

    regions = pd.read_csv(region_path)
    df = df.merge(regions[["country", "un_region", "global_north_south"]],
                  on="country", how="left")
    df["developer_origin"] = df["model"].map(DEVELOPER_ORIGIN)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — VF Rate by Model × Global North/South
# ─────────────────────────────────────────────────────────────────────────────

def fig1_vf_by_model_ns(df: pd.DataFrame, out_path: str):
    """
    Grouped bar chart: VF rate per model, split GN vs GS.
    Paper §V-B values:
      Mistral: GN=29.7%, GS=44.5%
      DeepSeek: GN=25.1%, GS=37.2%
      Qwen3: GN=19.6%, GS=30.6%
      Llama4: GN=11.0%, GS=15.9%
    """
    grp = (df.groupby(["model", "global_north_south"])["vf"]
             .mean()
             .reset_index()
             .rename(columns={"vf": "vf_rate"}))
    grp["vf_pct"] = grp["vf_rate"] * 100

    x     = np.arange(len(MODEL_ORDER))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 6))

    for i, geo in enumerate(["Global North", "Global South"]):
        sub = grp[grp["global_north_south"] == geo]
        vals = [sub[sub["model"] == m]["vf_pct"].values[0]
                if m in sub["model"].values else 0 for m in MODEL_ORDER]
        offset = (i - 0.5) * width
        color  = GN_COLOR if geo == "Global North" else "white"
        hatch  = None if geo == "Global North" else GS_HATCH
        edgecolor = GN_COLOR if geo == "Global North" else GS_COLOR
        bars = ax.bar(x + offset, vals, width, label=geo,
                      color=color, hatch=hatch,
                      edgecolor=edgecolor, linewidth=1.2)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in MODEL_ORDER], fontsize=9)
    ax.set_xlabel("Model", fontsize=10)
    ax.set_ylabel("Verified Factual Rate (%)", fontsize=10)
    ax.set_ylim(0, 62)
    ax.legend(fontsize=9)
    ax.set_title("Fig. 1.  VF Rate by Model and Global North/South (2010–2023)", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Fig. 1 saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — DiD Line Plot
# ─────────────────────────────────────────────────────────────────────────────

def fig2_did(df: pd.DataFrame, out_path: str):
    """
    Line plot: HF rate by developer origin for Global North vs Global South.
    Paper §V-D: Western GN=78.8%, GS=68.8%; Chinese GN=77.2%, GS=65.2%.
    """
    grp = (df.groupby(["developer_origin", "global_north_south"])["hf"]
             .mean()
             .reset_index()
             .rename(columns={"hf": "hf_rate"}))
    grp["hf_pct"] = grp["hf_rate"] * 100

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    styles = {"Western": ("o-", "#1f77b4"), "Chinese": ("s--", "#ff7f0e")}

    for origin, (style, color) in styles.items():
        sub = grp[grp["developer_origin"] == origin].sort_values("global_north_south")
        xs  = [0, 1]  # Global North=0, Global South=1
        ys  = []
        for geo in ["Global North", "Global South"]:
            row = sub[sub["global_north_south"] == geo]
            ys.append(row["hf_pct"].values[0] if len(row) else np.nan)
        label = f"{'Western':<8} (Llama4 + Mistral)" if origin == "Western" \
                else f"Chinese  (Qwen3 + DeepSeek)"
        ax.plot(xs, ys, style, color=color, linewidth=2, markersize=8, label=label)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f}%", (x, y),
                        textcoords="offset points", xytext=(8, 4), fontsize=9)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Global North", "Global South"], fontsize=10)
    ax.set_ylabel("HF Rate (%)", fontsize=10)
    ax.set_ylim(55, 97)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title(
        "Fig. 2.  DiD: Developer Origin × Country Income (2010–2023).\n"
        "Raw DiD = +2.07 pp",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Fig. 2 saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — VF Rate by Theme × Model (heatmap)
# ─────────────────────────────────────────────────────────────────────────────

def fig3_vf_by_theme_model(df: pd.DataFrame, out_path: str):
    """
    Heatmap / grouped bar: VF rate by IEEE IRAI theme and model.
    """
    grp = (df.groupby(["irai_theme", "model"])["vf"]
             .mean()
             .reset_index()
             .rename(columns={"vf": "vf_pct"}))
    grp["vf_pct"] *= 100

    themes  = [t for t in THEME_ORDER if t in grp["irai_theme"].unique()]
    models  = MODEL_ORDER
    n_themes = len(themes)
    n_models = len(models)

    x      = np.arange(n_themes)
    width  = 0.18
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(12, 6))

    for mi, model in enumerate(models):
        vals = []
        for theme in themes:
            row = grp[(grp["model"] == model) & (grp["irai_theme"] == theme)]
            vals.append(row["vf_pct"].values[0] if len(row) else 0)
        offset = (mi - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=MODEL_SHORT[model].replace("\n", " "),
                      color=colors[mi], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{int(round(val))}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(themes, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("VF Rate (%)", fontsize=10)
    ax.set_ylim(0, 72)
    ax.legend(title="Model", fontsize=8, title_fontsize=9,
              loc="upper right", ncol=2)
    ax.set_title("Fig. 3.  VF Rate (%) by IEEE IRAI Theme and Model (2010–2023)", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"Fig. 3 saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 & 5 — delegate to pca.py
# ─────────────────────────────────────────────────────────────────────────────

def fig4_fig5_pca(results_path: str, region_path: str,
                  scree_out: str, biplot_out: str):
    """Call pca.py functions to regenerate Figs 4 and 5."""
    from src.analysis.pca import build_vf_matrix, run_pca, plot_scree, plot_biplot
    matrix, region_df = build_vf_matrix(results_path, region_path)
    result = run_pca(matrix)
    plot_scree(result, output_path=scree_out)
    plot_biplot(result, region_df, output_path=biplot_out)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce all paper figures")
    parser.add_argument("--results", default="results/classified_results.csv")
    parser.add_argument("--regions", default="data/region_lookup.csv")
    parser.add_argument("--out",     default="figures/")
    parser.add_argument("--fig",     default="all",
                        help="Which figure(s) to build: 1,2,3,4,5 or 'all'")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    figs = (set(args.fig.split(","))
            if args.fig != "all"
            else {"1", "2", "3", "4", "5"})

    if "1" in figs or "2" in figs or "3" in figs:
        df = _load(args.results, args.regions)

    if "1" in figs:
        fig1_vf_by_model_ns(df, str(out / "fig1_vf_by_model_ns.png"))
    if "2" in figs:
        fig2_did(df, str(out / "fig2_did.png"))
    if "3" in figs:
        fig3_vf_by_theme_model(df, str(out / "fig3_vf_by_theme_model.png"))
    if "4" in figs or "5" in figs:
        fig4_fig5_pca(
            args.results, args.regions,
            scree_out  = str(out / "fig4_pca_scree.png"),
            biplot_out = str(out / "fig5_pca_biplot.png"),
        )

    log.info("All requested figures generated.")
