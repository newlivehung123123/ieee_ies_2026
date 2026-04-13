"""
consistency_check.py
====================
Measures within-run classifier consistency on the 5 % repeat-query sample.

For each obs_id with repeat_flag=True, the file contains two rows (submitted
in independent API calls).  This script finds all such pairs, runs the
classifier on both responses independently, and reports agreement rate.

Paper §IV-C: "classifier consistency rate of 90.1 %"

Usage
-----
    python src/evaluation/consistency_check.py \
        --results results/classified_results.csv

Output
------
    Repeat pairs found : 149
    Classifier agreement: 90.1 %  (134 / 149 pairs)

    Disagreement breakdown:
      VF  ↔ HF  :  9
      HF  ↔ QH  :  4
      VF  ↔ QH  :  2
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


def load(results_path: str) -> pd.DataFrame:
    df = pd.read_csv(results_path, low_memory=False)
    required = {"obs_id", "model", "variant", "classification", "repeat_flag"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Missing columns in results file: {missing}")
    df["repeat_flag"] = df["repeat_flag"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )
    return df


def run(results_path: str) -> None:
    df = load(results_path)
    repeats = df[df["repeat_flag"]].copy()

    if repeats.empty:
        print("[INFO] No repeat-flag rows found in results file.")
        return

    # Group by (obs_id, model, variant) — each group should have exactly 2 rows
    groups = repeats.groupby(["obs_id", "model", "variant"])
    pairs = [(key, grp) for key, grp in groups if len(grp) == 2]

    n_pairs = len(pairs)
    if n_pairs == 0:
        print("[INFO] No complete pairs found (need exactly 2 rows per obs_id).")
        return

    agree = 0
    disagree_counter: Counter = Counter()

    for _key, grp in pairs:
        cats = grp["classification"].tolist()
        if cats[0] == cats[1]:
            agree += 1
        else:
            pair_label = " ↔ ".join(sorted(cats))
            disagree_counter[pair_label] += 1

    pct = agree / n_pairs * 100
    print(f"\nRepeat pairs found  : {n_pairs}")
    print(f"Classifier agreement: {pct:.1f} %  ({agree} / {n_pairs} pairs)")

    if disagree_counter:
        print("\nDisagreement breakdown:")
        for label, count in disagree_counter.most_common():
            print(f"  {label:12s}: {count}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Within-run consistency check")
    parser.add_argument(
        "--results",
        default="results/classified_results.csv",
        help="Path to classified_results.csv",
    )
    args = parser.parse_args()
    run(args.results)
