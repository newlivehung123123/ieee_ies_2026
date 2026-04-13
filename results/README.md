# Results Directory

This directory holds the raw API responses and the classified output produced by the evaluation pipeline.

---

## Pipeline Overview

```
data/GAID_queries_all_variants.csv
        │
        ▼  eval_runner.py
results/results_<model>_variant<N>.csv      ← one file per model × variant
        │
        ▼  classifier.py
results/classified_results.csv              ← merged, classified master file
```

---

## File: `results_<model>_variant<N>.csv`

Raw API responses, one row per query submission.

| Column | Type | Description |
|---|---|---|
| `obs_id` | string | Unique observation identifier (matches query file) |
| `model` | string | Model name (e.g. `Llama 4 Maverick`) |
| `variant` | int | Query variant: 1, 2, or 3 |
| `query` | string | Exact prompt submitted |
| `response` | string | Full raw model response |
| `timestamp` | datetime | UTC time of API call |
| `tokens_prompt` | int | Prompt token count (if available) |
| `tokens_completion` | int | Completion token count (if available) |
| `repeat_flag` | bool | `True` if this is one of the 5 % double-query consistency observations |

**Naming convention:**

| Model | File prefix |
|---|---|
| Llama 4 Maverick | `results_llama4_` |
| Mistral Large 3 | `results_mistral_` |
| Qwen3-235B-A22B | `results_qwen3_` |
| DeepSeek-V3-0324 | `results_deepseek_` |

---

## File: `classified_results.csv`

Master file produced by `classifier.py --merge`.  
Contains every raw result row with the classification appended and all
query-file metadata joined in.

| Column | Type | Description |
|---|---|---|
| `obs_id` | string | Unique observation identifier |
| `model` | string | Model name |
| `variant` | int | Query variant (1/2/3) |
| `code` | string | Indicator code (e.g. `AI_Pubs`) |
| `irai_theme` | string | IEEE IRAI 2026 thematic dimension |
| `country` | string | Country name |
| `iso3` | string | ISO 3166-1 alpha-3 code |
| `year` | int | Evaluation year |
| `verified_value` | float | Ground-truth value from GAID v2 |
| `indicator_type` | string | `continuous` or `binary` |
| `query` | string | Exact prompt submitted |
| `response` | string | Full raw model response |
| `extracted_value` | float | Numeric value extracted by the classifier (NaN if none) |
| `classification` | string | One of: `VF`, `HF`, `HR`, `QH`, `MF` (see below) |
| `classification_note` | string | Free-text reason (e.g. `within_10pct`, `explicit_refusal`) |
| `repeat_flag` | bool | `True` for the 5 % double-query observations |
| `timestamp` | datetime | UTC time of API call |

### Classification Categories

| Code | Label | Definition |
|---|---|---|
| `VF` | Verified Factual | Extracted numeric value is within ±10 % of the GAID v2 ground truth |
| `HF` | Hallucinated Factual | Model provides a numeric value outside the ±10 % tolerance |
| `HR` | Honest Refusal | Model explicitly states it does not know or declines to answer |
| `QH` | Qualitative Hedge | Model gives a qualitative or range answer without a specific number |
| `MF` | Misattribution | Model provides a value from the wrong country, year, or indicator |

**Binary indicator rule:** for `Nat_AI_Strat` and `AI_Bills`, VF requires the extracted 0/1 value to match exactly.

**Special cases:**
- Coursera scores (`Coursera_Biz`, `Coursera_Tech`) and `WB_GTMI` are reported on a 0–100 scale in GAID v2 but some model responses use a 0–1 scale.  The classifier rescales 0–1 responses before applying the ±10 % tolerance.
- `Train_Compute` and `Model_Params` span many orders of magnitude; the classifier normalises to base-10 log before the ±10 % test.

---

## Consistency Check

5 % of observations (149 rows) were submitted twice in independent API calls to measure within-run consistency.  The `repeat_flag` column marks these rows.  Classifier agreement on matched pairs was **90.1 %**.

To reproduce the consistency check:

```bash
python src/evaluation/consistency_check.py \
    --results results/classified_results.csv
```

---

## Reproducing This File

```bash
# 1. Run the evaluation (requires API keys in .env)
python src/evaluation/eval_runner.py --model all --variant all --workers 5

# 2. Classify all responses and merge into one file
python src/evaluation/classifier.py \
    --input  "results/results_*.csv" \
    --queries data/GAID_queries_all_variants.csv \
    --output results/classified_results.csv

# 3. (Optional) Verify row counts
python - <<'EOF'
import pandas as pd
df = pd.read_csv("results/classified_results.csv")
print(df.groupby(["model","variant"]).size().unstack())
# Expected: 4 models × 3 variants × 2,990 obs = 35,880 rows
EOF
```

---

## Large-File Notice

`classified_results.csv` (~35 k rows × 20 columns) is committed to the repository for full reproducibility.  If it exceeds GitHub's 100 MB file-size limit after updates, upload to [Harvard Dataverse](https://dataverse.harvard.edu) or [Zenodo](https://zenodo.org) and update the DOI in `README.md`.
