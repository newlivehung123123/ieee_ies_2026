# 🤖 Benchmarking Open-Weight Foundation Models for Global AI Technical Governance

**IEEE IES Generative AI Challenge 2026 — Open Replication Repository**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Data: Harvard Dataverse](https://img.shields.io/badge/data-Harvard%20Dataverse-red.svg)](https://dataverse.harvard.edu)
[![Paper: arXiv](https://img.shields.io/badge/paper-arXiv-green.svg)](https://arxiv.org)

---

## 📌 Overview

This repository contains the complete, open-access replication package for:

> **Hung, J. (2026). Benchmarking Open-Weight Foundation Models for Global AI Technical Governance.** *IEEE IES Generative AI Challenge 2026.*

The study evaluates four leading open-weight large language models on their ability to accurately recall structured, numeric AI governance indicators across 177 countries and six evaluation years (2010–2023), using the **Global AI Dataset v2 (GAID v2)** as ground truth.

It identifies systematic geographic bias — Global South countries are disproportionately misrepresented — and shows that this bias is consistent across both Western and Chinese model families, with a statistically significant Difference-in-Differences estimate of **+2.07 percentage points**.

---

## 🏆 Key Results

| Model | 🌍 VF Rate — Global North | 🌏 VF Rate — Global South | 📉 Gap |
|---|---|---|---|
| Llama 4 Maverick | 11.0 % | 15.9 % | +4.9 pp |
| Mistral Large 3 | 29.7 % | 44.5 % | +14.8 pp |
| DeepSeek-V3-0324 | 25.1 % | 37.2 % | +12.1 pp |
| Qwen3-235B-A22B | 19.6 % | 30.6 % | +11.0 pp |

> ✅ **VF = Verified Factual** — model response within ±10 % of GAID v2 ground truth. Higher = better recall accuracy.

---

## 📁 Repository Structure

```
ieee_ies_2026/
├── 📄 README.md                          ← This file
├── ⚖️  LICENSE                            ← MIT
├── 📦 requirements.txt                   ← Python dependencies
├── 🚫 .gitignore
│
├── 📰 paper/
│   └── Jason_Hung_IEEE_condensed_v4.pdf  ← Published conference paper
│
├── 🗄️  data/
│   ├── README.md                         ← Data dictionary & GAID v2 docs
│   ├── GAID_queries_all_variants.csv     ← 8,970 structured prompts
│   └── region_lookup.csv                 ← Country → UN region + GN/GS flag
│
├── 📊 results/
│   ├── README.md                         ← CSV schema & classification codes
│   ├── classified_results.csv            ← Full classified output (35,880 rows)
│   ├── results_llama4.csv
│   ├── results_mistral.csv
│   ├── results_qwen3.csv
│   └── results_deepseek.csv
│
├── 🖼️  figures/
│   ├── fig1_vf_by_model_ns.png           ← Fig 1: VF Rate by Model × Geography
│   ├── fig2_did.png                      ← Fig 2: DiD Line Plot
│   ├── fig3_vf_by_theme_model.png        ← Fig 3: VF Rate by IRAI Theme × Model
│   ├── fig4_pca_scree.png                ← Fig 4: PCA Scree Plot
│   └── fig5_pca_biplot.png               ← Fig 5: Country PCA Scores
│
└── 🐍 src/
    ├── evaluation/
    │   ├── eval_runner.py                ← Submit queries to four model APIs
    │   ├── classifier.py                 ← Five-category rule-based classifier
    │   └── consistency_check.py         ← Within-run 5 % repeat-query analysis
    ├── data/
    │   ├── region_lookup.py              ← Country → UN M49 region mapping
    │   └── build_queries.py             ← Regenerate query CSV from GAID v2
    └── analysis/
        ├── regression.py                 ← Mixed-effects logistic regression
        ├── did.py                        ← Difference-in-Differences estimation
        ├── pca.py                        ← PCA of country × indicator VF matrix
        └── figures.py                   ← Reproduce all five paper figures
```

---

## ⚡ Quick Start

### 1️⃣ Clone the repository

```bash
git clone https://github.com/newlivehung123123/ieee_ies_2026.git
cd ieee_ies_2026
```

### 2️⃣ Set up a Python environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires **Python 3.10 or later**.

### 3️⃣ Reproduce all figures (no API keys needed)

```bash
python src/analysis/figures.py \
    --results results/classified_results.csv \
    --regions data/region_lookup.csv \
    --out figures/
```

### 4️⃣ Re-run the statistical analyses

```bash
# Mixed-effects logistic regression (Table II)
python src/analysis/regression.py \
    --results results/classified_results.csv \
    --regions data/region_lookup.csv

# Difference-in-Differences (Table III)
python src/analysis/did.py \
    --results results/classified_results.csv \
    --regions data/region_lookup.csv

# PCA (Section V-E)
python src/analysis/pca.py \
    --results results/classified_results.csv \
    --regions data/region_lookup.csv
```

---

## 🔬 Study Design

### 🎯 Research Questions

| # | Question |
|---|---|
| RQ1 | How accurately do open-weight LLMs recall structured AI governance indicators? |
| RQ2 | Does recall accuracy differ systematically between Global North and Global South countries? |
| RQ3 | Does the geographic gap interact with developer origin (Western vs. Chinese models)? |
| RQ4 | Which IEEE IRAI 2026 thematic dimensions show the largest accuracy deficits? |

### 🗃️ Ground-Truth Dataset: GAID v2

| Property | Value |
|---|---|
| 📦 Total rows | 259,546 |
| 📐 Indicators | 24,453 |
| 🌍 Countries | 227 |
| 📅 Years covered | 1998–2025 |
| 📚 Data sources | 11 curated international sources |

**Citation:**
> Hung, J. (2026). *Global AI Dataset v2 (GAID v2)*. Harvard Dataverse. [https://doi.org/[doi]](https://doi.org/10.7910/DVN/PUMGYU)

### 🤖 Models Evaluated

| Model | Developer | 🌐 Origin | API Provider |
|---|---|---|---|
| Llama 4 Maverick | Meta | Western | OpenRouter |
| Mistral Large 3 | Mistral AI | Western | Mistral AI |
| Qwen3-235B-A22B | Alibaba | Chinese | Together AI |
| DeepSeek-V3-0324 | DeepSeek | Chinese | Together AI |

All models queried at **temperature = 0** for deterministic output.

### 📝 Query Design

Each of 2,990 (country, indicator, year) observations was submitted using **three query variants**:

| Variant | Type | Description |
|---|---|---|
| 1 | 🎯 Direct numeric | `"What was [indicator] for [country] in [year]? Please provide a specific numeric value."` |
| 2 | 🔀 Comparative | Embeds the target in a regional comparison context |
| 3 | 🏛️ Contextual | Frames the query within an AI governance analysis scenario |

**Total:** 8,970 prompts × 4 models = **35,880 API calls**

A random 5 % of observations were double-submitted for consistency measurement. Classifier agreement: **90.1 %**.

### 📅 Primary Analysis Window

Primary: **2010, 2013, 2016, 2019** (within confirmed training windows of all four models)
Robustness check: **2022–2023**

---

## 🏷️ Response Classification

| Code | Label | Rule |
|---|---|---|
| ✅ `VF` | Verified Factual | Extracted numeric value within ±10 % of GAID v2 ground truth |
| ❌ `HF` | Hallucinated Factual | Numeric value provided but outside ±10 % tolerance |
| 🤐 `HR` | Honest Refusal | Model explicitly declines or states it does not know |
| 🌫️ `QH` | Qualitative Hedge | Qualitative or range answer without a specific number |
| 🔀 `MF` | Misattribution | Value from the wrong country, year, or indicator |

---

## 📈 Statistical Methods

### Mixed-Effects Logistic Regression (§V-C)

```
logit(P[HF_ijkmt]) = β₀ + β₁ Region_j + β₂ Model_k
                       + β₃ Year_t + β₄ Theme_m
                       + u_j + ε_ijkmt
```

Key result: **Africa OR = 2.41** (95 % CI [2.18, 2.67], p < 0.001)

### Difference-in-Differences (§V-D)

```
DiD = (C_GN − C_GS) − (W_GN − W_GS) = +2.07 pp
```

Chinese models show a 2.07 pp larger North–South HF gap than Western models.

### PCA (§V-E)

| Component | Variance Explained |
|---|---|
| PC1 | 11.3 % |
| PC2 | 10.4 % |
| n₉₀ | 15 components |
| Cumulative (15 PCs) | 92.9 % |

---

## 🚀 Running the Full Evaluation

### 🔑 API Keys

Create a `.env` file in the repo root (never commit this):

```bash
OPENROUTER_API_KEY=sk-or-...      # Llama 4 Maverick
TOGETHER_API_KEY=...               # Qwen3, DeepSeek
MISTRAL_API_KEY=...                # Mistral Large 3
```

### ▶️ Run

```bash
# All models, all variants
python src/evaluation/eval_runner.py --model all --variant all --workers 5

# Single model
python src/evaluation/eval_runner.py --model llama4 --variant all --workers 5

# Resume interrupted run
python src/evaluation/eval_runner.py --model deepseek --variant 1 --resume
```

### 🏷️ Classify responses

```bash
python src/evaluation/classifier.py \
    --input  "results/results_*.csv" \
    --queries data/GAID_queries_all_variants.csv \
    --output results/classified_results.csv
```

### ⏱️ Estimated Cost and Runtime

| Model | Provider | 💰 Cost (~8,970 calls) | ⏳ Time (5 workers) |
|---|---|---|---|
| Llama 4 Maverick | OpenRouter | ~$4–6 | ~3 h |
| Mistral Large 3 | Mistral AI | ~$8–12 | ~4 h |
| Qwen3-235B-A22B | Together AI | ~$5–8 | ~3 h |
| DeepSeek-V3-0324 | Together AI | ~$4–6 | ~3 h |

---

## ⚠️ Limitations

- **Training-data contamination**: GAID v2 values may be present in model training corpora; VF classifications are not evidence of genuine reasoning.
- **Single prompt per variant**: Prompt sensitivity beyond the three variants is not assessed.
- **Rule-based classifier**: The ±10 % tolerance is a pragmatic choice; edge cases near the boundary may be misclassified.
- **Model version sensitivity**: Results are tied to specific model releases (April 2026).
- **English-only queries**: Non-English model capabilities are not tested.

---

## 📖 Citation

```bibtex
@inproceedings{hung2026benchmarking,
  title     = {Benchmarking Open-Weight Foundation Models for Global AI Technical Governance},
  author    = {Hung, Jason},
  booktitle = {IEEE IES Generative AI Challenge 2026},
  year      = {2026}
}

@data{hung2026gaidv2,
  author    = {Hung, Jason},
  title     = {{Global AI Dataset v2 (GAID v2)}},
  publisher = {Harvard Dataverse},
  year      = {2026},
  doi       = {[doi]}
}
```

---

## ⚖️ License

Released under the [MIT License](LICENSE). Data sourced from GAID v2, Stanford AI Index, WIPO, World Bank, Epoch AI, Coursera, and OECD.ai remain subject to their respective licenses.

---

## 📬 Contact

**Jason Hung** — jasehung123@gmail.com
GitHub: [newlivehung123123](https://github.com/newlivehung123123)
