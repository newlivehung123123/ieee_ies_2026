# Data Directory

This directory contains the query file and supporting lookup tables used in the GAID v2 benchmark evaluation.

---

## Files

### `GAID_queries_all_variants.csv`

The complete set of **8,970 structured prompts** (2,990 observations × 3 query variants) submitted to each model.

| Column | Description |
|---|---|
| `obs_id` | Unique observation identifier |
| `code` | Indicator code (e.g. `AI_Pubs`, `Train_Compute`) |
| `irai_theme` | IEEE IRAI 2026 thematic dimension |
| `country` | Country name (as in GAID v2) |
| `iso3` | ISO 3166-1 alpha-3 code |
| `year` | Evaluation year (2010, 2013, 2016, 2019, 2022, 2023) |
| `verified_value` | Ground-truth value from GAID v2 |
| `data_source` | Primary data source (e.g. Stanford AI Index / Scopus) |
| `indicator_name` | Human-readable indicator name |
| `indicator_type` | `continuous` or `binary` |
| `variant` | Query variant: 1 (direct), 2 (comparative), 3 (contextual) |
| `variant_type` | `direct_numeric`, `comparative`, or `contextual` |
| `query` | The exact prompt submitted to each model |

**Query variants:**
- **Variant 1 (direct numeric):** `"What was [indicator] for [country] in [year]? Please provide a specific numeric value."`
- **Variant 2 (comparative):** Embeds the target in a regional comparison context.
- **Variant 3 (contextual):** Frames the query within an AI governance analysis scenario.

---

### `region_lookup.csv`

Maps every country to its UN M49 macro-region and Global North/South classification.

| Column | Description |
|---|---|
| `country` | Country name (matches GAID v2) |
| `iso3` | ISO 3166-1 alpha-3 code |
| `un_region` | UN M49 region: Africa, Americas, Asia, Europe, Oceania |
| `global_north_south` | `Global North` or `Global South` |

**Global North definition** (UN/OECD convention): Europe + North America + Australia + New Zealand + Japan + South Korea + Israel + Singapore + Taiwan + Hong Kong + Macau.

---

## Ground-Truth Source: GAID v2

The verified values come from the **Global AI Dataset v2 (GAID v2)**, published on Harvard Dataverse in January 2026.

| Property | Value |
|---|---|
| Total rows | 259,546 |
| Indicators | 24,453 |
| Countries | 227 |
| Years covered | 1998–2025 |
| Data sources | 11 curated international sources |

**Harvard Dataverse citation:**
> Hung, J. (2026). *Global AI Dataset v2 (GAID v2)*. Harvard Dataverse. https://doi.org/[doi]

**The 18 indicators used in this study:**

| IRAI Theme | Code | Indicator Name | Source |
|---|---|---|---|
| Transparency | AI_Pubs | Number of AI Publications: All | Stanford AI Index / Scopus |
| Transparency | WIPO_Patents | Total AI-Related Patent Publications | WIPO |
| Transparency | FWCI | Field Weighted Citation Impact: All | Stanford AI Index / Scopus |
| Fairness | CS_Grad_F | Share of CS Bachelor's Graduates Who Are Female | Stanford AI Index / OECD |
| Fairness | CS_PhD_F | Share of CS Doctoral Graduates Who Are Female | Stanford AI Index / OECD |
| Adoption | WB_GTMI | World Bank GovTech Overall Maturity Index | World Bank GTMI 2022 |
| Adoption | Coursera_Biz | Coursera Business Skills Proficiency Score | Coursera Global Skills Report |
| Adoption | Coursera_Tech | Coursera Technology Skills Proficiency Score | Coursera Global Skills Report |
| Adoption | BigData_Biz | Proportion of Businesses Performing Big Data Analysis | OECD.ai |
| Regulation | Nat_AI_Strat | National AI Strategy Published (binary) | Stanford AI Index |
| Regulation | AI_Bills | Number of AI-Related Bills Enacted into Law | Stanford AI Index |
| Accountability | AI_Legis | AI References in National Legislative Proceedings | Stanford AI Index |
| Safety | Train_Compute | Total Training Compute for AI Models Released (FLOP) | Epoch AI |
| Safety | Model_Params | Total Parameters of AI Models Released | Epoch AI |
| Security | ICT_Sec_All | Proportion of Businesses That Experienced ICT Security Breaches | OECD.ai |
| Security | ICT_Sec_ICT | ICT Sector Businesses That Experienced ICT Security Breaches | OECD.ai |
| Ethics | AI_Benefits | Survey Respondents: AI Offers More Benefits Than Drawbacks | Stanford AI Index / Ipsos |
| Ethics | AI_Nervous | Survey Respondents: AI Products Make Them Nervous | Stanford AI Index / Ipsos |

**Indicator selection criteria (three-stage screening):**
1. Thematic mapping to the eight IEEE IRAI 2026 dimensions.
2. Coverage: verified values for ≥ 10 countries in ≥ 1 evaluation year.
3. Redundancy: Spearman r < 0.90 with any already-selected indicator.
