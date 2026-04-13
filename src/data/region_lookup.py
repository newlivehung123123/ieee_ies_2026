"""
region_lookup.py
================
Country → world region mapping for the GAID v2 geographic-bias benchmark.

Paper: §III-G — mixed-effects logistic regression uses 'Region_j' as a
fixed-effect predictor; DiD uses a binary Global North / Global South split.

Regional classification scheme
-------------------------------
UN M49 five-region grouping (as used by the UN Statistics Division):
  1. Africa
  2. Americas
  3. Asia
  4. Europe
  5. Oceania

Global North / Global South binary
-----------------------------------
Following the UN/OECD convention:
  Global North: Europe + North America + Australia + New Zealand + Japan +
                South Korea + Israel + Singapore + Taiwan + Hong Kong + Macau
  Global South: All others

Usage
-----
  python region_lookup.py
  → Produces: region_lookup.csv  (country, iso3, un_region, global_north_south)
"""

import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# REGION MAPPING  (UN M49)
# ─────────────────────────────────────────────────────────────────────────────

# Format: country_name_as_in_GAID_v2 → (un_region, is_global_north)
# is_global_north: 1 = Global North, 0 = Global South
REGION_MAP = {
    # ── AFRICA ───────────────────────────────────────────────────────────────
    "Algeria":                       ("Africa", 0),
    "Angola":                        ("Africa", 0),
    "Benin":                         ("Africa", 0),
    "Botswana":                      ("Africa", 0),
    "Burkina Faso":                  ("Africa", 0),
    "Burundi":                       ("Africa", 0),
    "Cabo Verde":                    ("Africa", 0),
    "Cameroon":                      ("Africa", 0),
    "Central African Republic":      ("Africa", 0),
    "Chad":                          ("Africa", 0),
    "Comoros":                       ("Africa", 0),
    "Congo Republic":                ("Africa", 0),
    "Cote D'Ivoire":                 ("Africa", 0),
    "Djibouti":                      ("Africa", 0),
    "Dr Congo":                      ("Africa", 0),
    "Egypt":                         ("Africa", 0),
    "Equatorial Guinea":             ("Africa", 0),
    "Eritrea":                       ("Africa", 0),
    "Eswatini":                      ("Africa", 0),
    "Ethiopia":                      ("Africa", 0),
    "Gabon":                         ("Africa", 0),
    "Gambia":                        ("Africa", 0),
    "Ghana":                         ("Africa", 0),
    "Guinea":                        ("Africa", 0),
    "Guinea-Bissau":                 ("Africa", 0),
    "Kenya":                         ("Africa", 0),
    "Lesotho":                       ("Africa", 0),
    "Liberia":                       ("Africa", 0),
    "Libya":                         ("Africa", 0),
    "Madagascar":                    ("Africa", 0),
    "Malawi":                        ("Africa", 0),
    "Mali":                          ("Africa", 0),
    "Mauritania":                    ("Africa", 0),
    "Mauritius":                     ("Africa", 0),
    "Morocco":                       ("Africa", 0),
    "Mozambique":                    ("Africa", 0),
    "Namibia":                       ("Africa", 0),
    "Niger":                         ("Africa", 0),
    "Nigeria":                       ("Africa", 0),
    "Reunion":                       ("Africa", 0),
    "Rwanda":                        ("Africa", 0),
    "São Tomé and Principe":         ("Africa", 0),
    "Senegal":                       ("Africa", 0),
    "Seychelles":                    ("Africa", 0),
    "Sierra Leone":                  ("Africa", 0),
    "Somalia":                       ("Africa", 0),
    "South Africa":                  ("Africa", 0),
    "South Sudan":                   ("Africa", 0),
    "Sudan":                         ("Africa", 0),
    "Tanzania":                      ("Africa", 0),
    "Togo":                          ("Africa", 0),
    "Tunisia":                       ("Africa", 0),
    "Uganda":                        ("Africa", 0),
    "Zambia":                        ("Africa", 0),
    "Zimbabwe":                      ("Africa", 0),
    "French Southern Territories":   ("Africa", 0),

    # ── AMERICAS ─────────────────────────────────────────────────────────────
    "Antigua and Barbuda":           ("Americas", 0),
    "Argentina":                     ("Americas", 0),
    "Bahamas":                       ("Americas", 0),
    "Barbados":                      ("Americas", 0),
    "Belize":                        ("Americas", 0),
    "Bermuda":                       ("Americas", 1),  # British overseas territory
    "Bolivia":                       ("Americas", 0),
    "Brazil":                        ("Americas", 0),
    "Canada":                        ("Americas", 1),
    "Cayman Islands":                ("Americas", 0),
    "Chile":                         ("Americas", 0),
    "Colombia":                      ("Americas", 0),
    "Costa Rica":                    ("Americas", 0),
    "Cuba":                          ("Americas", 0),
    "Curacao":                       ("Americas", 0),
    "Dominica":                      ("Americas", 0),
    "Dominican Republic":            ("Americas", 0),
    "Ecuador":                       ("Americas", 0),
    "El Salvador":                   ("Americas", 0),
    "Grenada":                       ("Americas", 0),
    "Guadeloupe":                    ("Americas", 0),
    "Guam":                          ("Americas", 0),
    "Guatemala":                     ("Americas", 0),
    "Guyana":                        ("Americas", 0),
    "Haiti":                         ("Americas", 0),
    "Honduras":                      ("Americas", 0),
    "Jamaica":                       ("Americas", 0),
    "Martinique":                    ("Americas", 0),
    "Mexico":                        ("Americas", 0),
    "Nicaragua":                     ("Americas", 0),
    "Northern Mariana Islands":      ("Americas", 0),
    "Panama":                        ("Americas", 0),
    "Paraguay":                      ("Americas", 0),
    "Peru":                          ("Americas", 0),
    "Puerto Rico":                   ("Americas", 0),
    "St. Kitts And Nevis":           ("Americas", 0),
    "St. Lucia":                     ("Americas", 0),
    "St. Vincent And The Grenadines":("Americas", 0),
    "Suriname":                      ("Americas", 0),
    "Trinidad And Tobago":           ("Americas", 0),
    "United States":                 ("Americas", 1),
    "United States Minor Outlying Islands": ("Americas", 0),
    "Uruguay":                       ("Americas", 0),
    "Venezuela":                     ("Americas", 0),

    # ── ASIA ─────────────────────────────────────────────────────────────────
    "Afghanistan":                   ("Asia", 0),
    "Armenia":                       ("Asia", 0),
    "Azerbaijan":                    ("Asia", 0),
    "Bahrain":                       ("Asia", 0),
    "Bangladesh":                    ("Asia", 0),
    "Bhutan":                        ("Asia", 0),
    "Brunei Darussalam":             ("Asia", 0),
    "Cambodia":                      ("Asia", 0),
    "China":                         ("Asia", 0),
    "Cyprus":                        ("Europe", 1),   # UN: Asia, but EU member → Europe GN
    "Georgia":                       ("Asia", 0),
    "Hong Kong":                     ("Asia", 1),
    "India":                         ("Asia", 0),
    "Indonesia":                     ("Asia", 0),
    "Iran":                          ("Asia", 0),
    "Iraq":                          ("Asia", 0),
    "Israel":                        ("Asia", 1),
    "Japan":                         ("Asia", 1),
    "Jordan":                        ("Asia", 0),
    "Kazakhstan":                    ("Asia", 0),
    "Kuwait":                        ("Asia", 0),
    "Kyrgyzstan":                    ("Asia", 0),
    "Laos":                          ("Asia", 0),
    "Lebanon":                       ("Asia", 0),
    "Macau":                         ("Asia", 1),
    "Malaysia":                      ("Asia", 0),
    "Maldives":                      ("Asia", 0),
    "Mongolia":                      ("Asia", 0),
    "Myanmar":                       ("Asia", 0),
    "Nepal":                         ("Asia", 0),
    "North Korea":                   ("Asia", 0),
    "Oman":                          ("Asia", 0),
    "Pakistan":                      ("Asia", 0),
    "Palestine":                     ("Asia", 0),
    "Philippines":                   ("Asia", 0),
    "Qatar":                         ("Asia", 0),
    "Saudi Arabia":                  ("Asia", 0),
    "Singapore":                     ("Asia", 1),
    "South Korea":                   ("Asia", 1),
    "Sri Lanka":                     ("Asia", 0),
    "Syria":                         ("Asia", 0),
    "Taiwan":                        ("Asia", 1),
    "Tajikistan":                    ("Asia", 0),
    "Thailand":                      ("Asia", 0),
    "Timor-Leste":                   ("Asia", 0),
    "Turkey":                        ("Asia", 0),   # spans Europe/Asia; UN = Asia
    "Turkmenistan":                  ("Asia", 0),
    "United Arab Emirates":          ("Asia", 0),
    "Uzbekistan":                    ("Asia", 0),
    "Vietnam":                       ("Asia", 0),
    "Yemen":                         ("Asia", 0),
    "Kosovo":                        ("Europe", 0),

    # ── EUROPE ───────────────────────────────────────────────────────────────
    "Albania":                       ("Europe", 0),
    "Andorra":                       ("Europe", 1),
    "Austria":                       ("Europe", 1),
    "Belarus":                       ("Europe", 0),
    "Belgium":                       ("Europe", 1),
    "Bosnia And Herzegovina":        ("Europe", 0),
    "Bulgaria":                      ("Europe", 1),
    "Croatia":                       ("Europe", 1),
    "Czechia":                       ("Europe", 1),
    "Denmark":                       ("Europe", 1),
    "Estonia":                       ("Europe", 1),
    "Faroe Islands":                 ("Europe", 1),
    "Finland":                       ("Europe", 1),
    "France":                        ("Europe", 1),
    "Germany":                       ("Europe", 1),
    "Gibraltar":                     ("Europe", 1),
    "Greece":                        ("Europe", 1),
    "Greenland":                     ("Europe", 1),
    "Hungary":                       ("Europe", 1),
    "Iceland":                       ("Europe", 1),
    "Ireland":                       ("Europe", 1),
    "Isle Of Man":                   ("Europe", 1),
    "Italy":                         ("Europe", 1),
    "Latvia":                        ("Europe", 1),
    "Liechtenstein":                 ("Europe", 1),
    "Lithuania":                     ("Europe", 1),
    "Luxembourg":                    ("Europe", 1),
    "Malta":                         ("Europe", 1),
    "Moldova":                       ("Europe", 0),
    "Monaco":                        ("Europe", 1),
    "Montenegro":                    ("Europe", 0),
    "Netherlands":                   ("Europe", 1),
    "North Macedonia":               ("Europe", 0),
    "Norway":                        ("Europe", 1),
    "Poland":                        ("Europe", 1),
    "Portugal":                      ("Europe", 1),
    "Romania":                       ("Europe", 1),
    "Russia":                        ("Europe", 0),  # UN: Europe; GS by convention
    "San Marino":                    ("Europe", 1),
    "Serbia":                        ("Europe", 0),
    "Slovakia":                      ("Europe", 1),
    "Slovenia":                      ("Europe", 1),
    "Spain":                         ("Europe", 1),
    "Sweden":                        ("Europe", 1),
    "Switzerland":                   ("Europe", 1),
    "Ukraine":                       ("Europe", 0),
    "United Kingdom":                ("Europe", 1),
    "Vatican":                       ("Europe", 1),

    # ── OCEANIA ──────────────────────────────────────────────────────────────
    "Australia":                     ("Oceania", 1),
    "Cocos (Keeling) Islands":       ("Oceania", 0),
    "Fiji":                          ("Oceania", 0),
    "French Southern Territories":   ("Oceania", 0),
    "Kiribati":                      ("Oceania", 0),
    "Marshall Islands":              ("Oceania", 0),
    "Micronesia":                    ("Oceania", 0),
    "Nauru":                         ("Oceania", 0),
    "New Caledonia":                 ("Oceania", 0),
    "New Zealand":                   ("Oceania", 1),
    "Palau":                         ("Oceania", 0),
    "Papua New Guinea":              ("Oceania", 0),
    "Samoa":                         ("Oceania", 0),
    "Solomon Islands":               ("Oceania", 0),
    "Timor-Leste":                   ("Asia", 0),    # duplicate; Asia wins
    "Tonga":                         ("Oceania", 0),
    "Tuvalu":                        ("Oceania", 0),
    "Vanuatu":                       ("Oceania", 0),
}


def build_lookup(query_csv: str = "GAID_queries_all_variants.csv",
                 out_path: str  = "region_lookup.csv") -> pd.DataFrame:
    """
    Build the region lookup table for all countries in the query CSV.
    Flags any country with no entry so it can be manually assigned.
    """
    q = pd.read_csv(query_csv, usecols=["country", "iso3"]).drop_duplicates()

    records = []
    missing = []
    for _, row in q.iterrows():
        country = row["country"]
        iso3    = row["iso3"]
        if country in REGION_MAP:
            un_region, gn = REGION_MAP[country]
            records.append({
                "country":           country,
                "iso3":              iso3,
                "un_region":         un_region,
                "global_north_south": "Global North" if gn else "Global South",
            })
        else:
            missing.append(country)
            records.append({
                "country":            country,
                "iso3":               iso3,
                "un_region":          "UNKNOWN",
                "global_north_south": "UNKNOWN",
            })

    df = pd.DataFrame(records).sort_values("country").reset_index(drop=True)
    df.to_csv(out_path, index=False)

    if missing:
        print(f"\n⚠  {len(missing)} countries have no region assigned — please edit region_lookup.csv:")
        for c in sorted(missing):
            print(f"   {c}")
    else:
        print("✓  All countries assigned to a region.")

    # Summary
    print("\n── UN region counts ──")
    print(df["un_region"].value_counts().to_string())
    print("\n── Global North / South counts ──")
    print(df["global_north_south"].value_counts().to_string())
    print(f"\nSaved to: {out_path}")
    return df


if __name__ == "__main__":
    script_dir = str(Path(__file__).parent)
    build_lookup(
        query_csv=f"{script_dir}/queries_all_variants.csv",
        out_path=f"{script_dir}/region_lookup.csv",
    )
