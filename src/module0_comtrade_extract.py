"""
SAPIR-Net Module 0: UN Comtrade Data Extraction
=================================================
Extracts U.S. import data for oncology precursor commodities
from the UN Comtrade API (v1/legacy endpoint).

Target HS Codes:
  - 284390: Platinum-group compounds (Cisplatin/Carboplatin precursors)
  - 293359: Heterocyclic compounds, nitrogen heteroatom (Methotrexate precursors)

Reporter: United States (842)
Partners: China (156), India (356), World (0)
Period: 2019–2023 (5 years)

Output: sapir_raw_comtrade.csv
"""

import requests
import pandas as pd
import time

# ============================================================
# CONFIGURATION
# ============================================================

# UN Comtrade API v1 (legacy/public) endpoint
# If you have a Comtrade API subscription key, paste it below.
# The legacy v1 endpoint allows limited unauthenticated access.
# For the v2 endpoint (comtradeapi.un.org), a key is required.
API_KEY = ""  # <-- Paste your Comtrade API key here if available

# Use v1 (legacy) endpoint — more permissive for unauthenticated access
BASE_URL_V1 = "https://comtrade.un.org/api/get"

# v2 endpoint (use if v1 fails or is retired)
BASE_URL_V2 = "https://comtradeapi.un.org/data/v1/get/C/A"

REPORTER = "842"          # United States
PARTNERS = ["156", "356", "0"]  # China, India, World
HS_CODES = ["284390", "293359"]
YEARS = ["2019", "2020", "2021", "2022", "2023"]
TRADE_FLOW = "1"          # 1 = Imports

# Rate limit: Comtrade allows ~1 request per second for unauthenticated users
RATE_LIMIT_SECONDS = 1.5

# ============================================================
# EXTRACTION FUNCTIONS
# ============================================================

def fetch_comtrade_v1(reporter, partner, hs_code, year):
    """
    Query the UN Comtrade v1 (legacy) API for a single
    reporter/partner/commodity/year combination.
    Returns parsed JSON or None on failure.
    """
    params = {
        "r":    reporter,
        "p":    partner,
        "ps":   year,
        "cc":   hs_code,
        "rg":   TRADE_FLOW,
        "type": "C",       # Commodities
        "freq": "A",       # Annual
        "px":   "HS",      # HS classification
        "fmt":  "json",
        "max":  500,
        "head": "M",       # Machine-readable headers
    }
    if API_KEY:
        params["token"] = API_KEY

    try:
        resp = requests.get(BASE_URL_V1, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        # Comtrade v1 returns {"validation": {...}, "dataset": [...]}
        if "dataset" in payload and len(payload["dataset"]) > 0:
            return payload["dataset"]
        else:
            print(f"  -> No data: {year} | Partner {partner} | HS {hs_code}")
            return None

    except requests.exceptions.Timeout:
        print(f"  -> TIMEOUT: {year} | Partner {partner} | HS {hs_code}")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"  -> HTTP ERROR {e.response.status_code}: {year} | Partner {partner} | HS {hs_code}")
        return None
    except Exception as e:
        print(f"  -> UNEXPECTED ERROR: {e}")
        return None


def fetch_comtrade_v2(reporter, partner, hs_code, year):
    """
    Fallback: Query the UN Comtrade v2 API.
    Requires API_KEY to be set.
    """
    if not API_KEY:
        print("  -> v2 endpoint requires an API key. Skipping.")
        return None

    url = f"{BASE_URL_V2}/{reporter}/{partner}/{year}/{hs_code}"
    headers = {"Ocp-Apim-Subscription-Key": API_KEY}

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        payload = resp.json()

        if "data" in payload and len(payload["data"]) > 0:
            return payload["data"]
        else:
            print(f"  -> No data (v2): {year} | Partner {partner} | HS {hs_code}")
            return None

    except Exception as e:
        print(f"  -> v2 ERROR: {e}")
        return None


# ============================================================
# MAIN EXTRACTION LOOP
# ============================================================

def extract_all():
    """
    Iterate over all year/partner/commodity combinations.
    Returns a list of normalized record dicts.
    """
    all_records = []
    total_queries = len(YEARS) * len(PARTNERS) * len(HS_CODES)
    query_count = 0

    print(f"SAPIR-Net Module 0: Starting extraction ({total_queries} queries)")
    print("=" * 60)

    for year in YEARS:
        for partner in PARTNERS:
            for hs_code in HS_CODES:
                query_count += 1
                print(f"[{query_count}/{total_queries}] Year={year} Partner={partner} HS={hs_code}")

                # Try v1 first, fall back to v2
                data = fetch_comtrade_v1(REPORTER, partner, hs_code, year)
                if data is None:
                    data = fetch_comtrade_v2(REPORTER, partner, hs_code, year)

                if data:
                    for record in data:
                        # v1 field names (machine-readable header mode)
                        normalized = {
                            "Year":            record.get("yr", record.get("period", year)),
                            "Period":          record.get("period", year),
                            "Reporter":        record.get("rtTitle", record.get("reporterDesc", "United States")),
                            "Reporter_Code":   record.get("rt", record.get("reporterCode", REPORTER)),
                            "Source_Country":   record.get("ptTitle", record.get("partnerDesc", "")),
                            "Partner_Code":    record.get("pt", record.get("partnerCode", partner)),
                            "HS_Code":         record.get("cmdCode", hs_code),
                            "Commodity_Desc":  record.get("cmdDescE", record.get("cmdDesc", "")),
                            "Trade_Flow":      record.get("rgDesc", record.get("flowDesc", "Import")),
                            "Trade_Value_USD": record.get("TradeValue", record.get("primaryValue", 0)),
                            "Net_Weight_KG":   record.get("NetWeight", record.get("netWgt", 0)),
                        }
                        all_records.append(normalized)

                time.sleep(RATE_LIMIT_SECONDS)

    print("=" * 60)
    print(f"Extraction complete. Total records retrieved: {len(all_records)}")
    return all_records


# ============================================================
# BUILD DATAFRAME & EXPORT
# ============================================================

if __name__ == "__main__":
    records = extract_all()

    if not records:
        print("\nWARNING: No records retrieved. Possible causes:")
        print("  1. Comtrade API may be temporarily unavailable.")
        print("  2. Rate limiting — try again after a few minutes.")
        print("  3. API key may be required for these queries.")
        print("  4. HS code may not have data at 6-digit level for these partners.")
        print("\nNo CSV generated.")
    else:
        df = pd.DataFrame(records)

        # Enforce correct dtypes
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
        df["Trade_Value_USD"] = pd.to_numeric(df["Trade_Value_USD"], errors="coerce").fillna(0)
        df["Net_Weight_KG"] = pd.to_numeric(df["Net_Weight_KG"], errors="coerce").fillna(0)
        df["HS_Code"] = df["HS_Code"].astype(str).str.strip()

        # Sort for readability
        df = df.sort_values(by=["HS_Code", "Year", "Partner_Code"]).reset_index(drop=True)

        # Select final columns (superset — includes audit columns)
        output_cols = [
            "Year", "Period", "Reporter", "Source_Country", "Partner_Code",
            "HS_Code", "Commodity_Desc", "Trade_Flow",
            "Trade_Value_USD", "Net_Weight_KG"
        ]
        df = df[[c for c in output_cols if c in df.columns]]

        # Export
        output_path = "sapir_raw_comtrade.csv"
        df.to_csv(output_path, index=False)

        print(f"\nCSV exported: {output_path}")
        print(f"Shape: {df.shape}")
        print(f"\nPreview:\n{df.head(10).to_string()}")

        # Quick audit summary
        print(f"\n--- AUDIT SUMMARY ---")
        print(f"Unique years:    {sorted(df['Year'].dropna().unique())}")
        print(f"Unique HS codes: {sorted(df['HS_Code'].unique())}")
        print(f"Unique partners: {sorted(df['Source_Country'].unique())}")
        print(f"Total trade value (USD): ${df['Trade_Value_USD'].sum():,.0f}")
        print(f"Total net weight (KG):   {df['Net_Weight_KG'].sum():,.0f}")
