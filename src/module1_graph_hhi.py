"""
SAPIR-Net Module 1: Graph Construction & HHI Analysis
======================================================
Ingests the cleaned Comtrade Plus CSV, computes Herfindahl-Hirschman
Index (HHI) concentration scores for L1->L2 edges, and builds the
three-layer Macro-Geopolitical Dependency Graph in NetworkX.

Input:  sapir_raw_comtrade.csv (Comtrade Plus export)
Output: NetworkX DiGraph + HHI scores printed for Red Team audit
"""

import csv
import pandas as pd
import networkx as nx

# ============================================================
# STEP 1: INGEST & NORMALIZE
# ============================================================

INPUT_PATH = "sapir_raw_comtrade.csv"  # adjust path if needed in Colab

# Comtrade Plus CSVs have a trailing comma that creates a phantom
# 48th field per row. We parse with csv.reader and trim.
with open(INPUT_PATH, newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = [r[: len(header)] for r in reader]

raw = pd.DataFrame(rows, columns=header)

# Map Comtrade Plus headers to our locked MPB schema
df = pd.DataFrame(
    {
        "Year": pd.to_numeric(raw["refYear"]),
        "Source_Country": raw["partnerDesc"].str.strip(),
        "Partner_Code": raw["partnerCode"].str.strip(),
        "HS_Code": raw["cmdCode"].str.strip(),
        "Commodity_Desc": raw["cmdDesc"].str.strip(),
        "Trade_Value_USD": pd.to_numeric(raw["primaryValue"], errors="coerce").fillna(0),
        "Net_Weight_KG": pd.to_numeric(raw["netWgt"], errors="coerce").fillna(0),
    }
)

print("=" * 65)
print("SAPIR-Net Module 1: Data Ingestion Complete")
print("=" * 65)
print(f"Rows: {len(df)}  |  Years: {sorted(df.Year.unique())}  |  HS Codes: {sorted(df.HS_Code.unique())}")
print(f"Partners: {sorted(df.Source_Country.unique())}")
print()

# ============================================================
# STEP 2: COMPUTE HHI (L1 -> L2 CONCENTRATION)
# ============================================================
# HHI = sum of squared market shares (x 10,000 scale)
# Computed per HS code, per year, using Trade_Value_USD.
#
# Market share for China and India is their individual value
# divided by the World total for that HS/year combination.
# "Rest of World" (ROW) = World - China - India.

def compute_hhi(df):
    """
    Compute HHI per HS_Code per Year.
    Returns a DataFrame with per-country shares and HHI scores.
    """
    results = []

    for hs in sorted(df.HS_Code.unique()):
        for year in sorted(df.Year.unique()):
            mask_hs_yr = (df.HS_Code == hs) & (df.Year == year)
            subset = df[mask_hs_yr]

            world_val = subset.loc[subset.Source_Country == "World", "Trade_Value_USD"]
            china_val = subset.loc[subset.Source_Country == "China", "Trade_Value_USD"]
            india_val = subset.loc[subset.Source_Country == "India", "Trade_Value_USD"]

            total = world_val.values[0] if len(world_val) > 0 else 0
            cn = china_val.values[0] if len(china_val) > 0 else 0
            ind = india_val.values[0] if len(india_val) > 0 else 0

            if total == 0:
                continue

            row_val = total - cn - ind  # Rest of World residual

            share_cn = (cn / total) * 100
            share_in = (ind / total) * 100
            share_row = (row_val / total) * 100

            hhi = share_cn**2 + share_in**2 + share_row**2

            results.append(
                {
                    "HS_Code": hs,
                    "Year": year,
                    "Total_USD": total,
                    "China_USD": cn,
                    "India_USD": ind,
                    "ROW_USD": row_val,
                    "Share_China_pct": round(share_cn, 2),
                    "Share_India_pct": round(share_in, 2),
                    "Share_ROW_pct": round(share_row, 2),
                    "HHI": round(hhi, 1),
                }
            )

    return pd.DataFrame(results)


hhi_df = compute_hhi(df)

print("=" * 65)
print("HHI CONCENTRATION ANALYSIS (L1 -> L2)")
print("=" * 65)
print("DOJ/FTC thresholds: <1500 = unconcentrated | 1500-2500 = moderate | >2500 = highly concentrated")
print()
print(hhi_df.to_string(index=False))
print()

# Summary per commodity across all years
print("-" * 65)
print("5-YEAR AVERAGE HHI PER COMMODITY:")
for hs in sorted(hhi_df.HS_Code.unique()):
    avg = hhi_df.loc[hhi_df.HS_Code == hs, "HHI"].mean()
    label = "HIGHLY CONCENTRATED" if avg > 2500 else ("MODERATE" if avg > 1500 else "UNCONCENTRATED")
    print(f"  HS {hs}: avg HHI = {avg:,.0f}  [{label}]")
print()

# ============================================================
# STEP 3: BUILD NETWORKX DIGRAPH
# ============================================================

G = nx.DiGraph()
G.graph["name"] = "SAPIR-Net Phase 1: Oncology Supply Chain Dependency Graph"

# --- Layer 1 nodes: Geopolitical sources ---
l1_nodes = {
    "CN": {"label": "China", "layer": 1, "type": "geopolitical"},
    "IN": {"label": "India", "layer": 1, "type": "geopolitical"},
    "ROW": {"label": "Rest of World", "layer": 1, "type": "geopolitical"},
}
for node_id, attrs in l1_nodes.items():
    G.add_node(node_id, **attrs)

# --- Layer 2 nodes: Chemical chokepoints ---
l2_nodes = {
    "PLAT": {
        "label": "Platinum Compounds (HS 284390)",
        "layer": 2,
        "type": "chemical",
        "hs_code": "284390",
    },
    "HETERO": {
        "label": "Heterocyclic Compounds (HS 293359)",
        "layer": 2,
        "type": "chemical",
        "hs_code": "293359",
    },
}
for node_id, attrs in l2_nodes.items():
    G.add_node(node_id, **attrs)

# --- Layer 3 nodes: Essential medicines ---
l3_nodes = {
    "CIS": {"label": "Cisplatin", "layer": 3, "type": "medicine", "drug_class": "platinum-based"},
    "CARB": {"label": "Carboplatin", "layer": 3, "type": "medicine", "drug_class": "platinum-based"},
    "MTX": {"label": "Methotrexate", "layer": 3, "type": "medicine", "drug_class": "folate antagonist"},
}
for node_id, attrs in l3_nodes.items():
    G.add_node(node_id, **attrs)

# --- L1 -> L2 edges: weighted by 5-year aggregate trade value ---
# Map partner names to graph node IDs
partner_to_node = {"China": "CN", "India": "IN"}
hs_to_node = {"284390": "PLAT", "293359": "HETERO"}

# Aggregate across all 5 years for edge weights
for hs in ["284390", "293359"]:
    l2_node = hs_to_node[hs]
    hs_data = df[(df.HS_Code == hs) & (df.Source_Country != "World")]

    # China and India direct edges
    for country in ["China", "India"]:
        l1_node = partner_to_node[country]
        country_total = hs_data.loc[hs_data.Source_Country == country, "Trade_Value_USD"].sum()
        country_weight = hs_data.loc[hs_data.Source_Country == country, "Net_Weight_KG"].sum()

        G.add_edge(
            l1_node,
            l2_node,
            trade_value_usd=country_total,
            net_weight_kg=country_weight,
            edge_type="empirical_trade_flow",
            years="2019-2023",
        )

    # ROW edge = World total minus China minus India
    world_total_usd = df[(df.HS_Code == hs) & (df.Source_Country == "World")]["Trade_Value_USD"].sum()
    world_total_kg = df[(df.HS_Code == hs) & (df.Source_Country == "World")]["Net_Weight_KG"].sum()
    cn_usd = hs_data.loc[hs_data.Source_Country == "China", "Trade_Value_USD"].sum()
    cn_kg = hs_data.loc[hs_data.Source_Country == "China", "Net_Weight_KG"].sum()
    in_usd = hs_data.loc[hs_data.Source_Country == "India", "Trade_Value_USD"].sum()
    in_kg = hs_data.loc[hs_data.Source_Country == "India", "Net_Weight_KG"].sum()

    G.add_edge(
        "ROW",
        l2_node,
        trade_value_usd=world_total_usd - cn_usd - in_usd,
        net_weight_kg=world_total_kg - cn_kg - in_kg,
        edge_type="empirical_trade_flow",
        years="2019-2023",
    )

# --- L2 -> L3 edges: Binary Stoichiometric Flags ---
stoichiometric_edges = [
    ("PLAT", "CIS", 1.0, "Platinum required for cisplatin synthesis"),
    ("PLAT", "CARB", 1.0, "Platinum required for carboplatin synthesis"),
    ("HETERO", "MTX", 1.0, "Heterocyclic intermediate required for methotrexate synthesis"),
]
for src, dst, flag, rationale in stoichiometric_edges:
    G.add_edge(
        src,
        dst,
        dependency_flag=flag,
        edge_type="stoichiometric",
        rationale=rationale,
    )

# ============================================================
# STEP 4: GRAPH AUDIT OUTPUT
# ============================================================

print("=" * 65)
print("NETWORKX GRAPH SUMMARY")
print("=" * 65)
print(f"Nodes: {G.number_of_nodes()}  |  Edges: {G.number_of_edges()}")
print()

print("NODES:")
for node, attrs in G.nodes(data=True):
    print(f"  [{node}] Layer {attrs['layer']} | {attrs['label']} ({attrs['type']})")
print()

print("EDGES (L1 -> L2: Trade Flows, 5-year aggregate):")
for u, v, attrs in G.edges(data=True):
    if attrs["edge_type"] == "empirical_trade_flow":
        print(
            f"  {u} -> {v}  |  ${attrs['trade_value_usd']:>15,.0f} USD  |  "
            f"{attrs['net_weight_kg']:>12,.0f} KG"
        )
print()

print("EDGES (L2 -> L3: Stoichiometric Dependencies):")
for u, v, attrs in G.edges(data=True):
    if attrs["edge_type"] == "stoichiometric":
        print(f"  {u} -> {v}  |  Flag: {attrs['dependency_flag']}  |  {attrs['rationale']}")
print()

# ============================================================
# STEP 5: EXPORT HHI TABLE FOR RED TEAM
# ============================================================

hhi_df.to_csv("sapir_hhi_analysis.csv", index=False)
print("Exported: sapir_hhi_analysis.csv")
print("\nModule 1 complete. Graph object ready for Module 2 (Disruption Probability Engine).")
