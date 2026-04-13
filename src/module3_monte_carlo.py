"""
SAPIR-Net Module 3: Monte Carlo Simulation Loop
=================================================
Runs N=10,000 iterations per scenario, propagating disruptions
from L1 (Geopolitical) through L2 (Chemical) to L3 (Medicines).

Integrates:
  - Module 1: Graph structure and empirical edge weights
  - Module 2: Scenario classes and distribution samplers

Output: L3 drug shortage probability matrix for Red Team audit.

Dependencies: numpy, scipy, pandas
"""

import numpy as np
from scipy.stats import pareto, lognorm
import pandas as pd

np.random.seed(42)

# ============================================================
# SECTION 1: GRAPH DATA (FROM MODULE 1 OUTPUT)
# ============================================================
# 5-year aggregate trade values (USD) from empirical Comtrade data.
# Keyed by (L1_source, L2_commodity).

BASELINE_TRADE_USD = {
    ("CN", "284390"):   8_969_476,
    ("IN", "284390"):   7_113_905,
    ("ROW", "284390"): 529_117_989,
    ("CN", "293359"):  746_763_068,
    ("IN", "293359"):  387_254_472,
    ("ROW", "293359"): 23_625_415_468,
}

# Baseline totals per commodity (sum of all sources)
BASELINE_TOTAL = {}
for hs in ["284390", "293359"]:
    BASELINE_TOTAL[hs] = sum(v for (s, h), v in BASELINE_TRADE_USD.items() if h == hs)

# L2 -> L3 stoichiometric mapping
# Each L3 drug depends on exactly one L2 commodity.
DRUG_DEPENDENCY = {
    "Cisplatin":     "284390",
    "Carboplatin":   "284390",
    "Methotrexate":  "293359",
}

# Severe shortage threshold (FDA-aligned)
SEVERE_SHORTAGE_THRESHOLD = 0.30  # >30% capacity loss

# ============================================================
# SECTION 2: MODULE 2 PARAMETERS (EMBEDDED)
# ============================================================

ROW_CHINA_EXPOSURE = {
    "284390": {"low": 0.10, "high": 0.30},
    "293359": {"low": 0.50, "high": 0.80},
}
PARETO_ALPHA = 2.5
LOGNORM_MU = np.log(0.20)
LOGNORM_SIGMA = 0.50


# ============================================================
# SECTION 3: DISRUPTION SAMPLERS
# ============================================================

def sample_pareto_degradation(hs_code, n):
    """Sample ROW degradation fractions from rescaled Pareto."""
    low = ROW_CHINA_EXPOSURE[hs_code]["low"]
    high = ROW_CHINA_EXPOSURE[hs_code]["high"]
    raw = pareto.rvs(b=PARETO_ALPHA, size=n)
    normalized = 1.0 - (1.0 / raw)
    return np.clip(low + normalized * (high - low), low, high)


def sample_lognorm_reduction(n):
    """Sample global capacity reduction fractions from log-normal."""
    raw = lognorm.rvs(s=LOGNORM_SIGMA, scale=np.exp(LOGNORM_MU), size=n)
    return np.clip(raw, 0.01, 0.95)


# ============================================================
# SECTION 4: SIMULATION ENGINE
# ============================================================

def compute_commodity_remaining(weights, hs_code):
    """Sum remaining trade value for a commodity across all sources."""
    return sum(v for (s, h), v in weights.items() if h == hs_code)


def run_scenario_baseline(n_iter):
    """No disruption. Returns 0% loss for all iterations."""
    return {drug: np.zeros(n_iter) for drug in DRUG_DEPENDENCY}


def run_scenario_a(n_iter):
    """Scenario A: CN + IN direct exports zeroed."""
    losses = {drug: np.zeros(n_iter) for drug in DRUG_DEPENDENCY}

    # Deterministic — same result every iteration
    weights = BASELINE_TRADE_USD.copy()
    for key in weights:
        source, _ = key
        if source in ("CN", "IN"):
            weights[key] = 0.0

    for drug, hs in DRUG_DEPENDENCY.items():
        remaining = compute_commodity_remaining(weights, hs)
        loss_frac = 1.0 - (remaining / BASELINE_TOTAL[hs])
        losses[drug][:] = loss_frac

    return losses


def run_scenario_b(n_iter):
    """Scenario B: CN zeroed + ROW Pareto degradation (commodity-specific)."""
    losses = {drug: np.zeros(n_iter) for drug in DRUG_DEPENDENCY}

    # Pre-sample ROW degradation for each commodity
    deg_284390 = sample_pareto_degradation("284390", n_iter)
    deg_293359 = sample_pareto_degradation("293359", n_iter)
    degradation_samples = {"284390": deg_284390, "293359": deg_293359}

    for i in range(n_iter):
        weights = BASELINE_TRADE_USD.copy()

        # Zero out CN
        for key in weights:
            if key[0] == "CN":
                weights[key] = 0.0

        # Degrade ROW by commodity-specific sampled fraction
        for key in list(weights.keys()):
            source, hs = key
            if source == "ROW":
                weights[key] = weights[key] * (1.0 - degradation_samples[hs][i])

        # Propagate to L3
        for drug, hs in DRUG_DEPENDENCY.items():
            remaining = compute_commodity_remaining(weights, hs)
            losses[drug][i] = 1.0 - (remaining / BASELINE_TOTAL[hs])

    return losses


def run_scenario_c(n_iter):
    """Scenario C: Log-normal capacity reduction on ALL edges."""
    losses = {drug: np.zeros(n_iter) for drug in DRUG_DEPENDENCY}

    reductions = sample_lognorm_reduction(n_iter)

    for i in range(n_iter):
        r = reductions[i]
        for drug, hs in DRUG_DEPENDENCY.items():
            # All edges reduced uniformly — loss equals the reduction fraction
            losses[drug][i] = r

    return losses


# ============================================================
# SECTION 5: RUN ALL SCENARIOS
# ============================================================

N_ITER = 10_000

SCENARIOS = {
    "Baseline (No Shock)":       run_scenario_baseline,
    "A: Direct Ban (CN+IN)":     run_scenario_a,
    "B: Cascading Upstream Shock": run_scenario_b,
    "C: Logistics Chokepoint":   run_scenario_c,
}


def run_all():
    """Execute all scenarios and compile results."""
    all_results = []

    print("=" * 75)
    print(f"SAPIR-Net Module 3: Monte Carlo Simulation (N = {N_ITER:,} iterations)")
    print("=" * 75)

    for scenario_name, runner in SCENARIOS.items():
        print(f"\nRunning: {scenario_name}...")
        losses = runner(N_ITER)

        for drug in DRUG_DEPENDENCY:
            loss_array = losses[drug]
            mean_loss = loss_array.mean()
            median_loss = np.median(loss_array)
            p5 = np.percentile(loss_array, 5)
            p95 = np.percentile(loss_array, 95)
            prob_severe = (loss_array > SEVERE_SHORTAGE_THRESHOLD).mean()

            all_results.append({
                "Scenario":                scenario_name,
                "Drug":                    drug,
                "HS_Dependency":           DRUG_DEPENDENCY[drug],
                "Mean_Capacity_Loss_pct":  round(mean_loss * 100, 2),
                "Median_Capacity_Loss_pct": round(median_loss * 100, 2),
                "P5_Loss_pct":             round(p5 * 100, 2),
                "P95_Loss_pct":            round(p95 * 100, 2),
                "Prob_Severe_Shortage_pct": round(prob_severe * 100, 2),
            })

    results_df = pd.DataFrame(all_results)
    return results_df


# ============================================================
# SECTION 6: OUTPUT & AUDIT
# ============================================================

if __name__ == "__main__":
    results_df = run_all()

    # --- Full results table ---
    print(f"\n{'=' * 75}")
    print("FULL RESULTS TABLE")
    print(f"{'=' * 75}")
    print(f"Severe Shortage Threshold: >{SEVERE_SHORTAGE_THRESHOLD:.0%} capacity loss")
    print()
    print(results_df.to_string(index=False))

    # --- Vulnerability matrix (the key deliverable) ---
    print(f"\n{'=' * 75}")
    print("L3 DRUG VULNERABILITY MATRIX")
    print("Probability of Severe Shortage (>30% capacity loss) by Scenario")
    print(f"{'=' * 75}")

    pivot = results_df.pivot_table(
        index="Drug",
        columns="Scenario",
        values="Prob_Severe_Shortage_pct",
        aggfunc="first",
    )
    # Reorder columns
    col_order = [c for c in SCENARIOS.keys() if c in pivot.columns]
    pivot = pivot[col_order]
    print(pivot.to_string())

    # --- Mean capacity loss matrix ---
    print(f"\n{'=' * 75}")
    print("MEAN CAPACITY LOSS (%) BY SCENARIO")
    print(f"{'=' * 75}")

    pivot_mean = results_df.pivot_table(
        index="Drug",
        columns="Scenario",
        values="Mean_Capacity_Loss_pct",
        aggfunc="first",
    )
    pivot_mean = pivot_mean[col_order]
    print(pivot_mean.to_string())

    # --- Key findings ---
    print(f"\n{'=' * 75}")
    print("KEY FINDINGS")
    print(f"{'=' * 75}")

    for drug in DRUG_DEPENDENCY:
        scen_b = results_df[
            (results_df.Drug == drug) &
            (results_df.Scenario == "B: Cascading Upstream Shock")
        ].iloc[0]
        print(
            f"  {drug}: Under cascading shock, mean capacity loss = "
            f"{scen_b['Mean_Capacity_Loss_pct']:.1f}%, "
            f"probability of severe shortage = "
            f"{scen_b['Prob_Severe_Shortage_pct']:.1f}%"
        )

    # --- Export ---
    results_df.to_csv("sapir_monte_carlo_results.csv", index=False)
    print(f"\nExported: sapir_monte_carlo_results.csv")
    print(f"\nModule 3 complete. Results ready for Module 4 (Visualization & White Paper).")
