"""
SAPIR-Net Module 4A: Data Visualization
=========================================
Generates two publication-ready charts from Monte Carlo results.

1. Vulnerability Heatmap: Severe shortage probability matrix
2. KDE Distribution Plot: Scenario B capacity loss, MTX vs CIS

Output: sapir_fig1_heatmap.png, sapir_fig2_kde.png (300 dpi)

Dependencies: numpy, scipy, pandas, matplotlib, seaborn
"""

import numpy as np
from scipy.stats import pareto, lognorm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

np.random.seed(42)

# ============================================================
# GLOBAL STYLE
# ============================================================

# Federal-grade muted palette
NAVY      = "#1B2A4A"
STEEL     = "#4A6274"
SLATE     = "#7A8B99"
LIGHT_BG  = "#F4F6F8"
DANGER    = "#C0392B"
AMBER     = "#D4740E"
SAFE_BLUE = "#2980B9"
WHITE     = "#FFFFFF"

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":         11,
    "axes.titlesize":    14,
    "axes.titleweight":  "bold",
    "axes.labelsize":    12,
    "axes.facecolor":    WHITE,
    "figure.facecolor":  WHITE,
    "axes.edgecolor":    SLATE,
    "axes.grid":         False,
    "xtick.color":       NAVY,
    "ytick.color":       NAVY,
    "text.color":        NAVY,
})


# ============================================================
# REBUILD SIMULATION DATA (avoids CSV dependency)
# ============================================================

BASELINE_TRADE_USD = {
    ("CN", "284390"):   8_969_476,
    ("IN", "284390"):   7_113_905,
    ("ROW", "284390"): 529_117_989,
    ("CN", "293359"):  746_763_068,
    ("IN", "293359"):  387_254_472,
    ("ROW", "293359"): 23_625_415_468,
}

BASELINE_TOTAL = {}
for hs in ["284390", "293359"]:
    BASELINE_TOTAL[hs] = sum(v for (s, h), v in BASELINE_TRADE_USD.items() if h == hs)

DRUG_DEP = {"Cisplatin": "284390", "Carboplatin": "284390", "Methotrexate": "293359"}
ROW_EXP = {"284390": (0.10, 0.30), "293359": (0.50, 0.80)}
PARETO_ALPHA = 2.5
N = 10_000


def sample_deg(hs, n):
    low, high = ROW_EXP[hs]
    raw = pareto.rvs(b=PARETO_ALPHA, size=n)
    norm = 1.0 - (1.0 / raw)
    return np.clip(low + norm * (high - low), low, high)


def run_scenario_b():
    """Returns per-drug loss arrays for Scenario B."""
    d284 = sample_deg("284390", N)
    d293 = sample_deg("293359", N)
    losses = {}
    for drug, hs in DRUG_DEP.items():
        arr = np.zeros(N)
        for i in range(N):
            w = BASELINE_TRADE_USD.copy()
            for k in w:
                if k[0] == "CN":
                    w[k] = 0.0
            for k in list(w.keys()):
                if k[0] == "ROW":
                    deg = d284[i] if k[1] == "284390" else d293[i]
                    w[k] = w[k] * (1.0 - deg)
            remaining = sum(v for (s, h), v in w.items() if h == hs)
            arr[i] = 1.0 - (remaining / BASELINE_TOTAL[hs])
        losses[drug] = arr
    return losses


# ============================================================
# FIGURE 1: VULNERABILITY HEATMAP
# ============================================================

def make_heatmap():
    # Data from Module 3 verified output
    drugs = ["Cisplatin", "Carboplatin", "Methotrexate"]
    scenarios = ["Baseline\n(No Shock)", "A: Direct Ban\n(CN + IN)",
                 "B: Cascading\nUpstream Shock", "C: Logistics\nChokepoint"]
    data = np.array([
        [0.0,  0.0,   0.01, 21.03],   # Cisplatin
        [0.0,  0.0,   0.01, 21.03],   # Carboplatin
        [0.0,  0.0, 100.00, 21.03],   # Methotrexate
    ])

    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Custom diverging colormap: white -> amber -> red
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "federal_risk",
        [WHITE, "#FFF3E0", "#FFCC80", AMBER, DANGER, "#7B1A1A"],
        N=256,
    )

    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=100)

    # Annotate cells
    for i in range(len(drugs)):
        for j in range(len(scenarios)):
            val = data[i, j]
            # Dark text on light cells, white on dark
            color = WHITE if val > 50 else NAVY
            fontweight = "bold" if val > 30 else "normal"
            label = f"{val:.0f}%" if val == int(val) else f"{val:.1f}%"
            if val == 0:
                label = "0%"
            ax.text(j, i, label, ha="center", va="center",
                    color=color, fontsize=13, fontweight=fontweight)

    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, fontsize=10)
    ax.set_yticks(range(len(drugs)))
    ax.set_yticklabels(drugs, fontsize=12, fontweight="bold")

    ax.set_title(
        "Probability of Severe Drug Shortage (>30% Capacity Loss)\n"
        "by Disruption Scenario — Monte Carlo Simulation (N = 10,000)",
        pad=16, fontsize=13,
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
    cbar.set_label("Probability of Severe Shortage (%)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Border cleanup
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    fig.tight_layout()
    fig.savefig("sapir_fig1_heatmap.png", dpi=300, bbox_inches="tight",
                facecolor=WHITE, edgecolor="none")
    print("Exported: sapir_fig1_heatmap.png")
    plt.close(fig)


# ============================================================
# FIGURE 2: KDE — METHOTREXATE VS CISPLATIN (SCENARIO B)
# ============================================================

def make_kde():
    losses = run_scenario_b()
    cis_loss = losses["Cisplatin"] * 100
    mtx_loss = losses["Methotrexate"] * 100

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # KDE plots
    sns.kdeplot(cis_loss, ax=ax, color=SAFE_BLUE, linewidth=2.5,
                fill=True, alpha=0.25, label="Cisplatin / Carboplatin")
    sns.kdeplot(mtx_loss, ax=ax, color=DANGER, linewidth=2.5,
                fill=True, alpha=0.25, label="Methotrexate")

    # 30% threshold line
    ax.axvline(x=30, color=AMBER, linestyle="--", linewidth=2, zorder=5)
    ax.text(31.5, ax.get_ylim()[1] * 0.85, "30% Severe\nShortage\nThreshold",
            color=AMBER, fontsize=10, fontweight="bold", va="top")

    # Shade the danger zone
    xlim = ax.get_xlim()
    ax.axvspan(30, xlim[1], alpha=0.06, color=DANGER, zorder=0)
    ax.set_xlim(xlim)

    ax.set_xlabel("Capacity Loss (%)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        "Scenario B (Cascading Upstream Shock): Distribution of Capacity Loss\n"
        "Methotrexate vs. Cisplatin/Carboplatin — N = 10,000 iterations",
        pad=14, fontsize=13,
    )

    ax.legend(loc="upper left", frameon=True, framealpha=0.9,
              edgecolor=SLATE, fontsize=11)

    # Annotation arrows
    cis_peak_x = cis_loss.mean()
    mtx_peak_x = mtx_loss.mean()

    ax.annotate(
        f"Mean: {cis_peak_x:.1f}%\nP(severe): <0.1%",
        xy=(cis_peak_x, 0),
        xytext=(cis_peak_x - 5, ax.get_ylim()[1] * 0.55),
        fontsize=9, color=SAFE_BLUE, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=SAFE_BLUE, lw=1.5),
        ha="center",
    )

    ax.annotate(
        f"Mean: {mtx_peak_x:.1f}%\nP(severe): 100%",
        xy=(mtx_peak_x, 0),
        xytext=(mtx_peak_x + 8, ax.get_ylim()[1] * 0.55),
        fontsize=9, color=DANGER, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=DANGER, lw=1.5),
        ha="center",
    )

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SLATE)
    ax.spines["bottom"].set_color(SLATE)

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))

    fig.tight_layout()
    fig.savefig("sapir_fig2_kde.png", dpi=300, bbox_inches="tight",
                facecolor=WHITE, edgecolor="none")
    print("Exported: sapir_fig2_kde.png")
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("SAPIR-Net Module 4A: Generating publication-ready figures...")
    print()
    make_heatmap()
    make_kde()
    print("\nModule 4A complete. Both figures ready for Red Team visual audit.")
