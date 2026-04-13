# SAPIR-Net

**Supply Chain Analysis for Pharmaceutical Infrastructure Resilience — Network Model**

*Quantifying cascading vulnerability in the U.S. generic oncology drug supply chain*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://www.python.org/) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19548610.svg)](https://doi.org/10.5281/zenodo.19548610)


\---

## Overview

SAPIR-Net is a three-layer directed graph model that maps upstream chemical precursor supply chains for three essential U.S. oncology drugs — **cisplatin**, **carboplatin**, and **methotrexate** — and quantifies their vulnerability to geopolitical and logistical disruption through Monte Carlo simulation.

The model integrates:

* Empirical bilateral trade data from the **UN Comtrade** database (2019–2023)
* Binary stoichiometric dependency mappings from pharmaceutical chemistry
* Parameterized indirect upstream exposure estimates grounded in federal and peer-reviewed sources
* Stochastic disruption scenarios using **Pareto** and **log-normal** distributions

### Key Findings

* **Methotrexate** faces a **100% probability of severe shortage** (>30% capacity loss) under a cascading China-specific upstream shock, driven by 50–80% indirect Chinese dependency in the heterocyclic compound supply chain.
* **Cisplatin and carboplatin** are resilient to the same geopolitical shock (<0.01% severe shortage probability) but face a **21% probability of severe shortage** under systemic logistics disruption.
* The model's Scenario A output (**2.95% capacity loss**) aligns with the observed **2.7% average reduction** during the 2023 U.S. platinum chemotherapy shortage (Reibel et al., *JNCI*, 2025; 95% CI: −4.4% to −0.9%).

These findings demonstrate that oncology drug supply chain risk is **commodity-specific** and requires differentiated policy responses — not monolithic reshoring.

\---

## Repository Structure

```
sapir-net/
├── README.md
├── LICENSE
├── white\_paper/
│   └── SAPIR-Net\_White\_Paper\_v1.pdf
├── data/
│   ├── sapir\_raw\_comtrade.csv          # UN Comtrade bilateral trade data (2019–2023)
│   └── sapir\_hhi\_analysis.csv          # Computed HHI concentration scores
├── src/
│   ├── module0\_comtrade\_extract.py     # Data extraction from UN Comtrade API
│   ├── module1\_graph\_hhi.py            # Graph construction \& HHI computation
│   ├── module2\_disruption\_engine.py    # Disruption scenario definitions \& distributions
│   ├── module3\_monte\_carlo.py          # Monte Carlo simulation loop (N=10,000)
│   └── module4a\_visualizations.py      # Publication-ready figures (300 dpi)
├── figures/
│   ├── sapir\_fig1\_heatmap.png          # Severe shortage probability heatmap
│   └── sapir\_fig2\_kde.png              # Scenario B capacity loss KDE distributions
└── results/
    └── sapir\_monte\_carlo\_results.csv   # Full simulation output matrix
```

\---

## Methodology

The model operates on a three-layer directed graph:

|Layer|Nodes|Description|
|-|-|-|
|**L1: Geopolitical Sources**|China, India, Rest of World|Countries exporting chemical precursors to the U.S.|
|**L2: Chemical Chokepoints**|HS 284390 (Platinum compounds), HS 293359 (Heterocyclics)|Commodity classes serving as drug precursors|
|**L3: Essential Medicines**|Cisplatin, Carboplatin, Methotrexate|Target oncology drugs|

**L1 → L2 edges** are weighted by empirical Comtrade trade values (USD). Geographic concentration is evaluated using the Herfindahl-Hirschman Index (HHI).

**L2 → L3 edges** are binary stoichiometric dependency flags — a drug either requires the upstream commodity for synthesis (1.0) or does not (0.0). No weights are fabricated.

Three disruption scenarios are simulated:

* **Scenario A:** Deterministic elimination of direct CN + IN exports
* **Scenario B:** CN elimination + Pareto-distributed ROW degradation (commodity-specific `row\_china\_exposure` parameter)
* **Scenario C:** Log-normal global logistics capacity reduction

\---

## Quick Start

### Requirements

```
Python 3.9+
pandas
networkx
numpy
scipy
matplotlib
seaborn
```

### Installation

```bash
git clone https://github.com/fahimkazmi911/sapir-net.git
cd sapir-net
pip install pandas networkx numpy scipy matplotlib seaborn
```

### Execution

The modules are designed to run sequentially. Place `sapir\_raw\_comtrade.csv` in the working directory before running Module 1.

```bash
# Module 1: Build graph and compute HHI scores
python src/module1\_graph\_hhi.py

# Module 2: Validate disruption distributions (audit output)
python src/module2\_disruption\_engine.py

# Module 3: Run Monte Carlo simulation (N=10,000)
python src/module3\_monte\_carlo.py

# Module 4A: Generate publication-ready figures
python src/module4a\_visualizations.py
```

Each module prints a structured audit summary to stdout. Module 3 exports `sapir\_monte\_carlo\_results.csv`. Module 4A exports two 300-dpi PNG figures.

\---

## Data Sources

|Source|Description|Access|
|-|-|-|
|UN Comtrade|Bilateral trade data: U.S. imports of HS 284390, HS 293359 from CN, IN, World (2019–2023)|[comtradeplus.un.org](https://comtradeplus.un.org/)|
|USGS MCS 2024|Platinum-group metals production and trade statistics|[pubs.usgs.gov](https://pubs.usgs.gov/periodicals/mcs2024/mcs2024-platinum-group.pdf)|
|FDA CDER|Drug Shortage Database — current shortage status|[accessdata.fda.gov](https://www.accessdata.fda.gov/scripts/drugshortages/)|

\---

## Citation

If you use SAPIR-Net in your research or policy work, please cite:

```
SAPIR-Net: Supply Chain Analysis for Pharmaceutical Infrastructure Resilience —
A Network Model for Quantifying Cascading Vulnerability in the U.S. Generic
Oncology Drug Supply Chain. Phase 1: Cisplatin, Carboplatin, and Methotrexate.
April 2026. https://doi.org/10.5281/zenodo.19548610
```

\---

## Related Work

* **NPAORF** — National Pharmaceutical Asset Optimization \& Resilience Framework: A complementary facility-level predictive maintenance platform for pharmaceutical manufacturing equipment. DOI: [10.5281/zenodo.19310969](https://doi.org/10.5281/zenodo.19310969)

\---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

\---

## Medical Disclaimer

This project is an independent research initiative. It is not affiliated with, endorsed by, or representative of any government agency, pharmaceutical manufacturer, or healthcare institution. The model outputs are computational estimates derived from publicly available trade data and parameterized assumptions. They do not constitute medical advice, clinical guidance, or official risk assessments. Drug shortage information should be verified through the FDA Drug Shortage Database and institutional pharmacy channels. Treatment decisions should be made in consultation with qualified healthcare providers.

\---

## Contact

For questions, collaboration proposals, or policy inquiries, open an issue on this repository or contact the author directly.

