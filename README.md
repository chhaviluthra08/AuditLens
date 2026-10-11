# 🛡️ AuditLens: Credit Model Reliability & Fair Lending Audit Tool

AuditLens is an independent credit model auditing tool built with Streamlit and Python. A risk officer uploads a CSV of credit model default probabilities and realized borrower outcomes, adjusts operational risk parameters, and receives an end-to-end evaluation: discrimination, calibration, population stability (PSI), per-group reliability via conformal prediction, human-review capacity optimization, and downloadable executive HTML reporting.

---

## 1. What It Does

1. **Performance & Calibration**: Computes ROC AUC, Kolmogorov-Smirnov (KS) statistic with two-sample p-value, Brier score, and empirical calibration reliability curves.
2. **Population Stability (PSI)**: Quantifies score distribution drift between calibration and test partitions using smoothed quantile binning.
3. **Conformal Reliability Audit**:
   - **Marginal Split Conformal**: Generates finite-sample valid prediction sets $\{0\}$, $\{1\}$, $\{0, 1\}$, or $\emptyset$ at nominal confidence $1 - \alpha$ using the finite-sample rank correction $\lceil (n+1)(1-\alpha) \rceil / n$.
   - **Mondrian (Group-Aware) Conformal**: Calibrates per-group nonconformity thresholds to guarantee valid group-conditional coverage across demographic and operational segments.
   - **Statistical Under-Coverage Flagging**: Subgroups are flagged if and only if:
     1. They are not unreliable ($n \ge 100$),
     2. Their Bonferroni-adjusted Wilson score interval upper bound is strictly below target coverage ($\text{Upper}_{\text{Wilson}} < 1 - \alpha$), and
     3. Their empirical coverage gap is material ($\text{Coverage} \le 1 - \alpha - \text{tolerance}$).
4. **Human Review Capacity & Cost Simulation**:
   - Maps sets: $\{0\} \to$ Approve, $\{1\} \to$ Refuse, $\{0, 1\}$ or $\emptyset \to$ Defer to human review.
   - Evaluates selective risk (error rate on automated decisions) and expected cost $J = r \cdot P(\text{Approve} \land \text{Default}) + P(\text{Refuse} \land \text{Repaid}) + \delta \cdot P(\text{Defer})$.
   - Searches an $\alpha$ grid strictly on calibration data to identify the smallest $\alpha$ fitting human review capacity, preventing test-set data leakage.
5. **Downloadable Executive Report**: Exports a self-contained, offline-compatible HTML report rendered with autoescape security via Jinja2.

---

## 2. Project Scope & Architecture

- **Scope**: Built as a clean, transparent student project demo. Purely local execution without authentication, databases, or API model uploads.
- **Strict Separation**:
  - `core/`: Pure, testable scientific functions without Streamlit dependencies.
  - `app/`: Pure Streamlit presentation layer calling `core/` functions with `st.cache_data`.
  - `scripts/`: Data ingestion and demo generation.
  - `tests/`: Comprehensive unit, property, acceptance, and UI smoke tests.

```text
AuditLens/
├── app/
│   ├── __init__.py
│   ├── main.py                  # Main Streamlit presentation layer & routing
│   └── tabs/
│       ├── tab_upload.py        # Tab 1: Upload & Quality overview
│       ├── tab_performance.py   # Tab 2: Discrimination & calibration plots
│       ├── tab_audit.py         # Tab 3: Split vs Mondrian reliability dot plot
│       └── tab_capacity.py      # Tab 4: Capacity constraint simulation
├── core/
│   ├── config.py                # AuditLensConfig frozen dataclass (no magic numbers)
│   ├── errors.py                # Custom exceptions with plain-English fix hints
│   ├── log_setup.py             # Centralized logging (console + logs/auditlens.log)
│   ├── validation.py            # CSV schema and input contract validation
│   ├── splits.py                # Stratified 50/50 calibration/test partitioning
│   ├── performance.py           # AUC, KS statistic, Brier score, PSI
│   ├── conformal.py             # Split & Mondrian conformal prediction sets
│   ├── decisions.py             # Decision mapping, risk, cost J, capacity search
│   ├── audit.py                 # Per-group Wilson CIs, Bonferroni adjustment, flagging
│   └── report.py                # Jinja2 HTML report generator with autoescaping
├── data/
│   ├── raw/                     # Raw UCI dataset (UCI_Credit_Card.csv)
│   └── demo/                    # Stored demo CSVs (good_model.csv, flawed_model.csv)
├── scripts/
│   └── make_demo_data.py        # LightGBM training on UCI data + intentional distortion
├── tests/                       # Unit, property, acceptance, and AppTest smoke tests
├── Makefile                     # make run, test, lint, format, demo-data
├── pyproject.toml               # pytest, ruff, black configuration
└── requirements.txt             # Pinned virtual environment dependencies
```

---

## 3. Installation & Setup

Requirements: **Python 3.11+**

```bash
# 1. Clone or navigate to the directory
cd AuditLens

# 2. Create virtual environment
python3 -m venv .venv

# 3. Activate virtual environment
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 4. One-Command Run

To launch the Streamlit application:

```bash
make run
```
*(Or directly: `.venv/bin/streamlit run app/main.py`)*

Open your browser at `http://localhost:8501`.

---

## 5. Demo Walkthrough: Good vs. Flawed Model

AuditLens includes two pre-computed demo models derived from the UCI *Default of Credit Card Clients* dataset (15,000 holdout observations):

### Demo 1: Good Model (`Good Model (UCI LightGBM)`)
1. In the sidebar under **Dataset Ingestion**, select `Good Model (UCI LightGBM)`.
2. Navigate to **Tab 1 (Upload & Quality)**: Observe 15,000 rows with 22.1% base default rate and zero schema errors.
3. In **Tab 2 (Performance & Stability)**: Observe strong discrimination ($\text{AUC} \approx 0.787$, $\text{KS} \approx 0.445$) and stable calibration ($\text{PSI} < 0.01$).
4. In **Tab 3 (Reliability Audit)**:
   - Select attribute `age_band`.
   - All subgroups (`< 25`, `25-34`, `35-49`, `50+`) achieve $\approx 90\%$ split coverage within statistical tolerance.
   - **Zero subgroups are flagged** for under-coverage.

### Demo 2: Flawed Model (`Flawed Model (Biased < 25)`)
1. In the sidebar under **Dataset Ingestion**, select `Flawed Model (Biased < 25)`.
2. Navigate to **Tab 3 (Reliability Audit)**:
   - Select attribute `age_band`.
   - **Alert Triggered**: A red alert highlights statistically clear under-coverage for subgroup `< 25`.
   - The dot plot shows Split Conformal coverage drops to **$\approx 79\text{--}81\%$** (with Wilson upper bound $< 90\%$).
   - **Mondrian Fix**: The green diamond shows Mondrian (group-aware) conformal prediction automatically raises the threshold for `< 25`, restoring empirical coverage back to $\ge 90\%$.
   - Switch attribute to `credit_limit_band`: Notice subgroup `< 50k` also exhibits under-coverage. Because 64.6% of applicants aged `< 25` hold credit limits under 50k, the demographic bias naturally propagates to correlated operational limits.

---

## 6. How to Regenerate Demo Data

If `data/demo/*.csv` are deleted or need refreshing:

```bash
make demo-data
```

This executes `scripts/make_demo_data.py`. If `data/raw/UCI_Credit_Card.csv` is present, it trains LightGBM on the 15,000-row training partition and evaluates on the disjoint 15,000-row holdout partition. If the raw file is absent, a clearly labelled synthetic credit dataset is generated as a fallback.

---

## 7. Regulatory Topic Mapping

> **Note**: Regulatory topic mapping is indicative only; verify against the final RBI guidance. This tool uses conceptual topic labels only and does not cite statutory clauses or paragraph numbers.

| Regulatory Topic | AuditLens Evidence & Implementation |
| :--- | :--- |
| **Independent Validation** | Stratified 50/50 calibration and test partitioning with distribution-free nonconformity quantile calibration. |
| **Ongoing Monitoring** | Population Stability Index (PSI) score drift tracking between calibration baseline and evaluation data. |
| **Explainability** | Empirical binned reliability curves and score distribution comparisons between good and bad borrowers. |
| **Bias and Fairness** | Side-by-side Split Conformal vs Mondrian reliability auditing across protected demographic subgroups. |
| **Model Inventory** | Consolidated self-contained HTML audit report capturing data provenance, configuration parameters, and findings. |

---

## 8. Known Scope & Limitations

1. **Sample Dependency**: The audit reflects only the uploaded sample observations and does not guarantee future operational invariance.
2. **Single-Column Mondrian**: Mondrian conformal prediction is currently evaluated per single group column at a time, not for combinatorial group intersections.
3. **Multiplicity Correction**: Subgroup flagging utilizes a dynamic Bonferroni adjustment ($1 - 0.05 / n_{\text{groups}}$) and a minimum gap threshold ($\text{tolerance} = 0.03$) to minimize false alarms.
4. **Human Review Assumption**: Expected cost $J$ assumes human reviewers correctly resolve all deferred cases without residual classification error.
5. **Data Provenance**: Demo data is derived from the 2005 Taiwan UCI Credit Card dataset.
6. **Out-of-Time Validation**: The system assumes exchangeability between calibration and test partitions and does not independently audit time stamps for out-of-time stability unless a timestamp column is provided.

---

## 9. Testing & Code Quality

```bash
# Run all unit, property, acceptance, and UI smoke tests
make test

# Run ruff lint and black format verification
make lint

# Auto-format codebase
make format
```
