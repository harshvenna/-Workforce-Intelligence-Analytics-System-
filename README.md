# Palo Alto Networks — Workforce Engagement & Burnout Intelligence Platform

An enterprise-grade Streamlit analytics platform covering workforce engagement,
burnout risk, career stagnation, attrition early-warning, ML prediction,
explainability (SHAP), segmentation, and an HR scenario simulator.

## ⚠️ Data Disclosure

The underlying dataset (`data/Palo_Alto_Networks.xlsx`) is structurally
identical to the widely-used public **IBM HR Analytics Attrition** dataset
(1,470 employees, 31 fields), relabeled for this project. It is used here as
**illustrative/synthetic data** to demonstrate platform architecture and
analytical methodology — it is not verified real Palo Alto Networks
personnel data. Every page in the app carries an "Illustrative Data" badge
for this reason.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires Python 3.10+. First load of the Machine Learning Hub / Explainable
AI Center will take a few seconds to train models (cached afterward via
`st.cache_resource`).

## Architecture

Single file (`app.py`), internally sectioned into 14 modules matching the
sidebar navigation, plus a shared Foundation layer (Module 1):

| Module | Section |
|---|---|
| 1 | Foundation — data loading, feature engineering, KPI engine, filters |
| 2 | Executive Overview |
| 3 | Employee Experience EDA |
| 4 | Engagement Intelligence Center |
| 5 | Burnout Risk Center |
| 6 | Workload Stress Analytics |
| 7 | Department Intelligence |
| 8 | Career & Growth Analytics |
| 9 | Attrition Early Warning Center |
| 10 | Machine Learning Hub |
| 11 | Explainable AI Center (SHAP) |
| 12 | Workforce Segmentation (K-Means) |
| 13 | HR Scenario Simulator |
| 14 | Workforce Risk Analytics + Manager Action Center |

## Key Design Decisions

- **Engineered KPIs** (Engagement Index, Burnout Risk Score, Workforce
  Health Score, Career Stagnation Score, Early-Warning Score) are all
  transparent, documented weighted formulas — not black-box scores. See
  inline comments in `engineer_features()` (Module 1.2) for exact weights
  and the empirical rationale behind each.
- **ML models train on the full org dataset**, never the sidebar-filtered
  selection — filtering to a small subset would starve models of data.
- **Burnout/Engagement ML classifiers deliberately exclude their own
  formula's direct inputs** to avoid a trivial circular prediction task.
  Expect modest performance (AUC ~0.55–0.65) on these two tasks — that's
  the honest, correct result, not a bug. The Attrition model (a real label,
  not a formula) performs better (AUC ~0.77–0.80).
- **The HR Scenario Simulator re-runs the exact same formulas** used
  everywhere else in the app (single source of truth) rather than a
  separate toy model. Note: the Promotion Frequency lever uses an
  asymmetric (above-peer-median-only) adjustment rather than uniform
  scaling, because Career Stagnation Score is a peer-relative z-score and
  is mathematically invariant to uniform linear rescaling — this was
  caught and fixed during build/testing (see build notes).

## Known Limitations

- Department Intelligence flagging (Module 7) uses a fixed 5-point margin
  threshold, which is more defensible than tertile ranking with only 3
  departments in the data — but conclusions from 3 groups should be read
  with appropriate caution regardless of methodology.
- K-Means segmentation (Module 12) achieves a modest silhouette score
  (~0.27) — reported honestly rather than implying strong natural cluster
  separation.
- HR department (63 employees) is a small-n slice in every department-level
  cut; treat HR-specific findings with wider error margins than R&D/Sales.
