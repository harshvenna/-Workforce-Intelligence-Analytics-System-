# Workforce Engagement & Burnout Intelligence: A Composite-Index Framework for HR Analytics

### Palo Alto Networks Workforce Intelligence Platform — Research Documentation

---

## Abstract

This paper documents the methodology behind a composite-index HR analytics
platform built on a 1,470-employee illustrative dataset. We construct five
transparent, weighted engineered indices (Engagement, Burnout Risk,
Career Stagnation, Workforce Health, Early-Warning) from 31 raw HR fields,
validate them against actual attrition outcomes, and layer machine learning,
explainability (SHAP), unsupervised segmentation, and a scenario simulator
on top. We report all model metrics honestly, including cases where
deliberately excluding circular predictors produces modest (not impressive)
results — a design choice favoring intellectual honesty over inflated
performance claims.

## 1. Introduction

Traditional HR dashboards report raw satisfaction survey averages in
isolation. This platform instead builds composite indices that combine
multiple correlated-but-distinct signals (e.g., four separate satisfaction
dimensions into one Engagement Index), on the premise that workforce
experience is multi-dimensional and no single field captures it.

## 2. Data & Methodology

- **Source**: 1,470 employees, 31 fields, 0% missing, 0 duplicates.
- **Disclosure**: structurally identical to the public IBM HR Analytics
  Attrition dataset, relabeled for this project. Treated as illustrative
  throughout, not verified real personnel data.
- **Validation approach**: every engineered formula's weights were checked
  against empirical attrition lift before being finalized (e.g., Overtime
  shows 10.4%→30.5% attrition lift, justifying its 35% weight in the
  Burnout Risk formula — the single largest weight among five inputs).

## 3. Feature Engineering & KPI Framework

| Index | Formula (weights) | Range |
|---|---|---|
| Engagement Index | 30% Job Involvement + 30% Job Satisfaction + 20% Environment Satisfaction + 20% Relationship Satisfaction | 0–100 |
| Burnout Risk Score | 35% Overtime + 25% Work-Life Balance (inv.) + 15% Travel + 10% Commute + 15% Role Tenure Fatigue | 0–100 |
| Career Stagnation Score | Peer-relative z-score of Years Since Last Promotion (same Dept + Job Level) | 0–100 |
| Workforce Health Score | 35% Engagement + 30% (100−Burnout) + 20% Work-Life Balance + 15% Satisfaction Stability | 0–100 |
| Early-Warning Score | 30% (100−Engagement) + 30% Burnout + 20% (100−WLB) + 20% Career Stagnation | 0–100 |

All formulas are fully transparent and documented in-code — no black-box
scoring. The Early-Warning Score is deliberately kept separate from, and
shown alongside, the ML attrition model (Section 6) rather than replacing it,
since a rule-based score is auditable/explainable to an employee in ways a
tree ensemble is not.

## 4. EDA Findings

- Attrition base rate: 16.1%.
- Sales shows the highest attrition rate (~20.6%) of the three departments;
  R&D the lowest (~13.8%).
- Relationship Satisfaction is the uniformly weakest engagement
  sub-component across all three departments (56.5–63.0/100 range).
- Department sample sizes are highly uneven (R&D 961, Sales 446, HR 63) —
  HR-specific findings should be read with wider error margins.

## 5. Statistical Findings

- Overtime shows the strongest single empirical link to both attrition
  (10.4%→30.5%) and Burnout Risk Score of any factor tested.
- Commute distance shows only a weak correlation with Burnout Risk
  (|r| typically < 0.2), consistent with its low (10%) formula weight.
- With only 3 departments, tertile/percentile-based "bottom-tier" flagging
  is not statistically meaningful — a fixed-margin threshold (>5 points
  below the best performer) was used instead in Department Intelligence.

## 6. Machine Learning Results

Three tasks, three algorithms each (Logistic Regression, Random Forest,
XGBoost), all trained on the full org dataset (not filtered subsets),
with `sample_weight` balancing for class imbalance:

| Task | Best Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|---|
| Attrition Prediction (real label) | Random Forest | 0.86 | 0.83 | 0.79–0.80 |
| Burnout Risk Classification* | Random Forest | 0.72 | 0.65 | 0.55–0.58 |
| Engagement Level Classification* | Random Forest | 0.57 | 0.55 | 0.61–0.64 |

*Burnout and Engagement tasks deliberately exclude their own formula's
direct inputs to avoid a trivial circular prediction problem. The modest
results are the honest, expected outcome of that choice — not a modeling
failure. Attrition prediction, using a genuine independent label, performs
meaningfully better.

## 7. Explainability Findings (SHAP)

SHAP TreeExplainer applied to the XGBoost model per task, computed on the
one-hot-encoded feature space (not collapsed back to raw columns, to avoid
misrepresenting per-category effects). Multi-class tasks (Engagement) use
class-averaged SHAP values for global summary views, with per-class
waterfall explanations available for individual predictions.

## 8. Segmentation

K-Means (k=4, selected via silhouette comparison across k=3,4,5; k=4 scored
highest at ~0.27 — a modest but best-available separation, reported without
overclaiming strong natural clusters). Clusters are labeled by a
profile-based rule (ranked on Engagement−Burnout "health score" and tenure),
not hardcoded cluster indices, since KMeans label order is arbitrary:

- **Champions** — best engagement/burnout combination.
- **Stable Contributors** / **Quiet Survivors** — differentiated by tenure
  among the two middle clusters.
- **Flight Risks** — worst engagement/burnout combination.

## 9. Simulation Methodology

The HR Scenario Simulator re-runs the exact same `engineer_features()`
formulas used everywhere else in the platform (single source of truth),
with modified inputs:

- **Overtime reduction**: targeted removal, highest-Burnout-Risk employees
  relieved first (not random).
- **Work-Life Balance boost**: flat additive adjustment, capped at scale max.
- **Promotion frequency**: an **asymmetric** adjustment applied only to
  employees above their peer-group median. This was a necessary fix — a
  uniform multiplicative or additive scaling of Years Since Last Promotion
  has **zero effect** on Career Stagnation Score, because that score is a
  peer-relative z-score, and z-scores are mathematically invariant under
  any uniform linear rescaling within a group. This was caught via direct
  numerical testing during development (see repository build notes) rather
  than shipped as a silent no-op lever.

## 10. Discussion & HR Recommendations

- Overtime reduction is the single highest-leverage lever available in this
  dataset, given its outsized empirical relationship to both burnout and
  attrition.
- Relationship Satisfaction, being uniformly the weakest engagement driver,
  warrants investigation independent of department — possibly a
  cross-department management-practice issue rather than a departmental one.
- Given the 3-department limitation, org-wide findings (Overtime, tenure
  patterns) are more statistically robust than department-comparison
  findings, which should be treated as directional rather than definitive.

## 11. Limitations

- Synthetic/illustrative dataset — findings are methodological
  demonstrations, not real organizational diagnostics.
- Small number of departments (3) limits statistical power for
  department-level comparisons.
- Burnout and Engagement ML classifiers are intentionally weakened by
  excluding their formula's direct inputs; this is a methodological choice
  for avoiding circularity, not a claim that burnout/engagement are
  unpredictable in general.
- HR department (n=63) is a small-sample slice in every department cut.

## 12. Conclusion

The platform demonstrates a full-stack approach to HR analytics: transparent
composite indices validated against real outcomes, genuine (not inflated)
ML performance reporting, explainability at the individual level, honest
unsupervised segmentation, and a simulator that is mathematically consistent
with the rest of the system — including a real invariance bug that was
caught and fixed during development rather than shipped unnoticed.
