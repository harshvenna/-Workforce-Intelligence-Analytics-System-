"""
Palo Alto Networks — Workforce Engagement & Burnout Intelligence Platform
============================================================================
An enterprise-grade HR analytics platform covering engagement, burnout,
career stagnation, and early-warning attrition intelligence.

DATA DISCLOSURE: The underlying dataset is a synthetic/illustrative HR
dataset (structurally identical to the widely-used IBM HR Analytics
Attrition dataset) relabeled for this project. It is used here to
demonstrate platform architecture and analytical methodology, not as
verified real-world Palo Alto Networks personnel data. See README.md.

Author: Built with Claude (Anthropic) as lead architect / implementation partner.
============================================================================

FILE MAP (single-file, sectioned for navigability):
    MODULE 1  - Foundation: config, data loading, feature engineering, KPI engine
                (THIS MODULE — sidebar filters + shell wired in, page router stubbed)
    MODULE 2  - Executive Overview                        [pending]
    MODULE 3  - Employee Experience EDA                   [pending]
    MODULE 4  - Engagement Intelligence Center             [pending]
    MODULE 5  - Burnout Risk Center                        [pending]
    MODULE 6  - Workload Stress Analytics                  [pending]
    MODULE 7  - Department Intelligence                    [pending]
    MODULE 8  - Career & Growth Analytics                  [pending]
    MODULE 9  - Attrition Early Warning Center              [pending]
    MODULE 10 - Machine Learning Hub                       [pending]
    MODULE 11 - Explainable AI Center                      [pending]
    MODULE 12 - Workforce Segmentation                     [pending]
    MODULE 13 - HR Scenario Simulator                      [pending]
    MODULE 14 - Workforce Risk Analytics + Manager Action Center [pending]
============================================================================
"""

# =============================================================================
# MODULE 1.0 — IMPORTS & PAGE CONFIG
# =============================================================================
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")

APP_TITLE = "Palo Alto Networks — Workforce Intelligence Platform"
DATA_PATH = Path(__file__).parent / "data" / "Palo_Alto_Networks.xlsx"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Cybersecurity-enterprise visual identity (CSS injection)
# Palette: near-black background, ember-orange accent (nods to PANW's brand
# without claiming to BE PANW's brand), cool greys for structure.
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    :root {
        --accent: #FA582D;
        --accent-dim: #C2431E;
        --bg-panel: #141922;
        --bg-panel-alt: #1A2029;
        --border-soft: #2A3140;
        --text-dim: #8A94A6;
    }
    .stApp { background-color: #0B0E14; }
    section[data-testid="stSidebar"] {
        background-color: #0D1117;
        border-right: 1px solid var(--border-soft);
    }
    h1, h2, h3 { letter-spacing: -0.02em; }
    h1 { color: #F2F4F7 !important; font-weight: 700 !important; }
    h2, h3 { color: #E8EAED !important; font-weight: 600 !important; }

    div[data-testid="stMetric"] {
        background-color: var(--bg-panel);
        border: 1px solid var(--border-soft);
        border-left: 3px solid var(--accent);
        border-radius: 6px;
        padding: 14px 16px 10px 16px;
    }
    div[data-testid="stMetric"] label { color: var(--text-dim) !important; }
    div[data-testid="stMetricValue"] { color: #F2F4F7 !important; }

    .platform-header {
        display: flex; align-items: center; gap: 14px;
        padding: 18px 22px; margin-bottom: 6px;
        background: linear-gradient(90deg, #141922 0%, #0B0E14 100%);
        border: 1px solid var(--border-soft); border-radius: 8px;
        border-left: 4px solid var(--accent);
    }
    .platform-header .badge {
        font-size: 11px; font-weight: 700; letter-spacing: 0.08em;
        color: var(--accent); background: rgba(250,88,45,0.12);
        border: 1px solid rgba(250,88,45,0.35);
        padding: 3px 9px; border-radius: 4px; text-transform: uppercase;
    }
    .platform-header .subtitle { color: var(--text-dim); font-size: 13px; margin-top: 2px; }

    .section-tag {
        display: inline-block; font-size: 11px; font-weight: 700;
        color: var(--accent); letter-spacing: 0.08em; text-transform: uppercase;
        border-bottom: 2px solid var(--accent); padding-bottom: 4px; margin-bottom: 10px;
    }
    .insight-box {
        background-color: var(--bg-panel-alt); border: 1px solid var(--border-soft);
        border-left: 3px solid var(--accent); border-radius: 6px;
        padding: 12px 16px; margin: 8px 0; font-size: 14px; color: #D7DBE2;
    }
    .risk-critical { color: #FF4D4D !important; font-weight: 700; }
    .risk-high { color: #FF9A3D !important; font-weight: 700; }
    .risk-medium { color: #FFD23D !important; font-weight: 600; }
    .risk-low { color: #3DD68C !important; font-weight: 600; }

    div[data-testid="stDataFrame"] { border: 1px solid var(--border-soft); border-radius: 6px; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background-color: var(--bg-panel); border-radius: 6px 6px 0 0;
        border: 1px solid var(--border-soft); border-bottom: none;
    }
    .stTabs [aria-selected="true"] { border-bottom: 3px solid var(--accent) !important; }
    footer, #MainMenu { visibility: hidden; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# MODULE 1.1 — DATA LOADING & VALIDATION
# =============================================================================
REQUIRED_COLUMNS = [
    "Age", "Attrition", "BusinessTravel", "DailyRate", "Department",
    "DistanceFromHome", "Education", "EducationField", "EnvironmentSatisfaction",
    "Gender", "HourlyRate", "JobInvolvement", "JobLevel", "JobRole",
    "JobSatisfaction", "MaritalStatus", "MonthlyIncome", "MonthlyRate",
    "NumCompaniesWorked", "OverTime", "PercentSalaryHike", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "TotalWorkingYears",
    "TrainingTimesLastYear", "WorkLifeBalance", "YearsAtCompany",
    "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager",
]

SATISFACTION_SCALE_MAP = {1: "Low", 2: "Medium", 3: "High", 4: "Very High"}
WLB_SCALE_MAP = {1: "Bad", 2: "Good", 3: "Better", 4: "Best"}


@st.cache_data(show_spinner=False)
def load_raw_data(path: Path) -> pd.DataFrame:
    """
    Load the source Excel file and run structural validation.
    Raises a clear error rather than failing silently downstream if the
    schema doesn't match what every other module assumes.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place 'Palo_Alto_Networks.xlsx' in the /data folder."
        )

    df = pd.read_excel(path)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    # Defensive dtype coercion — the platform must not silently misbehave
    # if a future data refresh introduces nulls or type drift, even though
    # the current file is 100% complete / 0 duplicates (verified at build time).
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip()

    # Drop fully-duplicate rows and rows missing on core identity fields,
    # defensively — a no-op on the current clean file, but this is what
    # makes the pipeline safe against a future messier data refresh.
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.dropna(subset=["Age", "Department", "JobRole"]).reset_index(drop=True)
    dropped = before - len(df)

    df["EmployeeID"] = np.arange(1, len(df) + 1)

    df.attrs["dropped_rows"] = dropped
    df.attrs["load_timestamp"] = pd.Timestamp.now()
    return df


def data_quality_report(df: pd.DataFrame) -> dict:
    """Lightweight, honest data-quality summary shown in the sidebar / footer."""
    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "missing_cells": int(df.isnull().sum().sum()),
        "missing_pct": round(df.isnull().sum().sum() / df.size * 100, 3),
        "dropped_rows": df.attrs.get("dropped_rows", 0),
        "duplicate_rows": int(df.duplicated().sum()),
    }


# =============================================================================
# MODULE 1.2 — FEATURE ENGINEERING
# =============================================================================
@st.cache_data(show_spinner=False)
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds every derived field used across the platform. Each formula and
    its weighting rationale is documented inline — these are NOT arbitrary;
    weights were set after checking each factor's empirical relationship
    to actual Attrition in this dataset (see build notes / research paper).
    """
    df = df.copy()

    # ---- Rescale helper: Likert 1-4 -> 0-100 ----
    def rescale_1_4(series):
        return (series - 1) / 3 * 100

    # -------------------------------------------------------------------
    # ENGAGEMENT INDEX (0-100)
    # Weights: JobInvolvement 0.30, JobSatisfaction 0.30,
    #          EnvironmentSatisfaction 0.20, RelationshipSatisfaction 0.20
    # Involvement + satisfaction weighted highest as the most direct proxies
    # for day-to-day engagement; environment/relationship are contextual.
    # -------------------------------------------------------------------
    df["EngagementIndex"] = (
        0.30 * rescale_1_4(df["JobInvolvement"])
        + 0.30 * rescale_1_4(df["JobSatisfaction"])
        + 0.20 * rescale_1_4(df["EnvironmentSatisfaction"])
        + 0.20 * rescale_1_4(df["RelationshipSatisfaction"])
    ).round(1)

    def engagement_category(score):
        if score >= 75:
            return "Thriving"
        elif score >= 50:
            return "Healthy"
        elif score >= 30:
            return "At Risk"
        else:
            return "Critical"

    df["EngagementCategory"] = df["EngagementIndex"].apply(engagement_category)

    # -------------------------------------------------------------------
    # BURNOUT RISK SCORE (0-100)
    # Weights derived from empirical attrition lift in this dataset:
    #   OverTime=Yes shows the strongest lift (10.4% -> 30.5% attrition) -> 0.35
    #   WorkLifeBalance=1 shows the next strongest lift (31.3% vs 14.2%) -> 0.25
    #   BusinessTravel (frequent travel) -> 0.15
    #   DistanceFromHome (commute strain, min-max normalized) -> 0.10
    #   Extended YearsInCurrentRole w/o change (role fatigue proxy) -> 0.15
    # -------------------------------------------------------------------
    overtime_component = np.where(df["OverTime"] == "Yes", 100.0, 0.0)
    wlb_inverted = 100 - rescale_1_4(df["WorkLifeBalance"])

    travel_map = {"Non-Travel": 0.0, "Travel_Rarely": 40.0, "Travel_Frequently": 100.0}
    travel_component = df["BusinessTravel"].map(travel_map).fillna(40.0)

    dist_min, dist_max = df["DistanceFromHome"].min(), df["DistanceFromHome"].max()
    commute_component = (df["DistanceFromHome"] - dist_min) / (dist_max - dist_min) * 100

    # Role fatigue: normalized YearsInCurrentRole, capped — long stretches
    # in the same role without progression is a burnout contributor, not
    # just a career-stagnation one (the two frameworks intentionally overlap
    # here; that's realistic, not a bug).
    role_fatigue = np.clip(df["YearsInCurrentRole"] / 10 * 100, 0, 100)

    df["BurnoutRiskScore"] = (
        0.35 * overtime_component
        + 0.25 * wlb_inverted
        + 0.15 * travel_component
        + 0.10 * commute_component
        + 0.15 * role_fatigue
    ).round(1)

    def burnout_category(score):
        if score >= 70:
            return "Critical"
        elif score >= 50:
            return "High"
        elif score >= 30:
            return "Medium"
        else:
            return "Low"

    df["BurnoutRiskCategory"] = df["BurnoutRiskScore"].apply(burnout_category)

    # -------------------------------------------------------------------
    # WORK-LIFE BALANCE INDEX (0-100) — direct rescale
    # -------------------------------------------------------------------
    df["WorkLifeBalanceIndex"] = rescale_1_4(df["WorkLifeBalance"]).round(1)
    df["WorkLifeBalanceLabel"] = df["WorkLifeBalance"].map(WLB_SCALE_MAP)

    # -------------------------------------------------------------------
    # SATISFACTION STABILITY SCORE (0-100)
    # 100 minus the normalized std-dev across the 4 satisfaction facets.
    # Someone rating 4/4/4/4 is "stable-high"; someone rating 4/1/4/1 has
    # the same mean as a 2.5-flat employee but a very different experience
    # — this catches that volatility.
    # -------------------------------------------------------------------
    satisfaction_cols = ["JobInvolvement", "JobSatisfaction",
                          "EnvironmentSatisfaction", "RelationshipSatisfaction"]
    row_std = df[satisfaction_cols].std(axis=1)
    max_possible_std = df[satisfaction_cols].std(axis=1).max()
    max_possible_std = max_possible_std if max_possible_std > 0 else 1.0
    df["SatisfactionStabilityScore"] = (100 - (row_std / max_possible_std * 100)).round(1)

    # -------------------------------------------------------------------
    # CAREER STAGNATION SCORE (0-100, higher = more stagnant)
    # Peer-relative: compares YearsSinceLastPromotion against the median
    # for the same Department + JobLevel, then z-scores and caps at ±3SD
    # mapped to 0-100. Peer-relative because 3 years without promotion
    # means something very different for a JobLevel 1 vs JobLevel 5 role.
    # -------------------------------------------------------------------
    peer_median = df.groupby(["Department", "JobLevel"])["YearsSinceLastPromotion"].transform("median")
    peer_std = df.groupby(["Department", "JobLevel"])["YearsSinceLastPromotion"].transform("std").fillna(1.0)
    peer_std = peer_std.replace(0, 1.0)
    z = (df["YearsSinceLastPromotion"] - peer_median) / peer_std
    z_clipped = np.clip(z, -3, 3)
    df["CareerStagnationScore"] = ((z_clipped + 3) / 6 * 100).round(1)

    def stagnation_category(score):
        if score >= 70:
            return "Severe"
        elif score >= 50:
            return "Elevated"
        elif score >= 30:
            return "Mild"
        else:
            return "None"

    df["CareerStagnationCategory"] = df["CareerStagnationScore"].apply(stagnation_category)

    # -------------------------------------------------------------------
    # WORKFORCE HEALTH SCORE (0-100) — umbrella KPI
    # -------------------------------------------------------------------
    df["WorkforceHealthScore"] = (
        0.35 * df["EngagementIndex"]
        + 0.30 * (100 - df["BurnoutRiskScore"])
        + 0.20 * df["WorkLifeBalanceIndex"]
        + 0.15 * df["SatisfactionStabilityScore"]
    ).round(1)

    # -------------------------------------------------------------------
    # EARLY-WARNING SCORE (0-100, higher = more concerning) + categories
    # Transparent rule-based blend — this is intentionally SEPARATE from
    # the ML attrition model (Module 10). The ML model learns patterns
    # from the labeled data; this score is a human-auditable formula HR
    # can explain to an employee if asked. Both are shown side by side
    # in Module 9 so neither is presented as the sole source of truth.
    # -------------------------------------------------------------------
    df["EarlyWarningScore"] = (
        0.30 * (100 - df["EngagementIndex"])
        + 0.30 * df["BurnoutRiskScore"]
        + 0.20 * (100 - df["WorkLifeBalanceIndex"])
        + 0.20 * df["CareerStagnationScore"]
    ).round(1)

    def warning_category(score):
        if score >= 65:
            return "Critical"
        elif score >= 50:
            return "High Concern"
        elif score >= 35:
            return "Watchlist"
        else:
            return "Stable"

    df["EarlyWarningCategory"] = df["EarlyWarningScore"].apply(warning_category)

    # -------------------------------------------------------------------
    # Convenience bands for filtering / cross-tabs
    # -------------------------------------------------------------------
    df["TenureBand"] = pd.cut(
        df["YearsAtCompany"],
        bins=[-0.1, 2, 5, 10, 20, 100],
        labels=["0-2 yrs", "3-5 yrs", "6-10 yrs", "11-20 yrs", "20+ yrs"],
    )
    df["AgeBand"] = pd.cut(
        df["Age"],
        bins=[17, 25, 35, 45, 55, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "56+"],
    )
    df["IncomeQuartile"] = pd.qcut(
        df["MonthlyIncome"], q=4, labels=["Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)"]
    )
    df["AttritionLabel"] = df["Attrition"].map({0: "Retained", 1: "Attrited"})

    return df


# =============================================================================
# MODULE 1.3 — KPI ENGINE
# =============================================================================
def compute_kpis(df: pd.DataFrame) -> dict:
    """
    Computes all Executive Overview KPIs for a given (already filtered)
    dataframe. Kept as a pure function so it can be called on the full
    org population AND the filtered selection, side by side.
    """
    if len(df) == 0:
        return {k: np.nan for k in [
            "engagement_index", "burnout_risk_score", "wlb_index",
            "stability_score", "workforce_health", "attrition_rate",
            "high_risk_count", "high_risk_pct", "overtime_ratio", "headcount",
        ]}

    high_risk_mask = (
        df["BurnoutRiskCategory"].isin(["High", "Critical"])
        | df["EarlyWarningCategory"].isin(["High Concern", "Critical"])
    )

    return {
        "engagement_index": df["EngagementIndex"].mean(),
        "burnout_risk_score": df["BurnoutRiskScore"].mean(),
        "wlb_index": df["WorkLifeBalanceIndex"].mean(),
        "stability_score": df["SatisfactionStabilityScore"].mean(),
        "workforce_health": df["WorkforceHealthScore"].mean(),
        "attrition_rate": df["Attrition"].mean() * 100,
        "high_risk_count": int(high_risk_mask.sum()),
        "high_risk_pct": high_risk_mask.mean() * 100,
        "overtime_ratio": (df["OverTime"] == "Yes").mean() * 100,
        "headcount": len(df),
    }


def kpi_delta(filtered_val: float, org_val: float) -> str:
    """Formats a delta string for st.metric comparing filtered selection to org baseline."""
    if pd.isna(filtered_val) or pd.isna(org_val) or org_val == 0:
        return None
    diff = filtered_val - org_val
    return f"{diff:+.1f} vs org"


# =============================================================================
# MODULE 1.4 — SIDEBAR: NAVIGATION + GLOBAL FILTERS
# =============================================================================
PAGES = [
    "Executive Overview",
    "Employee Experience EDA",
    "Engagement Intelligence Center",
    "Burnout Risk Center",
    "Workload Stress Analytics",
    "Department Intelligence",
    "Career & Growth Analytics",
    "Attrition Early Warning Center",
    "Machine Learning Hub",
    "Explainable AI Center",
    "Workforce Segmentation",
    "HR Scenario Simulator",
    "Workforce Risk Analytics",
    "Manager Action Center",
]


def render_sidebar(df: pd.DataFrame):
    """
    Renders navigation + the global filter set specified in requirements:
    Department, Role, Overtime, Engagement threshold, Tenure.
    Returns (selected_page, filtered_dataframe, filter_summary_dict).
    """
    with st.sidebar:
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">
                <div style="font-size:22px;">🛡️</div>
                <div>
                    <div style="font-weight:700; font-size:15px; color:#F2F4F7; line-height:1.2;">
                        PANW Workforce
                    </div>
                    <div style="font-size:11px; color:#8A94A6; letter-spacing:0.05em;">
                        INTELLIGENCE PLATFORM
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        page = st.radio("Navigate", PAGES, label_visibility="collapsed")

        st.divider()
        st.markdown("**Global Filters**")

        departments = sorted(df["Department"].unique().tolist())
        sel_departments = st.multiselect("Department", departments, default=departments)

        roles = sorted(df["JobRole"].unique().tolist())
        sel_roles = st.multiselect("Job Role", roles, default=roles)

        overtime_opt = st.selectbox("Overtime Status", ["All", "Yes", "No"], index=0)

        eng_min, eng_max = st.slider(
            "Engagement Index Threshold",
            min_value=0, max_value=100, value=(0, 100),
            help="Filter employees by Engagement Index range (0-100).",
        )

        tenure_bands = df["TenureBand"].cat.categories.tolist() if hasattr(df["TenureBand"], "cat") else sorted(df["TenureBand"].dropna().unique().tolist())
        sel_tenure = st.multiselect("Tenure Band", tenure_bands, default=tenure_bands)

        st.divider()
        with st.expander("Data Quality Snapshot"):
            dq = data_quality_report(df)
            st.caption(f"Records: {dq['n_rows']:,}")
            st.caption(f"Missing cells: {dq['missing_cells']} ({dq['missing_pct']}%)")
            st.caption(f"Duplicate rows removed: {dq['dropped_rows']}")
            st.caption("Source: synthetic/illustrative HR dataset. See README.")

    # ---- Apply filters ----
    filtered = df[
        df["Department"].isin(sel_departments)
        & df["JobRole"].isin(sel_roles)
        & df["EngagementIndex"].between(eng_min, eng_max)
        & df["TenureBand"].isin(sel_tenure)
    ]
    if overtime_opt != "All":
        filtered = filtered[filtered["OverTime"] == overtime_opt]

    filter_summary = {
        "departments": sel_departments,
        "roles": sel_roles,
        "overtime": overtime_opt,
        "engagement_range": (eng_min, eng_max),
        "tenure_bands": sel_tenure,
    }

    return page, filtered, filter_summary


# =============================================================================
# MODULE 1.5 — SHARED HEADER + PLACEHOLDER PAGE RENDERER
# =============================================================================
def render_header():
    st.markdown(
        f"""
        <div class="platform-header">
            <div style="font-size:30px;">🛡️</div>
            <div>
                <div style="font-size:20px; font-weight:700; color:#F2F4F7;">
                    {APP_TITLE}
                </div>
                <div class="subtitle">
                    Workforce Engagement · Burnout Risk · Career Stagnation · Attrition Early Warning
                    <span class="badge" style="margin-left:10px;">Illustrative Data</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_placeholder(page_name: str, filtered_df: pd.DataFrame, kpis: dict):
    """
    Temporary stand-in for pages not yet built (Modules 2-14).
    Confirms the foundation layer (filters, KPI engine, engineered fields)
    is wired correctly end-to-end before any dashboard-specific code is added.
    Will be replaced module by module.
    """
    st.markdown(f'<div class="section-tag">{page_name}</div>', unsafe_allow_html=True)
    st.info(
        f"**{page_name}** is scheduled for the next build module. "
        f"Foundation layer is active — {kpis['headcount']:,} employees match current filters."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Engagement Index", f"{kpis['engagement_index']:.1f}")
    c2.metric("Burnout Risk Score", f"{kpis['burnout_risk_score']:.1f}")
    c3.metric("Workforce Health", f"{kpis['workforce_health']:.1f}")
    c4.metric("Attrition Rate", f"{kpis['attrition_rate']:.1f}%")

    with st.expander("Preview engineered dataset (first 25 rows of filtered selection)"):
        preview_cols = [
            "EmployeeID", "Department", "JobRole", "EngagementIndex", "EngagementCategory",
            "BurnoutRiskScore", "BurnoutRiskCategory", "WorkforceHealthScore",
            "CareerStagnationCategory", "EarlyWarningCategory", "Attrition",
        ]
        st.dataframe(filtered_df[preview_cols].head(25), use_container_width=True, hide_index=True)


# =============================================================================
# MODULE 2 — EXECUTIVE OVERVIEW
# =============================================================================
# Design decisions (locked in with stakeholder before building):
#   - Gauges: only the 3 most critical KPIs (Engagement, Burnout, Workforce
#     Health). Each gauge shows the FILTERED selection as the needle value,
#     and the ORG-WIDE average as a white threshold line — this is the
#     "benchmark" the gauge is read against, so filtering by department
#     immediately shows over/under-performance vs the whole company.
#   - Remaining 5 KPIs as metric cards with delta-vs-org.
#   - Department Comparison: single ranked bar chart with a metric toggle,
#     rather than a dense multi-metric grid — prioritizes readability.
#   - Workforce Snapshot: 2x2 demographic breakdown (dept / role / tenure / age).
# =============================================================================

GAUGE_BANDS = {
    "Engagement Index": {
        "bands": [(0, 30, "#FF4D4D"), (30, 50, "#FF9A3D"), (50, 75, "#FFD23D"), (75, 100, "#3DD68C")],
        "higher_is_better": True,
    },
    "Burnout Risk Score": {
        "bands": [(0, 30, "#3DD68C"), (30, 50, "#FFD23D"), (50, 70, "#FF9A3D"), (70, 100, "#FF4D4D")],
        "higher_is_better": False,
    },
    "Workforce Health Score": {
        "bands": [(0, 30, "#FF4D4D"), (30, 50, "#FF9A3D"), (50, 75, "#FFD23D"), (75, 100, "#3DD68C")],
        "higher_is_better": True,
    },
}


def render_gauge(title: str, filtered_value: float, org_value: float) -> go.Figure:
    """
    Gauge with the filtered-selection value as the needle and the org-wide
    average rendered as a white threshold line — this is the benchmark the
    number should be read against. Color bands reflect the same category
    thresholds used in feature engineering (Module 1.2), so the gauge zones
    are consistent with EngagementCategory / BurnoutRiskCategory labels
    used everywhere else in the platform.
    """
    config = GAUGE_BANDS[title]
    steps = [{"range": [s, e], "color": c} for s, e, c in config["bands"]]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(filtered_value, 1),
        number={"suffix": "", "font": {"size": 30, "color": "#F2F4F7"}},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8A94A6", "tickfont": {"color": "#8A94A6", "size": 10}},
            "bar": {"color": "#FA582D", "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": steps,
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.85,
                "value": round(org_value, 1),
            },
        },
    ))
    fig.update_layout(
        height=210,
        margin=dict(l=20, r=20, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E8EAED"},
        title={"text": title, "font": {"size": 13, "color": "#8A94A6"}, "x": 0.5},
    )
    return fig


DEPT_COMPARISON_METRICS = {
    "Workforce Health Score": ("WorkforceHealthScore", True),
    "Engagement Index": ("EngagementIndex", True),
    "Burnout Risk Score": ("BurnoutRiskScore", False),
    "Work-Life Balance Index": ("WorkLifeBalanceIndex", True),
    "Attrition Rate (%)": ("Attrition", False),  # special-cased: mean*100
    "Overtime Ratio (%)": ("OverTime", False),   # special-cased: % Yes
}


def render_department_comparison(df: pd.DataFrame):
    st.markdown("**Department Comparison**")
    metric_label = st.selectbox(
        "Metric", list(DEPT_COMPARISON_METRICS.keys()), key="dept_comparison_metric"
    )
    col, higher_is_better = DEPT_COMPARISON_METRICS[metric_label]

    if col == "Attrition":
        grouped = df.groupby("Department")["Attrition"].mean().mul(100).reset_index(name="value")
    elif col == "OverTime":
        grouped = df.groupby("Department")["OverTime"].apply(lambda s: (s == "Yes").mean() * 100).reset_index(name="value")
    else:
        grouped = df.groupby("Department")[col].mean().reset_index(name="value")

    grouped = grouped.sort_values("value", ascending=higher_is_better)

    fig = px.bar(
        grouped, x="value", y="Department", orientation="h",
        text=grouped["value"].round(1),
        color="value",
        color_continuous_scale=(
            ["#FF4D4D", "#FFD23D", "#3DD68C"] if higher_is_better
            else ["#3DD68C", "#FFD23D", "#FF4D4D"]
        ),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=280, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E8EAED"}, coloraxis_showscale=False,
        xaxis_title=metric_label, yaxis_title=None,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Auto-generated insight — computed from the actual grouped result, not templated text.
    best_dept, worst_dept = grouped.iloc[0]["Department"], grouped.iloc[-1]["Department"]
    best_val, worst_val = grouped.iloc[0]["value"], grouped.iloc[-1]["value"]
    if higher_is_better:
        insight = (f"**{worst_dept}** trails on {metric_label} ({worst_val:.1f} vs "
                    f"{best_dept}'s {best_val:.1f}) — the largest gap in the current selection.")
    else:
        insight = (f"**{best_dept}** carries the highest {metric_label} ({best_val:.1f}) — "
                    f"an intervention priority relative to {worst_dept} ({worst_val:.1f}).")
    st.markdown(f'<div class="insight-box">📊 {insight}</div>', unsafe_allow_html=True)


def render_workforce_snapshot(df: pd.DataFrame):
    st.markdown("**Workforce Snapshot**")
    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)

    with r1c1:
        dept_counts = df["Department"].value_counts().reset_index()
        dept_counts.columns = ["Department", "Headcount"]
        fig = px.pie(dept_counts, names="Department", values="Headcount", hole=0.55,
                     color_discrete_sequence=["#FA582D", "#4D8CFF", "#3DD68C", "#FFD23D"])
        fig.update_layout(height=240, margin=dict(l=10, r=10, t=25, b=10),
                           paper_bgcolor="rgba(0,0,0,0)", font={"color": "#E8EAED"},
                           title={"text": "Headcount by Department", "font": {"size": 12, "color": "#8A94A6"}},
                           legend={"font": {"size": 10}})
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        role_counts = df["JobRole"].value_counts().reset_index().head(8)
        role_counts.columns = ["JobRole", "Headcount"]
        fig = px.bar(role_counts.sort_values("Headcount"), x="Headcount", y="JobRole", orientation="h",
                     color_discrete_sequence=["#FA582D"])
        fig.update_layout(height=240, margin=dict(l=10, r=10, t=25, b=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font={"color": "#E8EAED"},
                           title={"text": "Headcount by Job Role (Top 8)", "font": {"size": 12, "color": "#8A94A6"}},
                           yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

    with r2c1:
        tenure_order = ["0-2 yrs", "3-5 yrs", "6-10 yrs", "11-20 yrs", "20+ yrs"]
        tenure_counts = df["TenureBand"].value_counts().reindex(tenure_order).reset_index()
        tenure_counts.columns = ["TenureBand", "Headcount"]
        fig = px.bar(tenure_counts, x="TenureBand", y="Headcount", color_discrete_sequence=["#4D8CFF"])
        fig.update_layout(height=240, margin=dict(l=10, r=10, t=25, b=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font={"color": "#E8EAED"},
                           title={"text": "Headcount by Tenure Band", "font": {"size": 12, "color": "#8A94A6"}},
                           xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

    with r2c2:
        age_order = ["18-25", "26-35", "36-45", "46-55", "56+"]
        age_counts = df["AgeBand"].value_counts().reindex(age_order).reset_index()
        age_counts.columns = ["AgeBand", "Headcount"]
        fig = px.bar(age_counts, x="AgeBand", y="Headcount", color_discrete_sequence=["#3DD68C"])
        fig.update_layout(height=240, margin=dict(l=10, r=10, t=25, b=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font={"color": "#E8EAED"},
                           title={"text": "Headcount by Age Band", "font": {"size": 12, "color": "#8A94A6"}},
                           xaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)


def render_executive_overview(df_filtered: pd.DataFrame, df_org: pd.DataFrame,
                               kpis_filtered: dict, kpis_org: dict):
    st.markdown('<div class="section-tag">Executive Overview</div>', unsafe_allow_html=True)
    st.caption(
        f"{kpis_filtered['headcount']:,} employees in current selection "
        f"(of {kpis_org['headcount']:,} total). Gauge threshold line = org-wide average."
    )

    # ---- Gauges: 3 most critical KPIs ----
    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(
            render_gauge("Engagement Index", kpis_filtered["engagement_index"], kpis_org["engagement_index"]),
            use_container_width=True,
        )
    with g2:
        st.plotly_chart(
            render_gauge("Burnout Risk Score", kpis_filtered["burnout_risk_score"], kpis_org["burnout_risk_score"]),
            use_container_width=True,
        )
    with g3:
        st.plotly_chart(
            render_gauge("Workforce Health Score", kpis_filtered["workforce_health"], kpis_org["workforce_health"]),
            use_container_width=True,
        )

    # ---- Metric cards: remaining 5 KPIs ----
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Work-Life Balance Index", f"{kpis_filtered['wlb_index']:.1f}",
              kpi_delta(kpis_filtered['wlb_index'], kpis_org['wlb_index']))
    m2.metric("Satisfaction Stability", f"{kpis_filtered['stability_score']:.1f}",
              kpi_delta(kpis_filtered['stability_score'], kpis_org['stability_score']))
    m3.metric("Attrition Rate", f"{kpis_filtered['attrition_rate']:.1f}%",
              kpi_delta(kpis_filtered['attrition_rate'], kpis_org['attrition_rate']), delta_color="inverse")
    m4.metric("High-Risk Employees", f"{kpis_filtered['high_risk_count']:,}",
              f"{kpis_filtered['high_risk_pct']:.1f}% of selection")
    m5.metric("Overtime Ratio", f"{kpis_filtered['overtime_ratio']:.1f}%",
              kpi_delta(kpis_filtered['overtime_ratio'], kpis_org['overtime_ratio']), delta_color="inverse")

    st.divider()

    col_left, col_right = st.columns([1, 1])
    with col_left:
        render_department_comparison(df_filtered)
    with col_right:
        render_workforce_snapshot(df_filtered)


# =============================================================================
# MODULE 3 — EMPLOYEE EXPERIENCE EDA
# =============================================================================
# Design decisions (locked in with stakeholder before building):
#   - Histogram/boxplot: single interactive field selector (Age, Gender,
#     Department, Education, Job Role, Job Level), not a fixed 6-panel grid.
#   - Histogram is stacked by EngagementCategory (not a plain count) so
#     demographic distribution and engagement mix are visible in one view —
#     otherwise this section is just headcount reporting, not "employee
#     experience" analysis.
#   - Boxplot shows WorkforceHealthScore split across the selected field's
#     categories — ties every demographic cut back to the umbrella KPI.
#   - Treemap: Department -> JobRole, sized by headcount, colored by
#     WorkforceHealthScore (avg, weighted) — surfaces which specific
#     role/department combinations are struggling, not just departments.
#   - Sunburst: Department -> JobLevel -> EngagementCategory — shows where
#     in the org hierarchy engagement breaks down.
#   - Correlation: curated rectangular heatmap (demographic/tenure fields
#     as rows x engineered KPIs as columns), not a full 20x20 matrix —
#     answers "what drives our KPIs" directly instead of burying it in noise.
# =============================================================================

EDUCATION_LABELS = {1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"}

ENGAGEMENT_COLOR_MAP = {
    "Critical": "#FF4D4D", "At Risk": "#FF9A3D", "Healthy": "#FFD23D", "Thriving": "#3DD68C",
}
ENGAGEMENT_ORDER = ["Critical", "At Risk", "Healthy", "Thriving"]

FIELD_EDA_CONFIG = {
    "Age": {"col": "Age", "box_x": "AgeBand", "kind": "continuous"},
    "Gender": {"col": "Gender", "box_x": "Gender", "kind": "categorical"},
    "Department": {"col": "Department", "box_x": "Department", "kind": "categorical"},
    "Education": {"col": "Education", "box_x": "Education", "kind": "ordinal"},
    "Job Role": {"col": "JobRole", "box_x": "JobRole", "kind": "categorical"},
    "Job Level": {"col": "JobLevel", "box_x": "JobLevel", "kind": "ordinal"},
}

DEMO_ROLE_NUMERIC_FIELDS = [
    "Age", "Education", "JobLevel", "DistanceFromHome", "NumCompaniesWorked",
    "TotalWorkingYears", "YearsAtCompany", "YearsInCurrentRole",
    "YearsSinceLastPromotion", "YearsWithCurrManager",
]
ENGINEERED_KPI_FIELDS = [
    "EngagementIndex", "BurnoutRiskScore", "WorkLifeBalanceIndex",
    "SatisfactionStabilityScore", "CareerStagnationScore",
    "WorkforceHealthScore", "EarlyWarningScore",
]


def render_field_distribution(df: pd.DataFrame, field_label: str):
    cfg = FIELD_EDA_CONFIG[field_label]
    plot_df = df.copy()

    hc1, hc2 = st.columns(2)

    # ---- Histogram: distribution stacked by EngagementCategory ----
    with hc1:
        if field_label == "Age":
            fig = px.histogram(
                plot_df, x="Age", nbins=20, color="EngagementCategory",
                category_orders={"EngagementCategory": ENGAGEMENT_ORDER},
                color_discrete_map=ENGAGEMENT_COLOR_MAP,
            )
        else:
            if field_label == "Education":
                plot_df["_x"] = plot_df["Education"].map(EDUCATION_LABELS)
                x_order = list(EDUCATION_LABELS.values())
            elif field_label == "Job Level":
                plot_df["_x"] = "Level " + plot_df["JobLevel"].astype(str)
                x_order = [f"Level {i}" for i in sorted(df["JobLevel"].unique())]
            else:
                plot_df["_x"] = plot_df[cfg["col"]]
                x_order = sorted(df[cfg["col"]].unique().tolist())

            fig = px.histogram(
                plot_df, x="_x", color="EngagementCategory",
                category_orders={"_x": x_order, "EngagementCategory": ENGAGEMENT_ORDER},
                color_discrete_map=ENGAGEMENT_COLOR_MAP,
            )
        fig.update_layout(
            height=320, margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#E8EAED"}, barmode="stack",
            title={"text": f"{field_label} Distribution (by Engagement Category)",
                   "font": {"size": 13, "color": "#8A94A6"}},
            xaxis_title=None, legend={"font": {"size": 10}, "title": None},
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---- Boxplot: WorkforceHealthScore across the field's categories ----
    with hc2:
        box_df = df.copy()
        box_x = cfg["box_x"]
        if field_label == "Education":
            box_df["_x"] = box_df["Education"].map(EDUCATION_LABELS)
            x_order = list(EDUCATION_LABELS.values())
        elif field_label == "Job Level":
            box_df["_x"] = "Level " + box_df["JobLevel"].astype(str)
            x_order = [f"Level {i}" for i in sorted(df["JobLevel"].unique())]
        elif field_label == "Age":
            box_df["_x"] = box_df["AgeBand"]
            x_order = ["18-25", "26-35", "36-45", "46-55", "56+"]
        else:
            box_df["_x"] = box_df[box_x]
            x_order = sorted(df[box_x].unique().tolist())

        fig2 = px.box(
            box_df, x="_x", y="WorkforceHealthScore", color="_x",
            category_orders={"_x": x_order}, points=False,
        )
        fig2.update_layout(
            height=320, margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#E8EAED"}, showlegend=False,
            title={"text": f"Workforce Health Score by {field_label}",
                   "font": {"size": 13, "color": "#8A94A6"}},
            xaxis_title=None,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ---- Auto-generated insight ----
    group_means = box_df.groupby("_x", observed=True)["WorkforceHealthScore"].mean().dropna()
    if len(group_means) >= 2:
        best_cat, worst_cat = group_means.idxmax(), group_means.idxmin()
        gap = group_means.max() - group_means.min()
        st.markdown(
            f'<div class="insight-box">📊 Within <b>{field_label}</b>, '
            f'<b>{best_cat}</b> shows the highest Workforce Health Score '
            f'({group_means.max():.1f}) while <b>{worst_cat}</b> is lowest '
            f'({group_means.min():.1f}) — a {gap:.1f}-point gap.</div>',
            unsafe_allow_html=True,
        )


def render_treemap(df: pd.DataFrame):
    plot_df = df.copy()
    plot_df["_count"] = 1
    fig = px.treemap(
        plot_df, path=["Department", "JobRole"], values="_count",
        color="WorkforceHealthScore", color_continuous_scale=["#FF4D4D", "#FFD23D", "#3DD68C"],
        color_continuous_midpoint=df["WorkforceHealthScore"].mean(),
    )
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#E8EAED"},
        title={"text": "Headcount by Department \u2192 Job Role (color = Workforce Health)",
               "font": {"size": 13, "color": "#8A94A6"}},
        coloraxis_colorbar={"title": None},
    )
    fig.update_traces(textinfo="label+value")
    st.plotly_chart(fig, use_container_width=True)


def render_sunburst(df: pd.DataFrame):
    plot_df = df.copy()
    plot_df["JobLevel"] = "Level " + plot_df["JobLevel"].astype(str)
    plot_df["_count"] = 1
    fig = px.sunburst(
        plot_df, path=["Department", "JobLevel", "EngagementCategory"], values="_count",
        color="EngagementCategory", color_discrete_map=ENGAGEMENT_COLOR_MAP,
    )
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#E8EAED"},
        title={"text": "Department \u2192 Job Level \u2192 Engagement Category",
               "font": {"size": 13, "color": "#8A94A6"}},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_correlation_analysis(df: pd.DataFrame):
    st.markdown("**Correlation: Demographic/Tenure Fields vs Engineered KPIs**")
    corr_full = df[DEMO_ROLE_NUMERIC_FIELDS + ENGINEERED_KPI_FIELDS].corr()
    corr = corr_full.loc[DEMO_ROLE_NUMERIC_FIELDS, ENGINEERED_KPI_FIELDS]

    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
    )
    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#E8EAED", "size": 11},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Auto-generated insight — strongest single correlation, by absolute value.
    stacked = corr.stack()
    stacked = stacked[stacked.index.get_level_values(0) != stacked.index.get_level_values(1)]
    if len(stacked) > 0:
        top_idx = stacked.abs().idxmax()
        top_val = stacked.loc[top_idx]
        direction = "positively" if top_val > 0 else "negatively"
        st.markdown(
            f'<div class="insight-box">📊 The strongest relationship in this selection: '
            f'<b>{top_idx[0]}</b> correlates {direction} with <b>{top_idx[1]}</b> '
            f'(r = {top_val:+.2f}). Treat this as descriptive, not causal — '
            f'correlation strength here is generally weak-to-moderate, consistent '
            f'with this being a multi-factor, composite-index framework rather than '
            f'a single-driver system.</div>',
            unsafe_allow_html=True,
        )


def render_employee_experience_eda(df: pd.DataFrame):
    st.markdown('<div class="section-tag">Employee Experience EDA</div>', unsafe_allow_html=True)

    field_label = st.selectbox("Analyze field:", list(FIELD_EDA_CONFIG.keys()), key="eda_field_selector")
    render_field_distribution(df, field_label)

    st.divider()
    t1, t2 = st.columns(2)
    with t1:
        render_treemap(df)
    with t2:
        render_sunburst(df)

    st.divider()
    render_correlation_analysis(df)


# =============================================================================
# MODULE 4 — ENGAGEMENT INTELLIGENCE CENTER
# =============================================================================
# Design decisions (locked in with stakeholder before building):
#   - Distribution: bar of headcount per EngagementCategory, stacked by
#     Department — literal "stacked bar," matches KPI card language
#     (Thriving/Healthy/At Risk/Critical), and the department stacking adds
#     a second layer of insight for free without complicating the read.
#   - Department Ranking and Role Ranking are two separate ranked bar
#     charts (not combined/faceted) — clean, unambiguous, easy to scan.
#   - Heatmap axes: the 4 engagement sub-components x Department. This
#     answers "WHICH input is dragging engagement down, and WHERE" — more
#     diagnostic than a plain Department x JobRole engagement grid, which
#     would just re-show what the ranking bar already shows.
# =============================================================================

ENGAGEMENT_SUBCOMPONENTS = {
    "Job Involvement": "JobInvolvement",
    "Job Satisfaction": "JobSatisfaction",
    "Environment Satisfaction": "EnvironmentSatisfaction",
    "Relationship Satisfaction": "RelationshipSatisfaction",
}


def render_engagement_distribution(df: pd.DataFrame):
    counts = df.groupby(["EngagementCategory", "Department"], observed=True).size().reset_index(name="Headcount")

    fig = px.bar(
        counts, x="EngagementCategory", y="Headcount", color="Department",
        category_orders={"EngagementCategory": ENGAGEMENT_ORDER},
        color_discrete_sequence=["#FA582D", "#4D8CFF", "#3DD68C", "#FFD23D"],
    )
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E8EAED"}, barmode="stack",
        title={"text": "Headcount by Engagement Category (stacked by Department)",
               "font": {"size": 13, "color": "#8A94A6"}},
        xaxis_title=None, legend={"font": {"size": 10}, "title": None},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Auto-generated insight
    total = len(df)
    cat_totals = df["EngagementCategory"].value_counts()
    thriving_pct = cat_totals.get("Thriving", 0) / total * 100
    critical_pct = cat_totals.get("Critical", 0) / total * 100
    at_risk_pct = cat_totals.get("At Risk", 0) / total * 100
    st.markdown(
        f'<div class="insight-box">📊 <b>{thriving_pct:.1f}%</b> of the current selection is '
        f'Thriving, while <b>{critical_pct + at_risk_pct:.1f}%</b> falls into At Risk or Critical '
        f'engagement bands ({critical_pct:.1f}% Critical) — this combined group is the addressable '
        f'population for engagement intervention.</div>',
        unsafe_allow_html=True,
    )


def render_engagement_rankings(df: pd.DataFrame):
    r1, r2 = st.columns(2)

    with r1:
        dept_rank = df.groupby("Department")["EngagementIndex"].mean().reset_index().sort_values("EngagementIndex")
        fig = px.bar(
            dept_rank, x="EngagementIndex", y="Department", orientation="h",
            text=dept_rank["EngagementIndex"].round(1), color="EngagementIndex",
            color_continuous_scale=["#FF4D4D", "#FFD23D", "#3DD68C"],
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            height=280, margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#E8EAED"}, coloraxis_showscale=False,
            title={"text": "Department Ranking \u2014 Mean Engagement Index",
                   "font": {"size": 13, "color": "#8A94A6"}},
            xaxis_title=None, yaxis_title=None,
        )
        st.plotly_chart(fig, use_container_width=True)

    with r2:
        role_rank = df.groupby("JobRole")["EngagementIndex"].mean().reset_index().sort_values("EngagementIndex")
        fig2 = px.bar(
            role_rank, x="EngagementIndex", y="JobRole", orientation="h",
            text=role_rank["EngagementIndex"].round(1), color="EngagementIndex",
            color_continuous_scale=["#FF4D4D", "#FFD23D", "#3DD68C"],
        )
        fig2.update_traces(textposition="outside")
        fig2.update_layout(
            height=280, margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#E8EAED"}, coloraxis_showscale=False,
            title={"text": "Role Ranking \u2014 Mean Engagement Index",
                   "font": {"size": 13, "color": "#8A94A6"}},
            xaxis_title=None, yaxis_title=None,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Auto-generated insight
    dept_worst, dept_worst_val = dept_rank.iloc[0]["Department"], dept_rank.iloc[0]["EngagementIndex"]
    role_worst, role_worst_val = role_rank.iloc[0]["JobRole"], role_rank.iloc[0]["EngagementIndex"]
    st.markdown(
        f'<div class="insight-box">📊 Lowest engagement: <b>{dept_worst}</b> department '
        f'({dept_worst_val:.1f}) and <b>{role_worst}</b> role ({role_worst_val:.1f}) — '
        f'cross-reference these against the heatmap below to see which specific input is driving it.</div>',
        unsafe_allow_html=True,
    )


def render_engagement_heatmap(df: pd.DataFrame):
    st.markdown("**Engagement Sub-Component Heatmap \u2014 Department \u00d7 Driver**")

    def rescale_1_4(s):
        return (s - 1) / 3 * 100

    rows = []
    for dept in sorted(df["Department"].unique()):
        dept_df = df[df["Department"] == dept]
        row = {"Department": dept}
        for label, col in ENGAGEMENT_SUBCOMPONENTS.items():
            row[label] = rescale_1_4(dept_df[col]).mean()
        rows.append(row)
    heat_df = pd.DataFrame(rows).set_index("Department")

    fig = px.imshow(
        heat_df, text_auto=".1f", color_continuous_scale="RdYlGn", zmin=0, zmax=100, aspect="auto",
    )
    fig.update_layout(
        height=280, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#E8EAED", "size": 11},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Auto-generated insight: weakest single (department, driver) cell
    stacked = heat_df.stack()
    weakest_dept, weakest_driver = stacked.idxmin()
    weakest_val = stacked.min()
    driver_means = heat_df.mean(axis=0).sort_values()
    weakest_driver_overall = driver_means.index[0]
    st.markdown(
        f'<div class="insight-box">📊 The single weakest point is <b>{weakest_driver}</b> in '
        f'<b>{weakest_dept}</b> ({weakest_val:.1f}/100). Org-wide, <b>{weakest_driver_overall}</b> '
        f'is the lowest-scoring driver on average ({driver_means.iloc[0]:.1f}) \u2014 the most '
        f'consistent lever across departments if a single intervention had to be prioritized.</div>',
        unsafe_allow_html=True,
    )


def render_engagement_intelligence_center(df: pd.DataFrame):
    st.markdown('<div class="section-tag">Engagement Intelligence Center</div>', unsafe_allow_html=True)
    st.caption(
        "Composite Engagement Index = 30% Job Involvement + 30% Job Satisfaction "
        "+ 20% Environment Satisfaction + 20% Relationship Satisfaction, rescaled 0\u2013100."
    )

    render_engagement_distribution(df)
    st.divider()
    render_engagement_rankings(df)
    st.divider()
    render_engagement_heatmap(df)


# =============================================================================
# MODULE 5 — BURNOUT RISK CENTER
# =============================================================================
# Design decisions (locked in with stakeholder before building):
#   - Heatmap: BurnoutRiskCategory x Department, row-normalized to % of
#     each department's headcount (NOT raw counts) — raw counts would be
#     dominated by R&D's size (961 vs HR's 63) and mask real concentration
#     differences. % within department is the honest comparison.
#   - Distribution: stacked bar by Department, mirroring Module 4's pattern
#     exactly for visual/analytical consistency across the platform.
#   - Employee Risk Segmentation: a ranked, actionable table of the
#     highest-burnout individuals (not a scatter) — this is meant for HR
#     to literally act on, one row at a time.
# =============================================================================

BURNOUT_ORDER = ["Low", "Medium", "High", "Critical"]
BURNOUT_COLOR_MAP = {"Low": "#3DD68C", "Medium": "#FFD23D", "High": "#FF9A3D", "Critical": "#FF4D4D"}


def render_burnout_heatmap(df: pd.DataFrame):
    st.markdown("**Burnout Risk Concentration \u2014 Department \u00d7 Risk Category (% of department headcount)**")

    cross = pd.crosstab(df["Department"], df["BurnoutRiskCategory"], normalize="index") * 100
    cross = cross.reindex(columns=BURNOUT_ORDER, fill_value=0)

    fig = px.imshow(
        cross, text_auto=".1f", color_continuous_scale="Reds", zmin=0, zmax=cross.values.max() if cross.values.max() > 0 else 100,
        aspect="auto",
    )
    fig.update_layout(
        height=260, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#E8EAED", "size": 11},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Auto-generated insight: department with highest combined High+Critical %
    high_critical_pct = (cross["High"] + cross["Critical"]).sort_values(ascending=False)
    top_dept = high_critical_pct.index[0]
    top_val = high_critical_pct.iloc[0]
    st.markdown(
        f'<div class="insight-box">🔥 <b>{top_dept}</b> has the highest concentration of '
        f'High/Critical burnout risk at <b>{top_val:.1f}%</b> of its headcount \u2014 '
        f'the top intervention priority by concentration (as opposed to raw headcount).</div>',
        unsafe_allow_html=True,
    )


def render_burnout_distribution(df: pd.DataFrame):
    counts = df.groupby(["BurnoutRiskCategory", "Department"], observed=True).size().reset_index(name="Headcount")

    fig = px.bar(
        counts, x="BurnoutRiskCategory", y="Headcount", color="Department",
        category_orders={"BurnoutRiskCategory": BURNOUT_ORDER},
        color_discrete_sequence=["#FA582D", "#4D8CFF", "#3DD68C", "#FFD23D"],
    )
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E8EAED"}, barmode="stack",
        title={"text": "Headcount by Burnout Risk Category (stacked by Department)",
               "font": {"size": 13, "color": "#8A94A6"}},
        xaxis_title=None, legend={"font": {"size": 10}, "title": None},
    )
    st.plotly_chart(fig, use_container_width=True)

    total = len(df)
    cat_totals = df["BurnoutRiskCategory"].value_counts()
    critical_pct = cat_totals.get("Critical", 0) / total * 100
    high_pct = cat_totals.get("High", 0) / total * 100
    st.markdown(
        f'<div class="insight-box">🔥 <b>{critical_pct:.1f}%</b> of the current selection is at '
        f'Critical burnout risk and <b>{high_pct:.1f}%</b> at High \u2014 combined, '
        f'<b>{critical_pct + high_pct:.1f}%</b> of this population warrants active monitoring.</div>',
        unsafe_allow_html=True,
    )


def render_burnout_department_ranking(df: pd.DataFrame):
    st.markdown("**Department Risk Ranking \u2014 Mean Burnout Risk Score**")
    dept_rank = df.groupby("Department")["BurnoutRiskScore"].mean().reset_index().sort_values(
        "BurnoutRiskScore", ascending=True
    )
    fig = px.bar(
        dept_rank, x="BurnoutRiskScore", y="Department", orientation="h",
        text=dept_rank["BurnoutRiskScore"].round(1), color="BurnoutRiskScore",
        color_continuous_scale=["#3DD68C", "#FFD23D", "#FF9A3D", "#FF4D4D"],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=260, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E8EAED"}, coloraxis_showscale=False,
        xaxis_title=None, yaxis_title=None,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_employee_risk_segmentation(df: pd.DataFrame):
    st.markdown("**Employee Risk Segmentation \u2014 Highest Individual Burnout Risk**")
    top_n = st.slider("Number of employees to show", min_value=10, max_value=30, value=20, key="burnout_top_n")

    cols = [
        "EmployeeID", "Department", "JobRole", "BurnoutRiskScore", "BurnoutRiskCategory",
        "OverTime", "WorkLifeBalanceLabel", "BusinessTravel", "DistanceFromHome",
        "YearsInCurrentRole", "EngagementIndex", "AttritionLabel",
    ]
    top_risk = df.nlargest(min(top_n, len(df)), "BurnoutRiskScore")[cols].reset_index(drop=True)

    st.dataframe(
        top_risk,
        use_container_width=True,
        hide_index=True,
        column_config={
            "EmployeeID": st.column_config.NumberColumn("ID", width="small"),
            "BurnoutRiskScore": st.column_config.ProgressColumn(
                "Burnout Score", min_value=0, max_value=100, format="%.1f"
            ),
            "EngagementIndex": st.column_config.ProgressColumn(
                "Engagement", min_value=0, max_value=100, format="%.1f"
            ),
            "BurnoutRiskCategory": st.column_config.TextColumn("Risk Tier"),
            "WorkLifeBalanceLabel": st.column_config.TextColumn("Work-Life Balance"),
            "DistanceFromHome": st.column_config.NumberColumn("Commute (mi)"),
            "YearsInCurrentRole": st.column_config.NumberColumn("Yrs in Role"),
            "AttritionLabel": st.column_config.TextColumn("Actual Outcome"),
        },
    )

    if len(top_risk) > 0:
        overtime_pct = (top_risk["OverTime"] == "Yes").mean() * 100
        attrited_pct = (top_risk["AttritionLabel"] == "Attrited").mean() * 100
        st.markdown(
            f'<div class="insight-box">🔥 Among these top {len(top_risk)} highest-risk employees, '
            f'<b>{overtime_pct:.0f}%</b> are on Overtime, and <b>{attrited_pct:.0f}%</b> have already '
            f'left the company \u2014 a useful (if retrospective) sanity check that this risk score '
            f'tracks real outcomes.</div>',
            unsafe_allow_html=True,
        )


def render_burnout_risk_center(df: pd.DataFrame):
    st.markdown('<div class="section-tag">Burnout Risk Center</div>', unsafe_allow_html=True)
    st.caption(
        "Burnout Risk Score = 35% Overtime + 25% Work-Life Balance (inverted) + 15% Travel Intensity "
        "+ 10% Commute Distance + 15% Role Fatigue (tenure-in-role), rescaled 0\u2013100."
    )

    render_burnout_heatmap(df)
    st.divider()
    render_burnout_distribution(df)
    st.divider()
    render_burnout_department_ranking(df)
    st.divider()
    render_employee_risk_segmentation(df)


# =============================================================================
# MODULE 6 — WORKLOAD STRESS ANALYTICS
# =============================================================================
# Design decisions (locked in with stakeholder before building):
#   - Overtime & Travel impact: violin plots (full distribution, not just
#     means) of Engagement and Burnout across each factor's levels.
#   - Commute impact: scatter + manually-fit OLS trendline (numpy polyfit,
#     not statsmodels — avoids adding a dependency for one trendline) of
#     raw DistanceFromHome vs both Engagement and Burnout.
#   - Factors are analyzed independently — no compounding/interaction view
#     in this section (that's deliberately reserved for later modules like
#     Workforce Risk Analytics, to avoid duplicating scope here).
# =============================================================================

def _violin_pair(df: pd.DataFrame, group_col: str, group_order: list, title_prefix: str):
    v1, v2 = st.columns(2)
    with v1:
        fig = px.violin(
            df, x=group_col, y="EngagementIndex", color=group_col, box=True, points=False,
            category_orders={group_col: group_order},
            color_discrete_sequence=["#4D8CFF", "#FA582D", "#3DD68C"],
        )
        fig.update_layout(
            height=320, margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#E8EAED"}, showlegend=False,
            title={"text": f"{title_prefix}: Engagement Index Distribution", "font": {"size": 13, "color": "#8A94A6"}},
            xaxis_title=None,
        )
        st.plotly_chart(fig, use_container_width=True)
    with v2:
        fig2 = px.violin(
            df, x=group_col, y="BurnoutRiskScore", color=group_col, box=True, points=False,
            category_orders={group_col: group_order},
            color_discrete_sequence=["#4D8CFF", "#FA582D", "#3DD68C"],
        )
        fig2.update_layout(
            height=320, margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#E8EAED"}, showlegend=False,
            title={"text": f"{title_prefix}: Burnout Risk Distribution", "font": {"size": 13, "color": "#8A94A6"}},
            xaxis_title=None,
        )
        st.plotly_chart(fig2, use_container_width=True)


def render_overtime_impact(df: pd.DataFrame):
    st.markdown("**Overtime Impact**")
    _violin_pair(df, "OverTime", ["No", "Yes"], "Overtime")

    eng_yes = df.loc[df["OverTime"] == "Yes", "EngagementIndex"].mean()
    eng_no = df.loc[df["OverTime"] == "No", "EngagementIndex"].mean()
    burn_yes = df.loc[df["OverTime"] == "Yes", "BurnoutRiskScore"].mean()
    burn_no = df.loc[df["OverTime"] == "No", "BurnoutRiskScore"].mean()
    if pd.notna(eng_yes) and pd.notna(eng_no) and pd.notna(burn_yes) and pd.notna(burn_no):
        st.markdown(
            f'<div class="insight-box">⚡ Employees on Overtime average <b>{eng_yes:.1f}</b> Engagement '
            f'vs <b>{eng_no:.1f}</b> for non-overtime ({eng_yes - eng_no:+.1f}), and '
            f'<b>{burn_yes:.1f}</b> Burnout Risk vs <b>{burn_no:.1f}</b> ({burn_yes - burn_no:+.1f}). '
            f'Overtime carries the single largest weight (35%) in the Burnout Risk formula, and this '
            f'gap is the empirical basis for that weighting.</div>',
            unsafe_allow_html=True,
        )


def render_travel_impact(df: pd.DataFrame):
    st.markdown("**Travel Impact**")
    order = ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
    order = [o for o in order if o in df["BusinessTravel"].unique()]
    _violin_pair(df, "BusinessTravel", order, "Travel")

    means = df.groupby("BusinessTravel")["BurnoutRiskScore"].mean()
    if len(means) >= 2:
        worst = means.idxmax()
        best = means.idxmin()
        st.markdown(
            f'<div class="insight-box">⚡ <b>{worst}</b> employees show the highest mean Burnout Risk '
            f'({means[worst]:.1f}) vs <b>{best}</b> at {means[best]:.1f} \u2014 a '
            f'{means[worst] - means[best]:.1f}-point spread across travel frequency.</div>',
            unsafe_allow_html=True,
        )


def render_commute_impact(df: pd.DataFrame):
    st.markdown("**Commute Impact**")
    c1, c2 = st.columns(2)

    def scatter_with_trend(x_col, y_col, color):
        x = df[x_col].values.astype(float)
        y = df[y_col].values.astype(float)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers", marker=dict(color=color, size=6, opacity=0.4), name="Employees",
        ))
        if len(x) >= 2 and np.std(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            x_line = np.array([x.min(), x.max()])
            y_line = slope * x_line + intercept
            fig.add_trace(go.Scatter(
                x=x_line, y=y_line, mode="lines", line=dict(color="white", width=2, dash="dash"),
                name="Trend (OLS)",
            ))
            r = np.corrcoef(x, y)[0, 1]
        else:
            r = np.nan
        fig.update_layout(
            height=320, margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#E8EAED"}, showlegend=False,
            title={"text": f"Distance from Home vs {y_col} (r={r:.2f})" if pd.notna(r) else f"Distance from Home vs {y_col}",
                   "font": {"size": 13, "color": "#8A94A6"}},
            xaxis_title="Distance from Home (mi)", yaxis_title=y_col,
        )
        return fig, r

    with c1:
        fig_e, r_e = scatter_with_trend("DistanceFromHome", "EngagementIndex", "#4D8CFF")
        st.plotly_chart(fig_e, use_container_width=True)
    with c2:
        fig_b, r_b = scatter_with_trend("DistanceFromHome", "BurnoutRiskScore", "#FA582D")
        st.plotly_chart(fig_b, use_container_width=True)

    if pd.notna(r_b):
        strength = "weak" if abs(r_b) < 0.2 else ("moderate" if abs(r_b) < 0.4 else "strong")
        st.markdown(
            f'<div class="insight-box">⚡ Commute distance shows a <b>{strength}</b> correlation with '
            f'Burnout Risk (r={r_b:+.2f}). Commute is intentionally weighted lowest (10%) in the '
            f'Burnout formula among the five inputs, consistent with this being a real but secondary '
            f'factor rather than a primary driver.</div>',
            unsafe_allow_html=True,
        )


def render_workload_stress_analytics(df: pd.DataFrame):
    st.markdown('<div class="section-tag">Workload Stress Analytics</div>', unsafe_allow_html=True)
    st.caption("Each factor analyzed independently against Engagement Index and Burnout Risk Score.")

    render_overtime_impact(df)
    st.divider()
    render_travel_impact(df)
    st.divider()
    render_commute_impact(df)


# =============================================================================
# MODULE 7 — DEPARTMENT INTELLIGENCE
# =============================================================================
# Design decisions (proceeded on best judgment per explicit "continue" —
# assumptions stated to the user, open to revision):
#   - Both a radar chart (shape comparison) AND a sortable scorecard table
#     (precise ranking) — with only 3 departments a radar alone reads thin,
#     and "rank departments best to worst" calls for an explicit table.
#   - Intervention priority: FLAG-based (bottom tier on 2+ of 4 metrics),
#     not a single averaged score — catches multi-dimensional problems a
#     simple average could mask.
#   - No job-role drill-down here — stays distinct from Module 4/5, which
#     already own role-level rankings.
# =============================================================================

DEPT_INTEL_METRICS = {
    "Engagement": ("EngagementIndex", True),
    "Burnout Risk": ("BurnoutRiskScore", False),
    "Satisfaction Stability": ("SatisfactionStabilityScore", True),
    "Workforce Health": ("WorkforceHealthScore", True),
}


def _department_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the department x metric table, plus per-metric bottom-tier flags.

    Flagging uses a FIXED MARGIN (>5 points worse than the best-performing
    department on that metric), not tertile/percentile ranking. With only
    3 departments in this dataset, tertile-based ranking always flags
    exactly one department per metric regardless of whether the gap is
    meaningful — a fixed margin only flags real, material gaps and scales
    correctly if more departments are added later.
    """
    FLAG_MARGIN = 5.0
    rows = []
    for dept in sorted(df["Department"].unique()):
        d = df[df["Department"] == dept]
        row = {"Department": dept}
        for label, (col, _) in DEPT_INTEL_METRICS.items():
            row[label] = d[col].mean()
        rows.append(row)
    scorecard = pd.DataFrame(rows)

    flag_cols = []
    for label, (col, higher_is_better) in DEPT_INTEL_METRICS.items():
        if higher_is_better:
            best_val = scorecard[label].max()
            gap = best_val - scorecard[label]
        else:
            best_val = scorecard[label].min()
            gap = scorecard[label] - best_val
        flag_col = f"_flag_{label}"
        scorecard[flag_col] = gap > FLAG_MARGIN
        flag_cols.append(flag_col)

    scorecard["FlagCount"] = scorecard[flag_cols].sum(axis=1)
    scorecard["InterventionPriority"] = scorecard["FlagCount"].apply(
        lambda n: "Critical" if n >= 3 else ("High" if n == 2 else ("Watch" if n == 1 else "Stable"))
    )
    return scorecard


def render_department_radar(df: pd.DataFrame, scorecard: pd.DataFrame):
    st.markdown("**Department Profile \u2014 Radar Comparison**")

    fig = go.Figure()
    colors = ["#FA582D", "#4D8CFF", "#3DD68C", "#FFD23D"]
    metric_labels = list(DEPT_INTEL_METRICS.keys())

    for i, (_, row) in enumerate(scorecard.iterrows()):
        # Normalize Burnout Risk for display so "further out" always means "better"
        # across all 4 axes (burnout is inverted since lower is better).
        values = []
        for label in metric_labels:
            v = row[label]
            if label == "Burnout Risk":
                v = 100 - v
            values.append(v)
        values.append(values[0])  # close the loop
        fig.add_trace(go.Scatterpolar(
            r=values, theta=metric_labels + [metric_labels[0]],
            fill="toself", name=row["Department"],
            line=dict(color=colors[i % len(colors)]),
            opacity=0.6,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="#2A3140", tickfont={"color": "#8A94A6"}),
            angularaxis=dict(tickfont={"color": "#E8EAED"}),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=400, margin=dict(l=40, r=40, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#E8EAED"},
        legend={"font": {"size": 11}},
    )
    st.caption("Note: Burnout Risk axis is inverted (100 \u2212 score) so that a larger shape always means better outcomes on every axis.")
    st.plotly_chart(fig, use_container_width=True)


def render_department_scorecard_table(scorecard: pd.DataFrame):
    st.markdown("**Department Scorecard \u2014 Ranked**")

    display_cols = ["Department"] + list(DEPT_INTEL_METRICS.keys()) + ["FlagCount", "InterventionPriority"]
    display_df = scorecard[display_cols].sort_values("FlagCount", ascending=False).reset_index(drop=True)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Engagement": st.column_config.ProgressColumn("Engagement", min_value=0, max_value=100, format="%.1f"),
            "Burnout Risk": st.column_config.ProgressColumn("Burnout Risk", min_value=0, max_value=100, format="%.1f"),
            "Satisfaction Stability": st.column_config.ProgressColumn("Satisfaction Stability", min_value=0, max_value=100, format="%.1f"),
            "Workforce Health": st.column_config.ProgressColumn("Workforce Health", min_value=0, max_value=100, format="%.1f"),
            "FlagCount": st.column_config.NumberColumn("Bottom-Tier Flags (of 4)", width="small"),
            "InterventionPriority": st.column_config.TextColumn("Priority"),
        },
    )

    critical_or_high = display_df[display_df["InterventionPriority"].isin(["Critical", "High"])]
    if len(critical_or_high) > 0:
        names = ", ".join(critical_or_high["Department"].tolist())
        st.markdown(
            f'<div class="insight-box">🎯 <b>Intervention priority:</b> {names} \u2014 flagged in the '
            f'bottom tier on 2 or more of the 4 core metrics simultaneously, indicating a '
            f'multi-dimensional (not single-factor) problem.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="insight-box">🎯 No department is flagged as bottom-tier on 2+ metrics '
            'simultaneously in the current selection \u2014 issues appear isolated to single metrics '
            'rather than compounding.</div>',
            unsafe_allow_html=True,
        )


def render_department_intelligence(df: pd.DataFrame):
    st.markdown('<div class="section-tag">Department Intelligence</div>', unsafe_allow_html=True)

    if df["Department"].nunique() < 2:
        st.info("Select 2 or more departments in the sidebar filter to enable department comparison.")
        return

    scorecard = _department_scorecard(df)
    render_department_radar(df, scorecard)
    st.divider()
    render_department_scorecard_table(scorecard)


GROWTH_BOTTLENECK_ORDER = [1, 2, 3, 4, 5]


def render_career_stagnation_bars(df: pd.DataFrame):
    st.markdown("**Career Stagnation Score \u2014 by Job Level & Department**")
    grouped = df.groupby(["JobLevel", "Department"])["CareerStagnationScore"].mean().reset_index()
    grouped["JobLevel"] = "Level " + grouped["JobLevel"].astype(str)
    fig = px.bar(
        grouped, x="JobLevel", y="CareerStagnationScore", color="Department", barmode="group",
        color_discrete_sequence=["#FA582D", "#4D8CFF", "#3DD68C", "#FFD23D"],
        category_orders={"JobLevel": [f"Level {i}" for i in GROWTH_BOTTLENECK_ORDER]},
    )
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E8EAED"}, xaxis_title=None, legend={"font": {"size": 10}, "title": None},
    )
    st.plotly_chart(fig, use_container_width=True)

    worst = grouped.loc[grouped["CareerStagnationScore"].idxmax()]
    st.markdown(
        f'<div class="insight-box">🪜 Highest stagnation: <b>{worst["Department"]}</b> at '
        f'<b>{worst["JobLevel"]}</b> (score {worst["CareerStagnationScore"]:.1f}/100, peer-relative '
        f'to same department + level).</div>',
        unsafe_allow_html=True,
    )


def render_growth_bottleneck_funnel(df: pd.DataFrame):
    st.markdown("**Growth Bottleneck \u2014 Headcount Funnel by Job Level**")
    counts = df["JobLevel"].value_counts().reindex(GROWTH_BOTTLENECK_ORDER, fill_value=0)
    fig = go.Figure(go.Funnel(
        y=[f"Level {i}" for i in GROWTH_BOTTLENECK_ORDER],
        x=counts.values,
        marker={"color": ["#FA582D", "#FF9A3D", "#FFD23D", "#4D8CFF", "#3DD68C"]},
        textinfo="value+percent initial",
    ))
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#E8EAED"},
    )
    st.plotly_chart(fig, use_container_width=True)

    if counts.iloc[0] > 0:
        narrowest_drop_idx = (counts.shift(1) - counts).iloc[1:].idxmax() if len(counts) > 1 else None
        if narrowest_drop_idx is not None and pd.notna(narrowest_drop_idx):
            drop_pct = (counts.shift(1) - counts) / counts.shift(1) * 100
            st.markdown(
                f'<div class="insight-box">🪜 The steepest headcount drop-off is entering '
                f'<b>Level {narrowest_drop_idx}</b> ({drop_pct.loc[narrowest_drop_idx]:.0f}% reduction '
                f'from the prior level) \u2014 the narrowest point in the career ladder.</div>',
                unsafe_allow_html=True,
            )


def render_promotion_delay_analysis(df: pd.DataFrame):
    st.markdown("**Promotion Delay \u2014 Years Since Last Promotion by Job Level**")
    plot_df = df.copy()
    plot_df["JobLevel"] = "Level " + plot_df["JobLevel"].astype(str)
    fig = px.box(
        plot_df, x="JobLevel", y="YearsSinceLastPromotion", color="JobLevel", points=False,
        category_orders={"JobLevel": [f"Level {i}" for i in GROWTH_BOTTLENECK_ORDER]},
    )
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E8EAED"}, showlegend=False, xaxis_title=None,
    )
    st.plotly_chart(fig, use_container_width=True)

    medians = df.groupby("JobLevel")["YearsSinceLastPromotion"].median()
    if len(medians) > 0:
        longest_level = medians.idxmax()
        st.markdown(
            f'<div class="insight-box">🪜 Level {longest_level} employees wait the longest for '
            f'promotion (median {medians[longest_level]:.1f} years since last promotion).</div>',
            unsafe_allow_html=True,
        )


def render_stagnation_category_breakdown(df: pd.DataFrame):
    st.markdown("**Stagnation Category Breakdown by Department**")
    order = ["None", "Mild", "Elevated", "Severe"]
    color_map = {"None": "#3DD68C", "Mild": "#FFD23D", "Elevated": "#FF9A3D", "Severe": "#FF4D4D"}
    counts = df.groupby(["CareerStagnationCategory", "Department"], observed=True).size().reset_index(name="Headcount")
    fig = px.bar(
        counts, x="CareerStagnationCategory", y="Headcount", color="Department",
        category_orders={"CareerStagnationCategory": order},
        color_discrete_sequence=["#FA582D", "#4D8CFF", "#3DD68C", "#FFD23D"],
    )
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E8EAED"}, barmode="stack", xaxis_title=None,
        legend={"font": {"size": 10}, "title": None},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_career_growth_analytics(df: pd.DataFrame):
    st.markdown('<div class="section-tag">Career & Growth Analytics</div>', unsafe_allow_html=True)
    st.caption(
        "Career Stagnation Score is peer-relative: compares Years Since Last Promotion "
        "against the median for the same Department + Job Level, z-scored 0\u2013100."
    )
    render_career_stagnation_bars(df)
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        render_growth_bottleneck_funnel(df)
    with c2:
        render_promotion_delay_analysis(df)
    st.divider()
    render_stagnation_category_breakdown(df)


# =============================================================================
# MODULE 9 — ATTRITION EARLY WARNING CENTER
# =============================================================================
EARLY_WARNING_ORDER = ["Stable", "Watchlist", "High Concern", "Critical"]
EARLY_WARNING_COLOR_MAP = {"Stable": "#3DD68C", "Watchlist": "#FFD23D", "High Concern": "#FF9A3D", "Critical": "#FF4D4D"}


def render_early_warning_distribution(df: pd.DataFrame):
    counts = df.groupby(["EarlyWarningCategory", "Department"], observed=True).size().reset_index(name="Headcount")
    fig = px.bar(
        counts, x="EarlyWarningCategory", y="Headcount", color="Department",
        category_orders={"EarlyWarningCategory": EARLY_WARNING_ORDER},
        color_discrete_sequence=["#FA582D", "#4D8CFF", "#3DD68C", "#FFD23D"],
    )
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E8EAED"}, barmode="stack", xaxis_title=None,
        legend={"font": {"size": 10}, "title": None},
    )
    st.plotly_chart(fig, use_container_width=True)

    total = len(df)
    critical_pct = (df["EarlyWarningCategory"] == "Critical").mean() * 100
    high_pct = (df["EarlyWarningCategory"] == "High Concern").mean() * 100
    st.markdown(
        f'<div class="insight-box">🚨 <b>{critical_pct:.1f}%</b> Critical, <b>{high_pct:.1f}%</b> '
        f'High Concern \u2014 together the addressable population for retention intervention '
        f'({(critical_pct + high_pct) / 100 * total:.0f} of {total} employees).</div>',
        unsafe_allow_html=True,
    )


def render_early_warning_driver_breakdown(df: pd.DataFrame):
    st.markdown("**What's Driving Early-Warning Status \u2014 by Category**")
    driver_cols = {
        "Engagement (inverted)": lambda d: 100 - d["EngagementIndex"],
        "Burnout Risk": lambda d: d["BurnoutRiskScore"],
        "Work-Life Balance (inverted)": lambda d: 100 - d["WorkLifeBalanceIndex"],
        "Career Stagnation": lambda d: d["CareerStagnationScore"],
    }
    rows = []
    for cat in EARLY_WARNING_ORDER:
        sub = df[df["EarlyWarningCategory"] == cat]
        if len(sub) == 0:
            continue
        row = {"Category": cat}
        for label, fn in driver_cols.items():
            row[label] = fn(sub).mean()
        rows.append(row)
    if not rows:
        st.info("No data available for the current selection.")
        return
    driver_df = pd.DataFrame(rows).set_index("Category")
    fig = px.imshow(
        driver_df, text_auto=".1f", color_continuous_scale="Reds", zmin=0, zmax=100, aspect="auto",
    )
    fig.update_layout(
        height=260, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#E8EAED", "size": 11},
    )
    st.plotly_chart(fig, use_container_width=True)


def render_early_warning_table(df: pd.DataFrame):
    st.markdown("**Highest Early-Warning Individuals**")
    top_n = st.slider("Number of employees to show", min_value=10, max_value=30, value=20, key="ew_top_n")
    cols = [
        "EmployeeID", "Department", "JobRole", "EarlyWarningScore", "EarlyWarningCategory",
        "EngagementIndex", "BurnoutRiskScore", "CareerStagnationCategory", "OverTime", "AttritionLabel",
    ]
    top = df.nlargest(min(top_n, len(df)), "EarlyWarningScore")[cols].reset_index(drop=True)
    st.dataframe(
        top, use_container_width=True, hide_index=True,
        column_config={
            "EarlyWarningScore": st.column_config.ProgressColumn("Early-Warning Score", min_value=0, max_value=100, format="%.1f"),
            "EngagementIndex": st.column_config.ProgressColumn("Engagement", min_value=0, max_value=100, format="%.1f"),
            "BurnoutRiskScore": st.column_config.ProgressColumn("Burnout", min_value=0, max_value=100, format="%.1f"),
            "AttritionLabel": st.column_config.TextColumn("Actual Outcome"),
        },
    )
    if len(top) > 0:
        attrited_pct = (top["AttritionLabel"] == "Attrited").mean() * 100
        st.markdown(
            f'<div class="insight-box">🚨 <b>{attrited_pct:.0f}%</b> of these top early-warning '
            f'employees have already left \u2014 a retrospective validity check on the score.</div>',
            unsafe_allow_html=True,
        )


def render_attrition_early_warning_center(df: pd.DataFrame):
    st.markdown('<div class="section-tag">Attrition Early Warning Center</div>', unsafe_allow_html=True)
    st.caption(
        "Early-Warning Score = 30% (100\u2212Engagement) + 30% Burnout Risk + 20% (100\u2212Work-Life Balance) "
        "+ 20% Career Stagnation. This is a transparent, rule-based score \u2014 shown alongside, not "
        "instead of, the ML attrition model in the Machine Learning Hub."
    )
    render_early_warning_distribution(df)
    st.divider()
    render_early_warning_driver_breakdown(df)
    st.divider()
    render_early_warning_table(df)


from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve,
)
import xgboost as xgb

# =============================================================================
# MODULE 10 — MACHINE LEARNING HUB
# =============================================================================
# IMPORTANT METHODOLOGY NOTE:
#   ML models here are ALWAYS trained on the full org dataset (unfiltered),
#   not the sidebar-filtered selection. Training on a filtered subset (e.g.
#   a single department) would starve the model of data and produce
#   meaningless, unstable metrics. This is stated explicitly in the UI.
#
#   FEATURE LEAKAGE HANDLING (deliberate, not an oversight):
#   - Attrition Prediction: uses the full feature set including engineered
#     KPIs (EngagementIndex, BurnoutRiskScore, etc.) as legitimate leading
#     indicators. Attrition is not algebraically derived from any other
#     field, so there's no formula leakage risk here.
#   - Burnout Risk Classification: EXCLUDES the burnout formula's own
#     direct inputs (OverTime, WorkLifeBalance, BusinessTravel,
#     DistanceFromHome, YearsInCurrentRole). Including them would make this
#     a trivial circular "recover the formula" exercise, not real ML.
#     Result: honestly weak (AUC ~0.56-0.58) — this is the correct,
#     expected outcome once the direct drivers are removed, not a bug.
#   - Engagement Classification: EXCLUDES the engagement formula's own
#     direct inputs (JobInvolvement, JobSatisfaction, EnvironmentSatisfaction,
#     RelationshipSatisfaction) for the same reason. Result: modest
#     (AUC ~0.60-0.64) — genuinely harder without direct survey data.
#   Both weaker results are reported as-is rather than swapped for an
#   easier (but circular) feature set that would inflate the numbers.
# =============================================================================

NON_FEATURE_COLS = {
    "EmployeeID", "Attrition", "AttritionLabel", "EngagementCategory", "BurnoutRiskCategory",
    "CareerStagnationCategory", "EarlyWarningCategory", "WorkLifeBalanceLabel",
    "TenureBand", "AgeBand", "IncomeQuartile", "EarlyWarningScore", "WorkforceHealthScore",
}
CATEGORICAL_CANDIDATES = ["BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime"]

BURNOUT_FORMULA_INPUTS = {"OverTime", "WorkLifeBalance", "BusinessTravel", "DistanceFromHome", "YearsInCurrentRole", "BurnoutRiskScore"}
ENGAGEMENT_FORMULA_INPUTS = {"JobInvolvement", "JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction", "EngagementIndex"}

ML_TASKS = {
    "Attrition Prediction": {
        "exclude": NON_FEATURE_COLS,
        "target_fn": lambda d: d["Attrition"],
        "positive_label": "Attrited",
        "task_type": "binary",
    },
    "Burnout Risk Classification": {
        "exclude": NON_FEATURE_COLS | BURNOUT_FORMULA_INPUTS,
        "target_fn": lambda d: (d["BurnoutRiskScore"] >= 50).astype(int),
        "positive_label": "High Burnout",
        "task_type": "binary",
    },
    "Engagement Level Classification": {
        "exclude": NON_FEATURE_COLS | ENGAGEMENT_FORMULA_INPUTS,
        "target_fn": lambda d: d["EngagementCategory"].map(
            {"Thriving": "High", "Healthy": "Medium", "At Risk": "Low", "Critical": "Low"}
        ),
        "positive_label": None,
        "task_type": "multiclass",
    },
}

MODEL_BUILDERS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1),
    "XGBoost": lambda: xgb.XGBClassifier(n_estimators=200, max_depth=4, eval_metric="logloss", random_state=42),
}


@st.cache_resource(show_spinner="Training models...")
def train_ml_task(_org_df: pd.DataFrame, task_name: str):
    """
    Trains all 3 models for a given task on the FULL org dataset.
    Cached by task_name (org_df is a fixed reference from st.cache_data
    upstream, so leading underscore skips hashing the large dataframe).
    Returns a dict with fitted pipelines, test data, metrics, and encoders
    — everything Module 11 (SHAP) needs downstream.
    """
    cfg = ML_TASKS[task_name]
    df = _org_df.copy()

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in cfg["exclude"]]
    cat_cols = [c for c in CATEGORICAL_CANDIDATES if c not in cfg["exclude"]]

    X = df[numeric_cols + cat_cols]
    y_raw = cfg["target_fn"](df)

    label_encoder = None
    if cfg["task_type"] == "multiclass":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
        class_names = list(label_encoder.classes_)
    else:
        y = y_raw
        class_names = ["No/Low", "Yes/High"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    sample_weight = compute_sample_weight("balanced", y_train)

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    results = {}
    for model_name, builder in MODEL_BUILDERS.items():
        pipe = Pipeline([("pre", preprocessor), ("clf", builder())])
        pipe.fit(X_train, y_train, clf__sample_weight=sample_weight)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, pred, average="weighted", zero_division=0),
        }
        if cfg["task_type"] == "binary":
            metrics["roc_auc"] = roc_auc_score(y_test, proba[:, 1])
            fpr, tpr, _ = roc_curve(y_test, proba[:, 1])
            metrics["roc_curve"] = (fpr, tpr)
        else:
            metrics["roc_auc"] = roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")
            metrics["roc_curve"] = None

        metrics["confusion_matrix"] = confusion_matrix(y_test, pred)
        results[model_name] = {"pipeline": pipe, "metrics": metrics}

    return {
        "results": results,
        "X_test": X_test,
        "y_test": y_test,
        "X_train": X_train,
        "y_train": y_train,
        "numeric_cols": numeric_cols,
        "cat_cols": cat_cols,
        "class_names": class_names,
        "label_encoder": label_encoder,
        "task_type": cfg["task_type"],
    }


def render_ml_metrics_table(task_result: dict):
    rows = []
    for model_name, r in task_result["results"].items():
        m = r["metrics"]
        rows.append({
            "Model": model_name, "Accuracy": m["accuracy"], "Precision": m["precision"],
            "Recall": m["recall"], "F1 Score": m["f1"], "ROC-AUC": m["roc_auc"],
        })
    metrics_df = pd.DataFrame(rows).set_index("Model")
    st.dataframe(
        metrics_df.style.format("{:.3f}").background_gradient(cmap="RdYlGn", vmin=0.4, vmax=1.0),
        use_container_width=True,
    )


def render_ml_confusion_and_roc(task_result: dict, model_name: str):
    r = task_result["results"][model_name]
    c1, c2 = st.columns(2)
    with c1:
        cm = r["metrics"]["confusion_matrix"]
        fig = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            x=task_result["class_names"], y=task_result["class_names"],
            labels=dict(x="Predicted", y="Actual"),
        )
        fig.update_layout(
            height=320, margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)", font={"color": "#E8EAED"},
            title={"text": f"{model_name} \u2014 Confusion Matrix", "font": {"size": 13, "color": "#8A94A6"}},
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        if task_result["task_type"] == "binary" and r["metrics"]["roc_curve"] is not None:
            fpr, tpr = r["metrics"]["roc_curve"]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color="#FA582D", width=2),
                                       name=f"{model_name} (AUC={r['metrics']['roc_auc']:.3f})"))
            fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                       line=dict(color="#8A94A6", width=1, dash="dash"), name="Random"))
            fig2.update_layout(
                height=320, margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#E8EAED"}, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                title={"text": f"{model_name} \u2014 ROC Curve", "font": {"size": 13, "color": "#8A94A6"}},
                legend={"font": {"size": 10}},
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("ROC curve is shown for binary tasks only (this is a multi-class task).")


def render_machine_learning_hub(org_df: pd.DataFrame):
    st.markdown('<div class="section-tag">Machine Learning Hub</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="insight-box">⚙️ Models are trained on the <b>full organization dataset</b> '
        '(1,470 employees), not the sidebar-filtered selection \u2014 training on a filtered subset '
        'would starve the model of data and produce unstable metrics. Sidebar filters do not affect this page.</div>',
        unsafe_allow_html=True,
    )

    task_name = st.selectbox("Prediction Task", list(ML_TASKS.keys()), key="ml_task_selector")

    if task_name != "Attrition Prediction":
        st.caption(
            "⚠️ This task deliberately excludes the engineered score's own direct formula inputs "
            "to avoid a trivial circular prediction problem. Expect modest performance \u2014 "
            "that's the honest, correct result of removing the direct drivers, not a bug."
        )

    with st.spinner("Training models on full dataset (cached after first run)..."):
        task_result = train_ml_task(org_df, task_name)

    st.markdown("**Model Comparison**")
    render_ml_metrics_table(task_result)

    st.divider()
    model_name = st.selectbox("Inspect model:", list(MODEL_BUILDERS.keys()), key="ml_model_inspect")
    render_ml_confusion_and_roc(task_result, model_name)

    st.session_state["_ml_task_result_cache"] = st.session_state.get("_ml_task_result_cache", {})
    st.session_state["_ml_task_result_cache"][task_name] = task_result


import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# MODULE 11 — EXPLAINABLE AI CENTER
# =============================================================================
# Uses shap.TreeExplainer on the XGBoost pipeline from Module 10 (tree
# models only — SHAP's TreeExplainer doesn't apply to Logistic Regression).
# Explains the ONE-HOT-ENCODED feature space directly (not the original
# columns) — this is intentional: SHAP attributions are only meaningful on
# the actual features the model sees, and collapsing one-hot categories
# back into a single "Department" bar would misrepresent per-category effects.
# =============================================================================

@st.cache_resource(show_spinner="Computing SHAP values...")
def compute_shap_values(_pipe, _X_test: pd.DataFrame):
    pre = _pipe.named_steps["pre"]
    clf = _pipe.named_steps["clf"]
    X_transformed = pre.transform(_X_test)
    feature_names = list(pre.get_feature_names_out())
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_transformed)

    # Multiclass tasks return a 3D array (n_samples, n_features, n_classes)
    # instead of the 2D (n_samples, n_features) binary tasks return.
    # This is handled explicitly everywhere downstream rather than assumed 2D.
    is_multiclass = np.asarray(shap_values).ndim == 3

    return {
        "shap_values": shap_values,
        "is_multiclass": is_multiclass,
        "X_transformed": X_transformed,
        "feature_names": feature_names,
        "expected_value": explainer.expected_value,
    }


def _global_importance_2d(shap_data: dict) -> np.ndarray:
    """
    Returns a (n_samples, n_features) view for global plots. For multiclass,
    this is the mean SIGNED shap value across classes — a simplification
    (loses per-class nuance) but gives a coherent overall-importance view.
    Explicitly labeled as such in the UI caption, not silently blended.
    """
    sv = np.asarray(shap_data["shap_values"])
    if shap_data["is_multiclass"]:
        return sv.mean(axis=2)
    return sv


def render_shap_summary(shap_data: dict):
    st.markdown("**SHAP Summary Plot \u2014 Global Feature Importance**")
    if shap_data["is_multiclass"]:
        st.caption("Multi-class task: values shown are averaged across all classes.")
    sv_2d = _global_importance_2d(shap_data)
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(
        sv_2d, shap_data["X_transformed"],
        feature_names=shap_data["feature_names"], show=False, max_display=12,
    )
    fig.patch.set_facecolor("#0B0E14")
    ax.set_facecolor("#0B0E14")
    ax.tick_params(colors="#E8EAED")
    ax.xaxis.label.set_color("#E8EAED")
    for text in ax.get_yticklabels() + ax.get_xticklabels():
        text.set_color("#E8EAED")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_shap_feature_importance_bar(shap_data: dict):
    st.markdown("**Mean |SHAP Value| \u2014 Feature Importance Ranking**")
    sv = np.asarray(shap_data["shap_values"])
    if shap_data["is_multiclass"]:
        mean_abs = np.abs(sv).mean(axis=(0, 2))  # average over samples AND classes
    else:
        mean_abs = np.abs(sv).mean(axis=0)
    imp_df = pd.DataFrame({"Feature": shap_data["feature_names"], "Mean |SHAP|": mean_abs})
    imp_df = imp_df.sort_values("Mean |SHAP|", ascending=True).tail(12)
    fig = px.bar(imp_df, x="Mean |SHAP|", y="Feature", orientation="h", color_discrete_sequence=["#FA582D"])
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E8EAED"}, yaxis_title=None,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_shap_waterfall(shap_data: dict, X_test: pd.DataFrame, class_names: list):
    st.markdown("**Waterfall \u2014 Individual Employee Explanation**")
    idx = st.slider("Test-set employee index", 0, len(X_test) - 1, 0, key="shap_waterfall_idx")

    sv = np.asarray(shap_data["shap_values"])
    if shap_data["is_multiclass"]:
        class_choice = st.selectbox("Explain prediction for class:", class_names, key="shap_waterfall_class")
        class_idx = class_names.index(class_choice)
        values = sv[idx, :, class_idx]
        base_value = shap_data["expected_value"][class_idx]
    else:
        values = sv[idx]
        base_value = shap_data["expected_value"]
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if len(base_value) > 1 else base_value[0]

    exp = shap.Explanation(
        values=values, base_values=base_value,
        data=shap_data["X_transformed"][idx], feature_names=shap_data["feature_names"],
    )
    fig = plt.figure(figsize=(8, 5))
    shap.waterfall_plot(exp, show=False, max_display=10)
    fig.patch.set_facecolor("#0B0E14")
    for ax in fig.get_axes():
        ax.set_facecolor("#0B0E14")
        ax.tick_params(colors="#E8EAED")
        for text in ax.get_yticklabels() + ax.get_xticklabels():
            text.set_color("#E8EAED")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_shap_dependence(shap_data: dict):
    st.markdown("**Dependence Plot**")
    sv_2d = _global_importance_2d(shap_data)
    if shap_data["is_multiclass"]:
        st.caption("Multi-class task: SHAP values averaged across all classes.")
    feature_choice = st.selectbox(
        "Feature", shap_data["feature_names"], key="shap_dependence_feature",
        index=int(np.argmax(np.abs(sv_2d).mean(axis=0))),
    )
    feat_idx = shap_data["feature_names"].index(feature_choice)
    x_vals = shap_data["X_transformed"][:, feat_idx]
    y_vals = sv_2d[:, feat_idx]
    fig = px.scatter(x=x_vals, y=y_vals, opacity=0.5, color_discrete_sequence=["#4D8CFF"])
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E8EAED"}, xaxis_title=feature_choice, yaxis_title="SHAP value",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_explainable_ai_center(org_df: pd.DataFrame):
    st.markdown('<div class="section-tag">Explainable AI Center</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="insight-box">🔍 Explanations use the XGBoost model from the Machine Learning '
        'Hub, trained on the full organization dataset. SHAP values are computed on the '
        'one-hot-encoded feature space the model actually sees, not the original raw columns.</div>',
        unsafe_allow_html=True,
    )

    task_name = st.selectbox("Explain task:", list(ML_TASKS.keys()), key="xai_task_selector")
    task_result = train_ml_task(org_df, task_name)
    pipe = task_result["results"]["XGBoost"]["pipeline"]
    X_test = task_result["X_test"]
    class_names = task_result["class_names"]

    shap_data = compute_shap_values(pipe, X_test)

    render_shap_summary(shap_data)
    st.divider()
    render_shap_feature_importance_bar(shap_data)
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        render_shap_waterfall(shap_data, X_test, class_names)
    with c2:
        render_shap_dependence(shap_data)


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =============================================================================
# MODULE 12 — WORKFORCE SEGMENTATION
# =============================================================================
# K=4 chosen after comparing silhouette scores for k=3,4,5 on this dataset
# (k=4 scored highest, ~0.27 — modest but the best available option; stated
# honestly rather than implying strong natural cluster separation).
#
# Cluster -> label mapping is PROFILE-BASED, not a hardcoded index mapping —
# KMeans cluster indices are arbitrary/unstable across runs, so labels are
# assigned by ranking clusters on a health score (Engagement - Burnout) and
# tenure, not by remembering "cluster 2 = Champions" from one run.
# =============================================================================

SEGMENTATION_FEATURES = ["EngagementIndex", "BurnoutRiskScore", "WorkLifeBalanceIndex", "TotalWorkingYears", "MonthlyIncome"]


@st.cache_resource(show_spinner="Running clustering...")
def run_workforce_segmentation(_org_df: pd.DataFrame, k: int = 4):
    df = _org_df.copy()
    X = StandardScaler().fit_transform(df[SEGMENTATION_FEATURES])
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    df["ClusterID"] = km.labels_
    sil = silhouette_score(X, km.labels_)

    profile = df.groupby("ClusterID")[SEGMENTATION_FEATURES].mean()
    profile["HealthScore"] = profile["EngagementIndex"] - profile["BurnoutRiskScore"]
    profile_sorted_by_health = profile.sort_values("HealthScore", ascending=False)

    labels = {}
    ids = list(profile_sorted_by_health.index)
    labels[ids[0]] = "Champions"          # best engagement-minus-burnout combo
    labels[ids[-1]] = "Flight Risks"      # worst combo
    middle_ids = ids[1:-1]
    if len(middle_ids) == 2:
        # Differentiate the two middle clusters by tenure: the longer-tenured
        # one is "Quiet Survivors" (staying despite mediocre scores), the
        # other is "Stable Contributors".
        middle_tenure = profile.loc[middle_ids, "TotalWorkingYears"]
        labels[middle_tenure.idxmax()] = "Quiet Survivors"
        labels[middle_tenure.idxmin()] = "Stable Contributors"
    else:
        for i, cid in enumerate(middle_ids):
            labels[cid] = f"Segment {i + 1}"

    df["SegmentLabel"] = df["ClusterID"].map(labels)
    profile["SegmentLabel"] = profile.index.map(labels)

    return {"df": df, "profile": profile, "silhouette": sil, "k": k}


def render_segmentation_scatter(seg_result: dict):
    st.markdown("**Segments \u2014 Engagement vs Burnout**")
    df = seg_result["df"]
    color_map = {
        "Champions": "#3DD68C", "Stable Contributors": "#4D8CFF",
        "Quiet Survivors": "#FFD23D", "Flight Risks": "#FF4D4D",
    }
    fig = px.scatter(
        df, x="EngagementIndex", y="BurnoutRiskScore", color="SegmentLabel",
        color_discrete_map=color_map, opacity=0.6,
        hover_data=["Department", "JobRole"],
    )
    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E8EAED"}, legend={"font": {"size": 11}, "title": None},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"K-Means, k={seg_result['k']}, silhouette score = {seg_result['silhouette']:.3f} "
               f"(modest separation \u2014 reported honestly, not inflated).")


def render_segmentation_profile_table(seg_result: dict):
    st.markdown("**Cluster Profiles**")
    profile = seg_result["profile"].set_index("SegmentLabel")[SEGMENTATION_FEATURES + ["HealthScore"]]
    st.dataframe(
        profile.style.format("{:.1f}").background_gradient(cmap="RdYlGn", subset=["HealthScore"]),
        use_container_width=True,
    )

    counts = seg_result["df"]["SegmentLabel"].value_counts()
    recommendations = {
        "Champions": "Retention focus: protect via recognition, growth paths, and avoiding burnout creep.",
        "Stable Contributors": "Maintain via consistent engagement practices; low urgency but monitor for drift.",
        "Quiet Survivors": "Investigate why long-tenured employees have mediocre scores \u2014 possible stagnation or disengagement masked by loyalty.",
        "Flight Risks": "Immediate intervention: highest combined burnout/low-engagement profile, prioritize for 1:1 outreach.",
    }
    for label, rec in recommendations.items():
        if label in counts.index:
            st.markdown(
                f'<div class="insight-box">🧩 <b>{label}</b> ({counts[label]:,} employees, '
                f'{counts[label]/counts.sum()*100:.1f}%): {rec}</div>',
                unsafe_allow_html=True,
            )


def render_workforce_segmentation(org_df: pd.DataFrame):
    st.markdown('<div class="section-tag">Workforce Segmentation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="insight-box">🧩 Clustering runs on the full organization dataset (not the '
        'sidebar filter) for stable cluster definitions.</div>',
        unsafe_allow_html=True,
    )
    seg_result = run_workforce_segmentation(org_df, k=4)
    render_segmentation_scatter(seg_result)
    st.divider()
    render_segmentation_profile_table(seg_result)


# =============================================================================
# MODULE 13 — HR SCENARIO SIMULATOR
# =============================================================================
# Re-applies the SAME formulas from Module 1.2 with user-adjusted inputs,
# so the simulator is mathematically consistent with the rest of the
# platform rather than a separate/decorative toy model.
# =============================================================================

def _simulate_scenario(df: pd.DataFrame, overtime_reduction_pct: float, wlb_boost: float,
                        promotion_frequency_mult: float, training_frequency_mult: float) -> pd.DataFrame:
    """
    Applies scenario adjustments to a copy of the dataframe and recomputes
    Engagement, Burnout, and Workforce Health using the identical formulas
    from engineer_features (Module 1.2), just with modified inputs:
      - overtime_reduction_pct: % of currently-overtime employees switched to No
      - wlb_boost: flat +N adjustment to WorkLifeBalance (capped at 4)
      - promotion_frequency_mult: multiplier on YearsSinceLastPromotion (lower = more frequent promotions)
      - training_frequency_mult: multiplier on TrainingTimesLastYear (higher = more training)
        (Training doesn't feed the current formulas directly, but is surfaced
        as a workforce-health-adjacent signal for the "before/after" narrative.)
    """
    sim = df.copy()

    # Overtime reduction: flip a proportion of "Yes" to "No", deterministically
    # (by BurnoutRiskScore descending, i.e. highest-risk employees relieved first)
    ot_yes_idx = sim[sim["OverTime"] == "Yes"].sort_values("BurnoutRiskScore", ascending=False).index
    n_to_flip = int(len(ot_yes_idx) * overtime_reduction_pct / 100)
    sim.loc[ot_yes_idx[:n_to_flip], "OverTime"] = "No"

    # Work-life balance boost (capped at the scale max of 4)
    sim["WorkLifeBalance"] = (sim["WorkLifeBalance"] + wlb_boost).clip(upper=4, lower=1)

    # Promotion frequency: uniform scaling of YearsSinceLastPromotion has ZERO
    # effect on CareerStagnationScore, because that score is a peer-relative
    # z-score — any uniform linear rescaling within a peer group is invariant
    # under z-scoring (verified numerically before shipping this). The fix:
    # asymmetrically pull ONLY above-peer-median employees toward their peer
    # median (proportional to how much more frequently promotions happen),
    # which changes the peer group's distribution shape, not just its scale.
    peer_median = df.groupby(["Department", "JobLevel"])["YearsSinceLastPromotion"].transform("median").astype(float)
    years = sim["YearsSinceLastPromotion"].astype(float)
    above_median_mask = years > peer_median
    years.loc[above_median_mask] = (
        peer_median[above_median_mask]
        + (years[above_median_mask] - peer_median[above_median_mask]) * promotion_frequency_mult
    )
    sim["YearsSinceLastPromotion"] = years.clip(lower=0)

    # Training frequency (not a direct formula input, kept for narrative completeness)
    sim["TrainingTimesLastYear"] = (sim["TrainingTimesLastYear"] * training_frequency_mult).clip(upper=6)

    # Recompute engineered fields using the SAME logic as engineer_features.
    # Re-running the full function ensures formula consistency (single source of truth).
    sim = engineer_features(sim.drop(columns=[
        c for c in sim.columns if c in {
            "EngagementIndex", "EngagementCategory", "BurnoutRiskScore", "BurnoutRiskCategory",
            "WorkLifeBalanceIndex", "WorkLifeBalanceLabel", "SatisfactionStabilityScore",
            "CareerStagnationScore", "CareerStagnationCategory", "WorkforceHealthScore",
            "EarlyWarningScore", "EarlyWarningCategory", "TenureBand", "AgeBand", "IncomeQuartile", "AttritionLabel",
        }
    ]))
    return sim


def render_hr_scenario_simulator(org_df: pd.DataFrame):
    st.markdown('<div class="section-tag">HR Scenario Simulator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="insight-box">🎛️ Simulates the full organization. Adjust the levers below and '
        'compare the resulting Engagement, Burnout, and Workforce Health against the current baseline. '
        'This re-runs the exact same formulas used everywhere else in the platform, just with modified inputs.</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        overtime_reduction = st.slider(
            "Reduce Overtime (% of overtime employees moved to no-overtime, highest-risk first)",
            0, 100, 0, step=5,
        )
        wlb_boost = st.slider("Work-Life Balance Boost (+N, capped at scale max of 4)", 0, 3, 0)
    with c2:
        promotion_freq = st.slider(
            "Promotion Frequency Multiplier (0.5 = promotions twice as often)", 0.3, 1.5, 1.0, step=0.1,
        )
        training_freq = st.slider(
            "Training Frequency Multiplier (2.0 = double current training frequency)", 0.5, 3.0, 1.0, step=0.1,
        )

    baseline_kpis = compute_kpis(org_df)

    if overtime_reduction == 0 and wlb_boost == 0 and promotion_freq == 1.0 and training_freq == 1.0:
        st.info("Adjust at least one lever above to see a before/after comparison.")
        return

    simulated_df = _simulate_scenario(org_df, overtime_reduction, wlb_boost, promotion_freq, training_freq)
    sim_kpis = compute_kpis(simulated_df)

    st.markdown("**Before vs After**")
    m1, m2, m3 = st.columns(3)
    m1.metric("Engagement Index", f"{sim_kpis['engagement_index']:.1f}",
              f"{sim_kpis['engagement_index'] - baseline_kpis['engagement_index']:+.1f}")
    m2.metric("Burnout Risk Score", f"{sim_kpis['burnout_risk_score']:.1f}",
              f"{sim_kpis['burnout_risk_score'] - baseline_kpis['burnout_risk_score']:+.1f}", delta_color="inverse")
    m3.metric("Workforce Health Score", f"{sim_kpis['workforce_health']:.1f}",
              f"{sim_kpis['workforce_health'] - baseline_kpis['workforce_health']:+.1f}")

    compare_df = pd.DataFrame({
        "Metric": ["Engagement Index", "Burnout Risk Score", "Workforce Health Score", "Work-Life Balance Index"],
        "Before": [baseline_kpis["engagement_index"], baseline_kpis["burnout_risk_score"],
                   baseline_kpis["workforce_health"], baseline_kpis["wlb_index"]],
        "After": [sim_kpis["engagement_index"], sim_kpis["burnout_risk_score"],
                  sim_kpis["workforce_health"], sim_kpis["wlb_index"]],
    })
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Before", x=compare_df["Metric"], y=compare_df["Before"], marker_color="#8A94A6"))
    fig.add_trace(go.Bar(name="After", x=compare_df["Metric"], y=compare_df["After"], marker_color="#FA582D"))
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=20, b=10), barmode="group",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E8EAED"}, legend={"font": {"size": 11}},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f'<div class="insight-box">🎛️ Under this scenario, high-risk-employee count moves from '
        f'<b>{baseline_kpis["high_risk_count"]:,}</b> to <b>{sim_kpis["high_risk_count"]:,}</b> '
        f'({sim_kpis["high_risk_count"] - baseline_kpis["high_risk_count"]:+,}).</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# MODULE 14 — WORKFORCE RISK ANALYTICS + MANAGER ACTION CENTER
# =============================================================================
def render_risk_matrix(df: pd.DataFrame):
    st.markdown("**Risk Matrix \u2014 Burnout vs Career Stagnation**")
    fig = px.density_heatmap(
        df, x="BurnoutRiskScore", y="CareerStagnationScore", nbinsx=20, nbinsy=20,
        color_continuous_scale="Reds",
    )
    fig.update_layout(
        height=360, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E8EAED"},
    )
    st.plotly_chart(fig, use_container_width=True)

    high_both = ((df["BurnoutRiskScore"] >= 50) & (df["CareerStagnationScore"] >= 50)).sum()
    st.markdown(
        f'<div class="insight-box">⚠️ <b>{high_both:,}</b> employees ({high_both/len(df)*100:.1f}%) '
        f'are simultaneously High+ on both Burnout Risk and Career Stagnation \u2014 the compounding-risk '
        f'population most likely to leave for reasons beyond a single fixable issue.</div>',
        unsafe_allow_html=True,
    )


def render_risk_summary_cards(df: pd.DataFrame):
    risk_defs = {
        "Burnout Risk": (df["BurnoutRiskCategory"].isin(["High", "Critical"])),
        "Attrition Risk (Early Warning)": (df["EarlyWarningCategory"].isin(["High Concern", "Critical"])),
        "Career Stagnation Risk": (df["CareerStagnationCategory"].isin(["Elevated", "Severe"])),
        "Workload Risk (Overtime)": (df["OverTime"] == "Yes"),
    }
    cols = st.columns(4)
    for col, (label, mask) in zip(cols, risk_defs.items()):
        col.metric(label, f"{mask.sum():,}", f"{mask.mean()*100:.1f}% of selection")


def render_manager_action_center(df: pd.DataFrame):
    st.markdown("**Manager Action Center \u2014 Data-Driven Recommendations**")

    recommendations = []

    dept_burnout = df.groupby("Department")["BurnoutRiskScore"].mean().sort_values(ascending=False)
    if len(dept_burnout) > 0 and dept_burnout.iloc[0] >= 40:
        recommendations.append(
            f"🔥 <b>{dept_burnout.index[0]}</b> has the highest average Burnout Risk "
            f"({dept_burnout.iloc[0]:.1f}/100) \u2014 prioritize workload review and overtime reduction here first."
        )

    ot_burnout_gap = df.loc[df["OverTime"] == "Yes", "BurnoutRiskScore"].mean() - df.loc[df["OverTime"] == "No", "BurnoutRiskScore"].mean()
    if pd.notna(ot_burnout_gap) and ot_burnout_gap > 15:
        n_ot = (df["OverTime"] == "Yes").sum()
        recommendations.append(
            f"⏰ <b>{n_ot:,} employees</b> are on overtime, showing a {ot_burnout_gap:.1f}-point higher "
            f"Burnout Risk than non-overtime peers \u2014 the single highest-leverage intervention available."
        )

    critical_ew = (df["EarlyWarningCategory"] == "Critical").sum()
    if critical_ew > 0:
        recommendations.append(
            f"🚨 <b>{critical_ew:,} employees</b> are in the Critical Early-Warning tier \u2014 "
            f"recommend 1:1 manager check-ins within the next 2 weeks."
        )

    stagnant = (df["CareerStagnationCategory"] == "Severe").sum()
    if stagnant > 0:
        recommendations.append(
            f"🪜 <b>{stagnant:,} employees</b> show Severe career stagnation relative to department/level peers "
            f"\u2014 review for overdue promotion conversations."
        )

    low_wlb_dept = df.groupby("Department")["WorkLifeBalanceIndex"].mean().sort_values().index
    if len(low_wlb_dept) > 0:
        recommendations.append(
            f"⚖️ <b>{low_wlb_dept[0]}</b> has the lowest Work-Life Balance Index in the current selection "
            f"\u2014 candidate for flexible-hours or workload redistribution pilots."
        )

    if not recommendations:
        st.info("No high-priority flags in the current selection \u2014 workforce indicators are within normal ranges.")
    else:
        for rec in recommendations:
            st.markdown(f'<div class="insight-box">{rec}</div>', unsafe_allow_html=True)


def render_workforce_risk_analytics(df: pd.DataFrame):
    st.markdown('<div class="section-tag">Workforce Risk Analytics</div>', unsafe_allow_html=True)
    render_risk_summary_cards(df)
    st.divider()
    render_risk_matrix(df)


def render_manager_action_center_page(df: pd.DataFrame):
    st.markdown('<div class="section-tag">Manager Action Center</div>', unsafe_allow_html=True)
    render_manager_action_center(df)


# =============================================================================
# MODULE 1.6 — MAIN ENTRYPOINT
# =============================================================================
def main():
    try:
        raw_df = load_raw_data(DATA_PATH)
    except (FileNotFoundError, ValueError) as e:
        st.error(f"**Data loading failed:** {e}")
        st.stop()

    df = engineer_features(raw_df)

    page, filtered_df, _filters = render_sidebar(df)

    render_header()

    if len(filtered_df) == 0:
        st.warning("No employees match the current filter combination. Adjust filters in the sidebar.")
        st.stop()

    kpis_filtered = compute_kpis(filtered_df)
    kpis_org = compute_kpis(df)  # unfiltered org-wide baseline, used for benchmarking

    # ---- Page router ----
    # Each elif will be replaced with its real module as the build proceeds.
    if page == "Executive Overview":
        render_executive_overview(filtered_df, df, kpis_filtered, kpis_org)
    elif page == "Employee Experience EDA":
        render_employee_experience_eda(filtered_df)
    elif page == "Engagement Intelligence Center":
        render_engagement_intelligence_center(filtered_df)
    elif page == "Burnout Risk Center":
        render_burnout_risk_center(filtered_df)
    elif page == "Workload Stress Analytics":
        render_workload_stress_analytics(filtered_df)
    elif page == "Department Intelligence":
        render_department_intelligence(filtered_df)
    elif page == "Career & Growth Analytics":
        render_career_growth_analytics(filtered_df)
    elif page == "Attrition Early Warning Center":
        render_attrition_early_warning_center(filtered_df)
    elif page == "Machine Learning Hub":
        render_machine_learning_hub(df)
    elif page == "Explainable AI Center":
        render_explainable_ai_center(df)
    elif page == "Workforce Segmentation":
        render_workforce_segmentation(df)
    elif page == "HR Scenario Simulator":
        render_hr_scenario_simulator(df)
    elif page == "Workforce Risk Analytics":
        render_workforce_risk_analytics(filtered_df)
    elif page == "Manager Action Center":
        render_manager_action_center_page(filtered_df)
    else:
        render_placeholder(page, filtered_df, kpis_filtered)


if __name__ == "__main__":
    main()
