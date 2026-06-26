import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import io

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp { background: #0B0F19; }
section[data-testid="stSidebar"] { background: #0F1420 !important; border-right: 1px solid #1E2433 !important; }

/* Sidebar */
.sb-logo { display:flex; align-items:center; gap:10px; padding:8px 0 20px; border-bottom:1px solid #1E2433; margin-bottom:20px; }
.sb-logo-mark { width:34px;height:34px;background:#4F46E5;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;color:#fff;letter-spacing:-0.5px; }
.sb-logo-text { font-size:14px;font-weight:600;color:#F1F5F9;letter-spacing:-0.3px; }
.sb-logo-sub { font-size:10px;color:#64748B;margin-top:1px; }
.sb-section { margin-bottom:24px; }
.sb-section-title { font-size:9px;font-weight:600;color:#4F46E5;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:10px; }
.sb-stat { background:#131827;border:1px solid #1E2433;border-radius:10px;padding:12px;margin-bottom:8px; }
.sb-stat-label { font-size:10px;color:#64748B;margin-bottom:4px; }
.sb-stat-val { font-size:18px;font-weight:700;color:#F1F5F9; }
.sb-stat-chip { display:inline-block;font-size:9px;font-weight:600;padding:2px 7px;border-radius:99px;margin-top:4px; }
.chip-green { background:#0D3321;color:#34D399; }
.chip-red { background:#3B0D0D;color:#F87171; }

/* Top bar */
.topbar { background:#0F1420;border:1px solid #1E2433;border-radius:14px;padding:22px 28px;display:flex;align-items:center;justify-content:space-between;margin-bottom:20px; }
.topbar-bar-accent { width:4px;height:44px;background:#4F46E5;border-radius:2px;flex-shrink:0; }
.topbar-title { font-size:20px;font-weight:700;color:#F1F5F9;letter-spacing:-0.5px;line-height:1; }
.topbar-tagline { font-size:11px;color:#64748B;margin-top:5px; }
.topbar-badge { background:#1E2433;border:1px solid #2D3748;color:#94A3B8;font-size:10px;font-weight:600;padding:4px 12px;border-radius:99px; }
.topbar-right { display:flex;flex-direction:column;align-items:flex-end;gap:6px; }

/* KPI row */
.kpi-grid { display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px; }
.kpi-card { background:#0F1420;border:1px solid #1E2433;border-radius:12px;padding:18px 20px; }
.kpi-label { font-size:10px;font-weight:600;color:#64748B;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px; }
.kpi-val { font-size:26px;font-weight:700;color:#F1F5F9;letter-spacing:-0.5px;line-height:1; }
.kpi-change { font-size:10px;font-weight:600;margin-top:6px; }
.kpi-change.up { color:#34D399; }
.kpi-change.down { color:#F87171; }
.kpi-change.neutral { color:#94A3B8; }

/* Section headers */
.section-hdr { display:flex;align-items:center;justify-content:space-between;margin-bottom:14px; }
.section-title { font-size:12px;font-weight:600;color:#F1F5F9;letter-spacing:-0.2px; }
.section-sub { font-size:10px;color:#64748B;margin-top:2px; }

/* Form panels */
.form-panel { background:#0F1420;border:1px solid #1E2433;border-radius:14px;padding:20px;margin-bottom:16px; }
.panel-title { font-size:10px;font-weight:600;color:#4F46E5;text-transform:uppercase;letter-spacing:1.2px;padding-bottom:12px;border-bottom:1px solid #1E2433;margin-bottom:16px; }

/* Inputs */
label, .stSelectbox label, .stSlider label, .stNumberInput label {
    font-size:11px !important;font-weight:500 !important;color:#94A3B8 !important;text-transform:uppercase;letter-spacing:0.8px !important;
}
.stSelectbox > div > div, .stNumberInput > div > div > input {
    background:#131827 !important;border:1px solid #1E2433 !important;border-radius:8px !important;color:#F1F5F9 !important;
}
.stSelectbox > div > div:hover, .stNumberInput > div > div > input:focus {
    border-color:#4F46E5 !important;
}

/* Button */
.stButton > button {
    background:#4F46E5 !important;color:#fff !important;border:none !important;
    border-radius:10px !important;padding:12px 28px !important;
    font-size:13px !important;font-weight:600 !important;letter-spacing:0.3px !important;
    width:100% !important;
}
.stButton > button:hover { background:#4338CA !important; }

/* Result cards */
.result-wrap { background:#0F1420;border:1px solid #1E2433;border-radius:14px;padding:24px;margin-bottom:16px; }
.result-stay { border-left:4px solid #10B981; }
.result-leave { border-left:4px solid #EF4444; }
.result-eyebrow { font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:2px;margin-bottom:6px; }
.result-eyebrow.stay { color:#10B981; }
.result-eyebrow.leave { color:#EF4444; }
.result-heading { font-size:20px;font-weight:700;color:#F1F5F9;margin-bottom:4px;letter-spacing:-0.4px; }
.result-desc { font-size:12px;color:#64748B;margin-bottom:14px;line-height:1.6; }
.result-pill { display:inline-block;font-size:13px;font-weight:700;padding:6px 18px;border-radius:99px; }
.pill-stay { background:#063620;color:#34D399;border:1px solid #10B981; }
.pill-leave { background:#3B0D0D;color:#F87171;border:1px solid #EF4444; }

.prob-box { background:#131827;border:1px solid #1E2433;border-radius:10px;padding:16px;text-align:center; }
.prob-box-label { font-size:9px;font-weight:600;color:#64748B;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px; }
.prob-box-val { font-size:28px;font-weight:700;letter-spacing:-0.5px; }
.val-stay { color:#34D399; }
.val-leave { color:#F87171; }

.risk-wrap { background:#131827;border:1px solid #1E2433;border-radius:10px;padding:16px;margin-top:14px; }
.risk-label { font-size:10px;font-weight:600;color:#94A3B8;text-transform:uppercase;letter-spacing:1px;margin-bottom:10px; }
.risk-bar-bg { height:6px;background:#1E2433;border-radius:3px;margin-bottom:8px; }
.risk-bar-fill-low { height:6px;background:#10B981;border-radius:3px; }
.risk-bar-fill-mid { height:6px;background:#F59E0B;border-radius:3px; }
.risk-bar-fill-high { height:6px;background:#EF4444;border-radius:3px; }
.risk-status { font-size:11px;font-weight:600; }
.risk-low { color:#34D399; }
.risk-med { color:#FBBF24; }
.risk-high { color:#F87171; }

/* Recommendation cards */
.rec-grid { display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:4px; }
.rec-card { background:#131827;border:1px solid #1E2433;border-radius:10px;padding:12px 14px;font-size:11px;color:#94A3B8;line-height:1.6; }
.rec-card strong { color:#F1F5F9;display:block;margin-bottom:3px;font-size:11px; }
.rec-warn { border-top:2px solid #F59E0B; }
.rec-info { border-top:2px solid #4F46E5; }

/* Chart containers */
.chart-panel { background:#0F1420;border:1px solid #1E2433;border-radius:14px;padding:20px;margin-bottom:16px; }
.chart-title { font-size:11px;font-weight:600;color:#F1F5F9;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:4px; }
.chart-sub { font-size:10px;color:#64748B;margin-bottom:16px; }

hr { border-color:#1E2433 !important;margin:20px 0 !important; }
.footer { text-align:center;font-size:10px;color:#374151;padding:16px 0 4px; }

div[data-testid="stMetricValue"] { color:#F1F5F9 !important;font-size:22px !important;font-weight:700 !important; }
div[data-testid="stMetricLabel"] { color:#64748B !important;font-size:10px !important; }
details summary { color:#64748B !important;font-size:11px !important; }
.stExpander { background:#0F1420 !important;border:1px solid #1E2433 !important;border-radius:10px !important; }
</style>
""", unsafe_allow_html=True)


# ── CHART HELPERS ──────────────────────────────────────────────────────────────

DARK_BG   = "#0F1420"
DARK_BG2  = "#131827"
BORDER    = "#1E2433"
TEXT_MAIN = "#F1F5F9"
TEXT_MUTE = "#64748B"
ACCENT    = "#4F46E5"
GREEN     = "#10B981"
RED       = "#EF4444"
AMBER     = "#F59E0B"


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130, facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf


def make_gauge(leave_prob: float) -> bytes:
    """Semicircular gauge showing attrition risk 0-100%."""
    fig, ax = plt.subplots(figsize=(4.2, 2.4), facecolor=DARK_BG2)
    ax.set_facecolor(DARK_BG2)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.2, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")

    # Background arc (full 180°)
    theta_bg = np.linspace(np.pi, 0, 300)
    r = 1.0
    ax.plot(r * np.cos(theta_bg), r * np.sin(theta_bg), color=BORDER, linewidth=14, solid_capstyle="round", zorder=1)

    # Colour segments: green 0-40, amber 40-70, red 70-100
    segments = [
        (np.pi,       np.pi * 0.6,  GREEN),   # 0–40%
        (np.pi * 0.6, np.pi * 0.3,  AMBER),   # 40–70%
        (np.pi * 0.3, 0,            RED),      # 70–100%
    ]
    for t1, t2, col in segments:
        th = np.linspace(t1, t2, 100)
        ax.plot(r * np.cos(th), r * np.sin(th), color=col, linewidth=14,
                solid_capstyle="butt", alpha=0.25, zorder=2)

    # Filled arc up to leave_prob
    fill_end = np.pi - (leave_prob / 100) * np.pi
    th_fill = np.linspace(np.pi, fill_end, 200)
    if leave_prob < 40:
        fill_col = GREEN
    elif leave_prob < 70:
        fill_col = AMBER
    else:
        fill_col = RED
    ax.plot(r * np.cos(th_fill), r * np.sin(th_fill), color=fill_col, linewidth=14,
            solid_capstyle="round", alpha=0.9, zorder=3)

    # Needle
    needle_angle = np.pi - (leave_prob / 100) * np.pi
    nx = 0.78 * np.cos(needle_angle)
    ny = 0.78 * np.sin(needle_angle)
    ax.annotate("", xy=(nx, ny), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=TEXT_MAIN, lw=1.6,
                                mutation_scale=12))
    ax.plot(0, 0, "o", color=TEXT_MAIN, markersize=6, zorder=6)

    # Labels
    ax.text(0, -0.12, f"{leave_prob:.1f}%", ha="center", va="center",
            fontsize=17, fontweight="700", color=fill_col, fontfamily="monospace")
    ax.text(0, -0.30, "Attrition Risk", ha="center", va="center",
            fontsize=8, color=TEXT_MUTE)
    ax.text(-1.15, -0.10, "0%",   ha="center", fontsize=7, color=TEXT_MUTE)
    ax.text( 1.15, -0.10, "100%", ha="center", fontsize=7, color=TEXT_MUTE)

    fig.tight_layout(pad=0.3)
    return fig_to_bytes(fig)


def make_prob_bar(stay_prob: float, leave_prob: float) -> bytes:
    """Horizontal stacked bar: stay vs leave probability."""
    fig, ax = plt.subplots(figsize=(5.5, 1.1), facecolor=DARK_BG2)
    ax.set_facecolor(DARK_BG2)

    ax.barh(0, stay_prob,  color=GREEN, height=0.45, label=f"Stay  {stay_prob:.1f}%")
    ax.barh(0, leave_prob, left=stay_prob, color=RED, height=0.45, label=f"Leave {leave_prob:.1f}%")

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")

    # Value labels inside bars
    if stay_prob > 10:
        ax.text(stay_prob / 2, 0, f"Stay\n{stay_prob:.1f}%", ha="center", va="center",
                fontsize=8, color="#fff", fontweight="600")
    if leave_prob > 10:
        ax.text(stay_prob + leave_prob / 2, 0, f"Leave\n{leave_prob:.1f}%",
                ha="center", va="center", fontsize=8, color="#fff", fontweight="600")

    fig.tight_layout(pad=0.2)
    return fig_to_bytes(fig)


def make_risk_factors_chart(input_values: dict) -> bytes:
    """
    Horizontal bar chart of key HR risk factors scored 0-1.
    Each factor is mapped to a normalised risk contribution.
    """
    factor_scores = {}

    # Overtime
    factor_scores["OverTime"] = 1.0 if input_values.get("OverTime") == "Yes" else 0.0

    # Job satisfaction (inverted, normalised)
    js = input_values.get("JobSatisfaction", 3)
    factor_scores["Job Satisfaction"] = (5 - js) / 4

    # Work-life balance (inverted)
    wlb = input_values.get("WorkLifeBalance", 3)
    factor_scores["Work-Life Balance"] = (5 - wlb) / 4

    # Years since promotion
    ysp = input_values.get("YearsSinceLastPromotion", 1)
    factor_scores["Promotion Gap"] = min(ysp / 10, 1.0)

    # Monthly income (inverted, relative to max 20000)
    mi = input_values.get("MonthlyIncome", 5000)
    factor_scores["Monthly Income"] = max(0, 1 - mi / 20000)

    # Environment satisfaction (inverted)
    es = input_values.get("EnvironmentSatisfaction", 3)
    factor_scores["Environment Satisfaction"] = (5 - es) / 4

    # Distance from home
    dfh = input_values.get("DistanceFromHome", 9)
    factor_scores["Distance from Home"] = min(dfh / 29, 1.0)

    # Job involvement (inverted)
    ji = input_values.get("JobInvolvement", 3)
    factor_scores["Job Involvement"] = (5 - ji) / 4

    # Sort by score descending
    labels  = list(factor_scores.keys())
    scores  = list(factor_scores.values())
    paired  = sorted(zip(scores, labels), reverse=True)
    scores, labels = zip(*paired)

    # Colours: red for high risk, amber for medium, green for low
    bar_colors = [
        RED if s > 0.65 else (AMBER if s > 0.35 else GREEN)
        for s in scores
    ]

    fig, ax = plt.subplots(figsize=(5.5, 3.8), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    y_pos = np.arange(len(labels))
    bars  = ax.barh(y_pos, scores, color=bar_colors, height=0.58, edgecolor="none")

    # Background reference lines
    for x in [0.25, 0.5, 0.75, 1.0]:
        ax.axvline(x, color=BORDER, linewidth=0.6, zorder=0)

    # Value labels
    for bar, s in zip(bars, scores):
        ax.text(s + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{s:.2f}", va="center", fontsize=8, color=TEXT_MUTE)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9, color=TEXT_MAIN)
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1.0"], fontsize=8, color=TEXT_MUTE)
    ax.tick_params(axis="both", which="both", length=0)
    ax.spines[:].set_visible(False)

    legend_handles = [
        mpatches.Patch(color=RED,   label="High Risk"),
        mpatches.Patch(color=AMBER, label="Moderate"),
        mpatches.Patch(color=GREEN, label="Low Risk"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=7,
              facecolor=DARK_BG2, edgecolor=BORDER, labelcolor=TEXT_MUTE,
              framealpha=1, handlelength=1.0, handleheight=0.8)

    ax.set_title("Risk Factor Contributions (normalised)", fontsize=9,
                 color=TEXT_MUTE, pad=10, loc="left")

    fig.tight_layout(pad=0.6)
    return fig_to_bytes(fig)


def make_model_comparison_chart() -> bytes:
    """Bar chart comparing model metrics — a static project-level overview."""
    models   = ["Logistic\nRegression", "Random\nForest", "XGBoost"]
    accuracy = [86.0, 87.0, 88.0]
    f1       = [52.0, 57.0, 62.0]
    roc_auc  = [81.0, 84.0, 87.0]

    x   = np.arange(len(models))
    w   = 0.24

    fig, ax = plt.subplots(figsize=(6, 3.2), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    ax.bar(x - w,     accuracy, width=w, color="#4F46E5", alpha=0.9, label="Accuracy (%)")
    ax.bar(x,         f1,       width=w, color=AMBER,     alpha=0.9, label="F1 Score (%)")
    ax.bar(x + w,     roc_auc,  width=w, color=GREEN,     alpha=0.9, label="ROC-AUC (%)")

    # Highlight best model
    ax.axvspan(1.5, 2.5, color=ACCENT, alpha=0.06, zorder=0)
    ax.text(2, 90.5, "Best Model", ha="center", fontsize=7.5, color=ACCENT, fontweight="600")

    for bars in [ax.containers[0], ax.containers[1], ax.containers[2]]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.0f}", ha="center", va="bottom", fontsize=7, color=TEXT_MUTE)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9, color=TEXT_MAIN)
    ax.set_ylim(40, 96)
    ax.set_ylabel("Score (%)", fontsize=8, color=TEXT_MUTE)
    ax.tick_params(axis="y", colors=TEXT_MUTE, labelsize=8)
    ax.tick_params(axis="x", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER)
    ax.spines["bottom"].set_color(BORDER)
    ax.yaxis.set_tick_params(color=BORDER)

    for y in [60, 70, 80, 90]:
        ax.axhline(y, color=BORDER, linewidth=0.5, zorder=0)

    ax.legend(fontsize=7.5, facecolor=DARK_BG2, edgecolor=BORDER,
              labelcolor=TEXT_MUTE, framealpha=1, loc="lower right")
    ax.set_title("Model Performance Comparison", fontsize=9, color=TEXT_MUTE, pad=10, loc="left")

    fig.tight_layout(pad=0.6)
    return fig_to_bytes(fig)


def make_attrition_driver_chart() -> bytes:
    """
    Horizontal bar chart of top dataset-level attrition drivers
    derived from the IBM HR dataset insights (from the EDA).
    """
    drivers = [
        "OverTime = Yes",
        "Sales Representative",
        "Single (Marital Status)",
        "Low Job Satisfaction",
        "Poor Work-Life Balance",
        "No Promotion (3+ yrs)",
        "Low Environment Satisfaction",
        "Frequent Business Travel",
    ]
    attrition_rates = [30.5, 39.8, 25.5, 22.8, 31.2, 18.6, 25.0, 24.9]
    baseline = 16.1

    fig, ax = plt.subplots(figsize=(6, 3.8), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    y_pos  = np.arange(len(drivers))
    colors = [RED if v > 25 else AMBER for v in attrition_rates]

    ax.barh(y_pos, attrition_rates, color=colors, height=0.55, edgecolor="none", alpha=0.85)
    ax.axvline(baseline, color=TEXT_MUTE, linewidth=1.1, linestyle="--", label=f"Dataset avg {baseline}%")

    for i, v in enumerate(attrition_rates):
        ax.text(v + 0.5, i, f"{v}%", va="center", fontsize=8, color=TEXT_MUTE)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(drivers, fontsize=8.5, color=TEXT_MAIN)
    ax.set_xlabel("Attrition Rate (%)", fontsize=8, color=TEXT_MUTE)
    ax.set_xlim(0, 48)
    ax.tick_params(axis="x", colors=TEXT_MUTE, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BORDER)

    for x in [10, 20, 30, 40]:
        ax.axvline(x, color=BORDER, linewidth=0.5, zorder=0)

    ax.legend(fontsize=7.5, facecolor=DARK_BG2, edgecolor=BORDER,
              labelcolor=TEXT_MUTE, framealpha=1, loc="lower right")
    ax.set_title("Attrition Rate by Risk Driver (IBM HR Dataset)", fontsize=9,
                 color=TEXT_MUTE, pad=10, loc="left")

    fig.tight_layout(pad=0.6)
    return fig_to_bytes(fig)


# ── MODEL LOADING ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    try:
        with open('Model/model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("model.pkl not found inside Model/ folder.")
        st.stop()

pkg            = load_model()
model          = pkg['model']
scaler         = pkg.get('scaler')
needs_scaling  = pkg.get('needs_scaling', False)
label_encoders = pkg.get('label_encoders', {})
feature_names  = pkg.get('feature_names', [])
model_name     = pkg.get('model_name', 'ML Model')


# ── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <div class="sb-logo-mark">AI</div>
        <div>
            <div class="sb-logo-text">AttritionAI</div>
            <div class="sb-logo-sub">HR Risk Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sb-section">
        <div class="sb-section-title">Model Info</div>
        <div class="sb-stat">
            <div class="sb-stat-label">Active Model</div>
            <div class="sb-stat-val" style="font-size:14px;margin-top:2px;">XGBoost</div>
            <span class="sb-stat-chip chip-green">Best F1 Score</span>
        </div>
        <div class="sb-stat">
            <div class="sb-stat-label">Dataset</div>
            <div class="sb-stat-val" style="font-size:13px;">IBM HR Analytics</div>
            <span class="sb-stat-chip chip-green">1,470 records</span>
        </div>
        <div class="sb-stat">
            <div class="sb-stat-label">Attrition Rate (dataset)</div>
            <div class="sb-stat-val">16.1%</div>
            <span class="sb-stat-chip chip-red">237 of 1,470</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section-title" style="margin-bottom:10px;">Dataset Overview</div>', unsafe_allow_html=True)
    st.image(make_attrition_driver_chart(), use_container_width=True)

    st.markdown("""
    <div style="font-size:10px;color:#374151;text-align:center;padding-top:10px;">
        AIML Internship Capstone 2026
    </div>
    """, unsafe_allow_html=True)


# ── TOP BAR ──────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="topbar">
    <div style="display:flex;align-items:center;gap:16px;">
        <div class="topbar-bar-accent"></div>
        <div>
            <div class="topbar-title">Employee Attrition Prediction</div>
            <div class="topbar-tagline">Predict flight risk · Retain top talent · Act before attrition occurs</div>
        </div>
    </div>
    <div class="topbar-right">
        <span class="topbar-badge">{model_name}</span>
        <span class="topbar-badge">IBM HR Analytics · 2026</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── KPI ROW ──────────────────────────────────────────────────────────────────

st.markdown("""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Models Trained</div>
        <div class="kpi-val">3</div>
        <div class="kpi-change neutral">LR · RF · XGBoost</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Best F1 Score</div>
        <div class="kpi-val">62%</div>
        <div class="kpi-change up">XGBoost</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Dataset Attrition</div>
        <div class="kpi-val">16.1%</div>
        <div class="kpi-change down">237 of 1,470 left</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Features Used</div>
        <div class="kpi-val">30</div>
        <div class="kpi-change neutral">HR profile inputs</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── MODEL COMPARISON CHART ────────────────────────────────────────────────────

st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
st.markdown('<div class="chart-title">Model Comparison</div>', unsafe_allow_html=True)
st.markdown('<div class="chart-sub">Accuracy, F1 Score and ROC-AUC across all three trained models</div>', unsafe_allow_html=True)
st.image(make_model_comparison_chart(), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)


# ── INPUT FORM ────────────────────────────────────────────────────────────────

st.markdown("""
<div class="section-hdr">
    <div>
        <div class="section-title">Employee Profile Input</div>
        <div class="section-sub">Fill in the employee details across the three sections below, then run the analysis.</div>
    </div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="form-panel"><div class="panel-title">Personal</div>', unsafe_allow_html=True)
    age             = st.slider("Age", 18, 60, 35)
    gender          = st.selectbox("Gender", ["Male", "Female"])
    marital_status  = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    education       = st.selectbox("Education Level", [1,2,3,4,5],
                        format_func=lambda x:{1:"Below College",2:"College",3:"Bachelor",4:"Master",5:"Doctor"}[x])
    education_field = st.selectbox("Education Field",
                        ["Life Sciences","Medical","Marketing","Technical Degree","Human Resources","Other"])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="form-panel"><div class="panel-title">Job</div>', unsafe_allow_html=True)
    department      = st.selectbox("Department", ["Research & Development","Sales","Human Resources"])
    job_role        = st.selectbox("Job Role", [
        "Sales Executive","Research Scientist","Laboratory Technician",
        "Manufacturing Director","Healthcare Representative","Manager",
        "Sales Representative","Research Director","Human Resources"])
    job_level       = st.selectbox("Job Level", [1,2,3,4,5])
    job_involvement = st.selectbox("Job Involvement", [1,2,3,4],
                        format_func=lambda x:{1:"Low",2:"Medium",3:"High",4:"Very High"}[x])
    business_travel = st.selectbox("Business Travel", ["Non-Travel","Travel_Rarely","Travel_Frequently"])
    overtime        = st.selectbox("OverTime", ["Yes","No"])
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="form-panel"><div class="panel-title">Compensation & Tenure</div>', unsafe_allow_html=True)
    monthly_income      = st.number_input("Monthly Income ($)", 1000, 20000, 5000, step=500)
    daily_rate          = st.number_input("Daily Rate", 100, 1500, 800, step=50)
    hourly_rate         = st.number_input("Hourly Rate", 30, 100, 65, step=5)
    percent_salary_hike = st.slider("Salary Hike (%)", 11, 25, 15)
    stock_option_level  = st.selectbox("Stock Option Level", [0,1,2,3])
    total_working_years = st.slider("Total Working Years", 0, 40, 10)
    years_at_company    = st.slider("Years at Company", 0, 40, 5)
    years_in_role       = st.slider("Years in Current Role", 0, 18, 3)
    years_since_promo   = st.slider("Years Since Promotion", 0, 15, 1)
    years_with_manager  = st.slider("Years with Current Manager", 0, 17, 3)
    num_companies       = st.slider("Companies Worked At", 0, 9, 2)
    training_times      = st.slider("Training Sessions Last Year", 0, 6, 3)
    st.markdown('</div>', unsafe_allow_html=True)

with st.expander("Satisfaction & Performance Details (optional)"):
    c1, c2 = st.columns(2)
    with c1:
        distance_home            = st.slider("Distance From Home (km)", 1, 29, 9)
        environment_satisfaction = st.selectbox("Environment Satisfaction", [1,2,3,4],
                                     format_func=lambda x:{1:"Low",2:"Medium",3:"High",4:"Very High"}[x])
        job_satisfaction         = st.selectbox("Job Satisfaction", [1,2,3,4],
                                     format_func=lambda x:{1:"Low",2:"Medium",3:"High",4:"Very High"}[x])
    with c2:
        relationship_satisfaction = st.selectbox("Relationship Satisfaction", [1,2,3,4],
                                      format_func=lambda x:{1:"Low",2:"Medium",3:"High",4:"Very High"}[x])
        work_life_balance         = st.selectbox("Work-Life Balance", [1,2,3,4],
                                      format_func=lambda x:{1:"Bad",2:"Good",3:"Better",4:"Best"}[x])
        performance_rating        = st.selectbox("Performance Rating", [3,4],
                                      format_func=lambda x:{3:"Excellent",4:"Outstanding"}[x])
        monthly_rate              = st.number_input("Monthly Rate", 2000, 27000, 14000, step=500)

st.markdown("---")

btn_col, _ = st.columns([1, 2])
with btn_col:
    predict_clicked = st.button("Run Attrition Analysis", use_container_width=True)


# ── PREDICTION & RESULTS ──────────────────────────────────────────────────────

if predict_clicked:
    input_dict = {
        'Age': age, 'BusinessTravel': business_travel, 'DailyRate': daily_rate,
        'Department': department, 'DistanceFromHome': distance_home, 'Education': education,
        'EducationField': education_field, 'EnvironmentSatisfaction': environment_satisfaction,
        'Gender': gender, 'HourlyRate': hourly_rate, 'JobInvolvement': job_involvement,
        'JobLevel': job_level, 'JobRole': job_role, 'JobSatisfaction': job_satisfaction,
        'MaritalStatus': marital_status, 'MonthlyIncome': monthly_income, 'MonthlyRate': monthly_rate,
        'NumCompaniesWorked': num_companies, 'OverTime': overtime,
        'PercentSalaryHike': percent_salary_hike, 'PerformanceRating': performance_rating,
        'RelationshipSatisfaction': relationship_satisfaction, 'StockOptionLevel': stock_option_level,
        'TotalWorkingYears': total_working_years, 'TrainingTimesLastYear': training_times,
        'WorkLifeBalance': work_life_balance, 'YearsAtCompany': years_at_company,
        'YearsInCurrentRole': years_in_role, 'YearsSinceLastPromotion': years_since_promo,
        'YearsWithCurrManager': years_with_manager,
    }
    for col, le in label_encoders.items():
        if col in input_dict:
            try: input_dict[col] = le.transform([input_dict[col]])[0]
            except ValueError: input_dict[col] = 0

    input_array = np.array([input_dict.get(f, 0) for f in feature_names]).reshape(1, -1)
    if needs_scaling and scaler:
        input_array = scaler.transform(input_array)

    prediction  = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0]
    stay_prob   = probability[0] * 100
    leave_prob  = probability[1] * 100

    st.markdown('<div class="section-title" style="color:#F1F5F9;font-size:13px;margin:14px 0 14px;">Analysis Result</div>', unsafe_allow_html=True)

    # ── Row 1: prediction card + gauge + probability bar ──────────────────────

    r1, r2 = st.columns([3, 2])

    with r1:
        if prediction == 0:
            st.markdown(f"""
            <div class="result-wrap result-stay">
                <div class="result-eyebrow stay">Prediction — Low Risk</div>
                <div class="result-heading">Employee likely to stay</div>
                <div class="result-desc">No significant attrition signals detected. Profile matches retained employee patterns in the training data.</div>
                <span class="result-pill pill-stay">{stay_prob:.1f}% confidence</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-wrap result-leave">
                <div class="result-eyebrow leave">Prediction — High Risk</div>
                <div class="result-heading">Employee at risk of leaving</div>
                <div class="result-desc">Multiple attrition signals detected. Immediate HR review and retention action recommended.</div>
                <span class="result-pill pill-leave">{leave_prob:.1f}% confidence</span>
            </div>""", unsafe_allow_html=True)

        # Probability breakdown bar
        st.markdown('<div class="chart-panel" style="padding:14px 16px;">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title" style="margin-bottom:8px;">Probability Breakdown</div>', unsafe_allow_html=True)
        st.image(make_prob_bar(stay_prob, leave_prob), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r2:
        # Gauge chart
        st.markdown('<div class="chart-panel" style="padding:14px 16px;text-align:center;">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title" style="margin-bottom:4px;">Risk Gauge</div>', unsafe_allow_html=True)
        st.image(make_gauge(leave_prob), use_container_width=True)

        # Risk level text
        if leave_prob < 30:
            bar_class = "risk-bar-fill-low"; status_class = "risk-low"; status_text = "Low risk — No immediate action required"
        elif leave_prob < 60:
            bar_class = "risk-bar-fill-mid"; status_class = "risk-med"; status_text = "Medium risk — Schedule a 1:1 check-in"
        else:
            bar_class = "risk-bar-fill-high"; status_class = "risk-high"; status_text = "High risk — Immediate HR intervention needed"

        st.markdown(f"""
        <div class="risk-bar-bg" style="margin-top:10px;"><div class="{bar_class}" style="width:{leave_prob:.0f}%"></div></div>
        <div class="risk-status {status_class}" style="margin-top:4px;">{status_text}</div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 2: Risk factors chart ─────────────────────────────────────────────

    st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">Individual Risk Factor Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-sub">Normalised risk contribution for each input factor (0 = no risk, 1 = maximum risk)</div>', unsafe_allow_html=True)

    # Reconstruct readable input dict for the chart (pre-encoding values)
    chart_input = {
        "OverTime": overtime,
        "JobSatisfaction": job_satisfaction,
        "WorkLifeBalance": work_life_balance,
        "YearsSinceLastPromotion": years_since_promo,
        "MonthlyIncome": monthly_income,
        "EnvironmentSatisfaction": environment_satisfaction,
        "DistanceFromHome": distance_home,
        "JobInvolvement": job_involvement,
    }
    st.image(make_risk_factors_chart(chart_input), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 3: HR Recommendations ─────────────────────────────────────────────

    recs = []
    if overtime == "Yes":
        recs.append(("warn", "Overtime Detected", "Redistribute workload or consider additional headcount to prevent burnout."))
    if work_life_balance <= 2:
        recs.append(("warn", "Poor Work-Life Balance", "Explore flexible or hybrid arrangements to improve retention odds."))
    if job_satisfaction <= 2:
        recs.append(("warn", "Low Job Satisfaction", "Schedule a structured career conversation and role review."))
    if years_since_promo > 3:
        recs.append(("info", "Stalled Progression", "No promotion in 3+ years — assess eligibility and create a growth plan."))
    if environment_satisfaction <= 2:
        recs.append(("info", "Environment Concerns", "Investigate team dynamics and physical/remote workplace conditions."))
    if monthly_income < 3000:
        recs.append(("info", "Below-Market Salary", "Benchmark compensation against current industry standards."))

    if recs:
        st.markdown('<div style="margin-top:16px;"><div class="section-title" style="color:#F1F5F9;font-size:12px;margin-bottom:10px;">HR Action Items</div><div class="rec-grid">', unsafe_allow_html=True)
        for rtype, title, body in recs:
            cls  = "rec-warn" if rtype == "warn" else "rec-info"
            icon = "!" if rtype == "warn" else "i"
            st.markdown(f'<div class="rec-card {cls}"><strong>{icon}  {title}</strong>{body}</div>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)


st.markdown('<div class="footer">AttritionAI · AIML Internship Capstone 2026 · IBM HR Analytics Dataset</div>', unsafe_allow_html=True)
