import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

/* ── Base ── */
.stApp { background: #0B0F19; }
section[data-testid="stSidebar"] {
    background: #0F1420 !important;
    border-right: 1px solid #1E2433 !important;
}

/* ── Sidebar ── */
.sb-logo {
    display:flex; align-items:center; gap:10px;
    padding:8px 0 20px; border-bottom:1px solid #1E2433; margin-bottom:20px;
}
.sb-logo-mark {
    width:36px; height:36px; background:#4F46E5; border-radius:9px;
    display:flex; align-items:center; justify-content:center;
    font-size:12px; font-weight:700; color:#fff; letter-spacing:-0.5px;
    box-shadow: 0 0 12px rgba(79,70,229,0.4);
}
.sb-logo-text { font-size:14px; font-weight:700; color:#F1F5F9; letter-spacing:-0.3px; }
.sb-logo-sub  { font-size:10px; color:#64748B; margin-top:1px; }
.sb-section   { margin-bottom:24px; }
.sb-section-title {
    font-size:9px; font-weight:700; color:#4F46E5;
    text-transform:uppercase; letter-spacing:1.8px; margin-bottom:10px;
}
.sb-stat {
    background:#131827; border:1px solid #1E2433; border-radius:10px;
    padding:12px; margin-bottom:8px;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.sb-stat:hover {
    border-color:#4F46E5 !important;
    box-shadow: 0 0 10px rgba(79,70,229,0.15);
}
.sb-stat-label { font-size:10px; color:#64748B; margin-bottom:4px; }
.sb-stat-val   { font-size:18px; font-weight:700; color:#F1F5F9; }
.sb-stat-chip  {
    display:inline-block; font-size:9px; font-weight:600;
    padding:2px 8px; border-radius:99px; margin-top:5px;
}
.chip-green { background:#0D3321; color:#34D399; }
.chip-red   { background:#3B0D0D; color:#F87171; }

/* ── Top bar ── */
.topbar {
    background: linear-gradient(135deg, #0F1420 0%, #131827 100%);
    border:1px solid #1E2433; border-radius:16px;
    padding:22px 28px; display:flex; align-items:center;
    justify-content:space-between; margin-bottom:20px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}
.topbar-bar-accent {
    width:4px; height:46px; background:linear-gradient(180deg,#6366F1,#4F46E5);
    border-radius:2px; flex-shrink:0;
    box-shadow: 0 0 10px rgba(99,102,241,0.5);
}
.topbar-title   { font-size:20px; font-weight:700; color:#F1F5F9; letter-spacing:-0.5px; line-height:1; }
.topbar-tagline { font-size:11px; color:#64748B; margin-top:5px; }
.topbar-badge {
    background:#1E2433; border:1px solid #2D3748; color:#94A3B8;
    font-size:10px; font-weight:600; padding:4px 12px; border-radius:99px;
    transition: border-color 0.2s, color 0.2s;
}
.topbar-badge:hover { border-color:#4F46E5; color:#F1F5F9; }
.topbar-right { display:flex; flex-direction:column; align-items:flex-end; gap:6px; }

/* ── KPI cards ── */
.kpi-grid {
    display:grid; grid-template-columns:repeat(4,1fr);
    gap:12px; margin-bottom:20px;
}
.kpi-card {
    background: linear-gradient(135deg, #0F1420 0%, #131827 100%);
    border:1px solid #1E2433; border-radius:14px; padding:18px 20px;
    transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
    cursor:default;
}
.kpi-card:hover {
    transform: translateY(-3px);
    border-color:#4F46E5;
    box-shadow: 0 8px 24px rgba(79,70,229,0.15);
}
.kpi-label {
    font-size:10px; font-weight:600; color:#64748B;
    text-transform:uppercase; letter-spacing:1px; margin-bottom:10px;
}
.kpi-val    { font-size:28px; font-weight:700; color:#F1F5F9; letter-spacing:-0.5px; line-height:1; }
.kpi-change { font-size:10px; font-weight:600; margin-top:7px; }
.kpi-change.up      { color:#34D399; }
.kpi-change.down    { color:#F87171; }
.kpi-change.neutral { color:#94A3B8; }
.kpi-accent-bar {
    height:2px; border-radius:1px; margin-bottom:14px;
    background:linear-gradient(90deg,#4F46E5,transparent);
}

/* ── Section header ── */
.section-hdr { display:flex; align-items:center; justify-content:space-between; margin-bottom:14px; }
.section-title { font-size:13px; font-weight:700; color:#F1F5F9; letter-spacing:-0.2px; }
.section-sub   { font-size:10px; color:#64748B; margin-top:2px; }
.section-divider {
    height:1px; background:linear-gradient(90deg,#4F46E5 0%,#1E2433 40%,transparent 100%);
    margin:6px 0 16px;
}

/* ── Form panels ── */
.form-panel {
    background:#0F1420; border:1px solid #1E2433; border-radius:14px;
    padding:20px; margin-bottom:16px;
    transition: border-color 0.2s;
}
.form-panel:hover { border-color:#2D3748; }
.panel-title {
    font-size:10px; font-weight:700; color:#4F46E5;
    text-transform:uppercase; letter-spacing:1.4px;
    padding-bottom:12px; border-bottom:1px solid #1E2433; margin-bottom:16px;
    display:flex; align-items:center; gap:7px;
}
.panel-dot {
    width:6px; height:6px; border-radius:50%; background:#4F46E5;
    box-shadow: 0 0 6px rgba(79,70,229,0.7); flex-shrink:0;
}

/* ── Inputs ── */
label, .stSelectbox label, .stSlider label, .stNumberInput label {
    font-size:11px !important; font-weight:500 !important;
    color:#94A3B8 !important; text-transform:uppercase; letter-spacing:0.8px !important;
}
.stSelectbox > div > div, .stNumberInput > div > div > input {
    background:#131827 !important; border:1px solid #1E2433 !important;
    border-radius:8px !important; color:#F1F5F9 !important;
    transition: border-color 0.2s !important;
}
.stSelectbox > div > div:hover, .stNumberInput > div > div > input:focus {
    border-color:#4F46E5 !important;
    box-shadow: 0 0 0 2px rgba(79,70,229,0.15) !important;
}
div[data-testid="stSlider"] > div { background: transparent !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background:#4F46E5 !important; border:none !important;
    box-shadow: 0 0 8px rgba(79,70,229,0.5) !important;
}

/* ── Expander fix — hide the default arrow text ── */
.streamlit-expanderHeader {
    background:#0F1420 !important;
    border:1px solid #1E2433 !important;
    border-radius:10px !important;
    color:#94A3B8 !important;
    font-size:11px !important;
    font-weight:600 !important;
    text-transform:uppercase !important;
    letter-spacing:0.8px !important;
    padding:12px 16px !important;
    transition: border-color 0.2s, color 0.2s !important;
}
.streamlit-expanderHeader:hover {
    border-color:#4F46E5 !important;
    color:#F1F5F9 !important;
}
.streamlit-expanderHeader p { display:none !important; }
.streamlit-expanderHeader::before {
    content: "Satisfaction & Performance Details";
    font-size:11px; font-weight:600; color:inherit;
    text-transform:uppercase; letter-spacing:0.8px;
}
.streamlit-expanderContent {
    background:#0F1420 !important;
    border:1px solid #1E2433 !important;
    border-top:none !important;
    border-radius:0 0 10px 10px !important;
    padding:16px !important;
}
/* Hide any stray arrow characters */
details > summary svg { display:none !important; }
details > summary::marker { display:none !important; content:''; }
details > summary::-webkit-details-marker { display:none !important; }

/* ── Predict button ── */
.stButton > button {
    background: linear-gradient(135deg, #4F46E5, #6366F1) !important;
    color:#fff !important; border:none !important;
    border-radius:12px !important; padding:14px 28px !important;
    font-size:13px !important; font-weight:700 !important;
    letter-spacing:0.5px !important; width:100% !important;
    box-shadow: 0 4px 16px rgba(79,70,229,0.35) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #4338CA, #4F46E5) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(79,70,229,0.45) !important;
}
.stButton > button:active { transform: translateY(0px) !important; }

/* ── Result cards ── */
.result-wrap {
    background: linear-gradient(135deg, #0F1420 0%, #131827 100%);
    border:1px solid #1E2433; border-radius:16px;
    padding:24px; margin-bottom:16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
}
.result-stay  { border-left:4px solid #10B981; }
.result-leave { border-left:4px solid #EF4444; }
.result-eyebrow {
    font-size:9px; font-weight:700; text-transform:uppercase;
    letter-spacing:2.5px; margin-bottom:8px;
}
.result-eyebrow.stay  { color:#10B981; }
.result-eyebrow.leave { color:#EF4444; }
.result-heading { font-size:22px; font-weight:700; color:#F1F5F9; margin-bottom:6px; letter-spacing:-0.4px; }
.result-desc    { font-size:12px; color:#64748B; margin-bottom:16px; line-height:1.7; }
.result-pill    { display:inline-block; font-size:13px; font-weight:700; padding:6px 18px; border-radius:99px; }
.pill-stay  { background:#063620; color:#34D399; border:1px solid #10B981; }
.pill-leave { background:#3B0D0D; color:#F87171; border:1px solid #EF4444; }

/* ── Probability boxes ── */
.prob-box {
    background:#131827; border:1px solid #1E2433; border-radius:12px;
    padding:18px 16px; text-align:center;
    transition: border-color 0.2s, transform 0.2s;
}
.prob-box:hover { transform:scale(1.02); }
.prob-box-label { font-size:9px; font-weight:600; color:#64748B; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px; }
.prob-box-val   { font-size:30px; font-weight:700; letter-spacing:-0.5px; }
.val-stay  { color:#34D399; }
.val-leave { color:#F87171; }

/* ── Risk bar ── */
.risk-bar-bg       { height:7px; background:#1E2433; border-radius:4px; margin-bottom:8px; overflow:hidden; }
.risk-bar-fill-low  { height:7px; background:linear-gradient(90deg,#059669,#10B981); border-radius:4px; }
.risk-bar-fill-mid  { height:7px; background:linear-gradient(90deg,#D97706,#F59E0B); border-radius:4px; }
.risk-bar-fill-high { height:7px; background:linear-gradient(90deg,#DC2626,#EF4444); border-radius:4px; }
.risk-status        { font-size:11px; font-weight:600; }
.risk-low  { color:#34D399; }
.risk-med  { color:#FBBF24; }
.risk-high { color:#F87171; }

/* ── Recommendation cards ── */
.rec-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:4px; }
.rec-card {
    background:#131827; border:1px solid #1E2433; border-radius:10px;
    padding:14px 16px; font-size:11px; color:#94A3B8; line-height:1.6;
    transition: transform 0.2s, border-color 0.2s;
}
.rec-card:hover { transform:translateY(-2px); border-color:#2D3748; }
.rec-card strong { color:#F1F5F9; display:block; margin-bottom:4px; font-size:11px; }
.rec-warn { border-top:2px solid #F59E0B; }
.rec-info { border-top:2px solid #4F46E5; }

/* ── Chart containers ── */
.chart-panel {
    background:#0F1420; border:1px solid #1E2433; border-radius:14px;
    padding:20px; margin-bottom:16px;
    transition: border-color 0.2s;
}
.chart-panel:hover { border-color:#2D3748; }
.chart-title { font-size:11px; font-weight:700; color:#F1F5F9; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px; }
.chart-sub   { font-size:10px; color:#64748B; margin-bottom:14px; }

/* ── Misc ── */
hr { border-color:#1E2433 !important; margin:20px 0 !important; }
.footer { text-align:center; font-size:10px; color:#374151; padding:20px 0 6px; }

div[data-testid="stMetricValue"] { color:#F1F5F9 !important; font-size:22px !important; font-weight:700 !important; }
div[data-testid="stMetricLabel"] { color:#64748B !important; font-size:10px !important; }
</style>
""", unsafe_allow_html=True)


# ── CHART HELPERS ─────────────────────────────────────────────────────────────

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
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140, facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf


def make_gauge(leave_prob: float) -> bytes:
    """Semicircular gauge — attrition risk 0-100%."""
    fig, ax = plt.subplots(figsize=(4.2, 2.5), facecolor=DARK_BG2)
    ax.set_facecolor(DARK_BG2)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.25, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")

    r = 1.0
    # Track background
    theta_bg = np.linspace(np.pi, 0, 300)
    ax.plot(r * np.cos(theta_bg), r * np.sin(theta_bg),
            color=BORDER, linewidth=16, solid_capstyle="round", zorder=1)

    # Coloured zone overlays (faint)
    for t1, t2, col in [
        (np.pi,       np.pi*0.6, GREEN),
        (np.pi*0.6,   np.pi*0.3, AMBER),
        (np.pi*0.3,   0,         RED),
    ]:
        th = np.linspace(t1, t2, 100)
        ax.plot(r*np.cos(th), r*np.sin(th), color=col, linewidth=16,
                solid_capstyle="butt", alpha=0.22, zorder=2)

    # Filled arc
    fill_end  = np.pi - (leave_prob/100)*np.pi
    th_fill   = np.linspace(np.pi, fill_end, 200)
    fill_col  = GREEN if leave_prob < 40 else (AMBER if leave_prob < 70 else RED)
    ax.plot(r*np.cos(th_fill), r*np.sin(th_fill), color=fill_col, linewidth=16,
            solid_capstyle="round", alpha=0.9, zorder=3)

    # Needle
    angle = np.pi - (leave_prob/100)*np.pi
    nx, ny = 0.76*np.cos(angle), 0.76*np.sin(angle)
    ax.annotate("", xy=(nx, ny), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=TEXT_MAIN, lw=1.8, mutation_scale=13))
    ax.plot(0, 0, "o", color=TEXT_MAIN, markersize=7, zorder=6)

    # Labels
    ax.text(0, -0.12, f"{leave_prob:.1f}%", ha="center", va="center",
            fontsize=18, fontweight="700", color=fill_col, fontfamily="monospace")
    ax.text(0, -0.30, "Attrition Risk", ha="center", va="center", fontsize=8, color=TEXT_MUTE)
    ax.text(-1.15, -0.12, "0%",   ha="center", fontsize=7, color=TEXT_MUTE)
    ax.text( 1.15, -0.12, "100%", ha="center", fontsize=7, color=TEXT_MUTE)

    # Zone labels
    ax.text(-0.78,  0.72, "LOW",    ha="center", fontsize=6.5, color=GREEN,  alpha=0.7)
    ax.text( 0.0,   1.06, "MED",    ha="center", fontsize=6.5, color=AMBER,  alpha=0.7)
    ax.text( 0.78,  0.72, "HIGH",   ha="center", fontsize=6.5, color=RED,    alpha=0.7)

    fig.tight_layout(pad=0.3)
    return fig_to_bytes(fig)


def make_prob_bar(stay_prob: float, leave_prob: float) -> bytes:
    """Segmented probability bar."""
    fig, ax = plt.subplots(figsize=(5.5, 1.0), facecolor=DARK_BG2)
    ax.set_facecolor(DARK_BG2)

    ax.barh(0, stay_prob,              color=GREEN,  height=0.5, edgecolor="none")
    ax.barh(0, leave_prob, left=stay_prob, color=RED, height=0.5, edgecolor="none")

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")

    if stay_prob > 12:
        ax.text(stay_prob/2, 0, f"Stay  {stay_prob:.1f}%",
                ha="center", va="center", fontsize=8.5, color="#fff", fontweight="600")
    if leave_prob > 12:
        ax.text(stay_prob + leave_prob/2, 0, f"Leave  {leave_prob:.1f}%",
                ha="center", va="center", fontsize=8.5, color="#fff", fontweight="600")

    fig.tight_layout(pad=0.2)
    return fig_to_bytes(fig)


def make_risk_factors_chart(inputs: dict) -> bytes:
    """Horizontal bar chart of normalised risk factors."""
    raw = {
        "OverTime":               1.0 if inputs.get("OverTime") == "Yes" else 0.0,
        "Job Satisfaction":       (5 - inputs.get("JobSatisfaction", 3)) / 4,
        "Work-Life Balance":      (5 - inputs.get("WorkLifeBalance", 3)) / 4,
        "Promotion Gap":          min(inputs.get("YearsSinceLastPromotion", 1) / 10, 1.0),
        "Monthly Income":         max(0, 1 - inputs.get("MonthlyIncome", 5000) / 20000),
        "Environment Satisfaction": (5 - inputs.get("EnvironmentSatisfaction", 3)) / 4,
        "Distance from Home":     min(inputs.get("DistanceFromHome", 9) / 29, 1.0),
        "Job Involvement":        (5 - inputs.get("JobInvolvement", 3)) / 4,
    }
    paired  = sorted(raw.items(), key=lambda x: x[1], reverse=True)
    labels  = [p[0] for p in paired]
    scores  = [p[1] for p in paired]
    colors  = [RED if s > 0.65 else (AMBER if s > 0.35 else GREEN) for s in scores]

    fig, ax = plt.subplots(figsize=(6.5, 4.0), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)

    y_pos = np.arange(len(labels))
    bars  = ax.barh(y_pos, scores, color=colors, height=0.55, edgecolor="none")

    for x in [0.25, 0.5, 0.75, 1.0]:
        ax.axvline(x, color=BORDER, linewidth=0.7, zorder=0)

    for bar, s in zip(bars, scores):
        ax.text(s + 0.025, bar.get_y() + bar.get_height()/2,
                f"{s:.2f}", va="center", fontsize=8.5, color=TEXT_MUTE)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.5, color=TEXT_MAIN)
    ax.set_xlim(0, 1.22)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1.0"], fontsize=8, color=TEXT_MUTE)
    ax.tick_params(axis="both", length=0)
    ax.spines[:].set_visible(False)

    legend_handles = [
        mpatches.Patch(color=RED,   label="High Risk  > 0.65"),
        mpatches.Patch(color=AMBER, label="Moderate  0.35–0.65"),
        mpatches.Patch(color=GREEN, label="Low Risk  < 0.35"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=7.5,
              facecolor=DARK_BG2, edgecolor=BORDER, labelcolor=TEXT_MUTE,
              framealpha=1, handlelength=1.1)

    ax.set_title("Risk Factor Contributions  (normalised 0–1)", fontsize=9,
                 color=TEXT_MUTE, pad=10, loc="left")
    fig.tight_layout(pad=0.7)
    return fig_to_bytes(fig)


def make_attrition_driver_chart() -> bytes:
    """Sidebar: dataset-level attrition rates by driver."""
    drivers = [
        "Frequent Business Travel",
        "Low Environment Satisfaction",
        "No Promotion (3+ yrs)",
        "Poor Work-Life Balance",
        "Low Job Satisfaction",
        "Single (Marital Status)",
        "Sales Representative",
        "OverTime = Yes",
    ]
    rates    = [24.9, 25.0, 18.6, 31.2, 22.8, 25.5, 39.8, 30.5]
    baseline = 16.1
    colors   = [RED if v > 25 else AMBER for v in rates]

    fig, ax = plt.subplots(figsize=(3.8, 4.0), facecolor=DARK_BG2)
    ax.set_facecolor(DARK_BG2)

    y_pos = np.arange(len(drivers))
    ax.barh(y_pos, rates, color=colors, height=0.52, edgecolor="none", alpha=0.88)
    ax.axvline(baseline, color=TEXT_MUTE, linewidth=1.2, linestyle="--",
               label=f"Avg {baseline}%", alpha=0.7)

    for i, v in enumerate(rates):
        ax.text(v + 0.4, i, f"{v}%", va="center", fontsize=7.5, color=TEXT_MUTE)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(drivers, fontsize=7.5, color=TEXT_MAIN)
    ax.set_xlabel("Attrition Rate (%)", fontsize=7.5, color=TEXT_MUTE)
    ax.set_xlim(0, 50)
    ax.tick_params(axis="x", colors=TEXT_MUTE, labelsize=7.5)
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BORDER)

    for x in [10, 20, 30, 40]:
        ax.axvline(x, color=BORDER, linewidth=0.5, zorder=0)

    ax.legend(fontsize=7, facecolor=DARK_BG2, edgecolor=BORDER,
              labelcolor=TEXT_MUTE, framealpha=1, loc="lower right")
    ax.set_title("Attrition Rate by Risk Driver", fontsize=8,
                 color=TEXT_MUTE, pad=8, loc="left")
    fig.tight_layout(pad=0.7)
    return fig_to_bytes(fig)


# ── MODEL LOADING ─────────────────────────────────────────────────────────────

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


# ── SIDEBAR ───────────────────────────────────────────────────────────────────

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
            <div class="sb-stat-val" style="font-size:15px;margin-top:2px;">XGBoost</div>
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

    st.markdown('<div class="sb-section-title" style="margin-bottom:10px;">Dataset Overview</div>',
                unsafe_allow_html=True)
    st.image(make_attrition_driver_chart(), use_container_width=True)

    st.markdown("""
    <div style="font-size:10px;color:#374151;text-align:center;padding-top:10px;">
        AIML Internship Capstone 2026
    </div>
    """, unsafe_allow_html=True)


# ── TOP BAR ───────────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="topbar">
    <div style="display:flex;align-items:center;gap:16px;">
        <div class="topbar-bar-accent"></div>
        <div>
            <div class="topbar-title">Employee Attrition Prediction</div>
            <div class="topbar-tagline">Predict flight risk &nbsp;·&nbsp; Retain top talent &nbsp;·&nbsp; Act before attrition occurs</div>
        </div>
    </div>
    <div class="topbar-right">
        <span class="topbar-badge">{model_name}</span>
        <span class="topbar-badge">IBM HR Analytics &nbsp;·&nbsp; 2026</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── KPI ROW ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-accent-bar"></div>
        <div class="kpi-label">Models Trained</div>
        <div class="kpi-val">3</div>
        <div class="kpi-change neutral">LR &nbsp;·&nbsp; RF &nbsp;·&nbsp; XGBoost</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-accent-bar"></div>
        <div class="kpi-label">Best F1 Score</div>
        <div class="kpi-val">62%</div>
        <div class="kpi-change up">XGBoost</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-accent-bar"></div>
        <div class="kpi-label">Dataset Attrition</div>
        <div class="kpi-val">16.1%</div>
        <div class="kpi-change down">237 of 1,470 left</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-accent-bar"></div>
        <div class="kpi-label">Features Used</div>
        <div class="kpi-val">30</div>
        <div class="kpi-change neutral">HR profile inputs</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── INPUT FORM ────────────────────────────────────────────────────────────────

st.markdown("""
<div class="section-hdr">
    <div>
        <div class="section-title">Employee Profile Input</div>
        <div class="section-sub">Fill in the employee details below, then run the analysis</div>
    </div>
</div>
<div class="section-divider"></div>
""", unsafe_allow_html=True)

with col1:
    st.markdown('<div class="form-panel"><div class="panel-title"><div class="panel-dot"></div>Personal</div>', unsafe_allow_html=True)
    age             = st.slider("Age", 18, 60, 35)
    gender          = st.selectbox("Gender", ["Male", "Female"])
    marital_status  = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    education       = st.selectbox("Education Level", [1,2,3,4,5],
                        format_func=lambda x:{1:"Below College",2:"College",3:"Bachelor",4:"Master",5:"Doctor"}[x])
    education_field = st.selectbox("Education Field",
                        ["Life Sciences","Medical","Marketing","Technical Degree","Human Resources","Other"])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="form-panel"><div class="panel-title"><div class="panel-dot"></div>Job</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="form-panel"><div class="panel-title"><div class="panel-dot"></div>Compensation & Tenure</div>', unsafe_allow_html=True)
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

# Expander — default values so variables always exist
distance_home             = 9
environment_satisfaction  = 3
job_satisfaction          = 3
relationship_satisfaction = 3
work_life_balance         = 3
performance_rating        = 3
monthly_rate              = 14000

st.markdown("""
<div style="background:#0F1420;border:1px solid #1E2433;border-radius:10px;
padding:14px 18px;margin-bottom:16px;">
<div style="font-size:10px;font-weight:700;color:#4F46E5;text-transform:uppercase;
letter-spacing:1.4px;margin-bottom:14px;border-bottom:1px solid #1E2433;
padding-bottom:10px;">Satisfaction & Performance Details (optional)</div>
""", unsafe_allow_html=True)

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

st.markdown('</div>', unsafe_allow_html=True)
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
            try:    input_dict[col] = le.transform([input_dict[col]])[0]
            except ValueError: input_dict[col] = 0

    input_array = np.array([input_dict.get(f, 0) for f in feature_names]).reshape(1, -1)
    if needs_scaling and scaler:
        input_array = scaler.transform(input_array)

    prediction  = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0]
    stay_prob   = probability[0] * 100
    leave_prob  = probability[1] * 100

    # Risk level
    if leave_prob < 30:
        bar_cls, st_cls, st_txt = "risk-bar-fill-low",  "risk-low",  "Low risk — No immediate action required"
    elif leave_prob < 60:
        bar_cls, st_cls, st_txt = "risk-bar-fill-mid",  "risk-med",  "Medium risk — Schedule a 1:1 check-in"
    else:
        bar_cls, st_cls, st_txt = "risk-bar-fill-high", "risk-high", "High risk — Immediate HR intervention needed"

    st.markdown("""
    <div class="section-hdr" style="margin-top:20px;">
        <div>
            <div class="section-title">Analysis Result</div>
            <div class="section-sub">Based on the submitted employee profile</div>
        </div>
    </div>
    <div class="section-divider"></div>
    """, unsafe_allow_html=True)

    # ── Row 1: result card | prob boxes | gauge ───────────────────────────────

    r1, r2, r3 = st.columns([4, 1, 2])

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

    with r2:
        st.markdown(f"""
        <div class="prob-box" style="margin-bottom:10px;">
            <div class="prob-box-label">Stay</div>
            <div class="prob-box-val val-stay">{stay_prob:.0f}%</div>
        </div>
        <div class="prob-box">
            <div class="prob-box-label">Leave</div>
            <div class="prob-box-val val-leave">{leave_prob:.0f}%</div>
        </div>""", unsafe_allow_html=True)

    with r3:
        st.markdown('<div class="chart-panel" style="padding:14px 16px;">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title" style="margin-bottom:4px;">Risk Gauge</div>', unsafe_allow_html=True)
        st.image(make_gauge(leave_prob), use_container_width=True)
        st.markdown(f"""
        <div class="risk-bar-bg" style="margin-top:10px;">
            <div class="{bar_cls}" style="width:{leave_prob:.0f}%"></div>
        </div>
        <div class="risk-status {st_cls}" style="margin-top:5px;">{st_txt}</div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 2: Probability bar + Risk factors side-by-side ────────────────────

    ch1, ch2 = st.columns([1, 2])

    with ch1:
        st.markdown('<div class="chart-panel" style="height:100%;">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Probability Split</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-sub">Stay vs Leave breakdown</div>', unsafe_allow_html=True)
        st.image(make_prob_bar(stay_prob, leave_prob), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with ch2:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Risk Factor Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-sub">Normalised risk contribution per factor (0 = no risk, 1 = maximum risk)</div>', unsafe_allow_html=True)
        chart_input = {
            "OverTime":                overtime,
            "JobSatisfaction":         job_satisfaction,
            "WorkLifeBalance":         work_life_balance,
            "YearsSinceLastPromotion": years_since_promo,
            "MonthlyIncome":           monthly_income,
            "EnvironmentSatisfaction": environment_satisfaction,
            "DistanceFromHome":        distance_home,
            "JobInvolvement":          job_involvement,
        }
        st.image(make_risk_factors_chart(chart_input), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 3: HR Action Items ─────────────────────────────────────────────────

    recs = []
    if overtime == "Yes":
        recs.append(("warn", "! Overtime Detected",
                     "Redistribute workload or consider additional headcount to prevent burnout."))
    if work_life_balance <= 2:
        recs.append(("warn", "! Poor Work-Life Balance",
                     "Explore flexible or hybrid arrangements to improve retention odds."))
    if job_satisfaction <= 2:
        recs.append(("warn", "! Low Job Satisfaction",
                     "Schedule a structured career conversation and role review."))
    if years_since_promo > 3:
        recs.append(("info", "i  Stalled Progression",
                     "No promotion in 3+ years — assess eligibility and create a growth plan."))
    if environment_satisfaction <= 2:
        recs.append(("info", "i  Environment Concerns",
                     "Investigate team dynamics and physical/remote workplace conditions."))
    if monthly_income < 3000:
        recs.append(("info", "i  Below-Market Salary",
                     "Benchmark compensation against current industry standards."))

    if recs:
        st.markdown("""
        <div class="section-hdr" style="margin-top:20px;">
            <div>
                <div class="section-title">HR Action Items</div>
                <div class="section-sub">Flags raised based on submitted profile</div>
            </div>
        </div>
        <div class="section-divider"></div>
        <div class="rec-grid">
        """, unsafe_allow_html=True)
        for rtype, title, body in recs:
            cls = "rec-warn" if rtype == "warn" else "rec-info"
            st.markdown(f'<div class="rec-card {cls}"><strong>{title}</strong>{body}</div>',
                        unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


st.markdown(
    '<div class="footer">AttritionAI &nbsp;·&nbsp; AIML Internship Capstone 2026 &nbsp;·&nbsp; IBM HR Analytics Dataset</div>',
    unsafe_allow_html=True
)
