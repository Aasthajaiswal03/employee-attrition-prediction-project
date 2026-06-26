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
    page_title="AttritionAI — HR Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* ═══════════════════════════════════════════
   BASE
═══════════════════════════════════════════ */
.stApp { background: #070B14; }

/* Remove default streamlit padding */
.block-container { padding: 1.2rem 1.8rem 2rem !important; max-width: 100% !important; }

/* ═══════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0C1022 0%, #080C18 100%) !important;
    border-right: 1px solid rgba(99,102,241,0.15) !important;
    width: 270px !important;
}
section[data-testid="stSidebar"] > div { padding: 1rem 1rem 1rem !important; }

.sb-brand {
    display: flex; align-items: center; gap: 10px;
    padding: 4px 0 18px; border-bottom: 1px solid rgba(99,102,241,0.2);
    margin-bottom: 18px;
}
.sb-brand-icon {
    width: 38px; height: 38px; flex-shrink: 0;
    background: linear-gradient(135deg, #4F46E5, #7C3AED);
    border-radius: 10px; display: flex; align-items: center; justify-content: center;
    font-size: 16px; box-shadow: 0 0 18px rgba(79,70,229,0.45);
}
.sb-brand-name { font-size: 15px; font-weight: 800; color: #F1F5F9; letter-spacing: -0.4px; }
.sb-brand-sub  { font-size: 10px; color: #475569; margin-top: 1px; }

.sb-label {
    font-size: 9px; font-weight: 700; color: #4F46E5;
    text-transform: uppercase; letter-spacing: 2px;
    margin: 0 0 10px; padding: 0;
}
.sb-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px; padding: 12px 14px; margin-bottom: 8px;
    transition: all 0.25s;
}
.sb-card:hover { border-color: rgba(99,102,241,0.4); background: rgba(99,102,241,0.05); }
.sb-card-lbl  { font-size: 10px; color: #475569; margin-bottom: 3px; }
.sb-card-val  { font-size: 17px; font-weight: 700; color: #E2E8F0; line-height: 1.2; }
.sb-chip {
    display: inline-block; font-size: 9px; font-weight: 600;
    padding: 2px 9px; border-radius: 99px; margin-top: 5px;
}
.chip-g { background: rgba(16,185,129,0.15); color: #34D399; border: 1px solid rgba(16,185,129,0.25); }
.chip-r { background: rgba(239,68,68,0.12);  color: #F87171; border: 1px solid rgba(239,68,68,0.25); }

/* ═══════════════════════════════════════════
   HERO HEADER
═══════════════════════════════════════════ */
.hero {
    background: linear-gradient(135deg, #0F1628 0%, #0C1022 50%, #111827 100%);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 20px; padding: 28px 32px;
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 18px; position: relative; overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-left { display: flex; align-items: center; gap: 18px; }
.hero-icon {
    width: 52px; height: 52px; flex-shrink: 0;
    background: linear-gradient(135deg, #4F46E5, #7C3AED);
    border-radius: 14px; display: flex; align-items: center; justify-content: center;
    font-size: 22px; box-shadow: 0 0 24px rgba(79,70,229,0.5);
}
.hero-title { font-size: 22px; font-weight: 800; color: #F8FAFC; letter-spacing: -0.6px; line-height: 1; }
.hero-sub   { font-size: 12px; color: #64748B; margin-top: 6px; }
.hero-badges { display: flex; flex-direction: column; align-items: flex-end; gap: 7px; }
.hero-badge {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.09);
    color: #94A3B8; font-size: 10px; font-weight: 600;
    padding: 5px 14px; border-radius: 99px;
    transition: all 0.2s;
}
.hero-badge:hover { border-color: rgba(99,102,241,0.5); color: #C7D2FE; }

/* ═══════════════════════════════════════════
   KPI STRIP
═══════════════════════════════════════════ */
.kpi-strip {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 12px; margin-bottom: 22px;
}
.kpi {
    background: linear-gradient(135deg, #0F1628 0%, #0D1220 100%);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 18px 20px;
    position: relative; overflow: hidden;
    transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
    cursor: default;
}
.kpi:hover {
    transform: translateY(-4px);
    border-color: rgba(99,102,241,0.4);
    box-shadow: 0 12px 32px rgba(79,70,229,0.15);
}
.kpi::after {
    content: ''; position: absolute;
    bottom: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #4F46E5, transparent);
    border-radius: 0 0 16px 16px;
}
.kpi-lbl  { font-size: 9px; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 1.4px; margin-bottom: 8px; }
.kpi-num  { font-size: 30px; font-weight: 800; color: #F1F5F9; letter-spacing: -1px; line-height: 1; }
.kpi-sub  { font-size: 10px; font-weight: 600; margin-top: 6px; }
.kpi-sub.g { color: #34D399; }
.kpi-sub.r { color: #F87171; }
.kpi-sub.n { color: #64748B; }

/* ═══════════════════════════════════════════
   SECTION HEADERS
═══════════════════════════════════════════ */
.sec-hdr { margin-bottom: 14px; }
.sec-title { font-size: 14px; font-weight: 700; color: #F1F5F9; letter-spacing: -0.3px; }
.sec-sub   { font-size: 11px; color: #475569; margin-top: 3px; }
.sec-line  {
    height: 1px; margin: 8px 0 16px;
    background: linear-gradient(90deg, rgba(99,102,241,0.6) 0%, rgba(99,102,241,0.1) 40%, transparent 100%);
}

/* ═══════════════════════════════════════════
   FORM PANELS
═══════════════════════════════════════════ */
.fpanel {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 18px 18px 20px;
    height: 100%;
    transition: border-color 0.25s;
}
.fpanel:hover { border-color: rgba(99,102,241,0.25); }
.fpanel-hdr {
    display: flex; align-items: center; gap: 8px;
    font-size: 9px; font-weight: 700; color: #6366F1;
    text-transform: uppercase; letter-spacing: 1.8px;
    padding-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 14px;
}
.fpanel-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: #4F46E5; flex-shrink: 0;
    box-shadow: 0 0 8px rgba(79,70,229,0.8);
}

/* ═══════════════════════════════════════════
   STREAMLIT WIDGET OVERRIDES
═══════════════════════════════════════════ */
label, .stSelectbox label, .stSlider label, .stNumberInput label {
    font-size: 10px !important; font-weight: 600 !important;
    color: #64748B !important; text-transform: uppercase !important;
    letter-spacing: 0.9px !important; margin-bottom: 4px !important;
}
div[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important; color: #E2E8F0 !important;
    transition: border-color 0.2s !important;
}
div[data-baseweb="select"] > div:hover,
div[data-baseweb="select"] > div:focus-within {
    border-color: rgba(99,102,241,0.6) !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.12) !important;
}
div[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important; color: #E2E8F0 !important;
}
div[data-testid="stNumberInput"] input:focus {
    border-color: rgba(99,102,241,0.6) !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.12) !important;
}

/* Slider track */
div[data-testid="stSlider"] > div > div > div {
    background: rgba(255,255,255,0.08) !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #4F46E5 !important;
    border: 2px solid #6366F1 !important;
    box-shadow: 0 0 10px rgba(79,70,229,0.6) !important;
}

/* Slider value label */
div[data-testid="stSlider"] > div > div > div > div { color: #94A3B8 !important; font-size: 11px !important; }

/* ═══════════════════════════════════════════
   EXPANDER — MODERN STREAMLIT
═══════════════════════════════════════════ */
div[data-testid="stExpander"] > details > summary {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    padding: 13px 18px !important;
    transition: border-color 0.2s, background 0.2s !important;
}
div[data-testid="stExpander"] > details > summary:hover {
    border-color: rgba(99,102,241,0.4) !important;
    background: rgba(99,102,241,0.06) !important;
}
div[data-testid="stExpander"] > details[open] > summary {
    border-radius: 12px 12px 0 0 !important;
    border-color: rgba(99,102,241,0.3) !important;
}
div[data-testid="stExpander"] summary p {
    font-size: 11px !important; font-weight: 600 !important;
    color: #94A3B8 !important; text-transform: uppercase !important;
    letter-spacing: 0.9px !important; margin: 0 !important;
}
div[data-testid="stExpander"] summary svg { display: none !important; }
div[data-testid="stExpander"] summary [data-testid="stExpanderToggleIcon"] { display: none !important; }
div[data-testid="stExpanderDetails"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
    padding: 18px !important;
}
/* Legacy */
.streamlit-expanderHeader { display: none !important; }
details > summary svg { display: none !important; }
details > summary::marker { display: none !important; content: '' !important; }
details > summary::-webkit-details-marker { display: none !important; }
[data-testid="stExpanderToggleIcon"] { display: none !important; }

/* ═══════════════════════════════════════════
   PREDICT BUTTON
═══════════════════════════════════════════ */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
    color: #fff !important; border: none !important;
    border-radius: 14px !important; padding: 15px 32px !important;
    font-size: 14px !important; font-weight: 700 !important;
    letter-spacing: 0.3px !important; width: 100% !important;
    box-shadow: 0 6px 24px rgba(79,70,229,0.4) !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 32px rgba(79,70,229,0.55) !important;
}
div[data-testid="stButton"] > button:active { transform: translateY(0) !important; }

/* ═══════════════════════════════════════════
   RESULTS
═══════════════════════════════════════════ */
.result-card {
    border-radius: 18px; padding: 26px 28px; margin-bottom: 0;
    position: relative; overflow: hidden;
}
.result-stay  {
    background: linear-gradient(135deg, #062219 0%, #041A13 100%);
    border: 1px solid rgba(16,185,129,0.3);
    box-shadow: 0 0 40px rgba(16,185,129,0.08);
}
.result-leave {
    background: linear-gradient(135deg, #1C0A0A 0%, #130606 100%);
    border: 1px solid rgba(239,68,68,0.3);
    box-shadow: 0 0 40px rgba(239,68,68,0.08);
}
.result-glow-stay  { position:absolute; top:-40px; right:-40px; width:160px; height:160px; background:radial-gradient(circle, rgba(16,185,129,0.15) 0%, transparent 70%); }
.result-glow-leave { position:absolute; top:-40px; right:-40px; width:160px; height:160px; background:radial-gradient(circle, rgba(239,68,68,0.15) 0%, transparent 70%); }
.result-eyebrow { font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: 2.5px; margin-bottom: 8px; }
.eyebrow-stay  { color: #34D399; }
.eyebrow-leave { color: #F87171; }
.result-heading { font-size: 24px; font-weight: 800; color: #F8FAFC; letter-spacing: -0.6px; margin-bottom: 8px; }
.result-desc { font-size: 12px; color: #64748B; line-height: 1.8; margin-bottom: 18px; }
.result-pill {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 13px; font-weight: 700; padding: 7px 20px;
    border-radius: 99px;
}
.pill-stay  { background: rgba(16,185,129,0.15); color: #34D399; border: 1px solid rgba(16,185,129,0.35); }
.pill-leave { background: rgba(239,68,68,0.12);  color: #F87171; border: 1px solid rgba(239,68,68,0.35); }

/* Probability boxes */
.prob-stack { display: flex; flex-direction: column; gap: 10px; height: 100%; }
.prob-box {
    flex: 1; background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 16px 14px; text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.prob-box:hover { transform: scale(1.03); }
.prob-lbl { font-size: 9px; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 8px; }
.prob-val { font-size: 28px; font-weight: 800; letter-spacing: -1px; }
.pv-stay  { color: #34D399; }
.pv-leave { color: #F87171; }

/* Risk bar */
.risk-wrap { margin-top: 12px; }
.risk-bar-bg { height: 8px; background: rgba(255,255,255,0.07); border-radius: 6px; overflow: hidden; margin-bottom: 8px; }
.rbl { height: 8px; border-radius: 6px; background: linear-gradient(90deg, #059669, #34D399); }
.rbm { height: 8px; border-radius: 6px; background: linear-gradient(90deg, #D97706, #FBBF24); }
.rbh { height: 8px; border-radius: 6px; background: linear-gradient(90deg, #DC2626, #F87171); }
.risk-txt { font-size: 10px; font-weight: 600; }
.rl { color: #34D399; } .rm { color: #FBBF24; } .rh { color: #F87171; }

/* Chart panel */
.chart-panel {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 18px 20px;
    transition: border-color 0.2s;
}
.chart-panel:hover { border-color: rgba(99,102,241,0.2); }
.chart-title { font-size: 11px; font-weight: 700; color: #E2E8F0; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 3px; }
.chart-sub   { font-size: 10px; color: #475569; margin-bottom: 14px; }

/* Action cards */
.action-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 4px; }
.action-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 14px 16px;
    font-size: 11px; color: #94A3B8; line-height: 1.7;
    transition: transform 0.2s, border-color 0.2s;
}
.action-card:hover { transform: translateY(-2px); }
.action-card strong { display: block; margin-bottom: 4px; font-size: 11px; font-weight: 700; }
.ac-warn { border-top: 2px solid #F59E0B; }
.ac-warn strong { color: #FBBF24; }
.ac-info { border-top: 2px solid #6366F1; }
.ac-info strong { color: #818CF8; }

/* Hide stray icon text */
button[data-testid="collapsedControl"] span { display: none !important; }
span[translate="no"] { font-size: 0 !important; width: 0 !important; overflow: hidden !important; }

/* Misc */
hr { border-color: rgba(255,255,255,0.07) !important; margin: 18px 0 !important; }
.footer {
    text-align: center; font-size: 10px; color: #1E293B;
    padding: 24px 0 4px; letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  CHART HELPERS
# ═══════════════════════════════════════════════════════════════
BG    = "#070B14"
BG2   = "#0D1220"
BD    = "rgba(255,255,255,0.08)"
TXT   = "#E2E8F0"
MUTE  = "#475569"
ACC   = "#4F46E5"
GRN   = "#10B981"
RED   = "#EF4444"
AMB   = "#F59E0B"

def fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf

def make_gauge(pct: float):
    fig, ax = plt.subplots(figsize=(4.4, 2.6), facecolor=BG2)
    ax.set_facecolor(BG2); ax.set_xlim(-1.3,1.3); ax.set_ylim(-0.3,1.15)
    ax.set_aspect("equal"); ax.axis("off")
    r = 1.0
    # Track
    th_bg = np.linspace(np.pi, 0, 300)
    ax.plot(r*np.cos(th_bg), r*np.sin(th_bg), color="#1E293B", linewidth=18, solid_capstyle="round", zorder=1)
    # Zone colours
    for t1, t2, col in [(np.pi, np.pi*0.6, GRN), (np.pi*0.6, np.pi*0.3, AMB), (np.pi*0.3, 0, RED)]:
        th = np.linspace(t1, t2, 100)
        ax.plot(r*np.cos(th), r*np.sin(th), color=col, linewidth=18, solid_capstyle="butt", alpha=0.18, zorder=2)
    # Fill arc
    fill_end = np.pi - (pct/100)*np.pi
    th_fill  = np.linspace(np.pi, fill_end, 200)
    fc = GRN if pct < 40 else (AMB if pct < 70 else RED)
    ax.plot(r*np.cos(th_fill), r*np.sin(th_fill), color=fc, linewidth=18, solid_capstyle="round", alpha=0.92, zorder=3)
    # Needle
    ang = np.pi - (pct/100)*np.pi
    ax.annotate("", xy=(0.76*np.cos(ang), 0.76*np.sin(ang)), xytext=(0,0),
                arrowprops=dict(arrowstyle="-|>", color=TXT, lw=2, mutation_scale=14))
    ax.plot(0, 0, "o", color=TXT, markersize=8, zorder=6)
    # Labels
    ax.text(0, -0.14, f"{pct:.1f}%", ha="center", va="center", fontsize=20, fontweight="800", color=fc, fontfamily="monospace")
    ax.text(0, -0.30, "Attrition Risk", ha="center", va="center", fontsize=8, color=MUTE)
    ax.text(-1.18, -0.14, "0%",   ha="center", fontsize=7, color=MUTE)
    ax.text( 1.18, -0.14, "100%", ha="center", fontsize=7, color=MUTE)
    ax.text(-0.80, 0.70, "LOW",  ha="center", fontsize=6.5, color=GRN, alpha=0.75)
    ax.text( 0.00, 1.05, "MED",  ha="center", fontsize=6.5, color=AMB, alpha=0.75)
    ax.text( 0.80, 0.70, "HIGH", ha="center", fontsize=6.5, color=RED, alpha=0.75)
    fig.tight_layout(pad=0.2)
    return fig_to_img(fig)

def make_prob_donut(stay: float, leave: float):
    fig, ax = plt.subplots(figsize=(3.6, 3.6), facecolor=BG2)
    ax.set_facecolor(BG2)
    sizes  = [stay, leave]
    colors = [GRN, RED]
    wedge_props = dict(width=0.42, edgecolor=BG2, linewidth=2.5)
    wedges, _ = ax.pie(sizes, colors=colors, wedgeprops=wedge_props, startangle=90)
    ax.text(0, 0.08, f"{leave:.0f}%", ha="center", va="center", fontsize=22, fontweight="800", color=RED, fontfamily="monospace")
    ax.text(0, -0.22, "leave risk", ha="center", va="center", fontsize=8, color=MUTE)
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=GRN, label=f"Stay  {stay:.1f}%"),
                       Patch(facecolor=RED, label=f"Leave {leave:.1f}%")]
    ax.legend(handles=legend_elements, loc="lower center", fontsize=8,
              facecolor=BG2, edgecolor="#1E293B", labelcolor=MUTE,
              framealpha=1, handlelength=1, ncol=2, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(pad=0.4)
    return fig_to_img(fig)

def make_risk_chart(inputs: dict):
    raw = {
        "Overtime":             1.0 if inputs.get("OverTime") == "Yes" else 0.0,
        "Job Satisfaction":     (5 - inputs.get("JobSatisfaction", 3)) / 4,
        "Work-Life Balance":    (5 - inputs.get("WorkLifeBalance", 3)) / 4,
        "Promotion Gap":        min(inputs.get("YearsSinceLastPromotion", 1) / 10, 1.0),
        "Monthly Income":       max(0, 1 - inputs.get("MonthlyIncome", 5000) / 20000),
        "Environment":          (5 - inputs.get("EnvironmentSatisfaction", 3)) / 4,
        "Distance from Home":   min(inputs.get("DistanceFromHome", 9) / 29, 1.0),
        "Job Involvement":      (5 - inputs.get("JobInvolvement", 3)) / 4,
    }
    paired = sorted(raw.items(), key=lambda x: x[1], reverse=True)
    labels = [p[0] for p in paired]
    scores = [p[1] for p in paired]
    colors = [RED if s > 0.65 else (AMB if s > 0.35 else GRN) for s in scores]

    fig, ax = plt.subplots(figsize=(6.8, 3.8), facecolor=BG)
    ax.set_facecolor(BG)
    y = np.arange(len(labels))
    bars = ax.barh(y, scores, color=colors, height=0.5, edgecolor="none", alpha=0.88)
    # Subtle bg bars
    ax.barh(y, [1]*len(labels), color="#0F1628", height=0.5, edgecolor="none", zorder=0)
    ax.barh(y, scores, color=colors, height=0.5, edgecolor="none", alpha=0.88, zorder=1)

    for x in [0.25, 0.5, 0.75, 1.0]:
        ax.axvline(x, color="#1E293B", linewidth=0.8, zorder=0)
    for bar, s in zip(bars, scores):
        ax.text(min(s + 0.03, 1.12), bar.get_y() + bar.get_height()/2,
                f"{s:.2f}", va="center", fontsize=8.5, color=MUTE)

    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9.5, color=TXT)
    ax.set_xlim(0, 1.25)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", ".25", ".50", ".75", "1.0"], fontsize=8, color=MUTE)
    ax.tick_params(length=0); ax.spines[:].set_visible(False)
    handles = [mpatches.Patch(color=RED, label=">0.65 High"),
               mpatches.Patch(color=AMB, label="0.35–0.65 Med"),
               mpatches.Patch(color=GRN, label="<0.35 Low")]
    ax.legend(handles=handles, loc="lower right", fontsize=7.5,
              facecolor=BG2, edgecolor="#1E293B", labelcolor=MUTE, framealpha=1, handlelength=1.1)
    fig.tight_layout(pad=0.6)
    return fig_to_img(fig)

def make_sidebar_chart():
    drivers = ["No Promotion 3+ yrs","Poor Work-Life Bal.","Low Environment Sat.",
               "Low Job Satisfaction","Single (Marital)","Frequent Travel",
               "Sales Representative","OverTime = Yes"]
    rates   = [18.6, 31.2, 25.0, 22.8, 25.5, 24.9, 39.8, 30.5]
    base    = 16.1
    colors  = [RED if v > 25 else AMB for v in rates]

    fig, ax = plt.subplots(figsize=(3.6, 3.8), facecolor=BG2)
    ax.set_facecolor(BG2)
    y = np.arange(len(drivers))
    ax.barh(y, rates, color=colors, height=0.5, edgecolor="none", alpha=0.85)
    ax.axvline(base, color=MUTE, linewidth=1.2, linestyle="--", alpha=0.6, label=f"Avg {base}%")
    for i, v in enumerate(rates):
        ax.text(v+0.5, i, f"{v}%", va="center", fontsize=7.5, color=MUTE)
    ax.set_yticks(y); ax.set_yticklabels(drivers, fontsize=7, color=TXT)
    ax.set_xlim(0, 52); ax.set_xlabel("Attrition %", fontsize=7.5, color=MUTE)
    ax.tick_params(axis="x", colors=MUTE, labelsize=7); ax.tick_params(axis="y", length=0)
    for sp in ["top","right","left"]: ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#1E293B")
    ax.legend(fontsize=7, facecolor=BG2, edgecolor="#1E293B", labelcolor=MUTE, framealpha=1, loc="lower right")
    ax.set_title("Attrition Rate by Risk Driver", fontsize=8, color=MUTE, pad=8, loc="left")
    fig.tight_layout(pad=0.6)
    return fig_to_img(fig)


# ═══════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    try:
        with open('Model/model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("❌ model.pkl not found in Model/ folder.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

pkg = load_model()
if pkg is None:
    st.stop()

model          = pkg.get('model')
scaler         = pkg.get('scaler')
needs_scaling  = pkg.get('needs_scaling', False)
label_encoders = pkg.get('label_encoders', {})
feature_names  = pkg.get('feature_names', [])
model_name     = pkg.get('model_name', 'ML Model')

if model is None:
    st.error("Model key not found in pkl. Check notebook pkl export.")
    st.stop()


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-brand-icon">🧠</div>
        <div>
            <div class="sb-brand-name">AttritionAI</div>
            <div class="sb-brand-sub">HR Risk Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-label">Model Info</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sb-card">
        <div class="sb-card-lbl">Active Model</div>
        <div class="sb-card-val">XGBoost</div>
        <span class="sb-chip chip-g">Best F1 Score</span>
    </div>
    <div class="sb-card">
        <div class="sb-card-lbl">Dataset</div>
        <div class="sb-card-val" style="font-size:14px;">IBM HR Analytics</div>
        <span class="sb-chip chip-g">1,470 records</span>
    </div>
    <div class="sb-card" style="margin-bottom:18px;">
        <div class="sb-card-lbl">Dataset Attrition Rate</div>
        <div class="sb-card-val">16.1%</div>
        <span class="sb-chip chip-r">237 of 1,470 left</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-label" style="margin-bottom:10px;">Dataset Overview</div>', unsafe_allow_html=True)
    st.image(make_sidebar_chart(), use_container_width=True)

    st.markdown("""
    <div style="font-size:9px;color:#1E293B;text-align:center;padding-top:12px;letter-spacing:0.5px;">
        AIML CAPSTONE 2026
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  HERO
# ═══════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
    <div class="hero-left">
        <div class="hero-icon">🧠</div>
        <div>
            <div class="hero-title">Employee Attrition Prediction</div>
            <div class="hero-sub">Predict flight risk &nbsp;·&nbsp; Retain top talent &nbsp;·&nbsp; Act before attrition occurs</div>
        </div>
    </div>
    <div class="hero-badges">
        <span class="hero-badge">{model_name}</span>
        <span class="hero-badge">IBM HR Analytics &nbsp;·&nbsp; 2026</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  KPI STRIP
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="kpi-strip">
    <div class="kpi">
        <div class="kpi-lbl">Models Trained</div>
        <div class="kpi-num">3</div>
        <div class="kpi-sub n">LR &nbsp;·&nbsp; RF &nbsp;·&nbsp; XGBoost</div>
    </div>
    <div class="kpi">
        <div class="kpi-lbl">Best F1 Score</div>
        <div class="kpi-num">62%</div>
        <div class="kpi-sub g">XGBoost</div>
    </div>
    <div class="kpi">
        <div class="kpi-lbl">Dataset Attrition</div>
        <div class="kpi-num">16.1%</div>
        <div class="kpi-sub r">237 of 1,470 left</div>
    </div>
    <div class="kpi">
        <div class="kpi-lbl">Features Used</div>
        <div class="kpi-num">30</div>
        <div class="kpi-sub n">HR profile inputs</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  INPUT FORM
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="sec-hdr">
    <div class="sec-title">Employee Profile Input</div>
    <div class="sec-sub">Fill in the employee details below, then run the analysis</div>
</div>
<div class="sec-line"></div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown('<div class="fpanel"><div class="fpanel-hdr"><div class="fpanel-dot"></div>Personal</div>', unsafe_allow_html=True)
    age             = st.slider("Age", 18, 60, 35)
    gender          = st.selectbox("Gender", ["Male", "Female"])
    marital_status  = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    education       = st.selectbox("Education Level", [1,2,3,4,5],
                        format_func=lambda x:{1:"Below College",2:"College",3:"Bachelor",4:"Master",5:"Doctor"}[x])
    education_field = st.selectbox("Education Field",
                        ["Life Sciences","Medical","Marketing","Technical Degree","Human Resources","Other"])
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="fpanel"><div class="fpanel-hdr"><div class="fpanel-dot"></div>Job Details</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="fpanel"><div class="fpanel-hdr"><div class="fpanel-dot"></div>Compensation & Tenure</div>', unsafe_allow_html=True)
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

# Defaults for optional fields
distance_home = 9; environment_satisfaction = 3; job_satisfaction = 3
relationship_satisfaction = 3; work_life_balance = 3; performance_rating = 3; monthly_rate = 14000

with st.expander("⚙️  Satisfaction & Performance Details (optional)"):
    c1, c2 = st.columns(2, gap="medium")
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

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
predict_clicked = st.button("🔍  Run Attrition Analysis", use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  PREDICTION & RESULTS
# ═══════════════════════════════════════════════════════════════
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

    if leave_prob < 30:   rb_cls, rt_cls, rt_txt = "rbl", "rl", "Low risk — No immediate action required"
    elif leave_prob < 60: rb_cls, rt_cls, rt_txt = "rbm", "rm", "Medium risk — Schedule a 1:1 check-in"
    else:                 rb_cls, rt_cls, rt_txt = "rbh", "rh", "High risk — Immediate HR intervention needed"

    st.markdown("""
    <div class="sec-hdr" style="margin-top:24px;">
        <div class="sec-title">Analysis Result</div>
        <div class="sec-sub">Based on the submitted employee profile</div>
    </div>
    <div class="sec-line"></div>
    """, unsafe_allow_html=True)

    # Row 1: result card | prob boxes | gauge
    r1, r2, r3 = st.columns([4, 1, 2], gap="medium")

    with r1:
        if prediction == 0:
            st.markdown(f"""
            <div class="result-card result-stay">
                <div class="result-glow-stay"></div>
                <div class="result-eyebrow eyebrow-stay">✦ Prediction — Low Risk</div>
                <div class="result-heading">Employee likely to stay</div>
                <div class="result-desc">No significant attrition signals detected. Profile closely matches retained employee patterns observed in the training dataset.</div>
                <span class="result-pill pill-stay">✓ &nbsp;{stay_prob:.1f}% confidence</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-leave">
                <div class="result-glow-leave"></div>
                <div class="result-eyebrow eyebrow-leave">⚠ Prediction — High Risk</div>
                <div class="result-heading">Employee at risk of leaving</div>
                <div class="result-desc">Multiple attrition signals detected. Immediate HR review and targeted retention intervention is strongly recommended.</div>
                <span class="result-pill pill-leave">⚠ &nbsp;{leave_prob:.1f}% confidence</span>
            </div>""", unsafe_allow_html=True)

    with r2:
        st.markdown(f"""
        <div class="prob-stack">
            <div class="prob-box">
                <div class="prob-lbl">Stay</div>
                <div class="prob-val pv-stay">{stay_prob:.0f}%</div>
            </div>
            <div class="prob-box">
                <div class="prob-lbl">Leave</div>
                <div class="prob-val pv-leave">{leave_prob:.0f}%</div>
            </div>
        </div>""", unsafe_allow_html=True)

    with r3:
        st.markdown('<div class="chart-panel" style="padding:14px 16px;">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title" style="margin-bottom:4px;">Risk Gauge</div>', unsafe_allow_html=True)
        st.image(make_gauge(leave_prob), use_container_width=True)
        st.markdown(f"""
        <div class="risk-wrap">
            <div class="risk-bar-bg"><div class="{rb_cls}" style="width:{leave_prob:.0f}%"></div></div>
            <div class="risk-txt {rt_cls}">{rt_txt}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 2: Donut + Risk factors
    ch1, ch2 = st.columns([1, 2], gap="medium")

    with ch1:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Probability Split</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-sub">Stay vs Leave breakdown</div>', unsafe_allow_html=True)
        st.image(make_prob_donut(stay_prob, leave_prob), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with ch2:
        st.markdown('<div class="chart-panel">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Risk Factor Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-sub">Normalised risk contribution per factor (0 = no risk, 1 = max risk)</div>', unsafe_allow_html=True)
        st.image(make_risk_chart({
            "OverTime": overtime, "JobSatisfaction": job_satisfaction,
            "WorkLifeBalance": work_life_balance, "YearsSinceLastPromotion": years_since_promo,
            "MonthlyIncome": monthly_income, "EnvironmentSatisfaction": environment_satisfaction,
            "DistanceFromHome": distance_home, "JobInvolvement": job_involvement,
        }), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 3: HR Action Items
    recs = []
    if overtime == "Yes":
        recs.append(("warn", "⚡ Overtime Detected", "Redistribute workload or add headcount to prevent burnout."))
    if work_life_balance <= 2:
        recs.append(("warn", "⚖ Poor Work-Life Balance", "Explore flexible or hybrid arrangements to improve retention odds."))
    if job_satisfaction <= 2:
        recs.append(("warn", "😟 Low Job Satisfaction", "Schedule a structured career conversation and role review."))
    if years_since_promo > 3:
        recs.append(("info", "📈 Stalled Progression", "No promotion in 3+ years — assess eligibility and create a growth plan."))
    if environment_satisfaction <= 2:
        recs.append(("info", "🏢 Environment Concerns", "Investigate team dynamics and workplace conditions."))
    if monthly_income < 3000:
        recs.append(("info", "💰 Below-Market Salary", "Benchmark compensation against current industry standards."))

    if recs:
        st.markdown("""
        <div class="sec-hdr" style="margin-top:22px;">
            <div class="sec-title">HR Action Items</div>
            <div class="sec-sub">Flags raised based on submitted profile</div>
        </div>
        <div class="sec-line"></div>
        <div class="action-grid">
        """, unsafe_allow_html=True)
        for rtype, title, body in recs:
            cls = "ac-warn" if rtype == "warn" else "ac-info"
            st.markdown(f'<div class="action-card {cls}"><strong>{title}</strong>{body}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    AttritionAI &nbsp;·&nbsp; AIML Internship Capstone 2026 &nbsp;·&nbsp; IBM HR Analytics Dataset
</div>
""", unsafe_allow_html=True)
