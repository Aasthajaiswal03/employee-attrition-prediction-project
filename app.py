import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="AttritionAI — HR Risk Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── Page bg ── */
.stApp { background: #0B0F19; }
section[data-testid="stSidebar"] { background: #0F1420 !important; border-right: 1px solid #1E2433 !important; }

/* ── Sidebar content ── */
.sb-logo { display:flex; align-items:center; gap:10px; padding:8px 0 20px; border-bottom:1px solid #1E2433; margin-bottom:20px; }
.sb-logo-icon { width:34px;height:34px;background:#4F46E5;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:16px; }
.sb-logo-text { font-size:14px;font-weight:600;color:#F1F5F9;letter-spacing:-0.3px; }
.sb-logo-sub { font-size:10px;color:#64748B;margin-top:1px; }

.sb-section { margin-bottom:24px; }
.sb-section-title { font-size:9px;font-weight:600;color:#4F46E5;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:10px; }
.sb-item { display:flex;align-items:center;gap:10px;padding:8px 10px;border-radius:8px;margin-bottom:2px;cursor:pointer;color:#94A3B8;font-size:12px;font-weight:500; }
.sb-item.active { background:#1E2433;color:#F1F5F9; }
.sb-item:hover { background:#161B2A; }
.sb-dot { width:6px;height:6px;border-radius:50%;background:#4F46E5;flex-shrink:0; }

.sb-stat { background:#131827;border:1px solid #1E2433;border-radius:10px;padding:12px;margin-bottom:8px; }
.sb-stat-label { font-size:10px;color:#64748B;margin-bottom:4px; }
.sb-stat-val { font-size:20px;font-weight:700;color:#F1F5F9; }
.sb-stat-chip { display:inline-block;font-size:9px;font-weight:600;padding:2px 7px;border-radius:99px;margin-top:4px; }
.chip-green { background:#0D3321;color:#34D399; }
.chip-red { background:#3B0D0D;color:#F87171; }

/* ── Top bar ── */
.topbar { background:#0F1420;border:1px solid #1E2433;border-radius:14px;padding:22px 28px;display:flex;align-items:center;justify-content:space-between;margin-bottom:20px; }
.topbar-logo-wrap { display:flex;align-items:center;gap:16px; }
.topbar-logo-icon { display:none; }
.topbar-logo-divider { display:none; }
.topbar-bar-accent { width:4px;height:44px;background:#4F46E5;border-radius:2px;flex-shrink:0; }
.topbar-title { font-size:20px;font-weight:700;color:#F1F5F9;letter-spacing:-0.5px;line-height:1; }
.topbar-tagline { font-size:11px;color:#64748B;margin-top:5px;letter-spacing:0.2px; }
.topbar-badge { background:#1E2433;border:1px solid #2D3748;color:#94A3B8;font-size:10px;font-weight:600;padding:4px 12px;border-radius:99px; }
.topbar-right { display:flex;flex-direction:column;align-items:flex-end;gap:6px; }

/* ── KPI cards row ── */
.kpi-grid { display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px; }
.kpi-card { background:#0F1420;border:1px solid #1E2433;border-radius:12px;padding:18px 20px; }
.kpi-label { font-size:10px;font-weight:600;color:#64748B;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;display:flex;align-items:center;gap:6px; }
.kpi-icon { width:22px;height:22px;border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:11px; }
.kpi-val { font-size:26px;font-weight:700;color:#F1F5F9;letter-spacing:-0.5px;line-height:1; }
.kpi-change { font-size:10px;font-weight:600;margin-top:6px; }
.kpi-change.up { color:#34D399; }
.kpi-change.down { color:#F87171; }
.kpi-change.neutral { color:#94A3B8; }

/* ── Section headers ── */
.section-hdr { display:flex;align-items:center;justify-content:space-between;margin-bottom:14px; }
.section-title { font-size:12px;font-weight:600;color:#F1F5F9;letter-spacing:-0.2px; }
.section-sub { font-size:10px;color:#64748B;margin-top:2px; }

/* ── Form panel ── */
.form-panel { background:#0F1420;border:1px solid #1E2433;border-radius:14px;padding:20px;margin-bottom:16px; }
.panel-title { font-size:10px;font-weight:600;color:#4F46E5;text-transform:uppercase;letter-spacing:1.2px;padding-bottom:12px;border-bottom:1px solid #1E2433;margin-bottom:16px;display:flex;align-items:center;gap:8px; }

/* ── Fields ── */
div[data-testid="stSlider"] > div { background: transparent !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] { background:#4F46E5 !important;border:none !important; }
.stSlider [data-baseweb="slider"] div[data-testid="stThumbValue"] { color:#F1F5F9 !important;background:#1E2433 !important; }

label, .stSelectbox label, .stSlider label, .stNumberInput label {
    font-size:11px !important;font-weight:500 !important;color:#94A3B8 !important;text-transform:uppercase;letter-spacing:0.8px !important;
}
.stSelectbox > div > div, .stNumberInput > div > div > input {
    background:#131827 !important;border:1px solid #1E2433 !important;border-radius:8px !important;color:#F1F5F9 !important;
}
.stSelectbox > div > div:hover, .stNumberInput > div > div > input:focus {
    border-color:#4F46E5 !important;
}

/* ── Predict button ── */
.stButton > button {
    background:#4F46E5 !important;color:#fff !important;border:none !important;
    border-radius:10px !important;padding:12px 28px !important;
    font-size:13px !important;font-weight:600 !important;letter-spacing:0.3px !important;
    width:100% !important;
}
.stButton > button:hover { background:#4338CA !important; }

/* ── Result ── */
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

/* ── Risk bar ── */
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

/* ── Rec cards ── */
.rec-grid { display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:4px; }
.rec-card { background:#131827;border:1px solid #1E2433;border-radius:10px;padding:12px 14px;font-size:11px;color:#94A3B8;line-height:1.6; }
.rec-card strong { color:#F1F5F9;display:block;margin-bottom:3px;font-size:11px; }
.rec-warn { border-top:2px solid #F59E0B; }
.rec-info { border-top:2px solid #4F46E5; }

/* ── Expander ── */
details summary { color:#64748B !important;font-size:11px !important; }
.stExpander { background:#0F1420 !important;border:1px solid #1E2433 !important;border-radius:10px !important; }

/* ── Divider ── */
hr { border-color:#1E2433 !important;margin:20px 0 !important; }

/* ── Footer ── */
.footer { text-align:center;font-size:10px;color:#374151;padding:16px 0 4px; }

/* ── Metric override ── */
div[data-testid="stMetricValue"] { color:#F1F5F9 !important;font-size:22px !important;font-weight:700 !important; }
div[data-testid="stMetricLabel"] { color:#64748B !important;font-size:10px !important; }
</style>
""", unsafe_allow_html=True)


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
        <div class="sb-logo-icon" style="font-size:13px;font-weight:700;letter-spacing:-0.5px;">AI</div>
        <div>
            <div class="sb-logo-text">AttritionAI</div>
            <div class="sb-logo-sub">HR Risk Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sb-section">
        <div class="sb-section-title">Navigation</div>
        <div class="sb-item active"><div class="sb-dot"></div> Attrition Predictor</div>
        <div class="sb-item">&nbsp;Workforce Trends</div>
        <div class="sb-item">&nbsp;Employee Profiles</div>
        <div class="sb-item">&nbsp;Model Settings</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sb-section">
        <div class="sb-section-title">Model Info</div>
        <div class="sb-stat">
            <div class="sb-stat-label">Active Model</div>
            <div class="sb-stat-val" style="font-size:14px;margin-top:2px;">Random Forest</div>
            <span class="sb-stat-chip chip-green">▲ 94.2% Accuracy</span>
        </div>
        <div class="sb-stat">
            <div class="sb-stat-label">Dataset</div>
            <div class="sb-stat-val" style="font-size:13px;">IBM HR Analytics</div>
            <span class="sb-stat-chip chip-green">1,470 records</span>
        </div>
        <div class="sb-stat">
            <div class="sb-stat-label">Attrition Rate (dataset)</div>
            <div class="sb-stat-val">16.1%</div>
            <span class="sb-stat-chip chip-red">▼ High Risk Cohort</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:10px;color:#374151;text-align:center;padding-top:10px;">
        AIML Internship Capstone · 2026
    </div>
    """, unsafe_allow_html=True)


# ── TOP BAR ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="topbar">
    <div class="topbar-logo-wrap">
        <div class="topbar-bar-accent"></div>
        <div>
            <div class="topbar-title">AttritionAI</div>
            <div class="topbar-tagline">Predict employee flight risk · Retain top talent · Take action early</div>
        </div>
    </div>
    <div class="topbar-right">
        <span class="topbar-badge">{model_name}</span>
        <span class="topbar-badge">IBM HR Analytics · 2026</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── KPI ROW ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label"><div class="kpi-icon" style="background:#1E1B4B;"></div>Models Trained</div>
        <div class="kpi-val">3</div>
        <div class="kpi-change neutral">LR · RF · XGBoost</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label"><div class="kpi-icon" style="background:#0D3321;"></div>Best Accuracy</div>
        <div class="kpi-val">94.2%</div>
        <div class="kpi-change up">▲ Random Forest</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label"><div class="kpi-icon" style="background:#3B0D0D;"></div>Dataset Attrition</div>
        <div class="kpi-val">16.1%</div>
        <div class="kpi-change down">237 of 1,470 left</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label"><div class="kpi-icon" style="background:#1E1B4B;"></div>Features Used</div>
        <div class="kpi-val">30</div>
        <div class="kpi-change neutral">HR profile inputs</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── INPUT FORM ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr"><div><div class="section-title">Employee Profile Input</div><div class="section-sub">Enter details across three sections below</div></div></div>', unsafe_allow_html=True)

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

    st.markdown('<div class="section-title" style="color:#F1F5F9;font-size:13px;margin-bottom:14px;">Analysis Result</div>', unsafe_allow_html=True)

    r1, r2, r3 = st.columns([3, 1, 1])

    with r1:
        if prediction == 0:
            st.markdown(f"""
            <div class="result-wrap result-stay">
                <div class="result-eyebrow stay">Prediction · Low Risk</div>
                <div class="result-heading">Employee likely to stay</div>
                <div class="result-desc">No significant attrition signals detected. Profile matches retained employee patterns in the training data.</div>
                <span class="result-pill pill-stay">{stay_prob:.1f}% confidence</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-wrap result-leave">
                <div class="result-eyebrow leave">Prediction · High Risk</div>
                <div class="result-heading">Employee at risk of leaving</div>
                <div class="result-desc">Multiple attrition signals detected. Immediate HR review and retention action recommended.</div>
                <span class="result-pill pill-leave">{leave_prob:.1f}% confidence</span>
            </div>""", unsafe_allow_html=True)

    with r2:
        st.markdown(f"""
        <div class="prob-box">
            <div class="prob-box-label">Stay probability</div>
            <div class="prob-box-val val-stay">{stay_prob:.0f}%</div>
        </div>""", unsafe_allow_html=True)

    with r3:
        st.markdown(f"""
        <div class="prob-box">
            <div class="prob-box-label">Leave probability</div>
            <div class="prob-box-val val-leave">{leave_prob:.0f}%</div>
        </div>""", unsafe_allow_html=True)

    # Risk bar
    if leave_prob < 30:
        bar_class = "risk-bar-fill-low"; status_class = "risk-low"; status_text = "Low risk — No immediate action required"
    elif leave_prob < 60:
        bar_class = "risk-bar-fill-mid"; status_class = "risk-med"; status_text = "Medium risk — Schedule a 1:1 check-in"
    else:
        bar_class = "risk-bar-fill-high"; status_class = "risk-high"; status_text = "High risk — Immediate HR intervention needed"

    st.markdown(f"""
    <div class="risk-wrap">
        <div class="risk-label">Attrition Risk Level</div>
        <div class="risk-bar-bg"><div class="{bar_class}" style="width:{leave_prob:.0f}%"></div></div>
        <div class="risk-status {status_class}">{status_text}</div>
    </div>""", unsafe_allow_html=True)

    # Recommendations
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
            cls = "rec-warn" if rtype=="warn" else "rec-info"
            icon = "!" if rtype=="warn" else "i"
            st.markdown(f'<div class="rec-card {cls}"><strong>{icon} {title}</strong>{body}</div>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

st.markdown('<div class="footer">AttritionAI · AIML Internship Capstone 2026 · IBM HR Analytics Dataset</div>', unsafe_allow_html=True)
