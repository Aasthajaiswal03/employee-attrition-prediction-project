import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        padding: 1rem 0 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box-stay {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border: 2px solid #28a745;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }
    .prediction-box-leave {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border: 2px solid #dc3545;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        border-left: 4px solid #667eea;
        margin-bottom: 10px;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #333;
        border-bottom: 2px solid #667eea;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 12px;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
    }
    .stButton>button:hover {
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open('Model/model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("⚠️ model.pkl not found! Make sure it's inside the 'Model/' folder.")
        st.stop()

pkg = load_model()
model        = pkg['model']
scaler       = pkg.get('scaler')
needs_scaling = pkg.get('needs_scaling', False)
label_encoders = pkg.get('label_encoders', {})
feature_names  = pkg.get('feature_names', [])
model_name     = pkg.get('model_name', 'ML Model')


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">👔 Employee Attrition Predictor</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-header">Powered by <b>{model_name}</b> | IBM HR Analytics Dataset</div>', unsafe_allow_html=True)

st.markdown("---")

# ── Sidebar — About ──────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/human-resources.png", width=80)
    st.title("About this App")
    st.info("""
    **Employee Attrition Prediction**
    
    This app predicts whether an employee is likely to leave the company based on HR factors.
    
    **Models trained:**
    - Logistic Regression
    - Random Forest
    - XGBoost
    
    **Dataset:** IBM HR Analytics
    """)
    st.markdown("---")
    st.markdown("**Project by:** AIML Intern 2026")
    st.markdown("**GitHub:** [View Code](#)")


# ── Input Form ────────────────────────────────────────────────────────────────
st.markdown("### 📝 Enter Employee Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="section-title">👤 Personal Info</div>', unsafe_allow_html=True)
    age              = st.slider("Age", 18, 60, 35)
    gender           = st.selectbox("Gender", ["Male", "Female"])
    marital_status   = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    education        = st.selectbox("Education Level", [1, 2, 3, 4, 5],
                                     format_func=lambda x: {1:"Below College", 2:"College",
                                                             3:"Bachelor", 4:"Master", 5:"Doctor"}[x])
    education_field  = st.selectbox("Education Field",
                                     ["Life Sciences", "Medical", "Marketing",
                                      "Technical Degree", "Human Resources", "Other"])

with col2:
    st.markdown('<div class="section-title">💼 Job Info</div>', unsafe_allow_html=True)
    department       = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
    job_role         = st.selectbox("Job Role", [
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative", "Manager",
        "Sales Representative", "Research Director", "Human Resources"
    ])
    job_level        = st.selectbox("Job Level", [1, 2, 3, 4, 5])
    job_involvement  = st.selectbox("Job Involvement", [1, 2, 3, 4],
                                     format_func=lambda x: {1:"Low", 2:"Medium", 3:"High", 4:"Very High"}[x])
    business_travel  = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
    overtime         = st.selectbox("OverTime", ["Yes", "No"])

with col3:
    st.markdown('<div class="section-title">📊 Work Metrics</div>', unsafe_allow_html=True)
    monthly_income       = st.number_input("Monthly Income ($)", 1000, 20000, 5000, step=500)
    daily_rate           = st.number_input("Daily Rate", 100, 1500, 800, step=50)
    hourly_rate          = st.number_input("Hourly Rate", 30, 100, 65, step=5)
    percent_salary_hike  = st.slider("Salary Hike (%)", 11, 25, 15)
    stock_option_level   = st.selectbox("Stock Option Level", [0, 1, 2, 3])
    total_working_years  = st.slider("Total Working Years", 0, 40, 10)
    years_at_company     = st.slider("Years at Company", 0, 40, 5)
    years_in_role        = st.slider("Years in Current Role", 0, 18, 3)
    years_since_promo    = st.slider("Years Since Promotion", 0, 15, 1)
    years_with_manager   = st.slider("Years with Current Manager", 0, 17, 3)
    num_companies        = st.slider("Num Companies Worked", 0, 9, 2)
    training_times       = st.slider("Training Times Last Year", 0, 6, 3)

# Additional sliders in expander
with st.expander("🔧 More Details (Optional)"):
    c1, c2 = st.columns(2)
    with c1:
        distance_home         = st.slider("Distance From Home (km)", 1, 29, 9)
        environment_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4],
                                                  format_func=lambda x: {1:"Low",2:"Medium",3:"High",4:"Very High"}[x])
        job_satisfaction      = st.selectbox("Job Satisfaction", [1, 2, 3, 4],
                                              format_func=lambda x: {1:"Low",2:"Medium",3:"High",4:"Very High"}[x])
    with c2:
        relationship_satisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4],
                                                    format_func=lambda x: {1:"Low",2:"Medium",3:"High",4:"Very High"}[x])
        work_life_balance     = st.selectbox("Work-Life Balance", [1, 2, 3, 4],
                                              format_func=lambda x: {1:"Bad",2:"Good",3:"Better",4:"Best"}[x])
        performance_rating    = st.selectbox("Performance Rating", [3, 4],
                                              format_func=lambda x: {3:"Excellent", 4:"Outstanding"}[x])
        monthly_rate          = st.number_input("Monthly Rate", 2000, 27000, 14000, step=500)


# ── Predict Button ────────────────────────────────────────────────────────────
st.markdown("---")
predict_clicked = st.button("🔍 Predict Attrition", use_container_width=True)

if predict_clicked:
    # Build input dict matching training feature order
    input_dict = {
        'Age': age,
        'BusinessTravel': business_travel,
        'DailyRate': daily_rate,
        'Department': department,
        'DistanceFromHome': distance_home,
        'Education': education,
        'EducationField': education_field,
        'EnvironmentSatisfaction': environment_satisfaction,
        'Gender': gender,
        'HourlyRate': hourly_rate,
        'JobInvolvement': job_involvement,
        'JobLevel': job_level,
        'JobRole': job_role,
        'JobSatisfaction': job_satisfaction,
        'MaritalStatus': marital_status,
        'MonthlyIncome': monthly_income,
        'MonthlyRate': monthly_rate,
        'NumCompaniesWorked': num_companies,
        'OverTime': overtime,
        'PercentSalaryHike': percent_salary_hike,
        'PerformanceRating': performance_rating,
        'RelationshipSatisfaction': relationship_satisfaction,
        'StockOptionLevel': stock_option_level,
        'TotalWorkingYears': total_working_years,
        'TrainingTimesLastYear': training_times,
        'WorkLifeBalance': work_life_balance,
        'YearsAtCompany': years_at_company,
        'YearsInCurrentRole': years_in_role,
        'YearsSinceLastPromotion': years_since_promo,
        'YearsWithCurrManager': years_with_manager,
    }

    # Encode categorical values
    for col, le in label_encoders.items():
        if col in input_dict:
            try:
                input_dict[col] = le.transform([input_dict[col]])[0]
            except ValueError:
                input_dict[col] = 0

    # Build input array in correct feature order
    input_array = np.array([input_dict.get(f, 0) for f in feature_names]).reshape(1, -1)

    # Scale if needed
    if needs_scaling and scaler:
        input_array = scaler.transform(input_array)

    # Predict
    prediction   = model.predict(input_array)[0]
    probability  = model.predict_proba(input_array)[0]

    stay_prob  = probability[0] * 100
    leave_prob = probability[1] * 100

    # ── Results ────────────────────────────────────────────────────────────
    st.markdown("### 🎯 Prediction Result")
    r_col1, r_col2, r_col3 = st.columns([2, 1, 1])

    with r_col1:
        if prediction == 0:
            st.markdown(f"""
            <div class="prediction-box-stay">
                <h2>✅ Employee Will STAY</h2>
                <p style="font-size:1.1rem; color:#155724;">
                    This employee is likely to remain with the company.
                </p>
                <h3 style="color:#155724;">Confidence: {stay_prob:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-box-leave">
                <h2>⚠️ Employee May LEAVE</h2>
                <p style="font-size:1.1rem; color:#721c24;">
                    This employee is at risk of leaving. HR intervention recommended.
                </p>
                <h3 style="color:#721c24;">Confidence: {leave_prob:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)

    with r_col2:
        st.metric("Stay Probability",   f"{stay_prob:.1f}%",  delta=None)
    with r_col3:
        st.metric("Leave Probability",  f"{leave_prob:.1f}%", delta=None)

    # Risk gauge
    st.markdown("#### Risk Level")
    st.progress(int(leave_prob))
    if leave_prob < 30:
        st.success("🟢 Low Risk")
    elif leave_prob < 60:
        st.warning("🟡 Medium Risk — Monitor this employee")
    else:
        st.error("🔴 High Risk — Immediate HR action recommended")

    # HR Recommendations
    st.markdown("---")
    st.markdown("### 💡 HR Recommendations")
    tips_col1, tips_col2 = st.columns(2)
    with tips_col1:
        if overtime == "Yes":
            st.warning("• Employee is working overtime — consider workload redistribution")
        if work_life_balance <= 2:
            st.warning("• Work-life balance is low — consider flexible work arrangements")
        if job_satisfaction <= 2:
            st.warning("• Low job satisfaction — schedule a review meeting")
    with tips_col2:
        if years_since_promo > 3:
            st.info("• No promotion in 3+ years — consider a performance review")
        if environment_satisfaction <= 2:
            st.info("• Environment satisfaction is low — check team dynamics")
        if monthly_income < 3000:
            st.info("• Below-average salary — benchmark against industry standards")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>AIML Internship 2026 | Employee Attrition Capstone Project</small></center>",
    unsafe_allow_html=True
)
