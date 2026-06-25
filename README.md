# 👔 Employee Attrition Prediction — AIML Capstone 2026

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?logo=streamlit)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20Random%20Forest%20%7C%20LogReg-green)
![Dataset](https://img.shields.io/badge/Dataset-IBM%20HR%20Analytics-orange)

> **Predicting whether an employee will leave the company using Machine Learning**

---

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

---

## 📌 Problem Statement

Employee attrition is a major challenge for HR departments. This project builds a classification model to predict whether an employee is likely to leave the company, using the **IBM HR Analytics Employee Attrition Dataset**.

- **Type:** Binary Classification (Attrition: Yes / No)
- **Dataset:** 1,470 employees × 35 features
- **Best Model Metric:** F1 Score (handles class imbalance well)

---

## 📁 Project Structure

```
Employee_Attrition_Project/
│
├── Dataset/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── Notebook/
│   └── Employee_Attrition_Colab.ipynb      ← Main Colab notebook
│
├── Model/
│   └── model.pkl                            ← Trained best model
│
├── Streamlit_App/
│   └── app.py                               ← Deployment app
│
├── requirements.txt
└── README.md
```

---

## 🔬 Project Phases

| Phase | Description |
|-------|-------------|
| **1** | Data Loading — IBM HR Dataset from Kaggle |
| **2** | EDA — Distribution plots, correlation heatmap, feature analysis |
| **3** | Preprocessing — Label encoding, train-test split (80/20), scaling |
| **4** | Model Building — Logistic Regression, Random Forest, XGBoost |
| **5** | Model Evaluation — Accuracy, Precision, Recall, F1, ROC-AUC |
| **6** | Deployment — Streamlit app with interactive predictions |

---

## 📊 Model Results

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | ~86% | ~52% | ~81% |
| Random Forest | ~87% | ~57% | ~84% |
| **XGBoost** | **~88%** | **~62%** | **~87%** |

> ✅ **XGBoost** selected as best model based on F1 Score

---

## ⚙️ How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Employee_Attrition_Project.git
cd Employee_Attrition_Project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Streamlit app
```bash
streamlit run Streamlit_App/app.py
```

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → Select your repo
4. Set **Main file path** to `Streamlit_App/app.py`
5. Click **Deploy** ✅

---

## 📦 Requirements

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

---

## 🗂️ Dataset

**IBM HR Analytics Employee Attrition Dataset**
- Source: [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- 1,470 rows × 35 columns
- Target: `Attrition` (Yes/No)
- No missing values

Key features: `Age`, `Department`, `JobRole`, `OverTime`, `MonthlyIncome`, `YearsAtCompany`, `JobSatisfaction`, `WorkLifeBalance`, and more.

---

## 📈 Key Insights from EDA

- Employees with **OverTime = Yes** have 3× higher attrition rate
- **Sales Representatives** have the highest attrition (~40%)
- **Single** employees leave more often than married employees
- Low **Job Satisfaction** and **Work-Life Balance** strongly predict attrition
- Employees who haven't been promoted in **3+ years** are higher risk

---

## 👨‍💻 Author

**[Your Name]**
AIML Summer Internship 2026

[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?logo=github)](https://github.com/YOUR_USERNAME)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/YOUR_PROFILE)

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).
