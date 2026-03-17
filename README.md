# 📊 Customer Churn Predictor

<div align="center">

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-churn-predictor-sjshumrhpnua86ukfam3j2.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.x-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

**An end-to-end Machine Learning web app to predict Telecom Customer Churn**

🚀 **[Live Demo](https://customer-churn-predictor-sjshumrhpnua86ukfam3j2.streamlit.app)**

</div>

---

## 🧠 About the Project

Customer churn is one of the biggest challenges in the telecom industry. This project builds a **full-stack ML web application** that helps businesses identify customers likely to leave, understand what features drive churn, predict churn individually or in bulk, and track prediction history over time.

The app uses real Telco data with **7,032 customers** and **21 features**, trained on 3 ML models with SHAP explainability.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🏠 Overview Dashboard | Key metrics, churn distribution, tenure analysis |
| 📈 EDA | Interactive charts — charges, contracts, correlations |
| 🤖 ML Models | 3 models with ROC curves, confusion matrix, feature importance |
| 🔍 SHAP Explainability | Model explainability using SHAP values |
| 🔮 Single Prediction | Predict churn for individual customer with gauge chart |
| 📤 Bulk Prediction | Upload CSV → get predictions for all customers |
| 📋 History | SQLite-powered prediction history with export |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.9+ |
| **Web Framework** | Streamlit |
| **ML Models** | Scikit-learn (Random Forest, Gradient Boosting, Logistic Regression) |
| **Explainability** | SHAP |
| **Visualization** | Plotly |
| **Database** | SQLite3 |

---

## 🤖 ML Models

| Model | Accuracy | AUC |
|---|---|---|
| Random Forest | ~80% | ~0.85 |
| Gradient Boosting | ~80% | ~0.85 |
| Logistic Regression | ~79% | ~0.84 |

> All models trained with 80/20 train-test split + 5-fold cross validation

---

## 📁 Dataset

- **Source:** IBM Telco Customer Churn (Kaggle)
- **Rows:** 7,032 customers
- **Features:** 21 (demographics, services, charges, contract)
- **Target:** `Churn` (Yes / No)

---

## 📂 Project Structure
```
customer-churn-predictor/
├── app.py
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── churn.db
└── requirements.txt
```

---

## 🚀 Run Locally
```bash
git clone https://github.com/Priyanshuu2008/customer-churn-predictor.git
cd customer-churn-predictor
pip install -r requirements.txt
streamlit run app.py
```

---

## 👨‍💻 Author

**Priyanshu Tiwari**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/priyanshuu20)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/Priyanshuu2008)

⭐ **If you found this useful, please give it a star!**
