# C.R.I.S.E.S  
**Crisis Risk Identification System for Economic Stability**

---

## Overview

This project builds a **global financial crisis early warning system** that detects and forecasts economic stress using:

- Gaussian Mixture Models (GMM)  
- Economic shock feature engineering  
- Regime probability smoothing  
- Time-series forecasting  
- Decision-support dashboard (Streamlit)

It analyzes macroeconomic indicators such as **bond yields, CDS spreads, and government deficits** to detect **hidden economic regimes** (Normal, Stress, Crisis) and generate **next-month crisis alerts** using machine learning.

The emphasis is on **clarity and interpretability** — showing how basic ML tools can complement economic reasoning for crisis forecasting.

---

##  Objectives

- Detect macro-financial stress periods across countries  
- Reveal **hidden economic regimes**  
- Predict **next-month crisis likelihood**  
- Build an **interactive dashboard** for analysis  
- Support interpretation using **feature importance, contagion maps, and regime evolution**

---

## Methodology

| Step | Description |
|------|-------------|
| **Feature Engineering** | Rolling mean and volatility shocks for bond, CDS, deficit |
| **Regime Detection** | GMM clustering reveals hidden economic states |
| **Smoothing** | 6-month rolling average improves stability |
| **Forecasting** | Random Forest predicts next-month regime |
| **Alerting** | Binary `crisis_alert` + traffic-light risk levels |
| **Validation** | TimeSeriesSplit prevents future data leakage |
| **Visualization** | Contagion maps, regime evolution, risk trends |
| **Dashboard** | Interactive Streamlit interface |

---

## Results

The system successfully identifies periods where **fiscal imbalance + market stress** align with **high crisis probability**.

### Key Predictive Features
- Deficit-to-GDP ratio  
- Debt-to-GDP ratio  
- Bond yield spreads (Greek vs German)

The model captures **pre-crisis stress buildup**, demonstrating early warning potential rather than perfect event prediction. This model **does not predict exact financial crises**. Instead, it acts as a **decision-support tool**, flagging **rising economic vulnerability** using market-based indicators and economic intuition.

---

## How to Run

**1. Install dependencies**
pip install -r requirements.txt

**2. Launch dashboard**
streamlit run src/dashboard_app.py

