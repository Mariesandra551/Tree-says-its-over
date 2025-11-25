# Crisis Risk Identification System for Economic Stability (C.R.I.S.E.S)

## Overview
This project builds a global financial crisis early warning system that detects and forecasts economic stress using Gaussian Mixture Models, regime probability smoothing, and a dashboard-based decision support tool.

Using macroeconomic shock indicators from bond yields, CDS spreads, and government deficits, the model identifies hidden economic regimes (Normal, Stress, Crisis) and generates next-month crisis alerts using machine learning and time series forecasting.

The emphasis is on **clarity and interpretability**, showing how even basic machine learning tools can complement economic reasoning for crisis forecasting.

---

## Objectives
- Detect macro-financial stress periods using unsupervised clustering
- Reveal hidden economic regimes across countries
- Predict regime transitions (next-month crisis likelihood)
- Build an interactive Streamlit dashboard for policymakers and analysts
- Support interpretation using feature importance, contagion maps, and regime evolution

---

## Methodology

| Step | Description |
|------------|-------------|
| Feature Engineering | Rolling mean and volatility shocks for bond, CDS, and deficit | 
| Regime Detection | Gaussian Mixture Model (GMM) identifies hidden economic states | 
| Smoothing | 6-month rolling average improves crisis probability stability | 
| Forecasting | Random Forest predicts next-month regime transitions | 
| Alerting | Binary crisis_alert + traffic-light risk levels |
| Validation | TimeSeriesSplit (chronological), prevents future leakage |
| Visualization |	Contagion matrix, regime evolution, probability trends | 
| Dashboard | Interactive, filterable, exportable Streamlit interface | 

---

## Results

- The model successfully identifies periods where **fiscal imbalance** and **market stress** coincide with high crisis probability.
- Key predictive features include:
  - Deficit-to-GDP ratio
  - Debt-to-GDP ratio
  - Bond yield spreads (Greek vs. German)
- The visualization highlights years that markets perceived as high-risk before the official crisis.

---

## Interpretation
The project demonstrates that even with basic data and a simple logistic regression, it is possible to detect **early warning signs** of financial distress.  
The model should not be viewed as a predictor of exact crises but as a **decision-support tool** that highlights periods of rising vulnerability.

---

## Challenges & Solutions

| Challenge | Mitigation |
|------------|-------------|
| Limited and inconsistent historical data | Used data cleaning and median imputation |
| Small dataset | Applied upsampling of crisis observations |
| Correlated variables | Chose interpretable indicators and simple models |
| Asymmetric information | Relied on market-based indicators like CDS spreads |
| Delayed reporting | Combined fiscal (slow) and market (fast) indicators |

---

License

This project is for educational and academic purposes only.
You may reuse or adapt the code with proper citation.

---

## How to Run

### Prerequisites
Install necessary dependencies:
```bash
Run the Streamlit dashboard:
streamlit run src/dashboard_app.py











