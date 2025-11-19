"""
dashboard_app.py
----------------
Run with:
    streamlit run dashboard_app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "gmm_predictions_dashboard.csv"

# -----------------------------
# 1. LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

st.title("📉 Financial Crisis Early Warning Dashboard")
st.caption("Powered by Gaussian Mixture Model (unsupervised regime detection)")

# -----------------------------
# 2. SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("Filters")
countries = df["country"].unique()
selected = st.sidebar.multiselect("Select countries", countries, default=countries[:3])

risk_level = st.sidebar.radio(
    "Filter by risk level",
    ["ALL", "GREEN", "YELLOW", "RED"]
)

df_filtered = df[df["country"].isin(selected)]

if risk_level != "ALL":
    df_filtered = df_filtered[df_filtered["risk_level"] == risk_level]

# -----------------------------
# 3. DATA TABLE
# -----------------------------
st.subheader("🔍 Filtered observations")
st.dataframe(
    df_filtered[
        ["country", "date", "crisis_prob", "regime", "risk_level"]
    ].sort_values(["country", "date"], ascending=[True, False]).head(20)
)

# -----------------------------
# 4. TREND PLOT
# -----------------------------
st.subheader("📈 Crisis Probability Over Time")
country_sel = st.selectbox("Select country for trend", countries)
subset = df[df["country"] == country_sel].copy()

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(subset["date"], subset["crisis_prob"], marker="o")
ax.set_title(f"Crisis Probability Trend: {country_sel}")
ax.set_xlabel("Date")
ax.set_ylabel("Probability")
plt.xticks(rotation=45)
st.pyplot(fig)

# -----------------------------
# 5. RISK LEVEL COUNTS
# -----------------------------
st.subheader("🚦 Risk Level Breakdown")
risk_counts = df_filtered["risk_level"].value_counts()
st.bar_chart(risk_counts)

# -----------------------------
# 6. DOWNLOAD DATA
# -----------------------------
st.subheader("💾 Export Results")
csv = df_filtered.to_csv(index=False).encode()
st.download_button(
    label="Download filtered dataset",
    data=csv,
    file_name="filtered_crisis_predictions.csv"
)

st.success("Dashboard ready — explore your early warning model!")

if __name__ == "__main__":
    st.write("Streamlit mode active")
