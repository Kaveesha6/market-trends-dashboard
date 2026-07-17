"""
app.py - Market Trends Dashboard (Streamlit)
-----------------------------------------------------
Self-contained dashboard covering the SAME 15-category SME industry
taxonomy used by the BizBuddyBot chatbot's guided journey, so a business
idea classified in the chatbot maps to a REAL, matching category here.

Deploy this as its own Streamlit Community Cloud app (one file +
requirements.txt is enough - data is generated and the forecast model is
trained and cached on first run, no external files needed).
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Shared taxonomy - MUST stay in sync with agentic/industry_classifier.py
# and trend_analysis/train_model.py in the chatbot backend.
# ---------------------------------------------------------------------------
INDUSTRY_TAXONOMY = [
    "Retail - Fashion & Apparel",
    "Retail - General & Convenience",
    "Food & Beverage",
    "Technology & Software",
    "Healthcare & Wellness",
    "Education & Training",
    "Beauty & Personal Care",
    "Home, Furniture & Decor",
    "Electronics & Gadgets",
    "Agriculture & Food Production",
    "Professional Services",
    "Hospitality & Tourism",
    "Handicrafts & Artisan Goods",
    "Transportation & Logistics",
    "Finance & Fintech",
]

# (monthly_growth_rate, seasonality_amplitude, noise_std)
INDUSTRY_PROFILES = {
    "Retail - Fashion & Apparel":       (1.035, 250, 150),
    "Retail - General & Convenience":   (1.020, 150, 100),
    "Food & Beverage":                  (1.030, 300, 180),
    "Technology & Software":            (1.070, 100, 220),
    "Healthcare & Wellness":            (1.025, 80,  90),
    "Education & Training":             (1.028, 350, 110),
    "Beauty & Personal Care":           (1.032, 180, 120),
    "Home, Furniture & Decor":          (1.022, 220, 140),
    "Electronics & Gadgets":            (1.040, 300, 200),
    "Agriculture & Food Production":    (1.015, 400, 160),
    "Professional Services":            (1.024, 60,  100),
    "Hospitality & Tourism":            (1.026, 450, 210),
    "Handicrafts & Artisan Goods":      (1.018, 200, 110),
    "Transportation & Logistics":       (1.027, 120, 130),
    "Finance & Fintech":                (1.045, 90,  170),
}

FEATURES = ["Customers", "Orders", "Profit", "Revenue_Lag_1", "Revenue_Lag_2", "Revenue_Lag_3"]
TARGET = "Revenue"


# ---------------------------------------------------------------------------
# Data generation + model training (cached - runs once per app session)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_dataset(seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    try:
        months = pd.date_range(start="2021-01-01", periods=48, freq="ME")
    except ValueError:
        months = pd.date_range(start="2021-01-01", periods=48, freq="M")

    rows = []
    base_revenue = 1000

    for industry in INDUSTRY_TAXONOMY:
        growth_rate, seasonality_amp, noise_std = INDUSTRY_PROFILES[industry]
        for i, month in enumerate(months):
            seasonal = seasonality_amp * np.sin(2 * np.pi * i / 12)
            revenue = base_revenue * (growth_rate ** i) + seasonal + np.random.normal(0, noise_std)
            revenue = max(revenue, 100)

            customers = revenue / 10 + np.random.normal(0, 20)
            orders = customers * 0.6 + np.random.normal(0, 10)
            profit = revenue * 0.25 + np.random.normal(0, 50)

            rows.append([industry, month, revenue, customers, orders, profit])

    df = pd.DataFrame(rows, columns=["Industry", "Month", "Revenue", "Customers", "Orders", "Profit"])
    return df


@st.cache_data(show_spinner=False)
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by=["Industry", "Month"]).copy()
    for lag in [1, 2, 3]:
        df[f"Revenue_Lag_{lag}"] = df.groupby("Industry")["Revenue"].shift(lag)
    return df.dropna()


@st.cache_resource(show_spinner=False)
def train_forecast_model(df: pd.DataFrame):
    # Per-industry chronological split (last 20% of months per industry held
    # out) - a single global row-order split leaves whole industries unseen
    # during training, which RandomForest can't extrapolate well from.
    train_frames, test_frames = [], []
    for industry, group in df.groupby("Industry"):
        group = group.sort_values("Month")
        cut = int(len(group) * 0.8)
        train_frames.append(group.iloc[:cut])
        test_frames.append(group.iloc[cut:])

    train_df, test_df = pd.concat(train_frames), pd.concat(test_frames)
    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return model, rmse, r2


def forecast_industry(model, df: pd.DataFrame, industry: str, months_ahead: int = 6) -> dict:
    industry_df = df[df["Industry"] == industry].sort_values("Month")
    last_rows = industry_df.tail(6).copy()
    forecasts, forecast_months = [], []

    last_month = industry_df["Month"].max()
    for i in range(months_ahead):
        X_future = last_rows[FEATURES].iloc[-1:]
        next_revenue = model.predict(X_future)[0]
        forecasts.append(round(float(next_revenue), 2))
        forecast_months.append(last_month + pd.DateOffset(months=i + 1))

        new_row = last_rows.iloc[-1:].copy()
        new_row["Revenue_Lag_3"] = new_row["Revenue_Lag_2"]
        new_row["Revenue_Lag_2"] = new_row["Revenue_Lag_1"]
        new_row["Revenue_Lag_1"] = next_revenue
        new_row["Revenue"] = next_revenue
        last_rows = pd.concat([last_rows, new_row])

    return {"months": forecast_months, "forecast": forecasts}


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Market Trends | BizBuddyBot", page_icon="📈", layout="wide")

raw_df = generate_dataset()
df = add_lag_features(raw_df)
model, rmse, r2 = train_forecast_model(df)

st.title("📈 Market Trends & Revenue Forecast")
st.caption("Part of the BizBuddyBot SME Toolkit - covers the same 15 industry categories used by the chatbot's guided business journey.")

# ── Sidebar controls ──
st.sidebar.header("Controls")
selected_industry = st.sidebar.selectbox("Industry", INDUSTRY_TAXONOMY, index=0)
months_ahead = st.sidebar.slider("Forecast horizon (months)", min_value=1, max_value=12, value=6)

industry_hist = df[df["Industry"] == selected_industry].sort_values("Month")
forecast_result = forecast_industry(model, df, selected_industry, months_ahead)

last_known_revenue = industry_hist["Revenue"].iloc[-1]
forecast_end_revenue = forecast_result["forecast"][-1]
pct_change = (forecast_end_revenue - last_known_revenue) / last_known_revenue * 100
trend_direction = "increasing" if pct_change > 0 else "decreasing"

# ── KPI row ──
col1, col2, col3, col4 = st.columns(4)
col1.metric("Last Known Monthly Revenue", f"LKR {last_known_revenue:,.0f}")
col2.metric(f"Projected Revenue (+{months_ahead}mo)", f"LKR {forecast_end_revenue:,.0f}", f"{pct_change:+.1f}%")
col3.metric("Trend Direction", trend_direction.capitalize())
col4.metric("Model R² Score", f"{r2:.2f}")

st.divider()

# ── Historical + Forecast chart ──
st.subheader(f"Revenue Trend: {selected_industry}")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=industry_hist["Month"], y=industry_hist["Revenue"],
    mode="lines", name="Historical Revenue", line=dict(color="#3b82f6", width=2)
))
fig.add_trace(go.Scatter(
    x=[industry_hist["Month"].iloc[-1]] + forecast_result["months"],
    y=[industry_hist["Revenue"].iloc[-1]] + forecast_result["forecast"],
    mode="lines+markers", name="Forecast", line=dict(color="#f97316", width=2, dash="dash")
))
fig.update_layout(xaxis_title="Month", yaxis_title="Revenue (LKR)", hovermode="x unified", height=450)
st.plotly_chart(fig, use_container_width=True)

# ── All-industries comparison ──
st.subheader("All Industries - Revenue Comparison")
fig2 = go.Figure()
for industry in INDUSTRY_TAXONOMY:
    ind_df = df[df["Industry"] == industry].sort_values("Month")
    fig2.add_trace(go.Scatter(
        x=ind_df["Month"], y=ind_df["Revenue"], mode="lines", name=industry,
        line=dict(width=3 if industry == selected_industry else 1),
        opacity=1.0 if industry == selected_industry else 0.35,
    ))
fig2.update_layout(xaxis_title="Month", yaxis_title="Revenue (LKR)", height=450, legend=dict(font=dict(size=9)))
st.plotly_chart(fig2, use_container_width=True)

# ── Forecast table ──
st.subheader(f"Forecast Detail - Next {months_ahead} Months")
forecast_table = pd.DataFrame({
    "Month": [m.strftime("%Y-%m") for m in forecast_result["months"]],
    "Forecasted Revenue (LKR)": [f"{v:,.0f}" for v in forecast_result["forecast"]],
})
st.dataframe(forecast_table, use_container_width=True, hide_index=True)

with st.expander("About this dashboard"):
    st.write(
        f"This dashboard uses a synthetically generated 48-month dataset covering all {len(INDUSTRY_TAXONOMY)} "
        "SME industry categories, matching the taxonomy used by the BizBuddyBot chatbot's guided business "
        f"journey. Forecasts come from a RandomForestRegressor trained on lag features "
        f"(RMSE: {rmse:,.0f}, R²: {r2:.2f} on held-out future months per industry)."
    )
