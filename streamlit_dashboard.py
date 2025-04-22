import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    return pd.read_csv("car_sales_cleaned.csv")

@st.cache_resource
def load_models():
    hgb = joblib.load("histgradient_model.pkl")
    lr = joblib.load("linear_regression_baseline.pkl")
    return hgb, lr

df = load_data()
hgb_model, lr_model = load_models()

st.set_page_config(layout="wide")
st.title("🚗 Car Sales Forecasting Dashboard")

tab1, tab2, tab3 = st.tabs(["📈 Model Predictions", "🌍 Regional Performance", "📊 Sales Trends & Summary"])

with tab1:
    st.header("Model-Based Price Predictions")
    
    with st.sidebar:
        st.subheader("Input Car Features")
        month = st.slider("Month of Sale", 1, 12, 6)
        car_age = st.slider("Car Age", 0, 20, 5)
        income = st.slider("Annual Income", 20000, 200000, 75000)

        region = st.selectbox("Dealer Region", [col for col in df.columns if "Dealer_Region_" in col])
        body_style = st.selectbox("Body Style", [col for col in df.columns if "Body Style_" in col])
        transmission = st.selectbox("Transmission", [col for col in df.columns if "Transmission_" in col])
        price_category = st.selectbox("Price Category", [col for col in df.columns if "Price_Category_" in col])

    input_data = {col: 0 for col in hgb_model.feature_names_in_}
    input_data["Month_Num"] = month
    input_data["Car_Age"] = car_age
    input_data["Annual Income"] = income
    for feature in [region, body_style, transmission, price_category]:
        if feature in input_data:
            input_data[feature] = 1

    input_df = pd.DataFrame([input_data])
    hgb_pred = hgb_model.predict(input_df)[0]
    lr_pred = lr_model.predict(input_df)[0]

    st.success(f"📈 Predicted Price (HistGradientBoosting): **${hgb_pred:,.2f}**")
    st.info(f"📉 Baseline Prediction (Linear Regression): **${lr_pred:,.2f}**")

with tab2:
    st.header("🌍 Regional Sales Performance")

    region_cols = [col for col in df.columns if "Dealer_Region_" in col]
    region_sales = df[region_cols].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    region_sales.plot(kind="bar", ax=ax)
    ax.set_title("Total Sales by Region (Encoded)")
    ax.set_ylabel("Total Sales")
    ax.set_xlabel("Region")
    st.pyplot(fig)

with tab3:
    st.header("📆 Monthly Trends and Sales Summary")

    st.subheader("Apply Filters")
    selected_body = st.selectbox("Filter by Body Style", ["All"] + [col for col in df.columns if "Body Style_" in col])
    selected_region = st.selectbox("Filter by Region", ["All"] + [col for col in df.columns if "Dealer_Region_" in col])
    selected_month = st.selectbox("Filter by Month", ["All"] + sorted(df["Month_Num"].unique()))

    df_filtered = df.copy()
    if selected_body != "All":
        df_filtered = df_filtered[df_filtered[selected_body] == 1]
    if selected_region != "All":
        df_filtered = df_filtered[df_filtered[selected_region] == 1]
    if selected_month != "All":
        df_filtered = df_filtered[df_filtered["Month_Num"] == int(selected_month)]

    st.markdown("### Monthly Sales Trend (Filtered)")
    monthly_sales = df_filtered.groupby("Month_Num")["Price ($)"].sum()
    fig2, ax2 = plt.subplots()
    monthly_sales.plot(marker="o", ax=ax2)
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Sales ($)")
    ax2.set_title("Monthly Sales Trend")
    st.pyplot(fig2)

    st.markdown("### Summary Statistics (Filtered Data)")
    st.dataframe(df_filtered[["Price ($)", "Annual Income", "Car_Age"]].describe().T)
