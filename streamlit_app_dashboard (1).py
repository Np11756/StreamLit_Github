import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import matplotlib.pyplot as plt

# Theme and layout settings
st.set_page_config(
    page_title="Car Sales Forecasting Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom dark theme CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #1e1e1e;
        color: white;
    }
    .css-1d391kg {
        background-color: #333333 !important;
    }
    .sidebar .sidebar-content {
        background-color: #2f2f2f;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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

st.title("🚗 Car Sales Forecasting Dashboard")

tab1, tab2, tab3 = st.tabs(["📈 Predictions", "🌍 Regional Overview", "📊 Trends & Summary"])

with tab1:
    st.header("Price Predictions with Machine Learning")
    
    st.sidebar.markdown("## 🎛️ Input Car Features")
    month = st.sidebar.slider("Month of Sale", 1, 12, 6)
    car_age = st.sidebar.slider("Car Age", 0, 20, 5)
    income = st.sidebar.slider("Annual Income", 20000, 200000, 75000)

    region = st.sidebar.selectbox("Dealer Region", [col for col in df.columns if "Dealer_Region_" in col])
    body_style = st.sidebar.selectbox("Body Style", [col for col in df.columns if "Body Style_" in col])
    transmission = st.sidebar.selectbox("Transmission", [col for col in df.columns if "Transmission_" in col])
    price_category = st.sidebar.selectbox("Price Category", [col for col in df.columns if "Price_Category_" in col])

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

    col1, col2 = st.columns(2)
    col1.metric("🚀 HistGradientBoosting Prediction", f"${hgb_pred:,.2f}")
    col2.metric("📉 Linear Regression Baseline", f"${lr_pred:,.2f}")

with tab2:
    st.header("Regional Sales Overview")
    region_cols = [col for col in df.columns if "Dealer_Region_" in col]
    region_sales = df[region_cols].sum().sort_values(ascending=False)
    region_df = region_sales.reset_index()
    region_df.columns = ["Region", "Sales"]

    fig = px.bar(region_df, x="Region", y="Sales", color="Region",
                 title="Total Sales by Region", color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Sales Trends & Summary")

    st.subheader("Filter Data")
    selected_body = st.selectbox("Body Style", ["All"] + [col for col in df.columns if "Body Style_" in col])
    selected_region = st.selectbox("Region", ["All"] + [col for col in df.columns if "Dealer_Region_" in col])
    selected_month = st.selectbox("Month", ["All"] + sorted(df["Month_Num"].unique()))

    df_filtered = df.copy()
    if selected_body != "All":
        df_filtered = df_filtered[df_filtered[selected_body] == 1]
    if selected_region != "All":
        df_filtered = df_filtered[df_filtered[selected_region] == 1]
    if selected_month != "All":
        df_filtered = df_filtered[df_filtered["Month_Num"] == int(selected_month)]

    # Monthly trend chart
    st.markdown("### 📆 Monthly Sales Trend")
    monthly_sales = df_filtered.groupby("Month_Num")["Price ($)"].sum().reset_index()
    fig2 = px.line(monthly_sales, x="Month_Num", y="Price ($)", markers=True,
                   title="Monthly Sales", color_discrete_sequence=["cyan"])
    st.plotly_chart(fig2, use_container_width=True)

    # Summary stats
    st.markdown("### 🧾 Summary Statistics")
    st.dataframe(df_filtered[["Price ($)", "Annual Income", "Car_Age"]].describe().T)
