
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="Car Sales Forecasting Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("car_sales_cleaned.csv")

    # Reconstruct columns from one-hot encoding
    def safe_reconstruct_column(df, prefix, new_col):
        onehot_cols = [col for col in df.columns if col.startswith(prefix)]
        if onehot_cols:
            df[new_col] = df[onehot_cols].idxmax(axis=1).str.replace(prefix, "", regex=False)
        else:
            df[new_col] = "Unknown"
        return df

    df = safe_reconstruct_column(df, "Dealer_Region_", "Dealer_Region")
    df = safe_reconstruct_column(df, "Body Style_", "Body Style")
    df = safe_reconstruct_column(df, "Transmission_", "Transmission")
    return df

@st.cache_resource
def load_models():
    hgb = joblib.load("histgradient_model.pkl")
    lr = joblib.load("linear_regression_baseline.pkl")
    return hgb, lr

df = load_data()
hgb_model, lr_model = load_models()

st.title("Car Sales Forecasting Dashboard")

tab1, tab2, tab3 = st.tabs(["Price Prediction Tool", "Dealer Insights", "Market Trends"])

with tab1:
    st.markdown("<h3 style='color:white;'>Price Predictions</h3>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Customize Car Features")
        month = st.slider("Month of Sale", 1, 12, 6)
        car_age = st.slider("Car Age", 0, 20, 5)
        income = st.slider("Annual Income", 20000, 200000, 75000)
        region = st.selectbox("Dealer Region", sorted(df["Dealer_Region"].unique()))
        body_style = st.selectbox("Body Style", sorted(df["Body Style"].unique()))
        transmission = st.selectbox("Transmission", sorted(df["Transmission"].unique()))

    # Safe prediction input matching model features
    input_data = {col: 0 for col in hgb_model.feature_names_in_}
    input_data["Month_Num"] = month
    input_data["Car_Age"] = car_age
    input_data["Annual Income"] = income

    if f"Dealer_Region_{region}" in input_data:
        input_data[f"Dealer_Region_{region}"] = 1
    if f"Body Style_{body_style}" in input_data:
        input_data[f"Body Style_{body_style}"] = 1
    if f"Transmission_{transmission}" in input_data:
        input_data[f"Transmission_{transmission}"] = 1

    input_df = pd.DataFrame([input_data])
    hgb_pred = hgb_model.predict(input_df)[0]
    lr_pred = lr_model.predict(input_df)[0]

    col1, col2 = st.columns(2)
    col1.metric("HistGradientBoosting Model", f"${hgb_pred:,.2f}")
    col2.metric("Linear Regression Baseline", f"${lr_pred:,.2f}")

with tab2:
    st.markdown("## Dealer Sales Overview")
    region_cols = [col for col in df.columns if "Dealer_Region_" in col]
    region_sales = df[region_cols].sum().sort_values(ascending=False).reset_index()
    region_sales.columns = ["Region", "Total Sales"]
    region_sales["Region"] = region_sales["Region"].str.replace("Dealer_Region_", "")

    fig_bar = px.bar(region_sales, x="Region", y="Total Sales", title="Sales Volume by Dealer Region")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### Dealership Locations Map")
    region_coords = {
        "Austin": (30.2672, -97.7431),
        "Greenville": (34.8526, -82.3940),
        "Janesville": (42.6828, -89.0187),
        "Middletown": (39.5151, -84.3983),
        "Pasco": (46.2396, -119.1006),
        "Scottsdale": (33.4942, -111.9261),
    }
    df["Latitude"] = df["Dealer_Region"].map(lambda r: region_coords.get(r, (0, 0))[0])
    df["Longitude"] = df["Dealer_Region"].map(lambda r: region_coords.get(r, (0, 0))[1])

    fig_map = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", color="Dealer_Region",
                                 hover_name="Dealer_Region", zoom=3,
                                 mapbox_style="open-street-map",
                                 title="Dealerships by Region")
    st.plotly_chart(fig_map, use_container_width=True)

with tab3:
    st.markdown("## Market Trends")
    body_filter = st.selectbox("Body Style Filter", ["All"] + [col for col in df.columns if "Body Style_" in col])
    region_filter = st.selectbox("Region Filter", ["All"] + [col for col in df.columns if "Dealer_Region_" in col])
    month_filter = st.selectbox("Month Filter", ["All"] + sorted(df["Month_Num"].unique()))

    df_filtered = df.copy()
    if body_filter != "All":
        df_filtered = df_filtered[df_filtered[body_filter] == 1]
    if region_filter != "All":
        df_filtered = df_filtered[df_filtered[region_filter] == 1]
    if month_filter != "All":
        df_filtered = df_filtered[df_filtered["Month_Num"] == int(month_filter)]

    trend_data = df_filtered.groupby("Month_Num")["Price ($)"].sum().reset_index()
    fig_trend = px.line(trend_data, x="Month_Num", y="Price ($)", markers=True, title="Monthly Sales Trend")
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("### Summary Statistics")
    st.dataframe(df_filtered[["Price ($)", "Annual Income", "Car_Age"]].describe().T)
