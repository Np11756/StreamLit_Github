import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Page config
st.set_page_config(page_title="Car Sales Forecast", layout="wide")
st.markdown("""
<style>
[data-testid="block-container"] {
    padding: 2rem 2rem 0rem 2rem;
}
[data-testid="stMetric"] {
    background-color: #1e1e1e;
    text-align: center;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# Load data and models
@st.cache_data
def load_data():
    df = pd.read_csv("car_sales_cleaned.csv")

    def safe_reconstruct_column(df, prefix, new_col):
        onehot_cols = [col for col in df.columns if col.startswith(prefix)]
        if onehot_cols:
            df[new_col] = df[onehot_cols].idxmax(axis=1).str.replace(prefix, "", regex=False)
        else:
            df[new_col] = "Unknown"
        return df

    df = safe_reconstruct_column(df, "Dealer_Region_", "Dealer_Region")
    return df

@st.cache_resource
def load_models():
    hgb = joblib.load("histgradient_model.pkl")
    lr = joblib.load("linear_regression_baseline.pkl")
    return hgb, lr

df = load_data()
hgb_model, lr_model = load_models()

# Title
st.title("🚗 Car Sales Forecasting Dashboard")
st.markdown("This dashboard predicts car sale prices using historical data and machine learning models. Adjust the inputs on the left to see real-time predictions and explore dealership performance across regions.")

# Layout with 3 columns
col1, col2, col3 = st.columns((1.5, 4, 2), gap="large")

# --- LEFT COLUMN: Input & Prediction ---
with col1:
    st.markdown("#### Predict Price")

    month = st.slider("Month of Sale", 1, 12, 6)
    income = st.slider("Annual Income", 20000, 200000, 75000)
    region = st.selectbox("Dealer Region", sorted(df["Dealer_Region"].unique()))

    # Build input data for prediction
    feature_list = list(hgb_model.feature_names_in_)
    input_data = {col: 0 for col in feature_list}
    input_data["Month_Num"] = month
    input_data["Annual Income"] = income
    if f"Dealer_Region_{region}" in input_data:
        input_data[f"Dealer_Region_{region}"] = 1

    input_df = pd.DataFrame([input_data])
    hgb_pred = hgb_model.predict(input_df)[0]
    lr_pred = lr_model.predict(input_df)[0]

    st.metric("HistGradientBoosting", f"${hgb_pred:,.2f}")
    st.metric("Linear Regression", f"${lr_pred:,.2f}")

# --- CENTER COLUMN: Dealership Map ---
with col2:
    st.markdown("#### Dealership Locations")

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

    map_fig = px.scatter_mapbox(
        df, lat="Latitude", lon="Longitude", color="Dealer_Region",
        zoom=3, mapbox_style="open-street-map",
        title="Dealer Locations by Region"
    )
    st.plotly_chart(map_fig, use_container_width=True)

# --- RIGHT COLUMN: Summary Stats ---
with col3:
    st.markdown("#### Market Summary")
    latest_month = df["Month_Num"].max()
    df_latest = df[df["Month_Num"] == latest_month]

    stats = df_latest.groupby("Dealer_Region")["Price ($)"].agg(["count", "mean", "sum"]).reset_index()
    stats.columns = ["Region", "Sales Count", "Avg Price", "Total Sales"]
    st.dataframe(stats.sort_values("Total Sales", ascending=False), use_container_width=True)
