import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from streamlit_plotly_events import plotly_events

st.set_page_config(page_title="Car Sales Forecast", layout="wide")

st.markdown("""
<style>
[data-testid="block-container"] {
    padding: 2rem 2rem 0rem 2rem;
}
[data-testid="stMetric"] {
    background-color: #393939;
    color: white;
    text-align: center;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 10px;
}
[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
  color: white;
}
</style>
""", unsafe_allow_html=True)

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

st.title("Car Sales Forecasting Dashboard")
st.markdown("Use this dashboard to explore machine learning-based car price predictions, dealership performance, and sales trends.")

tab1, tab2, tab3 = st.tabs(["Price Prediction", "Dealership Map", "Market Summary"])

# --- TAB 1: Price Prediction ---
with tab1:
    st.markdown("#### Customize Inputs")
    col1, col2 = st.columns(2)
    with col1:
        month = st.slider("Month of Sale", 1, 12, 6)
        income = st.slider("Annual Income", 20000, 200000, 75000)
    with col2:
        region = st.selectbox("Dealer Region", sorted(df["Dealer_Region"].unique()))

    feature_list = list(hgb_model.feature_names_in_)
    input_data = {col: 0 for col in feature_list}
    input_data["Month_Num"] = month
    input_data["Annual Income"] = income
    if f"Dealer_Region_{region}" in input_data:
        input_data[f"Dealer_Region_{region}"] = 1

    input_df = pd.DataFrame([input_data])
    hgb_pred = hgb_model.predict(input_df)[0]
    lr_pred = lr_model.predict(input_df)[0]

    st.markdown("#### Model Predictions")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<div style='background-color:#2a2a2a; padding:20px; border-radius:10px'><h4 style='color:white;'>HistGradientBoosting Model</h4><h2 style='color:white;'>${hgb_pred:,.2f}</h2></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div style='background-color:#2a2a2a; padding:20px; border-radius:10px'><h4 style='color:white;'>Linear Regression Model</h4><h2 style='color:white;'>${lr_pred:,.2f}</h2></div>", unsafe_allow_html=True)

# --- TAB 2: Dealership Map + Bar Chart ---
with tab2:
    st.markdown("#### Sales by Region")

    region_cols = [col for col in df.columns if "Dealer_Region_" in col]
    sales_data = []
    for r in region_cols:
        total = df[df[r] == 1]["Price ($)"].sum()
        region = r.replace("Dealer_Region_", "")
        sales_data.append({"Region": region, "Total Sales": total})
    sales_df = pd.DataFrame(sales_data).sort_values("Total Sales", ascending=False)
    fig_bar = px.bar(sales_df, x="Region", y="Total Sales", title="Click a Region to Filter Map")
    selected = plotly_events(fig_bar, click_event=True, key="barclick")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("#### Dealership Locations Map")
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
    if selected:
        selected_region = selected[0]["x"]
        filtered_df = df[df[f"Dealer_Region_{selected_region}"] == 1]
    else:
        filtered_df = df
    map_fig = px.scatter_mapbox(
        filtered_df, lat="Latitude", lon="Longitude", color="Dealer_Region",
        zoom=3, mapbox_style="open-street-map", title="Dealer Map"
    )
    st.plotly_chart(map_fig, use_container_width=True)

# --- TAB 3: Market Summary with Line Chart ---
with tab3:
    st.markdown("#### Market Trends")
    region_list = sorted(df["Dealer_Region"].unique())
    selected_region = st.selectbox("Select Region", ["All"] + region_list)

    trend_df = df.copy()
    if selected_region != "All":
        trend_df = trend_df[trend_df["Dealer_Region"] == selected_region]

    trend = trend_df.groupby(["Month_Num", "Dealer_Region"])["Price ($)"].mean().reset_index()
    fig_trend = px.line(trend, x="Month_Num", y="Price ($)", color="Dealer_Region", markers=True, title="Monthly Avg Price")
    fig_trend.update_layout(plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e', font_color='white')
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("#### Summary Table")
    latest_month = df["Month_Num"].max()
    df_latest = df[df["Month_Num"] == latest_month]
    stats = df_latest.groupby("Dealer_Region")["Price ($)"].agg(["count", "mean", "sum"]).reset_index()
    stats.columns = ["Region", "Sales Count", "Avg Price", "Total Sales"]
    st.dataframe(stats.sort_values("Total Sales", ascending=False), use_container_width=True)
