import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from streamlit_plotly_events import plotly_events

st.set_page_config(
    page_title="Car Sales Forecasting Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {
    background-color: #1e1e1e;
    color: white;
}
.sidebar .sidebar-content {
    background-color: #2f2f2f;
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
            st.warning(f"Column reconstruction skipped: No columns found with prefix '{prefix}'")
        return df

    df = safe_reconstruct_column(df, "Dealer_Region_", "Dealer_Region")
    df = safe_reconstruct_column(df, "Body Style_", "Body Style")
    df = safe_reconstruct_column(df, "Transmission_", "Transmission")
    # You can re-enable this if your dataset has Price_Category_
    # df = safe_reconstruct_column(df, "Price_Category_", "Price Category")

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

# ---------------------- TAB 1 ----------------------
with tab1:
    st.markdown("<h3 style='color:white;'>Price Predictions</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color:white;'>This tool uses two machine learning models to estimate car prices based on the selected features.</p>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Customize Car Features")
        month = st.slider("Month of Sale", 1, 12, 6)
        car_age = st.slider("Car Age", 0, 20, 5)
        income = st.slider("Annual Income", 20000, 200000, 75000)

        region = st.selectbox("Dealer Region", sorted(df["Dealer_Region"].unique()))
        body_style = st.selectbox("Body Style", sorted(df["Body Style"].unique()))
        transmission = st.selectbox("Transmission", sorted(df["Transmission"].unique()))

    input_data = {col: 0 for col in hgb_model.feature_names_in_}
    input_data["Month_Num"] = month
    input_data["Car_Age"] = car_age
    input_data["Annual Income"] = income

    categorical_prefixes = {
        "Dealer_Region": region,
        "Body Style": body_style,
        "Transmission": transmission
    }

    for prefix, value in categorical_prefixes.items():
        col_name = f"{prefix}_{value}"
        if col_name in input_data:
            input_data[col_name] = 1

    input_df = pd.DataFrame([input_data])

    hgb_pred = hgb_model.predict(input_df)[0]
    lr_pred = lr_model.predict(input_df)[0]

    st.markdown("### <span style='color:white;'>Model Predictions</span>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
            <div style='background-color:#2a2a2a; padding:20px; border-radius:10px'>
                <h4 style='color:white;'>HistGradientBoosting Model</h4>
                <h2 style='color:white;'>${hgb_pred:,.2f}</h2>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div style='background-color:#2a2a2a; padding:20px; border-radius:10px'>
                <h4 style='color:white;'>Linear Regression Baseline</h4>
                <h2 style='color:white;'>${lr_pred:,.2f}</h2>
            </div>
        """, unsafe_allow_html=True)

# ---------------------- TAB 2 ----------------------
with tab2:
    st.markdown("## Dealer Sales Overview")

    region_cols = [col for col in df.columns if "Dealer_Region_" in col]
    region_sales = df[region_cols].sum().sort_values(ascending=False).reset_index()
    region_sales.columns = ["Region_Encoded", "Total Sales"]
    region_sales["Region"] = region_sales["Region_Encoded"].str.replace("Dealer_Region_", "")

    fig_bar = px.bar(region_sales, x="Region", y="Total Sales", title="Click a Region to Filter the Map")
    selected_region = plotly_events(fig_bar, click_event=True, key="bar")

    st.markdown("### Dealership Locations Map")
    if selected_region:
        selected = selected_region[0]["x"]
        filtered_df = df[df["Dealer_Region"] == selected]
    else:
        filtered_df = df

    fig_map = px.scatter_mapbox(
        filtered_df, lat="Latitude", lon="Longitude", color="Dealer_Region",
        hover_name="Dealer_Region", zoom=3, mapbox_style="open-street-map",
        title="Dealerships by Region"
    )
    st.plotly_chart(fig_map, use_container_width=True)

# ---------------------- TAB 3 ----------------------
with tab3:
    st.markdown("## Market Trends")
    st.markdown("Explore how car prices vary across different body styles, regions, and months.")

    df["Body Style"] = df["Body Style"].str.strip().str.title()
    df["Dealer_Region"] = df["Dealer_Region"].str.strip().str.title()

    col1, col2, col3 = st.columns(3)
    with col1:
        body_val = st.selectbox("Filter by Body Style", ["All"] + sorted(df["Body Style"].dropna().unique()))
    with col2:
        region_val = st.selectbox("Filter by Region", ["All"] + sorted(df["Dealer_Region"].dropna().unique()))
    with col3:
        month_val = st.selectbox("Filter by Month", ["All"] + sorted(df["Month_Num"].dropna().unique()))

    df_filtered = df.copy()
    if body_val != "All":
        df_filtered = df_filtered[df_filtered["Body Style"] == body_val]
    if region_val != "All":
        df_filtered = df_filtered[df_filtered["Dealer_Region"] == region_val]
    if month_val != "All":
        df_filtered = df_filtered[df_filtered["Month_Num"] == int(month_val)]

    all_months = pd.DataFrame({"Month_Num": range(1, 13)})
    trend_data = df_filtered.groupby("Month_Num")["Price ($)"].sum().reset_index()
    trend_data = pd.merge(all_months, trend_data, on="Month_Num", how="left").fillna(0)

    fig_trend = px.line(trend_data, x="Month_Num", y="Price ($)", markers=True, title="Monthly Sales Trend")
    fig_trend.update_layout(plot_bgcolor='#1e1e1e', paper_bgcolor='#1e1e1e', font_color='white')
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("### Summary Statistics (Filtered Data)")
    summary_stats = df_filtered[["Price ($)", "Annual Income"]].describe().T
    summary_stats.rename(index={"Price ($)": "Price", "Annual Income": "Income"}, inplace=True)
    st.dataframe(summary_stats.style.format(precision=2))
