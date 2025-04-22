import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Car Sales Dashboard", layout="wide")

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

tab1, tab2, tab3 = st.tabs(["Price Prediction", "Dealership Map", "Market Trends"])

# --- TAB 1: Price Prediction ---
with tab1:
    st.header("💰 Predict Car Price")

    col1, col2 = st.columns(2)
    with col1:
        income = st.slider("Annual Income", 20000, 200000, 60000)
        car_age = st.slider("Car Age", 0, 20, 5)
        month = st.slider("Month of Sale", 1, 12, 6)
    
    with col2:
        transmission = st.selectbox("Transmission", ["Manual"])
        color = st.selectbox("Color", ["Pale White", "Red"])
        body_style = st.selectbox("Body Style", ["Hatchback", "Passenger", "SUV", "Sedan"])
        region = st.selectbox("Dealer Region", ["Austin", "Greenville", "Janesville", "Middletown", "Pasco", "Scottsdale"])

    # Initialize input row with zero
    input_dict = {col: 0 for col in hgb_model.feature_names_in_}
    input_dict["Annual Income"] = income
    input_dict["Car_Age"] = car_age
    input_dict["Month_Num"] = month
    input_dict[f"Transmission_Manual"] = 1 if transmission == "Manual" else 0
    input_dict[f"Color_Pale White"] = 1 if color == "Pale White" else 0
    input_dict[f"Color_Red"] = 1 if color == "Red" else 0
    input_dict[f"Body Style_{body_style}"] = 1
    input_dict[f"Dealer_Region_{region}"] = 1

    input_df = pd.DataFrame([input_dict])
    hgb_pred = hgb_model.predict(input_df)[0]
    lr_pred = lr_model.predict(input_df)[0]

    st.subheader("Predicted Prices")
    st.metric("HistGradientBoosting", f"${hgb_pred:,.2f}")
    st.metric("Linear Regression", f"${lr_pred:,.2f}")

# --- TAB 2: Dealership Map ---
with tab2:
    st.header("🗺️ Dealership Locations")
    region_coords = {
        "Austin": (30.2672, -97.7431),
        "Greenville": (34.8526, -82.3940),
        "Janesville": (42.6828, -89.0187),
        "Middletown": (39.5151, -84.3983),
        "Pasco": (46.2396, -119.1006),
        "Scottsdale": (33.4942, -111.9261),
    }

    coords_df = pd.DataFrame([
        {"Region": region, "Lat": lat, "Lon": lon}
        for region, (lat, lon) in region_coords.items()
    ])

    sales_data = []
    for region in region_coords:
        mask = df[f"Dealer_Region_{region}"] == 1
        total = df[mask]["Price ($)"].sum()
        sales_data.append({"Region": region, "Total Sales": total})

    sales_df = pd.DataFrame(sales_data)
    map_df = coords_df.merge(sales_df, on="Region")

    fig = px.scatter_mapbox(
        map_df,
        lat="Lat",
        lon="Lon",
        size="Total Sales",
        color="Region",
        mapbox_style="open-street-map",
        zoom=3,
        title="Dealership Locations and Sales"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: Market Trends ---
with tab3:
    st.header("📈 Market Trends")
    selected_body = st.selectbox("Filter by Body Style", ["All"] + ["Hatchback", "Passenger", "SUV", "Sedan"])
    filtered = df.copy()
    if selected_body != "All":
        filtered = filtered[filtered[f"Body Style_{selected_body}"] == 1]

    trend = filtered.groupby("Month_Num")["Price ($)"].sum().reset_index()
    fig_trend = px.line(trend, x="Month_Num", y="Price ($)", markers=True, title="Monthly Sales Trend")
    st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(filtered[["Price ($)", "Annual Income", "Car_Age"]].describe().T)
