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
        transmission_options = sorted([col.replace("Transmission_", "") for col in hgb_model.feature_names_in_ if col.startswith("Transmission_")])
        transmission = st.selectbox("Transmission", transmission_options)

        color_options = sorted([col.replace("Color_", "") for col in hgb_model.feature_names_in_ if col.startswith("Color_")])
        color = st.selectbox("Color", color_options)

        body_style_options = sorted([col.replace("Body Style_", "") for col in hgb_model.feature_names_in_ if col.startswith("Body Style_")])
        body_style = st.selectbox("Body Style", body_style_options)

        region_options = sorted([col.replace("Dealer_Region_", "") for col in hgb_model.feature_names_in_ if col.startswith("Dealer_Region_")])
        region = st.selectbox("Dealer Region", region_options)

    # Safe feature construction
    feature_list = list(hgb_model.feature_names_in_)
    input_data = {col: 0 for col in feature_list}
    input_data["Annual Income"] = income
    input_data["Car_Age"] = car_age
    input_data["Month_Num"] = month

    # Set one-hot encoded flags only if feature exists
    for prefix, value in {
        "Transmission": transmission,
        "Color": color,
        "Body Style": body_style,
        "Dealer_Region": region
    }.items():
        col_name = f"{prefix}_{value}"
        if col_name in input_data:
            input_data[col_name] = 1

    input_df = pd.DataFrame([input_data])

    # DEBUG: Show the input and model columns
    with st.expander("🛠️ Debug Info"):
        st.write("Input DataFrame Sent to Model:")
        st.dataframe(input_df)
        st.write("Expected Model Columns:")
        st.write(feature_list)

    # Predict
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
    available_body_styles = sorted([col.replace("Body Style_", "") for col in df.columns if col.startswith("Body Style_")])
    selected_body = st.selectbox("Filter by Body Style", ["All"] + available_body_styles)
    filtered = df.copy()
    if selected_body != "All":
        filtered = filtered[filtered[f"Body Style_{selected_body}"] == 1]

    trend = filtered.groupby("Month_Num")["Price ($)"].sum().reset_index()
    fig_trend = px.line(trend, x="Month_Num", y="Price ($)", markers=True, title="Monthly Sales Trend")
    st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(filtered[["Price ($)", "Annual Income", "Car_Age"]].describe().T)
