import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import folium
import streamlit.components.v1 as components
import pydeck as pdk
from sklearn.linear_model import LogisticRegression

# ----------------------------
# Functions & Data Generation
# ----------------------------

@st.cache_data(show_spinner=False)
def generate_data(num_days=60):
    """
    Generate synthetic flood-related data for Chennai.
    Columns:
      - Date: Consecutive dates.
      - Rainfall_mm: Daily rainfall in millimeters.
      - River_Level_m: Simulated river level (in meters) that partly depends on rainfall.
      - Flood_Risk: Binary label (0 for low, 1 for high risk) based on simple thresholds.
    """
    base_date = datetime(2023, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(num_days)]
    np.random.seed(42)
    rainfall = np.random.randint(0, 250, size=num_days)
    river_level = 2 + rainfall * 0.02 + np.random.normal(0, 0.3, size=num_days)
    flood_risk = ((rainfall > 150) & (river_level > 5)).astype(int)
    df = pd.DataFrame({
        'Date': dates,
        'Rainfall_mm': rainfall,
        'River_Level_m': river_level,
        'Flood_Risk': flood_risk
    })
    return df

@st.cache_resource(show_spinner=False)
def train_model(df):
    """
    Train a logistic regression model using synthetic data.
    """
    X = df[['Rainfall_mm', 'River_Level_m']]
    y = df['Flood_Risk']
    model = LogisticRegression()
    model.fit(X, y)
    return model

# Synthetic flood extent polygons for different return periods.
# Each polygon is defined in [longitude, latitude] order.
flood_polygons = [
    {
      "name": "10-year Flood",
      "flood_depth": 2,  # synthetic flood depth in meters
      "elevation": 2 * 5,  # extrusion height (multiplied for visualization)
      "polygon": [
          [80.2707 - 0.01, 13.0827 - 0.01],
          [80.2707 - 0.01, 13.0827 + 0.01],
          [80.2707 + 0.01, 13.0827 + 0.01],
          [80.2707 + 0.01, 13.0827 - 0.01]
      ],
      "fillColor": [255, 255, 0]  # Yellow
    },
    {
      "name": "50-year Flood",
      "flood_depth": 4,
      "elevation": 4 * 5,
      "polygon": [
          [80.2707 - 0.02, 13.0827 - 0.02],
          [80.2707 - 0.02, 13.0827 + 0.02],
          [80.2707 + 0.02, 13.0827 + 0.02],
          [80.2707 + 0.02, 13.0827 - 0.02]
      ],
      "fillColor": [255, 165, 0]  # Orange
    },
    {
      "name": "100-year Flood",
      "flood_depth": 6,
      "elevation": 6 * 5,
      "polygon": [
          [80.2707 - 0.03, 13.0827 - 0.03],
          [80.2707 - 0.03, 13.0827 + 0.03],
          [80.2707 + 0.03, 13.0827 + 0.03],
          [80.2707 + 0.03, 13.0827 - 0.03]
      ],
      "fillColor": [255, 0, 0]  # Red
    }
]

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", 
                        ["Home", "Data & Trends", "Interactive Prediction", 
                         "Forecast Scenario", "3D Flood Extent Map", 
                         "Satellite Flood Map", "Adyar River Map"])

# ----------------------------
# Page: Home (Overview)
# ----------------------------
if page == "Home":
    st.title("Artificial Intelligence in Disaster Risk Management")
    st.markdown("""
    **Overview:**
    
    This website demonstrates how Artificial Intelligence (AI) can be used in Disaster Risk Management (DRM).  
    In particular, it shows a sample application for flood risk management in Chennai, India.
    
    **Features of this demo:**
    - **Data & Trends:** Visualizations of synthetic historical flood data (rainfall and river levels).
    - **Interactive Prediction:** Adjust conditions to predict current flood risk.
    - **Forecast Scenario:** See future flood risk forecasts with reasoning.
    - **Flood Extent Maps:** Explore two types of flood maps:
        - A **3D map** with extruded polygons representing flood extents.
        - A **Satellite map** using Esri imagery.
    - **Adyar River Map:** An interactive map of the Adyar River, a key factor in Chennai floods.
    
    Use the sidebar to navigate through the different sections.
    """)

# ----------------------------
# Page: Data & Trends
# ----------------------------
elif page == "Data & Trends":
    st.title("Data & Trends")
    df = generate_data()
    st.markdown("### Synthetic Flood Data")
    if st.checkbox("Show Dataset"):
        st.dataframe(df)
    
    st.markdown("#### Daily Rainfall (mm)")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df['Date'], df['Rainfall_mm'], marker='o', linestyle='-', color='blue')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Rainfall (mm)")
    ax1.set_title("Daily Rainfall in Chennai")
    plt.xticks(rotation=45)
    st.pyplot(fig1)
    
    st.markdown("#### Daily River Level (m)")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(df['Date'], df['River_Level_m'], marker='o', linestyle='-', color='green')
    ax2.set_xlabel("Date")
    ax2.set_ylabel("River Level (m)")
    ax2.set_title("Daily River Level in Chennai")
    plt.xticks(rotation=45)
    st.pyplot(fig2)
    
    st.markdown("#### Rainfall vs. River Level")
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    colors = df['Flood_Risk'].map({0: 'blue', 1: 'red'})
    ax3.scatter(df['Rainfall_mm'], df['River_Level_m'], c=colors, s=50, alpha=0.7)
    ax3.set_xlabel("Rainfall (mm)")
    ax3.set_ylabel("River Level (m)")
    ax3.set_title("Scatter Plot: Rainfall vs. River Level")
    ax3.grid(True)
    st.pyplot(fig3)

# ----------------------------
# Page: Interactive Prediction
# ----------------------------
elif page == "Interactive Prediction":
    st.title("Interactive Flood Risk Prediction")
    st.markdown("""
    Adjust the sliders below to simulate current conditions in Chennai and view the predicted flood risk.
    """)
    df = generate_data()
    model = train_model(df)
    
    rain_input = st.slider("Rainfall (mm)", min_value=0, max_value=250, value=100)
    river_input = st.slider("River Level (m)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    
    user_features = np.array([[rain_input, river_input]])
    prediction = model.predict(user_features)[0]
    prediction_proba = model.predict_proba(user_features)[0][1]
    risk_text = "High Flood Risk" if prediction == 1 else "Low Flood Risk"
    
    st.write(f"### Predicted Flood Risk: **{risk_text}**")
    st.write(f"Probability of High Flood Risk: **{prediction_proba:.2f}**")

# ----------------------------
# Page: Forecast Scenario
# ----------------------------
elif page == "Forecast Scenario":
    st.title("Forecast Scenario & Reasoning")
    st.markdown("""
    **Forecast Assumption:**  
    If the rainfall continues for **3 more days at 10 mm per day**, the forecasted rainfall and river levels are updated accordingly.  
    The river level is assumed to increase by approximately 0.2 m for every 10 mm increase in rainfall.
    """)
    df = generate_data()
    model = train_model(df)
    
    rain_input = st.number_input("Current Rainfall (mm)", min_value=0, max_value=250, value=100)
    river_input = st.number_input("Current River Level (m)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    
    # Forecast for current day plus 3 days
    forecast_days = [0, 1, 2, 3]
    forecast_dates = [datetime.now() + timedelta(days=d) for d in forecast_days]
    forecast_rainfall = [rain_input + 10 * d for d in forecast_days]
    forecast_river = [river_input + 0.2 * d for d in forecast_days]
    
    forecast_probas = [
        model.predict_proba(np.array([[rf, rv]]))[0][1]
        for rf, rv in zip(forecast_rainfall, forecast_river)
    ]
    
    st.markdown("#### Forecasted Values:")
    for d, rf, rv, fp in zip(forecast_dates, forecast_rainfall, forecast_river, forecast_probas):
        st.write(f"{d.strftime('%Y-%m-%d')}: Rainfall = {rf} mm, River Level = {rv:.2f} m, Flood Risk Probability = {fp:.2f}")
    
    fig4, ax4 = plt.subplots(2, 1, figsize=(10,8), sharex=True)
    ax4[0].plot(forecast_dates, forecast_rainfall, marker='o', linestyle='-', color='blue')
    ax4[0].set_ylabel("Forecast Rainfall (mm)")
    ax4[0].set_title("Forecasted Rainfall over Next 3 Days")
    ax4[0].grid(True)
    ax4[1].plot(forecast_dates, forecast_probas, marker='o', linestyle='-', color='red')
    ax4[1].set_xlabel("Date")
    ax4[1].set_ylabel("Flood Risk Probability")
    ax4[1].set_title("Forecasted Flood Risk Probability")
    ax4[1].grid(True)
    st.pyplot(fig4)
    
    st.markdown("#### Reasoning and Recommendation:")
    if forecast_probas[-1] > 0.5:
        st.markdown(
            f"*Based on the forecast, with rainfall reaching **{forecast_rainfall[-1]} mm** and river level around **{forecast_river[-1]:.2f} m**, "
            "the predicted flood risk is high. **A flood notification should be sent.***"
        )
    else:
        st.markdown(
            f"*Even with a forecasted rainfall of **{forecast_rainfall[-1]} mm** and river level around **{forecast_river[-1]:.2f} m**, "
            "the predicted flood risk remains low. No flood notification is required at this time.*"
        )

# ----------------------------
# Page: 3D Flood Extent Map
# ----------------------------
elif page == "3D Flood Extent Map":
    st.title("3D Flood Extent Map")
    st.markdown("""
    This 3D map visualizes synthetic flood extents as extruded polygons.  
    Each polygon's height is proportional to its flood depth.
    """)
    # Create a PolygonLayer for the 3D visualization.
    poly_layer = pdk.Layer(
        "PolygonLayer",
        data=flood_polygons,
        get_polygon="polygon",
        get_fill_color="fillColor",
        get_elevation="elevation",
        extruded=True,
        pickable=True,
        auto_highlight=True
    )
    view_state = pdk.ViewState(
        longitude=80.2707,
        latitude=13.0827,
        zoom=12,
        pitch=45,
        bearing=0
    )
    deck = pdk.Deck(
        layers=[poly_layer],
        initial_view_state=view_state,
        tooltip={"text": "{name}\nFlood Depth: {flood_depth} m"}
    )
    st.pydeck_chart(deck)

# ----------------------------
# Page: Satellite Flood Map
# ----------------------------
elif page == "Satellite Flood Map":
    st.title("Satellite Flood Extent Map")
    st.markdown("""
    The map below displays a synthetic satellite view of the flood extents using Esri imagery.
    """)
    # Create a Folium map with Esri satellite tiles.
    sat_map = folium.Map(
        location=[13.0827, 80.2707],
        zoom_start=12,
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri'
    )
    # Color mapping for polygons
    color_mapping = {
        "10-year Flood": "yellow",
        "50-year Flood": "orange",
        "100-year Flood": "red"
    }
    def style_function(feature):
        name = feature["properties"]["name"]
        return {
            "fillColor": color_mapping.get(name, "gray"),
            "color": color_mapping.get(name, "gray"),
            "weight": 2,
            "fillOpacity": 0.5
        }
    # Add each flood polygon as a GeoJSON feature.
    for fp in flood_polygons:
        coords = fp["polygon"] + [fp["polygon"][0]]  # Close the polygon.
        feature = {
            "type": "Feature",
            "properties": {
                "name": fp["name"],
                "flood_depth": f"{fp['flood_depth']} m"
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords]
            }
        }
        folium.GeoJson(
            feature,
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(fields=["name", "flood_depth"])
        ).add_to(sat_map)
    sat_map_html = sat_map._repr_html_()
    components.html(sat_map_html, height=600)

# ----------------------------
# Page: Adyar River Map
# ----------------------------
elif page == "Adyar River Map":
    st.title("Adyar River Map")
    st.markdown("""
    The Adyar River is one of the main contributors to flooding in Chennai.  
    Below is an interactive map showing a synthetic course of the Adyar River.
    """)
    # Synthetic coordinates for the Adyar River.
    adyar_coords = [
        [13.0500, 80.2300],
        [13.0550, 80.2350],
        [13.0600, 80.2400],
        [13.0650, 80.2450],
        [13.0700, 80.2500],
        [13.0750, 80.2550],
        [13.0800, 80.2600],
        [13.0850, 80.2650],
        [13.0900, 80.2700],
        [13.0950, 80.2750]
    ]
    map_center = [13.0725, 80.2525]
    river_map = folium.Map(location=map_center, zoom_start=12)
    folium.PolyLine(adyar_coords, color="blue", weight=5, opacity=0.7, tooltip="Adyar River").add_to(river_map)
    folium.Marker(adyar_coords[0], popup="River Source", icon=folium.Icon(color="green")).add_to(river_map)
    folium.Marker(adyar_coords[-1], popup="River Mouth", icon=folium.Icon(color="red")).add_to(river_map)
    river_map_html = river_map._repr_html_()
    components.html(river_map_html, height=500)
