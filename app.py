import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px

# Page configuration
st.set_page_config(page_title="California House Price Predictor",
                   page_icon="🏡",
                   layout="wide")

# Load model
model = joblib.load("house_price_model.pkl")

# Custom CSS for modern look
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.stButton>button {
    background-color: #00ADB5;
    color: white;
    font-weight: bold;
}
.big-font {
    font-size:30px !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("🏡 California House Price Predictor")
st.markdown("### AI-Powered Real Estate Valuation using XGBoost")

st.divider()

# Layout in columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Enter Property Details")

    MedInc = st.slider("Median Income", 0.0, 15.0, 5.0)
    HouseAge = st.slider("House Age", 1, 60, 20)
    AveRooms = st.slider("Average Rooms", 1.0, 15.0, 5.0)
    AveBedrms = st.slider("Average Bedrooms", 0.5, 5.0, 1.0)
    Population = st.slider("Population", 1, 10000, 1000)
    AveOccup = st.slider("Average Occupancy", 0.5, 10.0, 3.0)
    Latitude = st.slider("Latitude", 32.0, 42.0, 37.0)
    Longitude = st.slider("Longitude", -125.0, -114.0, -120.0)

    input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                            Population, AveOccup, Latitude, Longitude]])

    if st.button("🔮 Predict Price"):
        prediction = model.predict(input_data)[0]
        st.success(f"💰 Estimated House Price: **${prediction * 100000:,.0f}**")

with col2:
    st.subheader("📊 Model Feature Importance")

    feature_names = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup", "Latitude", "Longitude"
    ]

    importances = model.feature_importances_

    df_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=True)

    fig = px.bar(df_importance,
                 x="Importance",
                 y="Feature",
                 orientation='h',
                 title="What Influences Price the Most?",
                 color="Importance",
                 color_continuous_scale="teal")

    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Built with ❤️ using Streamlit & XGBoost")
