import streamlit as st
import pandas as pd
import joblib
import os

# Load model
model_path = os.path.join(os.path.dirname(__file__), "..", "rf_model.pkl")
model = joblib.load(model_path)

st.title("🛒 Customer Conversion Predictor")

# Inputs
month = st.selectbox("Month", [4,5,6,7,8])
day = st.slider("Day", 1, 31, 15)
country = st.number_input("Country (1-47)", 1, 47, 1)
page1_main_category = st.number_input("Main Category (1-4)", 1, 4, 1)
page2_clothing_model = st.number_input("Unique Clothing Models", 1, 50, 5)
colour = st.number_input("Unique Colours", 1, 14, 2)
location = st.number_input("Unique Locations", 1, 6, 1)
model_photography = st.radio("Model Photography", [1, 2])
price_mean = st.number_input("Avg Price Seen", 1, 100, 30)
price_max = st.number_input("Max Price Seen", 1, 100, 50)
price_2_mean = st.number_input("Avg Price_2", 1, 3, 1)
page_max = st.number_input("Last Page Reached", 1, 5, 3)

# Predict button
if st.button("Predict Conversion"):
    input_data = pd.DataFrame([{
        "month_first": month,
        "day_first": day,
        "country_first": country,
        "page1_main_category_agg": page1_main_category,
        "page2_clothing_model_agg": page2_clothing_model,
        "colour_agg": colour,
        "location_agg": location,
        "model_photography_agg": model_photography,
        "price_mean": price_mean,
        "price_max": price_max,
        "price_2_mean": price_2_mean,
        "page_max": page_max
    }])

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if pred == 1:
        st.success(f"✅ Likely to Convert (Probability: {prob:.2f})")
    else:
        st.error(f"❌ Unlikely to Convert (Probability: {prob:.2f})")
