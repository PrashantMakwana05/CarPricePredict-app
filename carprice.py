import streamlit as st
import pandas as pd
import joblib
from babel.numbers import format_currency

# Load model and reference data
@st.cache_resource
def load_model_and_data():
    model = joblib.load('car_price_model.pkl')
    reference_data = pd.read_csv('reference_data.csv')
    return model, reference_data

model, reference_data = load_model_and_data()

st.set_page_config(page_title="Car Price Predictor")
st.title("🚗 Used Car Price Prediction")
st.write("Enter car details below to estimate the price.")

# Dropdown options
brands = sorted(reference_data['Brand'].unique())
models = sorted(reference_data['model'].unique())
transmissions = sorted(reference_data['Transmission'].unique())
owners = sorted(reference_data['Owner'].unique())
fuel_types = sorted(reference_data['FuelType'].unique())

# --- Input form ---
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        selected_brand = st.selectbox("Brand", brands)
        selected_model = st.selectbox("Model", models)
        selected_transmission = st.selectbox("Transmission", transmissions)

    with col2:
        selected_owner = st.selectbox("Owner Type", owners)
        selected_fuel = st.selectbox("Fuel Type", fuel_types)
        year = st.number_input("Year", min_value=1990, max_value=2025, value=2015, step=1)

    km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000, step=1000)

    submitted = st.form_submit_button("Predict Price")

if submitted:
    user_input = pd.DataFrame([{
        'Brand': selected_brand,
        'model': selected_model,
        'Transmission': selected_transmission,
        'Owner': selected_owner,
        'FuelType': selected_fuel,
        'Year': year,
        'kmDriven': km_driven
    }])

    try:
        prediction = model.predict(user_input)[0]
        indian_price = format_currency(prediction, 'INR', locale='en_IN')
        st.success(f"💰 Estimated Car Price: {indian_price}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
