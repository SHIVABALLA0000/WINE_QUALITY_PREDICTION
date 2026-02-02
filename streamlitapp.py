import streamlit as st
import requests
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
API_URL = "http://127.0.0.1:8000/predict"
API_KEY = "my-secret-api-key"

HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="centered"
)

# -----------------------------
# HEADER
# -----------------------------
st.title("Wine Quality Prediction System")
st.markdown(
    """
This application predicts **wine quality** using a machine learning model.
The frontend is built with **Streamlit** and calls a **FastAPI inference service**
that handles preprocessing, confidence scoring, and prediction.
"""
)

st.divider()

# -----------------------------
# INPUT FORM
# -----------------------------
st.subheader("Wine Chemical Properties")

with st.form("wine_form"):

    col1, col2 = st.columns(2)

    with col1:
        fixed_acidity = st.number_input("Fixed Acidity", 0.0, 20.0, 7.4)
        volatile_acidity = st.number_input("Volatile Acidity", 0.0, 2.0, 0.7)
        citric_acid = st.number_input("Citric Acid", 0.0, 2.0, 0.0)
        residual_sugar = st.number_input("Residual Sugar", 0.0, 50.0, 1.9)
        chlorides = st.number_input("Chlorides", 0.0, 1.0, 0.076)

    with col2:
        free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 0, 300, 11)
        total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", 0, 500, 34)
        density = st.number_input("Density", 0.9900, 1.0050, 0.9978, format="%.4f")
        pH = st.number_input("pH", 2.0, 4.5, 3.51)
        sulphates = st.number_input("Sulphates", 0.0, 2.0, 0.56)
        alcohol = st.number_input("Alcohol (%)", 5.0, 20.0, 9.4)

    wine_type = st.selectbox("Wine Type", ["red", "white"])

    submitted = st.form_submit_button("Predict Wine Quality")

# -----------------------------
# PREDICTION
# -----------------------------
if submitted:

    payload = {
        "fixed_acidity": fixed_acidity,
        "volatile_acidity": volatile_acidity,
        "citric_acid": citric_acid,
        "residual_sugar": residual_sugar,
        "chlorides": chlorides,
        "free_sulfur_dioxide": free_sulfur_dioxide,
        "total_sulfur_dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol,
        "wine_type": wine_type
    }

    with st.spinner("Sending data to inference service..."):
        response = requests.post(API_URL, json=payload, headers=HEADERS)

    st.divider()
    st.subheader("Prediction Result")

    if response.status_code == 200:
        result = response.json()

        if result.get("status") == "rejected":
            st.error(f"Prediction Rejected: {result['reason']}")

        else:
            predicted_quality = result["predicted_quality"]
            confidence = result["confidence"]

            if confidence >= 0.80:
                st.success(f"Predicted Wine Quality: **{predicted_quality}**")
            elif confidence >= 0.65:
                st.warning(f"Predicted Wine Quality: **{predicted_quality}**")
            else:
                st.error(f"Low Confidence Prediction: **{predicted_quality}**")

            st.markdown(f"**Confidence Score:** `{confidence:.2f}`")

            # Probability chart
            probs = result["class_probabilities"]
            prob_df = pd.DataFrame.from_dict(
                probs, orient="index", columns=["Probability"]
            ).sort_index()

            st.subheader("Class Probability Distribution")
            st.bar_chart(prob_df)

    else:
        st.error("API Error")
        st.code(response.text)
