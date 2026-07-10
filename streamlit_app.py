
import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

st.title("Insurance Premium Category Predictor")
st.markdown("Enter your details below to get your premium category prediction:")

age = st.number_input("Age", min_value=1, max_value=119, value=30)
weight = st.number_input("Weight (kg)", min_value=1.0, value=65.0)
height = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.7)
income_lpa = st.number_input("Annual Income (LPA)", min_value=0.1, value=10.0)
smoker = st.selectbox("Are you a smoker?", options=[False, True])
city = st.text_input("City", value="Mumbai")
occupation = st.selectbox(
    "Occupation",
    options=[
        "retired", "freelancer", "student", "government_job",
        "business_owner", "unemployed", "private_job"
    ]
)

# Prediction button
if st.button("Predict Premium Category"):
    input_data = {
        "age": age,
        "weight": weight,
        "height": height,
        "income_lpa": income_lpa,
        "smoker": smoker,
        "city": city,
        "occupation": occupation
    }

    try:
        # Send request to API
        response = requests.post(API_URL, json=input_data)
        result = response.json()

        if response.status_code == 200:
            prediction = result["response"]
            st.success(f"Predicted Insurance Premium Category: **{prediction['predicted_category']}**")

            st.write(f"Confidence: {prediction['confidence']:.2%}")

            st.subheader("Class Probabilities")
            st.bar_chart(prediction["class_probabilities"])
        else:
            st.error(f"API Error: {response.status_code}")
            st.write(result)

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the FastAPI server. Make sure it's running on "
                  "http://localhost:8000 (uvicorn app:app --reload --port 8000).")