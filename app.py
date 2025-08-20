import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open("model/stress_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Stress Detection ML App")

# Example input form
user_input = st.text_input("Enter your stress-related data (comma-separated features):")

if st.button("Predict"):
    if user_input:
        # Convert input to a DataFrame (modify as per your features)
        input_data = [float(x) for x in user_input.split(",")]
        df = pd.DataFrame([input_data])
        prediction = model.predict(df)
        st.success(f"Predicted Stress Level: {prediction[0]}")
    else:
        st.warning("Please enter input data.")
