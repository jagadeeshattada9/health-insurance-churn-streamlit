import streamlit as st
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# Load saved models
scaler = pickle.load(open("../saved_models/scaler.pkl", "rb"))
features = pickle.load(open("../saved_models/features.pkl", "rb"))
encoder = load_model("../saved_models/encoder.h5")
model = load_model("../saved_models/churn_model.h5")

st.title("Health Insurance Customer Churn Prediction")

st.write("Enter customer details to predict churn")

# -------- USER INPUT --------
company = st.selectbox("Company Name", ["Aetna", "MaxLife", "StarHealth", "ICICI", "HDFC"])
claim_reason = st.selectbox("Claim Reason", ["Accident", "Surgery", "Routine", "Critical Illness"])
conf = st.selectbox("Data Confidentiality", ["Yes", "No"])
claim_amount = st.number_input("Claim Amount", min_value=0)
premium = st.selectbox("Category Premium", ["Silver", "Gold", "Platinum"])
ratio = st.number_input("Premium / Amount Ratio", min_value=0.0)
claim_out = st.selectbox("Claim Request Output", ["Approved", "Rejected"])
bmi = st.number_input("BMI", min_value=10.0)

# -------- ENCODING --------
company_map = {"Aetna":0, "MaxLife":1, "StarHealth":2, "ICICI":3, "HDFC":4}
claim_map = {"Accident":0, "Surgery":1, "Routine":2, "Critical Illness":3}
conf_map = {"No":0, "Yes":1}
premium_map = {"Silver":0, "Gold":1, "Platinum":2}
claim_out_map = {"Rejected":0, "Approved":1}

input_data = {
    "Company Name": company_map[company],
    "Claim Reason": claim_map[claim_reason],
    "Data confidentiality": conf_map[conf],
    "Claim Amount": claim_amount,
    "Category Premium": premium_map[premium],
    "Premium/Amount Ratio": ratio,
    "Claim Request output": claim_out_map[claim_out],
    "BMI": bmi
}

df = pd.DataFrame([input_data])

# Align with training features
df = df.reindex(columns=features, fill_value=0)

# -------- PREDICTION --------
if st.button("Predict Churn"):
    scaled = scaler.transform(df)
    encoded = encoder.predict(scaled)
    pred = model.predict(encoded)[0][0]

    if pred > 0.5:
        st.error(f"❌ Customer WILL CHURN (Probability: {pred:.2f})")
    else:
        st.success(f"✅ Customer will NOT churn (Probability: {pred:.2f})")
