# Streamlit App Setup
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide"
)

st.title("📉 Customer Churn Prediction & Risk Analysis")
st.write("Predict customer churn risk using a trained Machine Learning model.")

# Load model, scaler, and data
model = joblib.load("../models/logistic_model.pkl")
scaler = joblib.load("../models/scaler.pkl")
data = pd.read_csv("../data/processed_churn_data.csv")

X_columns = data.drop("Churn", axis=1).columns

# Sidebar Inputs
st.sidebar.header("Enter Customer Details")

def user_input():
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.sidebar.slider("Monthly Charges", 20, 120, 70)
    total_charges = tenure * monthly_charges

    contract_type = st.sidebar.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

    payment_method = st.sidebar.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    input_dict = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract_One year": 1 if contract_type == "One year" else 0,
        "Contract_Two year": 1 if contract_type == "Two year" else 0,
        "PaymentMethod_Electronic check": 1 if payment_method == "Electronic check" else 0,
    }

    return input_dict

# Prepare input
user_data = user_input()
input_df = pd.DataFrame([user_data])

# Add missing columns
for col in X_columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[X_columns]


st.subheader("🔍 Prediction Result")

if st.button("Predict Churn Risk"):
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    probability = model.predict_proba(input_df)[0][1]

    if probability < 0.45:
        st.success(f"🟢 Low Churn Risk (Score: {probability:.2f})")
        st.write("👉 Customer is stable. Maintain service quality.")

    elif probability < 0.6:
        st.warning(f"🟡 Medium Churn Risk (Score: {probability:.2f})")
        st.write("👉 Monitor closely and apply soft retention strategies.")

    else:
        st.error(f"🔴 High Churn Risk (Score: {probability:.2f})")
        st.write("👉 Immediate retention action recommended.")


