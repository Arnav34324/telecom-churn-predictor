import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

# ─────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="centered"
)

# ─────────────────────────────────────────────
# Load the saved XGBoost model and feature list
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("xgboost_churn_model.pkl")   # ← updated to XGBoost
    features = joblib.load("model_features.pkl")
    return model, features

model, feature_columns = load_model()

# ─────────────────────────────────────────────
# SHARED HELPER FUNCTION
# Core prediction logic reused by both tabs.
# Takes a raw DataFrame, encodes it, aligns columns, returns probabilities.
# ─────────────────────────────────────────────
def predict_churn(raw_df):
    categorical_cols = raw_df.select_dtypes(include=["object"]).columns
    encoded = pd.get_dummies(raw_df, columns=categorical_cols, drop_first=True)
    aligned = encoded.reindex(columns=feature_columns, fill_value=0)
    probabilities = model.predict_proba(aligned)[:, 1]
    return probabilities


# ─────────────────────────────────────────────
# APP TITLE
# ─────────────────────────────────────────────
st.title("📡 Telco Customer Churn Predictor")
st.markdown("""
This app uses an **XGBoost AI model** (GridSearch tuned, ROC-AUC: 0.8526)  
trained on 7,032 real telecom customers to predict whether a customer  
is likely to leave (churn).
""")
st.divider()

# ─────────────────────────────────────────────
# TWO TABS
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Single Customer Prediction", "📂 Batch Prediction (CSV Upload)"])


# ═══════════════════════════════════════════════════════════════
# TAB 1 — SINGLE CUSTOMER PREDICTOR
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("📋 Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.slider("Tenure (months with company)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)
        senior_citizen = st.selectbox("Senior Citizen?", ["No", "Yes"])
        partner = st.selectbox("Has a Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
        phone_service = st.selectbox("Phone Service?", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines?", ["No", "Yes", "No phone service"])

    with col2:
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online Security?", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup?", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection?", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support?", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV?", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies?", ["No", "Yes", "No internet service"])

    st.subheader("💳 Billing & Contract")
    col3, col4 = st.columns(2)

    with col3:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing?", ["Yes", "No"])

    with col4:
        payment_method = st.selectbox("Payment Method", [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ])
        gender = st.selectbox("Gender", ["Male", "Female"])

    st.divider()

    if st.button("🔮 Predict Churn Risk", use_container_width=True, type="primary"):

        raw_input = {
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "gender": gender,
            "Partner": partner,
            "Dependents": dependents,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
        }

        input_df = pd.DataFrame([raw_input])
        churn_prob = predict_churn(input_df)[0]
        churn_risk_percent = round(churn_prob * 100, 1)

        st.subheader("🎯 Prediction Result")

        if churn_prob >= 0.3:
            st.error(f"⚠️ **HIGH CHURN RISK — {churn_risk_percent}% probability of leaving**")
            st.markdown("""
            **Recommended Actions:**
            - Offer a discounted upgrade to a 1 or 2-year contract
            - Proactively reach out with a personalised retention call
            - Consider offering free add-ons (Tech Support, Online Security)
            """)
        else:
            st.success(f"✅ **LOW CHURN RISK — {churn_risk_percent}% probability of leaving**")
            st.markdown("This customer appears stable. No immediate action needed.")

        st.markdown(f"**Churn Probability: {churn_risk_percent}%**")
        st.progress(float(churn_prob))

        st.info(f"""
        📊 **Key factors the AI considered most:**
        - Tenure: {tenure} months (longer = more loyal)
        - Monthly Charges: ${monthly_charges} (higher = higher risk)
        - Contract Type: {contract} (month-to-month = highest risk)
        - Internet Service: {internet_service} (Fiber optic users churn more)
        """)


# ═══════════════════════════════════════════════════════════════
# TAB 2 — BATCH PREDICTION
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📂 Upload a Customer CSV File")
    st.markdown("""
    Upload a CSV file containing multiple customers. The app will predict churn 
    risk for every row and give you back a downloadable file with the results.
    
    **Your CSV must contain these columns** (same names as the original Telco dataset):  
    `tenure`, `MonthlyCharges`, `SeniorCitizen`, `gender`, `Partner`, `Dependents`,
    `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`,
    `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, 
    `Contract`, `PaperlessBilling`, `PaymentMethod`
    """)

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df_batch = pd.read_csv(uploaded_file)

        st.markdown(f"**✅ File uploaded successfully — {len(df_batch)} customers found.**")
        st.markdown("**Preview of uploaded data (first 5 rows):**")
        st.dataframe(df_batch.head())

        if st.button("🔮 Predict Churn for All Customers", use_container_width=True, type="primary"):

            # Drop columns that weren't in training
            if 'customerID' in df_batch.columns:
                df_batch = df_batch.drop('customerID', axis=1)
            if 'Churn' in df_batch.columns:
                df_batch = df_batch.drop('Churn', axis=1)
            if 'TotalCharges' in df_batch.columns:
                df_batch['TotalCharges'] = pd.to_numeric(df_batch['TotalCharges'], errors='coerce')
                df_batch = df_batch.drop('TotalCharges', axis=1)

            # Convert SeniorCitizen to numeric if it came in as Yes/No text
            if df_batch['SeniorCitizen'].dtype == object:
                df_batch['SeniorCitizen'] = df_batch['SeniorCitizen'].apply(
                    lambda x: 1 if x == 'Yes' else 0
                )

            # Run predictions
            probabilities = predict_churn(df_batch)

            # Add results as new columns
            df_results = df_batch.copy()
            df_results['Churn_Probability_%'] = (probabilities * 100).round(1)
            df_results['Churn_Risk_Label'] = df_results['Churn_Probability_%'].apply(
                lambda p: 'HIGH RISK' if p >= 30 else 'LOW RISK'
            )

            # Sort highest risk first — retention team sees urgent cases immediately
            df_results = df_results.sort_values('Churn_Probability_%', ascending=False)

            st.divider()
            st.subheader("📊 Batch Prediction Results")

            high_risk_count = (df_results['Churn_Risk_Label'] == 'HIGH RISK').sum()
            low_risk_count = (df_results['Churn_Risk_Label'] == 'LOW RISK').sum()
            avg_churn_prob = df_results['Churn_Probability_%'].mean()

            m1, m2, m3 = st.columns(3)
            m1.metric("🔴 High Risk Customers", high_risk_count)
            m2.metric("🟢 Low Risk Customers", low_risk_count)
            m3.metric("📈 Avg Churn Probability", f"{avg_churn_prob:.1f}%")

            st.markdown("**Full results (sorted by highest churn risk first):**")
            result_cols = ['Churn_Probability_%', 'Churn_Risk_Label'] + [
                c for c in df_results.columns
                if c not in ['Churn_Probability_%', 'Churn_Risk_Label']
            ]
            st.dataframe(df_results[result_cols])

            # Download button — results as CSV
            csv_buffer = io.BytesIO()
            df_results.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)

            st.download_button(
                label="⬇️ Download Results as CSV",
                data=csv_buffer,
                file_name="churn_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.success("Done! Your results are ready to download and share with your retention team.")

st.divider()
st.caption("Model: XGBoost (GridSearch Tuned) | ROC-AUC: 0.8526 | Trained on 7,032 Telco customers | Threshold: 0.30")
