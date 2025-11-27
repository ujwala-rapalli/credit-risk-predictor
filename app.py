# app.py
"""
Streamlit app for Credit Risk Predictor.
Expecting a saved pipeline 'model.pkl' (joblib) in the same folder.
The pipeline should accept the following feature columns in this exact order:
['Credit_Limit','Gender_Code','Education_Level','Marital_Status','Age_Years',
 'Repay_Sep','Repay_Aug','Repay_Jul','Repay_Jun','Repay_May','Repay_Apr',
 'BillAmt_Sep','BillAmt_Aug','BillAmt_Jul','BillAmt_Jun','BillAmt_May','BillAmt_Apr',
 'PaidAmt_Sep','PaidAmt_Aug','PaidAmt_Jul','PaidAmt_Jun','PaidAmt_May','PaidAmt_Apr']
If your pipeline includes preprocessing (recommended), it will handle scaling/encoding.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

st.set_page_config(page_title="Credit Risk Predictor", layout="wide")
st.title("Credit Risk Predictor — Streamlit Demo")
st.markdown("This app predicts whether an applicant will default next month. The model pipeline must be saved as `model.pkl` in the same folder as this `app.py`.")

# -------------------------
# Load model
# -------------------------
@st.cache_resource
def load_model(path="model.pkl"):
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        return None

model = load_model()

if model is None:
    st.error("model.pkl not found in the app folder. Save your trained pipeline as 'model.pkl' and place it here.")
    st.stop()

# exact feature order expected by the original training pipeline
FEATURES = [
    "Credit_Limit","Gender_Code","Education_Level","Marital_Status","Age_Years",
    "Repay_Sep","Repay_Aug","Repay_Jul","Repay_Jun","Repay_May","Repay_Apr",
    "BillAmt_Sep","BillAmt_Aug","BillAmt_Jul","BillAmt_Jun","BillAmt_May","BillAmt_Apr",
    "PaidAmt_Sep","PaidAmt_Aug","PaidAmt_Jul","PaidAmt_Jun","PaidAmt_May","PaidAmt_Apr"
]

# -------------------------
# Single input form
# -------------------------
st.header("Single Applicant Prediction")
st.info("Fill the fields below. Use the same units/encodings as used during model training (e.g., Gender_Code 0/1).")

with st.form("single_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        Credit_Limit = st.number_input("Credit_Limit", min_value=0.0, value=50000.0, step=1000.0)
        Gender_Code = st.number_input("Gender_Code (e.g., 0/1)", min_value=0, max_value=10, value=1, step=1)
        Education_Level = st.number_input("Education_Level (encoded)", min_value=0, max_value=10, value=2, step=1)
        Marital_Status = st.number_input("Marital_Status (encoded)", min_value=0, max_value=10, value=1, step=1)
        Age_Years = st.number_input("Age_Years", min_value=18, max_value=120, value=30, step=1)

    with col2:
        Repay_Sep = st.number_input("Repay_Sep", value=0.0, step=1.0)
        Repay_Aug = st.number_input("Repay_Aug", value=0.0, step=1.0)
        Repay_Jul = st.number_input("Repay_Jul", value=0.0, step=1.0)
        Repay_Jun = st.number_input("Repay_Jun", value=0.0, step=1.0)
        Repay_May = st.number_input("Repay_May", value=0.0, step=1.0)

    with col3:
        Repay_Apr = st.number_input("Repay_Apr", value=0.0, step=1.0)
        BillAmt_Sep = st.number_input("BillAmt_Sep", min_value=0.0, value=10000.0, step=500.0)
        BillAmt_Aug = st.number_input("BillAmt_Aug", min_value=0.0, value=10000.0, step=500.0)
        BillAmt_Jul = st.number_input("BillAmt_Jul", min_value=0.0, value=10000.0, step=500.0)
        BillAmt_Jun = st.number_input("BillAmt_Jun", min_value=0.0, value=10000.0, step=500.0)

    # additional bill / paid columns below form to keep layout tidy
    BillAmt_May = st.number_input("BillAmt_May", min_value=0.0, value=10000.0, step=500.0)
    BillAmt_Apr = st.number_input("BillAmt_Apr", min_value=0.0, value=10000.0, step=500.0)

    PaidAmt_Sep = st.number_input("PaidAmt_Sep", min_value=0.0, value=5000.0, step=100.0)
    PaidAmt_Aug = st.number_input("PaidAmt_Aug", min_value=0.0, value=5000.0, step=100.0)
    PaidAmt_Jul = st.number_input("PaidAmt_Jul", min_value=0.0, value=5000.0, step=100.0)
    PaidAmt_Jun = st.number_input("PaidAmt_Jun", min_value=0.0, value=5000.0, step=100.0)
    PaidAmt_May = st.number_input("PaidAmt_May", min_value=0.0, value=5000.0, step=100.0)
    PaidAmt_Apr = st.number_input("PaidAmt_Apr", min_value=0.0, value=5000.0, step=100.0)

    submitted = st.form_submit_button("Predict Single Applicant")

if submitted:
    input_dict = {
        "Credit_Limit": Credit_Limit,
        "Gender_Code": int(Gender_Code),
        "Education_Level": int(Education_Level),
        "Marital_Status": int(Marital_Status),
        "Age_Years": int(Age_Years),
        "Repay_Sep": Repay_Sep,
        "Repay_Aug": Repay_Aug,
        "Repay_Jul": Repay_Jul,
        "Repay_Jun": Repay_Jun,
        "Repay_May": Repay_May,
        "Repay_Apr": Repay_Apr,
        "BillAmt_Sep": BillAmt_Sep,
        "BillAmt_Aug": BillAmt_Aug,
        "BillAmt_Jul": BillAmt_Jul,
        "BillAmt_Jun": BillAmt_Jun,
        "BillAmt_May": BillAmt_May,
        "BillAmt_Apr": BillAmt_Apr,
        "PaidAmt_Sep": PaidAmt_Sep,
        "PaidAmt_Aug": PaidAmt_Aug,
        "PaidAmt_Jul": PaidAmt_Jul,
        "PaidAmt_Jun": PaidAmt_Jun,
        "PaidAmt_May": PaidAmt_May,
        "PaidAmt_Apr": PaidAmt_Apr
    }

    # build DataFrame with columns in the expected order
    X_single = pd.DataFrame([input_dict], columns=FEATURES)

    try:
        pred = model.predict(X_single)[0]
        proba = model.predict_proba(X_single)[0][1] if hasattr(model, "predict_proba") else None

        st.write("### Result")
        if pred == 1:
            st.error(f"High risk to default next month. (probability = {proba:.3f})" if proba is not None else "High risk to default next month.")
        else:
            st.success(f"Low risk to default next month. (probability = {proba:.3f})" if proba is not None else "Low risk to default next month.")
    except Exception as e:
        st.error("Prediction failed. Likely feature mismatch between UI and trained pipeline.")
        st.write(e)

st.markdown("---")

# -------------------------
# Batch predictions via CSV
# -------------------------
st.header("Batch Predictions (CSV upload)")
st.markdown("Upload a CSV with the same feature columns. Column names must match exactly.")

uploaded = st.file_uploader("Upload applicants CSV (columns must match)", type=["csv"], key="batch")

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Preview of uploaded file:")
    st.dataframe(df.head())

    # check for missing required columns
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        st.error(f"Uploaded CSV is missing these required columns: {missing}")
    else:
        try:
            preds = model.predict(df[FEATURES])
            probs = model.predict_proba(df[FEATURES])[:, 1] if hasattr(model, "predict_proba") else None
            df_out = df.copy()
            df_out["prediction"] = preds
            if probs is not None:
                df_out["risk_probability"] = probs
            st.success("Batch prediction finished.")
            st.dataframe(df_out.head())

            csv_bytes = df_out.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", csv_bytes, "predictions.csv", "text/csv")
        except Exception as e:
            st.error("Prediction failed on uploaded data — see error below.")
            st.write(e)

st.markdown("---")

# -------------------------
# Optional: Evaluate on labelled test data
# -------------------------
st.header("Evaluate Model (Optional)")
st.markdown("Upload a labelled CSV that includes the column `Will_Default_Next_Month` to run evaluation metrics.")

eval_file = st.file_uploader("Upload labelled test CSV", type=["csv"], key="eval")

if eval_file:
    test_df = pd.read_csv(eval_file)
    if "Will_Default_Next_Month" not in test_df.columns:
        st.error("Label column 'Will_Default_Next_Month' not found in uploaded file.")
    else:
        X_test = test_df[FEATURES]
        y_test = test_df["Will_Default_Next_Month"]

        try:
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=False)
            st.text(report)

            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            if y_prob is not None:
                st.subheader("ROC Curve")
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                fig2, ax2 = plt.subplots()
                ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
                ax2.plot([0, 1], [0, 1], "k--")
                ax2.set_xlabel("False Positive Rate")
                ax2.set_ylabel("True Positive Rate")
                ax2.legend()
                st.pyplot(fig2)
        except Exception as e:
            st.error("Evaluation failed — check that the uploaded file's columns and types match the training data.")
            st.write(e)

st.markdown("---")
st.caption("Tip: Save your final pipeline (preprocessor + model) as 'model.pkl' using joblib.dump(pipeline, 'model.pkl'). The pipeline should accept raw feature columns as listed and handle scaling/encoding internally.")
