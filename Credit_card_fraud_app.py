import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix

st.set_page_config(page_title="Bank Marketing Classifier", layout="wide")

st.title("📊 Bank Marketing Prediction App")
st.write("Upload test CSV and evaluate trained ML models")

# ================= LOAD SAVED OBJECTS =================
model_dir = "model"

scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
columns = joblib.load(os.path.join(model_dir, "columns.pkl"))

models = {
    "Logistic Regression": joblib.load(os.path.join(model_dir, "Logistic_Regression.pkl")),
    "Decision Tree": joblib.load(os.path.join(model_dir, "Decision_Tree.pkl")),
    "KNN": joblib.load(os.path.join(model_dir, "KNN.pkl")),
    "Naive Bayes": joblib.load(os.path.join(model_dir, "Naive_Bayes.pkl")),
    "Random Forest": joblib.load(os.path.join(model_dir, "Random_Forest.pkl")),
    "XGBoost": joblib.load(os.path.join(model_dir, "XGBoost.pkl"))
}

# ================= MODEL SELECT =================
model_name = st.selectbox("Select Model", list(models.keys()))
model = models[model_name]

# ================= FILE UPLOAD =================
uploaded_file = st.file_uploader("Upload CSV Test File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=None, engine="python")
    df.columns = df.columns.str.strip()

    if "y" not in df.columns:
        st.error("Uploaded file must contain target column 'y'")
    else:
        # Convert target
        y_true = df["y"].astype(str).str.lower().map({"yes": 1, "no": 0})

        # Drop target
        X = df.drop("y", axis=1)

        # Handle unknown values
        X.replace("unknown", np.nan, inplace=True)

        for col in X.select_dtypes(include=np.number).columns:
            X[col].fillna(X[col].median(), inplace=True)

        for col in X.select_dtypes(include="object").columns:
            X[col].fillna(X[col].mode()[0], inplace=True)

        # Encoding
        X = pd.get_dummies(X)

        # Align columns with training columns
        X = X.reindex(columns=columns.drop("y"), fill_value=0)

        # Scale
        X_scaled = scaler.transform(X)

        # Predict
        preds = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)[:, 1]

        # Metrics
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        auc = roc_auc_score(y_true, probs)
        mcc = matthews_corrcoef(y_true, preds)

        # Display metrics
        st.subheader("📈 Model Performance")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", round(acc, 3))
        col2.metric("Precision", round(prec, 3))
        col3.metric("Recall", round(rec, 3))

        col4, col5, col6 = st.columns(3)
        col4.metric("F1 Score", round(f1, 3))
        col5.metric("AUC", round(auc, 3))
        col6.metric("MCC", round(mcc, 3))

        # Confusion Matrix
        st.subheader("🔍 Confusion Matrix")
        cm = confusion_matrix(y_true, preds)
        st.write(pd.DataFrame(cm, columns=["Pred No", "Pred Yes"], index=["Actual No", "Actual Yes"]))
