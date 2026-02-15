import streamlit as st
import pandas as pd

st.set_page_config(page_title="Bank Marketing ML App")

st.title("📊 Bank Marketing Model Comparison")

st.write("This app shows performance of 6 ML models trained on Bank Marketing Dataset.")

# Load saved metrics
metrics = pd.read_csv("model/model_metrics.csv")

# Model dropdown
model_name = st.selectbox("Select Model", metrics["Model"])

# Show selected model metrics
selected = metrics[metrics["Model"] == model_name].iloc[0]

st.subheader("📈 Performance Metrics")

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", round(selected["Accuracy"], 3))
col2.metric("Precision", round(selected["Precision"], 3))
col3.metric("Recall", round(selected["Recall"], 3))

col4, col5, col6 = st.columns(3)
col4.metric("F1 Score", round(selected["F1"], 3))
col5.metric("AUC", round(selected["AUC"], 3))
col6.metric("MCC", round(selected["MCC"], 3))

st.subheader("📋 Full Comparison Table")
st.dataframe(metrics.sort_values("F1", ascending=False))
