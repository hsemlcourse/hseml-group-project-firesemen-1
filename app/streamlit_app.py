from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="ETHUSD CP3 Demo", layout="centered")
st.title("ETHUSD one-minute forecast")
st.write(
    "Upload a CSV with at least 61 recent one-minute candles. "
    "The app sends the last candles to the FastAPI service."
)

uploaded = st.file_uploader("CSV file", type=["csv"])
api_url = st.text_input("API URL", API_URL)

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write("Preview", df.tail(5))
    payload = {"candles": df.tail(120).to_dict(orient="records")}
    if st.button("Predict next close"):
        response = requests.post(api_url, json=payload, timeout=20)
        if response.ok:
            result = response.json()
            st.metric("Recommended next close", f"{result['recommended_prediction']:.4f}")
            st.metric("Model prediction", f"{result['model_prediction']:.4f}")
            st.metric(
                "Persistence baseline",
                f"{result['persistence_baseline_prediction']:.4f}",
            )
            st.json(result)
        else:
            st.error(response.text)
else:
    st.info("Use examples/sample_candles.csv or export recent candles from the dataset.")
