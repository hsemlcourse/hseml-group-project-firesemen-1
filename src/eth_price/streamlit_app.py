"""Streamlit interface for the ETH/USD prediction model."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from eth_price.config import DEFAULT_MODEL_PATH, DEFAULT_SAMPLE_PATH
from eth_price.predict import load_artifact, predict_from_raw_candles

st.set_page_config(page_title="ETH/USD predictor", layout="wide")
st.title("ETH/USD: прогноз Close на следующую минуту")
st.write(
    "Загрузите CSV с последними минутными свечами Binance или используйте встроенный sample. "
    "Модель строит лаги и rolling-признаки, поэтому нужно минимум 70 строк."
)

uploaded = st.file_uploader("CSV с OHLCV свечами", type=["csv"])
if uploaded is not None:
    raw_df = pd.read_csv(uploaded)
else:
    raw_df = pd.read_csv(DEFAULT_SAMPLE_PATH).tail(300)
    st.info("Используется встроенный sample из data/sample/ethusd_1m_sample.csv")

if st.button("Сделать прогноз"):
    if not Path(DEFAULT_MODEL_PATH).exists():
        st.error("Файл модели не найден. Сначала выполните: python scripts/train.py")
    else:
        artifact = load_artifact(DEFAULT_MODEL_PATH)
        predictions = predict_from_raw_candles(raw_df, artifact)
        latest = predictions.iloc[-1]
        col1, col2, col3 = st.columns(3)
        col1.metric("Последняя свеча", str(latest["Open time"]))
        col2.metric("Текущий Close", f"{latest['Close']:.2f}")
        col3.metric("Прогноз Close(t+1)", f"{latest['prediction_next_close']:.2f}")
        chart_df = predictions.tail(120).set_index("Open time")
        st.line_chart(chart_df[["Close", "prediction_next_close"]])
        st.dataframe(predictions.tail(20), use_container_width=True)
