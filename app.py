import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ta import add_all_ta_features

# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(page_title="Stock Analytics Dashboard", layout="wide")

st.title("📊 Professional Stock Market Analysis Dashboard")

# -----------------------------------
# Sidebar Controls
# -----------------------------------

st.sidebar.header("Settings")

ticker = st.sidebar.text_input("Stock Ticker", "AAPL").upper()

period = st.sidebar.selectbox(
    "Time Period",
    ["1mo","3mo","6mo","1y","2y","5y","10y","max"]
)

interval = st.sidebar.selectbox(
    "Interval",
    ["1d","1wk","1mo"]
)

# -----------------------------------
# Data Loader
# -----------------------------------

@st.cache_data(show_spinner=False)
def load_data(ticker, period, interval):

    try:

        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False
        )

        if df.empty:
            return None

        # Flatten columns if multi-index
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.dropna(inplace=True)

        return df

    except Exception:
        return None


# -----------------------------------
# Download Data
# -----------------------------------

with st.spinner("Downloading market data..."):
    data = load_data(ticker, period, interval)

if data is None or data.empty:
    st.error("Failed to download data. Check ticker symbol or internet connection.")
    st.stop()

# -----------------------------------
# Technical Indicators
# -----------------------------------

try:

    data = add_all_ta_features(
        data,
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        fillna=True
    )

except Exception:
    st.warning("Technical indicators could not be calculated.")


# -----------------------------------
# Performance Metrics
# -----------------------------------

try:

    close_prices = data["Close"]

    returns = close_prices.pct_change().dropna()

    total_return = (close_prices.iloc[-1] / close_prices.iloc[0]) - 1

    volatility = returns.std() * np.sqrt(252)

    std = returns.std()

    if pd.isna(std) or std == 0:
        sharpe = 0
    else:
        sharpe = (returns.mean() / std) * np.sqrt(252)

except Exception:

    total_return = 0
    volatility = 0
    sharpe = 0


# -----------------------------------
# Metrics Display
# -----------------------------------

col1, col2, col3 = st.columns(3)

col1.metric("Total Return", f"{total_return*100:.2f}%")
col2.metric("Annual Volatility", f"{volatility*100:.2f}%")
col3.metric("Sharpe Ratio", f"{sharpe:.2f}")

# -----------------------------------
# Candlestick Chart
# -----------------------------------

fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=data.index,
    open=data["Open"],
    high=data["High"],
    low=data["Low"],
    close=data["Close"],
    name="Price"
))

fig.update_layout(
    title=f"{ticker} Price Chart",
    height=600,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------
# Moving Averages
# -----------------------------------

data["MA50"] = data["Close"].rolling(50).mean()
data["MA200"] = data["Close"].rolling(200).mean()

st.subheader("Moving Averages")

st.line_chart(data[["Close","MA50","MA200"]])

# -----------------------------------
# RSI
# -----------------------------------

if "momentum_rsi" in data.columns:

    st.subheader("RSI Indicator")
    st.line_chart(data["momentum_rsi"])

# -----------------------------------
# MACD
# -----------------------------------

if "trend_macd" in data.columns:

    st.subheader("MACD")

    st.line_chart(
        data[[
            "trend_macd",
            "trend_macd_signal"
        ]]
    )

# -----------------------------------
# Bollinger Bands
# -----------------------------------

if "volatility_bbm" in data.columns:

    st.subheader("Bollinger Bands")

    st.line_chart(
        data[[
            "volatility_bbm",
            "volatility_bbh",
            "volatility_bbl"
        ]]
    )

# -----------------------------------
# Volume Chart
# -----------------------------------

st.subheader("Volume")

st.bar_chart(data["Volume"])

# -----------------------------------
# Raw Data Table
# -----------------------------------

st.subheader("Raw Data")

st.dataframe(data.tail(50))

# -----------------------------------
# Download Button
# -----------------------------------

csv = data.to_csv().encode("utf-8")

st.download_button(
    "Download Dataset",
    csv,
    f"{ticker}_stock_data.csv",
    "text/csv"
)