import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AI 주가 예측", layout="wide")

st.title("📈 AI 주가 예측 웹앱")

# 시장 선택
market = st.radio(
    "시장 선택",
    ["🇰🇷 국내 주식", "🇺🇸 미국 주식"]
)

# 종목 목록
if market == "🇰🇷 국내 주식":
    stocks = {
        "삼성전자": "005930.KS",
        "SK하이닉스": "000660.KS",
        "NAVER": "035420.KS",
        "카카오": "035720.KS",
        "현대차": "005380.KS",
        "LG에너지솔루션": "373220.KS",
        "셀트리온": "068270.KS",
        "기아": "000270.KS"
    }
else:
    stocks = {
        "NVIDIA": "NVDA",
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Amazon": "AMZN",
        "Alphabet": "GOOGL",
        "Meta": "META",
        "Tesla": "TSLA",
        "Broadcom": "AVGO"
    }

company = st.selectbox(
    "종목 선택",
    list(stocks.keys())
)

ticker = stocks[company]

# 데이터 다운로드
df = yf.download(
    ticker,
    period="1y",
    auto_adjust=True,
    progress=False
)

if not df.empty:

    close = df["Close"]

    if len(close.shape) > 1:
        close = close.iloc[:, 0]

    close = close.values.flatten()

    # 모델 학습
    x = np.arange(len(close)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(x, close)

    # 미래 30일 예측
    future_days = 30

    future_x = np.arange(
        len(close),
        len(close)+future_days
    ).reshape(-1,1)

    pred = model.predict(future_x)

    actual_dates = df.index

    future_dates = pd.date_range(
        start=actual_dates[-1],
        periods=future_days+1,
        freq="B"
    )[1:]

    fig = go.Figure()

    # 실제 주가
    fig.add_trace(
        go.Scatter(
            x=actual_dates,
            y=close,
            mode="lines",
            name="실제 주가"
        )
    )

    # 예측 주가
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=pred,
            mode="lines",
            name="30일 예측"
        )
    )

    fig.update_layout(
        title=f"{company} 주가 및 30일 예측",
        xaxis_title="날짜",
        yaxis_title="주가",
        hovermode="x unified",
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("데이터를 불러올 수 없습니다.")
