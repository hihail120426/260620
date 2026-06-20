import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AI 주가 예측", layout="wide")

st.title("📈 AI 주가 예측 웹앱")

ticker = st.text_input(
    "종목 코드 입력",
    value="NVDA"
).upper()

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

    close = close.values

    # 날짜 번호 생성
    x = np.arange(len(close)).reshape(-1,1)

    # 모델 학습
    model = LinearRegression()
    model.fit(x, close)

    # 미래 30일 예측
    future_days = 30
    future_x = np.arange(
        len(close),
        len(close)+future_days
    ).reshape(-1,1)

    predictions = model.predict(future_x)

    # 실제 날짜
    actual_dates = df.index

    # 미래 날짜
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
            y=predictions,
            mode="lines",
            name="예측 주가"
        )
    )

    fig.update_layout(
        title=f"{ticker} 주가 및 향후 30일 예측",
        xaxis_title="날짜",
        yaxis_title="주가",
        hovermode="x unified",
        height=700
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("데이터를 불러올 수 없습니다.")
