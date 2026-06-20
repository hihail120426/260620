import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="AI Stock Forecast",
    page_icon="📈",
    layout="wide"
)

# -----------------------------
# 제목
# -----------------------------
st.markdown("""
# 📈 AI Stock Forecast
### 실시간 주가 분석 및 미래 예측
""")

# -----------------------------
# 사이드바
# -----------------------------
st.sidebar.header("설정")

market = st.sidebar.radio(
    "시장 선택",
    ["🇰🇷 국내 주식", "🇺🇸 미국 주식"]
)

if market == "🇰🇷 국내 주식":
    stocks = {
        "삼성전자":"005930.KS",
        "SK하이닉스":"000660.KS",
        "NAVER":"035420.KS",
        "카카오":"035720.KS",
        "현대차":"005380.KS",
        "기아":"000270.KS",
        "셀트리온":"068270.KS",
        "LG에너지솔루션":"373220.KS"
    }
else:
    stocks = {
        "NVIDIA":"NVDA",
        "Apple":"AAPL",
        "Microsoft":"MSFT",
        "Amazon":"AMZN",
        "Alphabet":"GOOGL",
        "Meta":"META",
        "Tesla":"TSLA",
        "Broadcom":"AVGO"
    }

company = st.sidebar.selectbox(
    "종목 선택",
    list(stocks.keys())
)

period_dict = {
    "1년":"1y",
    "3년":"3y",
    "5년":"5y",
    "10년":"10y",
    "최대":"max"
}

period_label = st.sidebar.selectbox(
    "과거 데이터 기간",
    list(period_dict.keys())
)

period = period_dict[period_label]

prediction_dict = {
    "7일":7,
    "30일":30,
    "90일":90,
    "180일":180,
    "1년":365,
    "3년":1095
}

prediction_label = st.sidebar.selectbox(
    "미래 예측 기간",
    list(prediction_dict.keys())
)

future_days = prediction_dict[prediction_label]

ticker = stocks[company]

# -----------------------------
# 데이터 다운로드
# -----------------------------
with st.spinner("데이터 불러오는 중..."):

    df = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False
    )

# Close 데이터 처리
close = df["Close"]

if len(close.shape) > 1:
    close = close.iloc[:,0]

close = close.values.flatten()

dates = df.index

# -----------------------------
# 이동평균선
# -----------------------------
df["MA20"] = df["Close"].rolling(20).mean()
df["MA60"] = df["Close"].rolling(60).mean()

# -----------------------------
# 선형 회귀 예측
# -----------------------------
x = np.arange(len(close)).reshape(-1,1)

model = LinearRegression()
model.fit(x, close)

future_x = np.arange(
    len(close),
    len(close)+future_days
).reshape(-1,1)

prediction = model.predict(future_x)

future_dates = pd.date_range(
    start=dates[-1],
    periods=future_days+1,
    freq="B"
)[1:]

# -----------------------------
# 상단 정보 카드
# -----------------------------
current_price = round(close[-1],2)

change = (
    (close[-1]-close[-2])
    / close[-2]
    *100
)

high_52 = round(df["High"].max(),2)
low_52 = round(df["Low"].min(),2)

col1,col2,col3,col4 = st.columns(4)

with col1:
    st.metric(
        "현재가",
        f"{current_price}"
    )

with col2:
    st.metric(
        "일일 변동률",
        f"{change:.2f}%"
    )

with col3:
    st.metric(
        "기간 최고가",
        f"{high_52}"
    )

with col4:
    st.metric(
        "기간 최저가",
        f"{low_52}"
    )

# -----------------------------
# 메인 차트
# -----------------------------
fig = go.Figure()

# 실제 주가
fig.add_trace(
    go.Scatter(
        x=dates,
        y=close,
        name="실제 주가",
        line=dict(width=3)
    )
)

# MA20
fig.add_trace(
    go.Scatter(
        x=dates,
        y=df["MA20"],
        name="MA20",
        line=dict(dash="dot")
    )
)

# MA60
fig.add_trace(
    go.Scatter(
        x=dates,
        y=df["MA60"],
        name="MA60",
        line=dict(dash="dash")
    )
)

# 미래 예측
fig.add_trace(
    go.Scatter(
        x=future_dates,
        y=prediction,
        name="AI 예측",
        line=dict(
            width=4,
            dash="dash"
        )
    )
)

fig.update_layout(
    title=f"{company} 주가 및 미래 예측",
    template="plotly_dark",
    height=750,
    hovermode="x unified",
    title_x=0.5,
    xaxis_title="날짜",
    yaxis_title="주가"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# -----------------------------
# 데이터 표
# -----------------------------
st.subheader("최근 데이터")

st.dataframe(
    df.tail(20),
    use_container_width=True
)
