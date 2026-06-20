import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI 주식 예측기",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }

    .main-title {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border-radius: 14px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }
    .info-box {
        background: #161b22;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #ccc;
    }
    .predict-result {
        background: linear-gradient(135deg, #0d1b2a, #1b263b);
        border: 1px solid #415a77;
        border-radius: 12px;
        padding: 1.4rem;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1a6b4a, #2ecc71);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        padding: 0.6rem 1.5rem;
        width: 100%;
        font-size: 1rem;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        filter: brightness(1.15);
        transform: translateY(-1px);
    }
    section[data-testid="stSidebar"] {
        background: #0d1117 !important;
    }
    .ticker-badge {
        display: inline-block;
        background: #1f6feb;
        color: white;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 8px;
        vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 헤더
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-title">
    <h1 style="margin:0; font-size:2rem; font-weight:700;">AI 주식 예측기</h1>
    <p style="margin:0.4rem 0 0; opacity:0.65; font-size:0.95rem;">
        Yahoo Finance 실시간 데이터 &nbsp;|&nbsp; Random Forest + Gradient Boosting 앙상블
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 유틸 함수
# ─────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def fetch_realtime(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame | None:
    """Yahoo Finance 실시간 데이터 가져오기"""
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index)
        return df.dropna()
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """기술적 지표 피처 생성"""
    d = df.copy()

    for w in [5, 10, 20, 60]:
        d[f"MA{w}"] = d["Close"].rolling(w).mean()
        d[f"MA{w}_ratio"] = d["Close"] / (d[f"MA{w}"] + 1e-9)

    # RSI
    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    # MACD
    d["MACD"] = d["Close"].ewm(span=12).mean() - d["Close"].ewm(span=26).mean()
    d["MACD_sig"] = d["MACD"].ewm(span=9).mean()

    # 볼린저 밴드
    bb_m = d["Close"].rolling(20).mean()
    bb_s = d["Close"].rolling(20).std()
    d["BB_upper_ratio"] = d["Close"] / (bb_m + 2 * bb_s + 1e-9)
    d["BB_lower_ratio"] = d["Close"] / (bb_m - 2 * bb_s + 1e-9)

    # 수익률 / 변동성
    d["ret1"] = d["Close"].pct_change()
    d["ret5"] = d["Close"].pct_change(5)
    d["ret20"] = d["Close"].pct_change(20)
    d["vol20"] = d["ret1"].rolling(20).std()

    # 거래량
    d["vol_ratio"] = d["Volume"] / (d["Volume"].rolling(20).mean() + 1e-9)

    # 고저 범위
    d["HL_ratio"] = (d["High"] - d["Low"]) / (d["Close"] + 1e-9)

    return d.dropna()


def build_xy(df: pd.DataFrame, look_back: int = 10):
    """지도학습용 X, y 생성"""
    feat_cols = [c for c in df.columns if c not in ["Open", "High", "Low", "Volume"]]
    data = df[feat_cols].values
    close_col = list(df[feat_cols].columns).index("Close")

    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i].flatten())
        y.append(data[i, close_col])
    return np.array(X), np.array(y), feat_cols, close_col


def predict_future(df_feat: pd.DataFrame, model, scaler: MinMaxScaler,
                   feat_cols: list, close_col: int,
                   look_back: int, forecast_days: int) -> np.ndarray:
    """미래 forecast_days일 예측"""
    scaled = scaler.transform(df_feat[feat_cols])
    current_window = scaled[-look_back:].copy()
    preds = []

    for _ in range(forecast_days):
        inp = current_window.flatten().reshape(1, -1)
        pred_scaled = model.predict(inp)[0]
        preds.append(pred_scaled)

        new_row = current_window[-1].copy()
        new_row[close_col] = pred_scaled
        current_window = np.vstack([current_window[1:], new_row])

    # 역정규화
    dummy = np.zeros((len(preds), len(feat_cols)))
    dummy[:, close_col] = preds
    return scaler.inverse_transform(dummy)[:, close_col]


def run_model(df_feat: pd.DataFrame, forecast_days: int, look_back: int = 10):
    feat_cols = [c for c in df_feat.columns if c not in ["Open", "High", "Low", "Volume"]]
    close_col = list(df_feat[feat_cols].columns).index("Close")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_feat[feat_cols])

    X_raw = [scaled[i - look_back:i].flatten() for i in range(look_back, len(scaled))]
    y_raw = [scaled[i, close_col] for i in range(look_back, len(scaled))]
    X, y = np.array(X_raw), np.array(y_raw)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 두 모델 학습
    rf = RandomForestRegressor(n_estimators=200, max_depth=10,
                                min_samples_split=5, random_state=42, n_jobs=-1)
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                   learning_rate=0.05, random_state=42)
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    # 검증 RMSE (역정규화)
    def inverse_close(arr):
        dummy = np.zeros((len(arr), len(feat_cols)))
        dummy[:, close_col] = arr
        return scaler.inverse_transform(dummy)[:, close_col]

    rf_val = inverse_close(rf.predict(X_test))
    gb_val = inverse_close(gb.predict(X_test))
    ensemble_val = 0.5 * rf_val + 0.5 * gb_val
    actual_val = inverse_close(y_test)

    rmse = float(np.sqrt(mean_squared_error(actual_val, ensemble_val)))

    # 미래 예측
    rf_fut = predict_future(df_feat, rf, scaler, feat_cols, close_col, look_back, forecast_days)
    gb_fut = predict_future(df_feat, gb, scaler, feat_cols, close_col, look_back, forecast_days)
    ensemble_fut = 0.5 * rf_fut + 0.5 * gb_fut

    return ensemble_fut, rmse, actual_val, ensemble_val


# ─────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 종목 설정")
    st.markdown("---")

    PRESETS = {
        "삼성전자": "005930.KS",
        "SK하이닉스": "000660.KS",
        "NAVER": "035420.KS",
        "카카오": "035720.KS",
        "현대차": "005380.KS",
        "POSCO홀딩스": "005490.KS",
        "Apple": "AAPL",
        "Tesla": "TSLA",
        "NVIDIA": "NVDA",
        "Microsoft": "MSFT",
        "Amazon": "AMZN",
        "직접 입력": "__custom__",
    }

    preset_choice = st.selectbox("종목 선택", list(PRESETS.keys()))

    if PRESETS[preset_choice] == "__custom__":
        ticker_input = st.text_input(
            "티커 심볼 입력",
            placeholder="예: AAPL  /  005930.KS",
        ).strip().upper()
    else:
        ticker_input = PRESETS[preset_choice]

    st.markdown("---")

    period_map = {"6개월": "6mo", "1년": "1y", "2년": "2y", "3년": "3y", "5년": "5y"}
    period_label = st.selectbox("학습 데이터 기간", list(period_map.keys()), index=2)
    period = period_map[period_label]

    interval_map = {"일봉 (1d)": "1d", "주봉 (1wk)": "1wk"}
    interval_label = st.selectbox("봉 주기", list(interval_map.keys()))
    interval = interval_map[interval_label]

    forecast_days = st.slider("예측 일수", min_value=5, max_value=30, value=14)

    st.markdown("---")
    run_btn = st.button("예측 시작", use_container_width=True)

    st.markdown("""
    <div style='font-size:0.72rem; color:#555; margin-top:1.5rem; line-height:1.6;'>
    * Yahoo Finance 실시간 데이터 기반<br>
    * 본 서비스는 투자 참고용입니다.<br>
    * 실제 투자 손익에 책임지지 않습니다.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 티커 검증
# ─────────────────────────────────────────────
if not ticker_input:
    st.info("사이드바에서 종목을 선택하거나 티커를 직접 입력하세요.")
    st.stop()

# ─────────────────────────────────────────────
# 실시간 데이터 로드
# ─────────────────────────────────────────────
with st.spinner(f"{ticker_input} 실시간 데이터 로딩 중..."):
    df_raw = fetch_realtime(ticker_input, period=period, interval=interval)
    info = get_info(ticker_input)

if df_raw is None or df_raw.empty:
    st.error(f"'{ticker_input}' 데이터를 가져올 수 없습니다. 티커 심볼을 확인하세요.")
    st.stop()

company_name = info.get("longName") or info.get("shortName") or ticker_input
currency = info.get("currency", "")
sector = info.get("sector", "")
market_cap = info.get("marketCap")

# ─────────────────────────────────────────────
# 종목 정보 헤더
# ─────────────────────────────────────────────
close_s = df_raw["Close"].squeeze()
cur_price = float(close_s.iloc[-1])
prev_price = float(close_s.iloc[-2])
change_amt = cur_price - prev_price
change_pct = (change_amt / prev_price) * 100
color_chg = "#00d4aa" if change_pct >= 0 else "#ff6b6b"
sign = "+" if change_pct >= 0 else ""

st.markdown(f"""
<div class="info-box">
    <span style="font-size:1.15rem; font-weight:700; color:white;">{company_name}</span>
    <span class="ticker-badge">{ticker_input}</span>
    {"&nbsp;&nbsp;<span style='color:#888;font-size:0.85rem;'>" + sector + "</span>" if sector else ""}
    <span style="float:right; font-size:1.05rem; font-weight:600; color:{color_chg};">
        {cur_price:,.2f} {currency} &nbsp;
        <span style="font-size:0.9rem;">({sign}{change_pct:.2f}%)</span>
    </span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 핵심 지표 카드
# ─────────────────────────────────────────────
high_52 = float(close_s.tail(252).max())
low_52 = float(close_s.tail(252).min())
vol_avg = float(df_raw["Volume"].squeeze().tail(20).mean()) if "Volume" in df_raw.columns else 0
returns = close_s.pct_change().dropna()
volatility = float(returns.tail(30).std() * np.sqrt(252) * 100)
mc_text = f"{market_cap / 1e12:.2f}T" if market_cap and market_cap >= 1e12 else \
          (f"{market_cap / 1e9:.1f}B" if market_cap else "-")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("현재가", f"{cur_price:,.2f}", f"{sign}{change_pct:.2f}%")
c2.metric("52주 최고", f"{high_52:,.2f}")
c3.metric("52주 최저", f"{low_52:,.2f}")
c4.metric("변동성 (연환산)", f"{volatility:.1f}%")
c5.metric("시가총액", mc_text)

st.markdown("---")

# ─────────────────────────────────────────────
# 탭 구성
# ─────────────────────────────────────────────
tab_chart, tab_indicator, tab_predict = st.tabs(["주가 차트", "기술적 지표", "AI 예측"])

# ── 탭 1: 캔들 차트 ──────────────────────────────────────────
with tab_chart:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.04, row_heights=[0.75, 0.25],
    )

    fig.add_trace(go.Candlestick(
        x=df_raw.index,
        open=df_raw["Open"].squeeze(),
        high=df_raw["High"].squeeze(),
        low=df_raw["Low"].squeeze(),
        close=close_s,
        name="주가",
        increasing_line_color="#00d4aa",
        decreasing_line_color="#ff6b6b",
    ), row=1, col=1)

    for ma, c in [(20, "#ffd700"), (60, "#ff8c00"), (120, "#da70d6")]:
        if len(df_raw) >= ma:
            fig.add_trace(go.Scatter(
                x=df_raw.index,
                y=close_s.rolling(ma).mean(),
                name=f"MA{ma}",
                line=dict(color=c, width=1.2),
                opacity=0.85,
            ), row=1, col=1)

    vol_colors = [
        "#00d4aa" if float(c) >= float(o) else "#ff6b6b"
        for c, o in zip(df_raw["Close"].squeeze(), df_raw["Open"].squeeze())
    ]
    fig.add_trace(go.Bar(
        x=df_raw.index, y=df_raw["Volume"].squeeze(),
        name="거래량", marker_color=vol_colors, opacity=0.55,
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark", height=560,
        xaxis_rangeslider_visible=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0d1117",
        font=dict(family="Noto Sans KR"),
        legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="#2d3748", borderwidth=1),
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── 탭 2: 기술적 지표 ────────────────────────────────────────
with tab_indicator:
    di = df_raw.copy()
    di["Close"] = close_s

    delta = di["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    di["RSI"] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    ema12 = di["Close"].ewm(span=12).mean()
    ema26 = di["Close"].ewm(span=26).mean()
    di["MACD"] = ema12 - ema26
    di["Signal"] = di["MACD"].ewm(span=9).mean()
    di["Hist"] = di["MACD"] - di["Signal"]

    bb_m = di["Close"].rolling(20).mean()
    bb_s = di["Close"].rolling(20).std()
    di["BB_upper"] = bb_m + 2 * bb_s
    di["BB_lower"] = bb_m - 2 * bb_s

    fig2 = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("볼린저 밴드", "RSI (14)", "MACD"),
        vertical_spacing=0.07,
        row_heights=[0.5, 0.25, 0.25],
    )

    fig2.add_trace(go.Scatter(x=di.index, y=di["Close"], name="종가",
                               line=dict(color="#00d4aa", width=1.8)), row=1, col=1)
    fig2.add_trace(go.Scatter(x=di.index, y=di["BB_upper"], name="상단",
                               line=dict(color="#ffd700", dash="dash", width=1.0)), row=1, col=1)
    fig2.add_trace(go.Scatter(x=di.index, y=di["BB_lower"], name="하단",
                               line=dict(color="#ffd700", dash="dash", width=1.0),
                               fill="tonexty", fillcolor="rgba(255,215,0,0.05)"), row=1, col=1)

    fig2.add_trace(go.Scatter(x=di.index, y=di["RSI"], name="RSI",
                               line=dict(color="#da70d6", width=1.5)), row=2, col=1)
    fig2.add_hline(y=70, line_dash="dash", line_color="#ff6b6b", opacity=0.4, row=2, col=1)
    fig2.add_hline(y=30, line_dash="dash", line_color="#00d4aa", opacity=0.4, row=2, col=1)

    hist_c = ["#00d4aa" if v >= 0 else "#ff6b6b" for v in di["Hist"].fillna(0)]
    fig2.add_trace(go.Bar(x=di.index, y=di["Hist"], name="Hist",
                           marker_color=hist_c, opacity=0.6), row=3, col=1)
    fig2.add_trace(go.Scatter(x=di.index, y=di["MACD"], name="MACD",
                               line=dict(color="#00d4aa", width=1.3)), row=3, col=1)
    fig2.add_trace(go.Scatter(x=di.index, y=di["Signal"], name="Signal",
                               line=dict(color="#ff8c00", width=1.3)), row=3, col=1)

    fig2.update_layout(
        template="plotly_dark", height=600,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
        font=dict(family="Noto Sans KR"),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0.3)"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── 탭 3: AI 예측 ────────────────────────────────────────────
with tab_predict:
    if not run_btn:
        st.markdown("""
        <div style='text-align:center; padding:5rem 1rem; color:#555;'>
            <div style='font-size:2.5rem; font-weight:300; letter-spacing:4px;'>- - -</div>
            <p style='margin-top:1.2rem; font-size:1rem;'>
                사이드바에서 설정 후 <strong style='color:#aaa;'>예측 시작</strong> 버튼을 누르세요.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("AI 모델 학습 중... (약 10~30초)"):
            try:
                df_feat = add_features(df_raw)

                if len(df_feat) < 80:
                    st.error("데이터가 너무 적습니다. 더 긴 기간을 선택하세요.")
                    st.stop()

                future_preds, rmse, actual_val, val_preds = run_model(
                    df_feat, forecast_days=forecast_days, look_back=10
                )

                # 날짜 인덱스
                last_date = df_raw.index[-1]
                if interval == "1wk":
                    future_dates = pd.date_range(
                        start=last_date + pd.Timedelta(weeks=1),
                        periods=forecast_days, freq="W-MON",
                    )
                else:
                    future_dates = pd.bdate_range(
                        start=last_date + pd.Timedelta(days=1),
                        periods=forecast_days,
                    )

                pred_df = pd.DataFrame({"predicted": future_preds}, index=future_dates)
                std_recent = float(close_s.tail(30).std())
                uncertainty = np.linspace(0.3, 1.2, forecast_days) * std_recent
                pred_df["upper"] = pred_df["predicted"] + uncertainty
                pred_df["lower"] = pred_df["predicted"] - uncertainty

                final_pred = float(pred_df["predicted"].iloc[-1])
                diff = final_pred - cur_price
                diff_pct = (diff / cur_price) * 100
                arrow_color = "#00d4aa" if diff_pct >= 0 else "#ff6b6b"
                dir_text = "상승" if diff_pct >= 0 else "하락"
                sign2 = "+" if diff_pct >= 0 else ""

                # 결과 요약
                r1, r2, r3 = st.columns(3)
                r1.metric("현재가", f"{cur_price:,.2f} {currency}")
                r2.metric(
                    f"{forecast_days}일 후 예측가",
                    f"{final_pred:,.2f} {currency}",
                    f"{sign2}{diff_pct:.2f}%",
                )
                r3.metric("모델 RMSE", f"{rmse:,.2f}")

                st.markdown(f"""
                <div class="predict-result">
                    <p style="color:#aaa; font-size:0.85rem; margin:0 0 0.4rem;">예측 요약</p>
                    <p style="font-size:1.1rem; margin:0; color:white;">
                        향후 <strong>{forecast_days}일</strong> 동안
                        <span style="color:{arrow_color}; font-weight:700;">
                            {dir_text} ({sign2}{diff_pct:.2f}%)
                        </span>이 예상됩니다.
                    </p>
                    <p style="color:#555; font-size:0.8rem; margin:0.5rem 0 0;">
                        모델: RF 50% + GradientBoosting 50% 앙상블 &nbsp;|&nbsp;
                        학습 기간: {period_label}
                    </p>
                </div>
                """, unsafe_allow_html=True)

                # 예측 차트
                fig3 = go.Figure()

                recent_n = min(90, len(df_raw))
                recent_close = close_s.tail(recent_n)

                fig3.add_trace(go.Scatter(
                    x=recent_close.index, y=recent_close.values,
                    name="실제 주가", line=dict(color="#00d4aa", width=2),
                ))

                # 경계 연결선 (실제 마지막 -> 예측 첫날)
                connect_x = [recent_close.index[-1], pred_df.index[0]]
                connect_y = [float(recent_close.iloc[-1]), float(pred_df["predicted"].iloc[0])]
                fig3.add_trace(go.Scatter(
                    x=connect_x, y=connect_y,
                    line=dict(color="#ffd700", width=2),
                    showlegend=False,
                ))

                fig3.add_trace(go.Scatter(
                    x=pred_df.index, y=pred_df["predicted"],
                    name="예측 주가",
                    line=dict(color="#ffd700", width=2.5, dash="dot"),
                    mode="lines+markers",
                    marker=dict(size=5, color="#ffd700"),
                ))

                # 신뢰 구간
                fig3.add_trace(go.Scatter(
                    x=list(pred_df.index) + list(pred_df.index[::-1]),
                    y=list(pred_df["upper"]) + list(pred_df["lower"][::-1]),
                    fill="toself",
                    fillcolor="rgba(255,215,0,0.08)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="신뢰 구간",
                ))

                fig3.update_layout(
                    template="plotly_dark", height=460,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0d1117",
                    font=dict(family="Noto Sans KR"),
                    legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="#2d3748", borderwidth=1),
                    title=dict(
                        text=f"{company_name} &nbsp;|&nbsp; {forecast_days}일 예측",
                        font=dict(size=15, color="#ccc"),
                    ),
                    margin=dict(l=10, r=10, t=50, b=10),
                )
                st.plotly_chart(fig3, use_container_width=True)

                # 상세 테이블
                with st.expander("예측 날짜별 상세 데이터"):
                    table_df = pred_df.copy()
                    table_df.index = table_df.index.strftime("%Y-%m-%d")
                    table_df.columns = ["예측가", "상단 (신뢰 구간)", "하단 (신뢰 구간)"]
                    st.dataframe(
                        table_df.style.format("{:,.2f}"),
                        use_container_width=True,
                    )

            except Exception as e:
                st.error(f"예측 중 오류 발생: {e}")

# ─────────────────────────────────────────────
# 푸터
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#444; font-size:0.78rem; padding:0.5rem;'>
    AI 주식 예측기 &nbsp;|&nbsp; Yahoo Finance 실시간 데이터 &nbsp;|&nbsp;
    본 서비스는 투자 참고용이며 실제 투자 결과에 책임지지 않습니다.
</div>
""", unsafe_allow_html=True)
