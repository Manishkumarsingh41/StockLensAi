import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from transformers import pipeline
print(pipeline)
from fpdf import FPDF
import pandas as pd
import io
import warnings

# ML imports
from sklearn.metrics import mean_absolute_percentage_error
try:
    from prophet import Prophet
except Exception as e:
    Prophet = None

warnings.filterwarnings("ignore")

# Load Sentiment Model (FinBERT)
sentiment_pipeline = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")

# Stock Ticker List for Autocomplete (100 tickers you've provided)
all_tickers = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "JPM", "V", "PG", "KO", "NESN", "ROG", "SAP",
    "MC", "ULVR", "HSBA", "005930.KS", "0700.HK", "BABA", "7203.T", "6758.T", "BRK.A",
    "NOVN", "RDSA", "BP", "SIE", "ASML", "NFLX", "PYPL", "NVDA", "MA", "OR", "TSM", "INTC",
    "DIS", "NKE", "MCD", "CRM", "ABBV", "COST", "TXN", "VZ", "WMT", "LLY", "ABB", "BHP",
    "SNY", "GILD", "CAT", "CHTR", "MDT", "ADP", "HON", "IBM", "AMGN", "T", "UL", "LIN",
    "PLD", "DE", "BLK", "AMAT", "ZM", "NOW", "SQ", "TSMC", "C", "GS", "BAC", "CSCO", "ORCL",
    "ADI", "EA", "QCOM", "LMT", "BA", "COP", "CVS", "PFE", "MMM", "GE", "RTX", "F", "MRNA",
    "BIIB", "SBUX", "RIO", "ADBE", "SPGI", "BKNG", "BMY"
]

# Streamlit Page Config
st.set_page_config(page_title="StockLens AI", layout="wide")
st.title("📊 StockLens AI - Live Stock Sentiment, Forecast & Market Insights")

# Sidebar - Autocomplete Stock Selection & Forecast settings
selected_tickers = st.sidebar.multiselect(
    "Select up to 10 Stock Tickers",
    options=all_tickers,
    default=["AAPL", "MSFT", "TSLA"],
    max_selections=10
)
period = st.sidebar.selectbox(
    "Select Period (history)",
    options=[
        "1d", "5d", "1wk", "1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "10y", "ytd", "max"
    ],
    index=4
)

interval = st.sidebar.selectbox(
    "Select Interval",
    options=[
        "1m", "2m", "5m", "15m", "30m",
        "1h", "2h", "3h", "4h", "6h", "12h",
        "1d", "1wk", "1mo", "3mo", "6mo", "1y"
    ],
    index=12
)


st.sidebar.markdown("---")
forecast_days = st.sidebar.number_input("Periods into future to predict", min_value=1, max_value=365, value=30, step=1)
freq_option = st.sidebar.selectbox(
    "Select Forecast Frequency",
    options=["Days", "Months", "Years"],
    index=0
)
st.sidebar.markdown("**Model**: Prophet (add `prophet` package).")

# Map selection to Prophet freq codes
freq_map = {
    "Days": "D",
    "Months": "M",
    "Years": "Y"
}
forecast_freq = freq_map[freq_option]

# Data storage for export
sentiment_export = []
technical_summary = []
chart_images = []
forecast_results = []  # store prediction summaries for optional export

# Helper: build prophet model and produce forecast + backtest accuracy
def run_prophet_forecast(df_close, periods, freq):
    """
    df_close: DataFrame with index (datetime) and 'Close' column
    periods: int periods to forecast into future
    freq: 'D', 'M', or 'Y' for daily/monthly/yearly frequency
    Returns: forecast_df (with ds, yhat, yhat_lower, yhat_upper), accuracy_pct (float), model
    """
    if Prophet is None:
        raise RuntimeError("Prophet not installed. pip install prophet")

    df = df_close.reset_index().rename(columns={df_close.index.name or 'Date': 'ds', 'Close': 'y'})

    # Remove timezone info from ds column here
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)

    # Need enough points
    if len(df) < 30:
        return None, None, None

    # backtest split: last 20% as test
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    m.fit(train_df)

    # Forecast for test period to compute accuracy
    future_test = m.make_future_dataframe(periods=len(test_df), freq=freq)
    forecast_test = m.predict(future_test)

    # align predicted vs actual for test period
    pred_test = forecast_test.set_index('ds').loc[test_df['ds'].values]
    y_true = test_df['y'].values
    y_pred = pred_test['yhat'].values

    # compute MAPE; handle zeros in y_true
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred)
    except Exception:
        mape = (abs(y_true - y_pred) / (abs(y_true) + 1e-8)).mean()

    accuracy_pct = max(0.0, (1 - mape) * 100)  # clamp to >=0

    # Full forecast including future periods
    future_full = m.make_future_dataframe(periods=periods, freq=freq)
    forecast_full = m.predict(future_full)

    # Return forecast (DataFrame), accuracy %
    return forecast_full, accuracy_pct, m

# Loop through each ticker
for ticker in selected_tickers:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)

        if hist.empty:
            st.warning(f"No historical data for {ticker}. Skipping.")
            continue

        # Ensure Close column exists and index is DatetimeIndex
        if 'Close' not in hist.columns:
            st.warning(f"No 'Close' column for {ticker}. Skipping.")
            continue

        hist = hist.sort_index()
        hist.index.name = 'Date'

        # Display Stock Info
        st.subheader(f"📈 {ticker} - {stock.info.get('shortName', 'N/A')}")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"${stock.info.get('currentPrice', 'N/A')}")
        col2.metric("Market Cap", f"{stock.info.get('marketCap', 'N/A')}")
        col3.metric("52W High", f"{stock.info.get('fiftyTwoWeekHigh', 'N/A')}")
        col4.metric("52W Low", f"{stock.info.get('fiftyTwoWeekLow', 'N/A')}")

        # Save technical analysis summary
        summary = (
            f"{ticker} Technical Summary:\n"
            f"- Current Price: {stock.info.get('currentPrice', 'N/A')}\n"
            f"- Market Cap: {stock.info.get('marketCap', 'N/A')}\n"
            f"- 52W High: {stock.info.get('fiftyTwoWeekHigh', 'N/A')}\n"
            f"- 52W Low: {stock.info.get('fiftyTwoWeekLow', 'N/A')}\n"
        )
        technical_summary.append(summary)

        # Price Chart (history)
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=hist.index, y=hist['Close'], mode='lines', name='Historical Close'
        ))

        # ML Forecast (Prophet)
        forecast_df, accuracy_pct, model = None, None, None
        try:
            # Build close-only dataframe
            df_close = hist[['Close']].copy()

            # Resample according to forecast frequency
            if forecast_freq == "D":
                df_resampled = df_close.resample('D').ffill()
            elif forecast_freq == "M":
                df_resampled = df_close.resample('M').ffill()
            else:  # "Y"
                df_resampled = df_close.resample('Y').ffill()

            forecast_df, accuracy_pct, model = run_prophet_forecast(df_resampled, periods=forecast_days, freq=forecast_freq)

            if forecast_df is None:
                st.info(f"Not enough historical points to train model for {ticker} (need >=30).")
            else:
                forecast_df['ds'] = pd.to_datetime(forecast_df['ds']).dt.tz_localize(None)
                last_hist_date = pd.to_datetime(df_resampled.index.max()).tz_localize(None)

                # predicted future rows
                future_rows = forecast_df[forecast_df['ds'] > last_hist_date]

                # Add forecast line to chart (combine historical + forecast yhat for plotting)
                fig_price.add_trace(go.Scatter(
                    x=forecast_df['ds'], y=forecast_df['yhat'],
                    mode='lines', name='Forecast (yhat)', line=dict(dash='dash')
                ))
                # Add confidence band
                fig_price.add_trace(go.Scatter(
                    x=forecast_df['ds'].tolist() + forecast_df['ds'].tolist()[::-1],
                    y=forecast_df['yhat_upper'].tolist() + forecast_df['yhat_lower'].tolist()[::-1],
                    fill='toself', fillcolor='rgba(200,200,200,0.2)', line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip", showlegend=True, name='Forecast uncertainty'
                ))

                # Show accuracy
                st.markdown(f"**Forecast (next {forecast_days} {freq_option.lower()})** — Estimated model accuracy: **{accuracy_pct:.2f}%** (based on simple backtest)")

                # Show next predicted price point as quick number
                if not future_rows.empty:
                    next_pred = future_rows.iloc[0]['yhat']
                    st.metric(label=f"Predicted price on {future_rows.iloc[0]['ds'].date()}", value=f"${next_pred:.2f}")

                # store forecast summary for export
                pred_df_for_export = future_rows[['ds','yhat','yhat_lower','yhat_upper']].copy().rename(columns={'ds':'date','yhat':'predicted','yhat_lower':'lower','yhat_upper':'upper'})
                pred_df_for_export['ticker'] = ticker
                forecast_results.append(pred_df_for_export)

        except Exception as e:
            st.error(f"Forecasting error for {ticker}: {e}")

        # Plot chart
        fig_price.update_layout(title=f"{ticker} Price & Forecast", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig_price, use_container_width=True)

        # Save chart image in memory (for PDF)
        try:
            img_bytes = fig_price.to_image(format="png")
            chart_images.append((ticker, img_bytes))
        except Exception:
            # if to_image fails (no kaleido), skip saving chart image
            pass

        # Sentiment Analysis (sample headlines)
        sample_headlines = [
            f"{ticker} stock hits new high after strong earnings",
            f"Analysts worry about {ticker}'s future growth",
            f"{ticker} announces new product launch next month"
        ]
        sentiments = sentiment_pipeline(sample_headlines)

        st.markdown("#### 📰 Sentiment Analysis from Sample News")
        for headline, sentiment in zip(sample_headlines, sentiments):
            st.write(f"**{headline}** — {sentiment['label']} ({round(sentiment['score']*100,2)}%)")

        # Prepare data for export
        for headline, sentiment in zip(sample_headlines, sentiments):
            sentiment_export.append({
                "Ticker": ticker,
                "Headline": headline,
                "Sentiment": sentiment['label'],
                "Confidence (%)": round(sentiment['score']*100, 2)
            })

    except Exception as e:
        st.error(f"Error loading {ticker}: {e}")

# Convert sentiment data to DataFrame
sentiment_df = pd.DataFrame(sentiment_export)

# Export CSV
csv_buffer = io.StringIO()
sentiment_df.to_csv(csv_buffer, index=False)
st.download_button(
    label="📥 Download Sentiment Data (CSV)",
    data=csv_buffer.getvalue(),
    file_name="sentiment_data.csv",
    mime="text/csv"
)

# Optionally export forecast CSV
if forecast_results:
    all_forecasts_df = pd.concat(forecast_results, ignore_index=True)
    csv_forecast_buf = io.StringIO()
    all_forecasts_df.to_csv(csv_forecast_buf, index=False)
    st.download_button(
        label="📥 Download Forecasts (CSV)",
        data=csv_forecast_buf.getvalue(),
        file_name="forecast_predictions.csv",
        mime="text/csv"
    )

# Export PDF (keeps previous PDF layout, then includes charts if available)
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "StockLens AI Report", ln=True, align="C")

    def chapter_title(self, title):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, ln=True, align="L")
        self.ln(4)

    def chapter_body(self, body):
        self.set_font("Arial", "", 10)
        self.multi_cell(0, 6, body)
        self.ln()

pdf = PDF()
pdf.add_page()

# Add Technical Analysis
pdf.chapter_title("Technical Analysis Summary")
pdf.chapter_body("\n".join(technical_summary))

# Add Sentiment Data Table (replace unsupported characters)
pdf.chapter_title("Sentiment Analysis Data")
for _, row in sentiment_df.iterrows():
    text = f"{row['Ticker']}: {row['Headline']} - {row['Sentiment']} ({row['Confidence (%)']}%)"
    pdf.cell(0, 6, text, ln=True)

# Add Forecasts summary (if any)
if forecast_results:
    pdf.chapter_title("Forecast Predictions (sample)")
    # show first few forecast rows
    sample_forecast_for_pdf = all_forecasts_df.head(20)
    for _, r in sample_forecast_for_pdf.iterrows():
        pdf.cell(0, 6, f"{r['ticker']} | {r['date']} | Pred: {r['predicted']:.2f} | Lower: {r['lower']:.2f} | Upper: {r['upper']:.2f}", ln=True)

# Add Charts
pdf.chapter_title("Price Charts")
for ticker, img_bytes in chart_images:
    try:
        img_stream = io.BytesIO(img_bytes)
        pdf.image(img_stream, w=170)
        pdf.ln(5)
    except Exception:
        continue

# Output PDF as bytes for Streamlit download
pdf_output = bytes(pdf.output(dest="S"))

st.download_button(
    label="📄 Download Report (PDF)",
    data=pdf_output,
    file_name="stock_report.pdf",
    mime="application/pdf"
)
