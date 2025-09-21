# trading_app.py
# To run this app, save it as a python file (e.g., app.py) and run: streamlit run app.py
# Make sure to set the environment variables before running.
# Make sure to install all necessary libraries:
# pip install streamlit pandas numpy scikit-learn statsmodels optuna yfinance requests matplotlib xlsxwriter openpyxl psycopg2-binary

import streamlit as st
import datetime
import matplotlib.pyplot as plt
import numpy as np
import optuna
import os
import pandas as pd
import requests
import statsmodels.api as sm
import time
import warnings
import yfinance as yf
import io
import re
import psycopg2
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.api import VAR
from decimal import Decimal

# --- Initial Setup ---
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
st.set_page_config(layout="wide", page_title="Stock Trading & Analysis Tool")

# --- Database Connection ---
@st.cache_resource
def init_connection():
    try:
        conn = psycopg2.connect(st.secrets["postgres_url"])
        return conn
    except Exception as e:
        st.error(f"Error connecting to the database: {e}")
        return None

conn = init_connection()

def create_trades_table():
    if conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(10) NOT NULL,
                    trade_type VARCHAR(4) NOT NULL,
                    quantity INT NOT NULL,
                    price NUMERIC(10, 2) NOT NULL,
                    trade_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()

# Run table creation at app start
create_trades_table()

# --- Email Function ---
def send_trade_notification(ticker, trade_type, quantity, current_price):
    try:
        sender_email = st.secrets["gmail_user"]
        receiver_email = st.secrets["gmail_user"]  # Sending to yourself
        password = st.secrets["gmail_password"]

        subject = f"New Trade Alert: {trade_type.upper()} {ticker}"
        body = f"""
        Scott has requested a new trade:

        Ticker: {ticker}
        Type: {trade_type.upper()}
        Quantity: {quantity}
        Price: ${current_price:,.2f}
        Total Value: ${quantity * current_price:,.2f}

        This is an automated notification from the Stock Trading Simulator.
        """

        message = MIMEMultipart()
        message["From"] = sender_email
        message["To"] = receiver_email
        message["Subject"] = subject
        message.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message.as_string())
        st.sidebar.success("Trade notification email sent!")
    except Exception as e:
        st.sidebar.error(f"Failed to send email: {e}")

# --- Helper Functions ---

# Throttling for API requests
MAX_REQUESTS_PER_HOUR = 10000
requests_made = 0
start_time = datetime.datetime.now()

def throttle_request():
    global requests_made, start_time
    requests_made += 1
    if requests_made > MAX_REQUESTS_PER_HOUR:
        time_elapsed = datetime.datetime.now() - start_time
        if time_elapsed.total_seconds() < 3600:
            wait_time = 3600 - time_elapsed.total_seconds()
            st.warning(f"Rate limit likely exceeded. Sleeping for {wait_time:.2f} seconds.")
            time.sleep(wait_time)
        requests_made = 1
        start_time = datetime.datetime.now()

def create_date_features(df):
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['day'] = df['Date'].dt.day
    df['day_of_week'] = df['Date'].dt.day_of_week
    df['is_month_end'] = df['Date'].dt.is_month_end.astype('int64')
    df['is_month_start'] = df['Date'].dt.is_month_start.astype('int64')
    df['is_quarter_end'] = df['Date'].dt.is_quarter_end.astype('int64')
    df['is_quarter_start'] = df['Date'].dt.is_quarter_start.astype('int64')
    return df

def parse_and_clean_tickers(input_data):
    if isinstance(input_data, list):
        text_data = ' '.join(map(str, input_data))
    else:
        text_data = str(input_data)
    tokens = re.split(r'[\s,;\t\n]+', text_data)
    cleaned_tickers = [
        token.strip().upper() for token in tokens
        if re.fullmatch(r'[A-Z]{1,5}', token.strip())
    ]
    seen = set()
    unique_tickers = [t for t in cleaned_tickers if not (t in seen or seen.add(t))]
    return unique_tickers

@st.cache_data(ttl=3600)
def get_top_200_active_tickers(tiingo_api_key):
    url = "https://api.tiingo.com/iex"
    headers = {"Authorization": f"Token {tiingo_api_key}"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        df = pd.DataFrame(response.json())
        df = df.sort_values(by="volume", ascending=False)
        return parse_and_clean_tickers(df['ticker'].head(200).tolist())
    except Exception as e:
        st.warning(f"Failed to fetch top active tickers: {e}")
        return ["AAPL", "MSFT", "GOOG", "AMZN"]

@st.cache_data(ttl=300) # Cache for 5 minutes
def get_current_price(stock_name, tiingo_api_key):
    # Request only the last few business days to get the most recent adjusted close
    url = f"https://api.tiingo.com/tiingo/daily/{stock_name.upper()}/prices"
    headers = {'Content-Type': 'application/json', 'Authorization': f'Token {tiingo_api_key}'}
    # Use a short window (last 5 business days) to avoid large payloads and ensure we get the latest available trading day
    end_dt = end_date if 'end_date' in globals() else datetime.date.today()
    start_dt = (pd.to_datetime(end_dt) - pd.offsets.BDay(7)).strftime('%Y-%m-%d')
    params = {'startDate': start_dt, 'endDate': pd.to_datetime(end_dt).strftime('%Y-%m-%d'), 'resampleFreq': 'daily'}
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        # Tiingo returns a list of daily price dicts. Prefer 'adjClose' when available, otherwise fall back to 'close' or 'last'
        if isinstance(data, list) and len(data) > 0:
            # Find the most recent entry (should already be ordered, but sort by date to be safe)
            try:
                sorted_data = sorted(data, key=lambda x: x.get('date', ''))
            except Exception:
                sorted_data = data
            latest = sorted_data[-1]
            for key in ('adjClose', 'close', 'last'):
                if key in latest and latest[key] is not None:
                    return latest[key]
    except requests.exceptions.HTTPError as e:
        st.warning(f"Tiingo API failed with HTTP error for {stock_name}: {e}. Status code: {e.response.status_code}. Trying yfinance.")
    except Exception as e:
        st.warning(f"Tiingo API failed with an unexpected error for {stock_name}: {e}. Trying yfinance.")
    
    try:
        stock = yf.Ticker(stock_name)
        # Request a few recent days to make sure we get an adjusted close if today is a holiday/weekend
        hist = stock.history(period="7d")
        if not hist.empty:
            # Prefer 'Adj Close' if available
            if 'Adj Close' in hist.columns:
                return hist['Adj Close'].dropna().iloc[-1]
            return hist['Close'].dropna().iloc[-1]
    except Exception as e:
        st.error(f"Could not get current price for {stock_name} from yfinance: {e}")
    return None

@st.cache_data(ttl=3600)
def get_data(stock_name, end_date, tiingo_api_key):
    try:
        st.info(f"[{stock_name}] Sourcing data from Tiingo...")
        throttle_request()
        url = f"https://api.tiingo.com/tiingo/daily/{stock_name}/prices"
        headers = {'Content-Type': 'application/json', 'Authorization': f'Token {tiingo_api_key}'}
        params = {'startDate': '2015-01-01', 'endDate': end_date.strftime('%Y-%m-%d'), 'resampleFreq': 'daily'}
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200: raise Exception(f"Tiingo error: {response.status_code}")
        data = response.json()
        if not data: raise ValueError("Tiingo returned empty data.")
        df = pd.DataFrame(data)
        df = df[['date', 'adjOpen', 'adjHigh', 'adjLow', 'adjClose', 'adjVolume']].rename(columns={
            'date': 'Date', 'adjOpen': 'Open', 'adjHigh': 'High', 'adjLow': 'Low', 
            'adjClose': 'Close', 'adjVolume': 'Volume'})
        df = create_date_features(df).set_index('Date').asfreq('B').dropna()
        return df
    except Exception as e:
        st.warning(f"Tiingo failed for {stock_name}: {e}. Trying yfinance.")
        try:
            df = yf.download(stock_name, start='2015-01-01', end=end_date, progress=False)
            if not df.empty:
                return create_date_features(df.reset_index()).set_index('Date').asfreq('B').dropna()
        except Exception as yf_e:
            st.error(f"[{stock_name}] yfinance failed: {yf_e}")
    return None

# --- Trading Functions ---
def execute_trade(ticker, trade_type, quantity, price):
    if conn:
        with conn.cursor() as cur:
            rounded_price = round(price, 2)
            cur.execute(
                "INSERT INTO trades (ticker, trade_type, quantity, price) VALUES (%s, %s, %s, %s)",
                (ticker, trade_type, quantity, rounded_price)
            )
            conn.commit()
        st.sidebar.success(f"Trade for {quantity} shares of {ticker} at ${rounded_price} recorded!")
        send_trade_notification(ticker, trade_type, quantity, rounded_price)
    else:
        st.sidebar.error("Database connection not available. Trade not recorded.")

def get_trade_history():
    if conn:
        with conn.cursor() as cur:
            cur.execute("SELECT ticker, trade_type, quantity, price, trade_date FROM trades ORDER BY trade_date DESC")
            return pd.DataFrame(cur.fetchall(), columns=['Ticker', 'Type', 'Quantity', 'Price', 'Date'])
    return pd.DataFrame()

def get_portfolio():
    history = get_trade_history()
    cash_balance = 100000.0  # Starting capital
    if history.empty:
        return pd.DataFrame(columns=['Ticker', 'Shares', 'Total Cost']), cash_balance
    
    portfolio = {}
    
    # Calculate portfolio holdings
    for _, row in history.iterrows():
        ticker, trade_type, quantity, price = row['Ticker'], row['Type'], row['Quantity'], row['Price']
        
        if isinstance(price, Decimal):
            price = float(price)

        if ticker not in portfolio:
            portfolio[ticker] = {'shares': 0, 'cost': 0.0}
        
        if trade_type == 'buy':
            portfolio[ticker]['shares'] += quantity
            portfolio[ticker]['cost'] += quantity * price
            cash_balance -= quantity * price
        elif trade_type == 'sell':
            # Reduce shares and cost proportionally to maintain correct average cost
            if portfolio[ticker]['shares'] > 0:
                avg_cost_per_share = portfolio[ticker]['cost'] / portfolio[ticker]['shares']
                portfolio[ticker]['cost'] -= quantity * avg_cost_per_share
            portfolio[ticker]['shares'] -= quantity
            cash_balance += quantity * price

    # Filter out closed positions and create DataFrame
    portfolio_df = pd.DataFrame([
        {'Ticker': t, 'Shares': p['shares'], 'Total Cost': p['cost']}
        for t, p in portfolio.items() if p['shares'] > 0.001 # Use tolerance for float precision
    ])
    
    return portfolio_df, cash_balance

# --- Forecasting Helper Functions ---
def get_significant_lags(series, alpha=0.15, nlags=None):
    acf_values, confint_acf = sm.tsa.stattools.acf(series, alpha=alpha, nlags=nlags)
    pacf_values, confint_pacf = sm.tsa.stattools.pacf(series, alpha=alpha, nlags=nlags)
    significant_acf_lags = np.where(np.abs(acf_values) > confint_acf[:, 1] - acf_values)[0]
    significant_pacf_lags = np.where(np.abs(pacf_values) > confint_pacf[:, 1] - pacf_values)[0]
    return significant_acf_lags, significant_pacf_lags

def create_lagged_features(df, interpolate='bfill'):
    significant_lags_dict = {}
    for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
        significant_acf, significant_pacf = get_significant_lags(df[col])
        significant_lags_dict[col] = {'acf': significant_acf, 'pacf': significant_pacf}
        for ma_lag in significant_acf:
            if ma_lag > 0:
                df[f'{col}_ma_lag{ma_lag}'] = df[col].shift(1).rolling(window=ma_lag).mean()
        for lag in significant_pacf:
            if lag > 0:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
    df.dropna(inplace=True)
    if df.isnull().values.any():
        df = df.interpolate(method=interpolate)
    return df, significant_lags_dict

def train_test_split(df, train_size=0.80):
    x_data, y_data = df.drop(columns=['Close', 'High', 'Low', 'Open', 'Volume']), df['Close']
    split_idx = int(len(x_data) * train_size)
    x_train, x_test = x_data.iloc[:split_idx], x_data.iloc[split_idx:]
    y_train, y_test = y_data.iloc[:split_idx], y_data.iloc[split_idx:]
    return x_data, y_data, x_train, x_test, y_train, y_test

def plot_actual_vs_predicted(y_train, y_test, y_pred, model_name, stock_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_train.index, y_train, label="Training Data", color="blue", linewidth=1)
    ax.plot(y_test.index, y_test, label="Test Data (Actuals)", color="green", linewidth=1)
    ax.plot(y_test.index, y_pred, label="Predicted Test Data", color="red", linewidth=1)
    ax.legend()
    ax.set_title(f"{stock_name} - Historical Actuals vs Predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Values")
    ax.grid(True)
    return fig

def save_plot_forecast(df, rolling_forecast_df, stock_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index[-180:], df['Close'][-180:], label="Actual Close", color='blue')
    ax.plot(rolling_forecast_df['Date'], rolling_forecast_df['Predicted_Close'], label="Rolling Forecast", color='red')
    ax.set_title(f"Predicted Close Prices for {stock_name} (as of {datetime.date.today()})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price")
    ax.grid(True)
    ax.legend()
    return fig

def rolling_forecast(df, best_model, n_periods, x_data, significant_lags_dict, stock_name):
    try:
        var_model = VAR(df[['Close', 'High', 'Low', 'Open', 'Volume']])
        var_fitted = var_model.fit(ic='aic')
        
        if len(df) < var_fitted.k_ar:
            st.warning(f"Skipping {stock_name}: Not enough data for VAR model.")
            return [], df
        
        rolling_df = df.copy()
        rolling_predictions = []
        progress_bar_text = f"Generating {n_periods}-day forecast for {stock_name}..."
        progress_bar = st.progress(0, text=progress_bar_text)

        for i in range(n_periods):
            last_date = rolling_df.index[-1]
            new_date = last_date + pd.offsets.BusinessDay(1)

            var_input = rolling_df[['Close', 'High', 'Low', 'Open', 'Volume']].iloc[-var_fitted.k_ar:]
            
            if var_input.shape[0] < var_fitted.k_ar:
                st.warning(f"Insufficient data for step {i+1}. Forecasting halted.")
                break
            
            var_forecast = var_fitted.forecast(y=var_input.values, steps=1)[0]
            predicted_close_var, predicted_high, predicted_low, predicted_open, predicted_volume = var_forecast

            next_period = pd.DataFrame({
                'Close': [max(predicted_close_var, 0.01)], 
                'High': [max(predicted_high, 0.01)],
                'Low': [max(predicted_low, 0.01)], 
                'Open': [max(predicted_open, 0.01)],
                'Volume': [max(predicted_volume, 0)]
            }, index=[new_date])

            latest_data = pd.concat([rolling_df[['Close', 'High', 'Low', 'Open', 'Volume']], next_period])
            new_row = latest_data.copy()

            for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
                for lag in significant_lags_dict[col]['pacf']:
                    if lag > 0: 
                        new_row[f'{col}_lag{lag}'] = new_row[col].shift(lag).iloc[-1]
                for ma_lag in significant_lags_dict[col]['acf']:
                    if ma_lag > 0: 
                        new_row[f'{col}_ma_lag{ma_lag}'] = new_row[col].shift(1).rolling(window=ma_lag).mean().iloc[-1]

            feature_cols = new_row.columns.difference(['Close', 'High', 'Low', 'Open', 'Volume'])
            new_row = pd.DataFrame(new_row[feature_cols].values, columns=feature_cols, index=new_row.index).tail(1)
            
            new_row = new_row.reset_index().rename(columns={'index': 'Date'})
            new_row = create_date_features(new_row)
            new_row = new_row.set_index('Date').asfreq('B').dropna()
            new_row = new_row[x_data.columns]

            predicted_value = max(best_model.predict(new_row)[0], 0.01)
            rolling_predictions.append(predicted_value)

            final_row = pd.DataFrame({
                'Close': [predicted_value], 
                'High': [predicted_high], 
                'Low': [predicted_low],
                'Open': [predicted_open], 
                'Volume': [predicted_volume]
            }, index=[new_date])
            
            rolling_df = pd.concat([rolling_df, final_row])
            if i % 5 == 0 or i == n_periods -1:
                progress_bar.progress((i + 1) / n_periods, text=f"Day {i+1}/{n_periods} forecasted...")
         
        progress_bar.empty()
        return rolling_predictions, rolling_df

    except Exception as e:
        st.error(f"Error during rolling forecast: {e}")
        return [], df

def finalize_forecast_and_metrics(stock_name, rolling_predictions, df, n_periods, rolling_df=None):
    rolling_forecast_df = pd.DataFrame({
        'Date': pd.date_range(start=df.index[-1], periods=n_periods + 1, freq='B')[1:],
        'Predicted_Close': rolling_predictions})

    horizon_df = rolling_forecast_df.head(15)
    predicted_high_15_days = max(round(horizon_df['Predicted_Close'].max(), 2), 0.01)
    predicted_low_15_days = max(round(horizon_df['Predicted_Close'].min(), 2), 0.01)
    predicted_avg_15_days = max(round(horizon_df['Predicted_Close'].mean(), 2), 0.01)
    predicted_volatility_15_days = round(horizon_df['Predicted_Close'].std() / predicted_avg_15_days, 3)

    # Extract predicted Open/High/Low for next (first forecasted) day if available in rolling_df
    predicted_next_open = predicted_next_high = predicted_next_low = None
    if rolling_df is not None and not rolling_df.empty:
        try:
            base_last_date = df.index[-1]
            # Find the first row in rolling_df that is strictly after the last real data date
            future_rows = rolling_df[rolling_df.index > base_last_date]
            if not future_rows.empty:
                next_row = future_rows.iloc[0]
                predicted_next_open = max(round(float(next_row.get('Open', np.nan)), 2), 0.01) if pd.notna(next_row.get('Open', np.nan)) else df['Close'].iloc[-1]
                predicted_next_high = max(round(float(next_row.get('High', np.nan)), 2), 0.01) if pd.notna(next_row.get('High', np.nan)) else df['Close'].iloc[-1]
                predicted_next_low = max(round(float(next_row.get('Low', np.nan)), 2), 0.01) if pd.notna(next_row.get('Low', np.nan)) else df['Close'].iloc[-1]
        except Exception:
            predicted_next_open = predicted_next_high = predicted_next_low = df['Close'].iloc[-1]

    # Calculate short-term buy/sell targets, predicted return, and recommendations
    target_buy_price = round(np.mean([predicted_next_open, predicted_next_low]), 2) if predicted_next_open and predicted_next_low else df['Close'].iloc[-1]
    target_sell_price = round(np.mean([predicted_next_open, predicted_next_high]), 2) if predicted_next_open and predicted_next_high else df['Close'].iloc[-1]
    predicted_return = ((target_sell_price / target_buy_price) - 1) if target_buy_price > 0 else 0

    daily_direction = 'flat'
    if target_sell_price > target_buy_price: 
        daily_direction = 'up' if horizon_df['Predicted_Close'].iloc[0] > df['Close'].iloc[-1] else 'flat'
    elif target_sell_price < target_buy_price: 
        daily_direction = 'down' if horizon_df['Predicted_Close'].iloc[0] < df['Close'].iloc[-1] else 'flat'

    daily_recommendation = 'avoid/sell'
    if daily_direction == 'up' and predicted_return > 0.015:
        daily_recommendation = 'buy' if predicted_volatility_15_days < 0.10 else 'hold'

    # Adjust recommendation for additional conditions
    if daily_direction == 'up' and predicted_return > 0.015:
        # If predicted range looks wide relative to avg, prefer hold for safety
        intraday_strength = 0
        if predicted_next_high and predicted_next_low:
            intraday_strength = (predicted_next_high - predicted_next_low) / np.mean([predicted_next_open, predicted_next_low, predicted_next_high])
        daily_recommendation = 'avoid/sell' if intraday_strength > 0.075 else 'buy'

    # Calculate long-term buy/sell targets, predicted return, and recommendations
    long_term_buy_price = max(round((predicted_avg_15_days * (1 - (0.5 * predicted_volatility_15_days))), 2), 0.01)
    long_term_sell_price = max(round((predicted_avg_15_days * (1 + (0.5 * predicted_volatility_15_days))), 2), 0.01)
    long_term_predicted_return = ((long_term_sell_price / long_term_buy_price) - 1) if long_term_buy_price > 0 else 0

    long_term_direction = 'flat'
    if horizon_df['Predicted_Close'].iloc[-1] > long_term_buy_price: 
        long_term_direction = 'up'
    elif horizon_df['Predicted_Close'].iloc[-1] < long_term_buy_price: 
        long_term_direction = 'down'

    # Adjust recommendation for additional conditions
    long_term_recommendation = 'avoid/sell'
    if long_term_direction == 'up' and long_term_predicted_return > 0.03:
        long_term_recommendation = 'buy' if predicted_volatility_15_days < 0.125 else 'hold'

    if long_term_direction == 'up' and predicted_return > 0.03:
        # If predicted range looks wide relative to avg, prefer hold for safety
        long_term_strength = 0
        if predicted_next_high and predicted_next_low and predicted_avg_15_days > 0:
            long_term_strength = (predicted_high_15_days - predicted_low_15_days) / predicted_avg_15_days
        long_term_recommendation = 'avoid/sell' if predicted_volatility_15_days > 0.15 or long_term_strength > 0.10 else 'buy'

    summary_df = pd.DataFrame({
        'ticker_symbol': [stock_name], 
        'daily_direction': [daily_direction], 
        'daily_recommendation': [daily_recommendation],
        'target_buy_price': [target_buy_price],
        'target_sell_price': [target_sell_price],
        'predicted_return_%': [predicted_return * 100],
        'predicted_open': [predicted_next_open],
        'predicted_high': [predicted_next_high],
        'predicted_low': [predicted_next_low],
        'long_term_direction': [long_term_direction],
        'long_term_recommendation': [long_term_recommendation],
        'long_term_buy_price': [long_term_buy_price],
        'long_term_sell_price': [long_term_sell_price],
        'long_term_predicted_return_%': [long_term_predicted_return * 100],
        'predicted_high_15_day': [predicted_high_15_days],
        'predicted_low_15_day': [predicted_low_15_days], 
        'predicted_avg_15_day': [predicted_avg_15_days],
        'predicted_volatility_%': [predicted_volatility_15_days * 100]})

    return rolling_forecast_df, summary_df

def autofit_columns(df, worksheet):
    for i, column in enumerate(df.columns):
        column_width = max(df[column].astype(str).map(len).max(), len(column)) + 3
        worksheet.set_column(i, i, column_width)

# --- Main App UI ---
st.title("📈 Newberry Stock Trading & Analysis Tool")

# Get API key from environment variable or secrets
tiingo_api_key = os.getenv("TIINGO_API_KEY", st.secrets.get("tiingo_api_key"))

# --- Sidebar ---
with st.sidebar:
    st.header("👦 Trading Terminal")

    trade_ticker = st.text_input("Ticker Symbol for Trade", "MSFT").upper()
    current_price = get_current_price(trade_ticker, tiingo_api_key)
    if current_price:
        st.session_state.current_price = current_price
        st.info(f"Most recent closing price for {trade_ticker}: ${current_price:,.2f}")

    trade_type = st.radio("Trade Type", ('buy', 'sell'))
    quantity = st.number_input("Quantity", min_value=1, value=10)

    if st.button("Submit Trade", type="primary"):
        if trade_ticker and trade_type and current_price and quantity > 0:
            execute_trade(trade_ticker, trade_type, quantity, current_price)
        elif trade_ticker and trade_type and not current_price and quantity > 0:
            current_price = get_current_price(trade_ticker, tiingo_api_key)
            execute_trade(trade_ticker, trade_type, quantity, current_price)
        elif quantity == 0:
            st.error("Quantity must be greater than 0.")
        else:
            st.error("Please fill in all trade details.")

    st.markdown("---")
    st.header("⚙️ Forecasting Configuration")
    
    st.subheader("Ticker Input")
    if tiingo_api_key:
        default_tickers = get_top_200_active_tickers(tiingo_api_key)
        default_stocks = ", ".join(default_tickers)
    else:
        default_stocks = "AAPL, MSFT, GOOG, AMZN"

    stock_list_str = st.text_area("Paste Tickers for Forecasting", default_stocks, height=150)
    
    st.subheader("Forecasting Parameters")
    n_periods = st.slider("Forecast Horizon (days)", 10, 100, 45)
    max_trials = st.slider("Max Optimization Trials", 10, 100, 20)


# --- Main Content Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Portfolio Dashboard", "📈 Forecasting", "📜 Trade History"])

with tab1:
    st.header("📊 Portfolio Dashboard")
    portfolio_df, cash_balance = get_portfolio()

    if not portfolio_df.empty:
        portfolio_df['Current Price'] = portfolio_df['Ticker'].apply(lambda x: get_current_price(x, tiingo_api_key))
        portfolio_df.dropna(subset=['Current Price'], inplace=True)
        portfolio_df['Market Value'] = portfolio_df['Shares'] * portfolio_df['Current Price']
        portfolio_df['Avg Cost/Share'] = portfolio_df['Total Cost'] / portfolio_df['Shares']
        portfolio_df['Unrealized P/L'] = portfolio_df['Market Value'] - portfolio_df['Total Cost']
        
        total_market_value = portfolio_df['Market Value'].sum()
        total_account_value = total_market_value + cash_balance
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Account Value", f"${total_account_value:,.2f}")
        col2.metric("Portfolio Value", f"${total_market_value:,.2f}")
        col3.metric("Cash Balance", f"${cash_balance:,.2f}")
        
        st.dataframe(portfolio_df, width='stretch')
    else:
        st.metric("Total Account Value", f"${cash_balance:,.2f}")
        st.info("No open positions. Place a trade to get started!")

with tab2:
    st.header("📈 Forecasting Tool")
    if st.button("🚀 Run Forecast"):
        stock_list = parse_and_clean_tickers(stock_list_str)
        
        if not tiingo_api_key:
            st.error("`TIINGO_API_KEY` is not set. Please configure it in your secrets.")
        elif not stock_list:
            st.error("No valid stock tickers found. Please enter tickers.")
        else:
            st.success(f"Found {len(stock_list)} unique tickers to process: {', '.join(stock_list)}")
            today = datetime.date.today()
            end_date = today + pd.offsets.BusinessDay(1)
            
            forecast_results = {}
            summary_results = []
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                for stock_name in stock_list:
                    st.subheader(f"Processing: {stock_name}")
                    
                    with st.spinner(f"Fetching data for {stock_name}..."):
                        df = get_data(stock_name, end_date, tiingo_api_key)
                    
                    if df is None or len(df) < 100:
                        st.error(f"Could not retrieve sufficient data for {stock_name}. Skipping.")
                        continue

                    with st.spinner(f"Creating features for {stock_name}..."):
                        df, s_lags_dict = create_lagged_features(df)
                        x_data, y_data, x_train, x_test, y_train, y_test = train_test_split(df)

                    def objective_elastic_net(trial):
                        alpha = trial.suggest_float('alpha', 1e-6, 1e-1, log=True)
                        l1_ratio = trial.suggest_float('l1_ratio', 0, 1)
                        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
                        model.fit(x_train, y_train)
                        return mean_absolute_error(y_test, model.predict(x_test))

                    study = optuna.create_study(direction='minimize')
                    with st.spinner(f"Optimizing model for {stock_name}..."):
                        study.optimize(objective_elastic_net, n_trials=max_trials)
                    
                    best_params = study.best_params
                    best_model = ElasticNet(**best_params)
                    best_model.fit(x_train, y_train)
                    mae = study.best_value
                    
                    y_pred = best_model.predict(x_test)
                    fig_actual = plot_actual_vs_predicted(y_train, y_test, y_pred, "ElasticNet", stock_name)
                    st.write(f"**Model Performance for {stock_name}** (MAE: {mae:.4f})")
                    st.pyplot(fig_actual)
                    
                    st.success(f"Best Model MAE for {stock_name}: **{mae:.4f}**")

                    with st.spinner(f"Forecasting {stock_name}..."):
                        X_full, y_full = df.drop(columns=['Close', 'High', 'Low', 'Open', 'Volume']), df['Close']
                        best_model.fit(X_full, y_full)
                        
                        rolling_predictions, rolling_df = rolling_forecast(df, best_model, n_periods, x_data, s_lags_dict, stock_name)
                        rolling_forecast_df, summary_df = finalize_forecast_and_metrics(stock_name, rolling_predictions, df, n_periods, rolling_df)

                    forecast_results[stock_name] = rolling_forecast_df
                    summary_results.append(summary_df)

                    fig_forecast = save_plot_forecast(df, rolling_forecast_df, stock_name)
                    st.pyplot(fig_forecast)
                    
                    sheet_name = re.sub(r'[\[\]\*:\?/\\ ]', '_', stock_name)[:31]
                    rolling_forecast_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    worksheet = writer.sheets[sheet_name]
                    autofit_columns(rolling_forecast_df, worksheet)

                    st.markdown("---")

                if summary_results:
                    st.header("📊 Consolidated Summary")
                    combined_summary = pd.concat(summary_results, ignore_index=True)
                    st.dataframe(combined_summary, width='stretch')

                    combined_summary.to_excel(writer, sheet_name='Summary_Stats', index=False)
                    summary_worksheet = writer.sheets["Summary_Stats"]
                    autofit_columns(combined_summary, summary_worksheet)
            
            if summary_results:
                output.seek(0)
                st.download_button(
                    label="📥 Download All Forecasts as Excel",
                    data=output,
                    file_name=f"stock_forecasts_{today}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

with tab3:
    st.header("📜 Trade History")
    trade_history_df = get_trade_history()
    if not trade_history_df.empty:
        st.dataframe(trade_history_df, width='stretch')
    else:
        st.info("No trades have been made yet.")
