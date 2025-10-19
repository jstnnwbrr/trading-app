# trading_app.py
# To run this app, save it as a python file (e.g., app.py) and run: streamlit run app.py
# Make sure to set the environment variables before running.
# Make sure to install all necessary libraries:
# pip install streamlit pandas numpy scikit-learn statsmodels optuna yfinance requests matplotlib xlsxwriter openpyxl psycopg2-binary plotly kaleido

import streamlit as st
import datetime
import matplotlib.pyplot as plt
import numpy as np
import optuna
import plotly.express as px
import plotly.graph_objects as go
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
    cash_balance = 52785.13  # Starting capital
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


@st.cache_data(ttl=3600)
def compute_account_performance(trades_df, tiingo_api_key, initial_cash=100000.0):
    """Simulate account value over time using trade history and index comparisons.
    Returns a DataFrame indexed by business day with columns: AccountValue, DJIA, S&P500, Nasdaq
    """
    if trades_df is None or trades_df.empty:
        return None

    # Ensure trade dates are datetimes without tz
    trades = trades_df.copy()
    trades['Date'] = pd.to_datetime(trades['Date']).dt.tz_localize(None)
    trades = trades.sort_values('Date')

    start_date = trades['Date'].dt.date.min()
    end_date = datetime.date.today()
    biz_index = pd.date_range(start=start_date, end=end_date, freq='B')

    # Unique tickers from history
    tickers = trades['Ticker'].unique().tolist()
    # Map index symbols to ETFs available via Tiingo
    index_map = {'^DJI': 'DIA', '^GSPC': 'SPY', '^IXIC': 'QQQ'}
    all_symbols = tickers + list(index_map.values())

    # Fetch adjusted close prices per symbol from Tiingo
    price_frames = []
    headers = {'Content-Type': 'application/json', 'Authorization': f'Token {tiingo_api_key}'}
    for sym in all_symbols:
        try:
            throttle_request()
            url = f"https://api.tiingo.com/tiingo/daily/{sym}/prices"
            params = {'startDate': start_date.strftime('%Y-%m-%d') if isinstance(start_date, (datetime.date, datetime.datetime)) else str(start_date),
                      'endDate': (end_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d'),
                      'resampleFreq': 'daily'}
            resp = requests.get(url, headers=headers, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                continue
            df_sym = pd.DataFrame(data)
            if 'date' in df_sym.columns:
                df_sym['date'] = pd.to_datetime(df_sym['date']).dt.tz_localize(None)
                # Prefer adjClose, then close
                if 'adjClose' in df_sym.columns:
                    series = df_sym.set_index('date')['adjClose']
                elif 'close' in df_sym.columns:
                    series = df_sym.set_index('date')['close']
                else:
                    continue
                series.name = sym
                price_frames.append(series)
        except Exception:
            # skip missing symbol
            continue

    if not price_frames:
        df = pd.DataFrame(index=biz_index)
        df['AccountValue'] = initial_cash
        df['DJIA'] = np.nan
        df['SP500'] = np.nan
        df['Nasdaq'] = np.nan
        return df

    price_df = pd.concat(price_frames, axis=1)
    # Reindex to biz_index (datetime) and forward/backfill
    price_df = price_df.reindex(pd.to_datetime(biz_index)).ffill().bfill()
    # Ensure no duplicate index entries (keep last) to avoid .loc returning a Series
    if price_df.index.duplicated().any():
        price_df = price_df[~price_df.index.duplicated(keep='last')]
    # Rename ETF index columns back to index labels for plotting
    # reverse map
    reverse_index_map = {v: k for k, v in index_map.items()}
    for etf_sym, idx_label in reverse_index_map.items():
        if etf_sym in price_df.columns:
            price_df.rename(columns={etf_sym: idx_label}, inplace=True)

    # Prepare simulation
    holdings = {t: 0 for t in tickers}
    cash = initial_cash
    account_values = []
    trade_iter = trades.to_dict('records')
    trade_idx = 0

    def _safe_extract(series, dt_index):
        """Return a single float value for series.loc[dt_index]. If multiple rows exist, return the last non-null value."""
        try:
            val = series.loc[dt_index]
            # If a Series/ndarray is returned due to duplicate index, pick the last non-null
            if isinstance(val, (pd.Series, pd.DataFrame, list, tuple)):
                # convert to Series and dropna
                s = pd.Series(val).dropna()
                if s.empty:
                    return np.nan
                return float(s.iloc[-1])
            # scalar
            if pd.isna(val):
                return np.nan
            return float(val)
        except KeyError:
            return np.nan
        except Exception:
            try:
                return float(val)
            except Exception:
                return np.nan

    for current_date in biz_index:
        # apply all trades up to and including current_date
        while trade_idx < len(trade_iter) and pd.to_datetime(trade_iter[trade_idx]['Date']).date() <= current_date.date():
            tr = trade_iter[trade_idx]
            sym = tr['Ticker']
            qty = int(tr['Quantity'])
            price = float(tr['Price'])
            if tr['Type'].lower() == 'buy':
                holdings[sym] = holdings.get(sym, 0) + qty
                cash -= qty * price
            else:
                holdings[sym] = holdings.get(sym, 0) - qty
                cash += qty * price
            trade_idx += 1

        # compute market value using Adj Close prices
        market_value = 0.0
        for sym, qty in holdings.items():
            if qty == 0:
                continue
            try:
                px = price_df.get(sym)
                if px is None:
                    px_val = np.nan
                else:
                    px_val = _safe_extract(px, current_date)
                if pd.isna(px_val):
                    px_val = 0.0
            except Exception:
                px_val = 0.0
            market_value += qty * float(px_val)

        total_value = cash + market_value
        account_values.append({'Date': current_date, 'AccountValue': total_value,
                               'DJIA': _safe_extract(price_df.get('^DJI'), current_date) if '^DJI' in price_df.columns else np.nan,
                               'SP500': _safe_extract(price_df.get('^GSPC'), current_date) if '^GSPC' in price_df.columns else np.nan,
                               'Nasdaq': _safe_extract(price_df.get('^IXIC'), current_date) if '^IXIC' in price_df.columns else np.nan})

    out = pd.DataFrame(account_values).set_index('Date')
    return out

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

    predicted_avg_3_days = max(round(horizon_df['Predicted_Close'].head(3).mean(), 2), 0.01)

    predicted_high_15_days = max(round(horizon_df['Predicted_Close'].max(), 2), 0.01)
    predicted_low_15_days = max(round(horizon_df['Predicted_Close'].min(), 2), 0.01)
    # Compute the second-lowest predicted close within the 15-day horizon to avoid overreacting to a single deep dip
    horizon_vals = horizon_df['Predicted_Close'].dropna().values
    if len(horizon_vals) >= 2:
        sorted_vals = np.sort(horizon_vals)
        # second-lowest is at index 1 (0-based); ensure it's at least a small positive number
        predicted_second_lowest_15_days = max(round(float(sorted_vals[1]), 2), 0.01)
    else:
        predicted_second_lowest_15_days = predicted_low_15_days
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
    target_buy_price = round(np.mean([(0.75 * predicted_next_open) + (0.25 * predicted_next_low), predicted_next_high/1.01]), 2) if predicted_next_open != 0.01 else df['Close'].iloc[-1]
    target_sell_price = round(np.mean([predicted_next_open, predicted_next_high]), 2) if predicted_next_open and predicted_next_high else df['Close'].iloc[-1]
    target_return_price = round(np.mean([target_sell_price, predicted_avg_3_days]), 2) if predicted_next_open and predicted_next_high else df['Close'].iloc[-1]
    predicted_return = ((target_return_price / target_buy_price) - 1) if target_buy_price > 0 else 0

    short_term_direction = 'flat'
    if predicted_return > 0: 
        short_term_direction = 'up' 
    elif predicted_return < 0: 
        short_term_direction = 'down'

    short_term_recommendation = 'avoid/sell'
    if short_term_direction == 'up' and predicted_return > 0.005:
        short_term_recommendation = 'buy' if predicted_volatility_15_days < 0.10 else 'hold'

    # Adjust recommendation for additional conditions
    if short_term_direction == 'up' and predicted_return > 0.005:
        # If predicted range looks wide relative to avg, prefer hold for safety
        intraday_strength = 0
        if predicted_next_high and predicted_next_low:
            intraday_strength = (predicted_next_high - predicted_next_low) / np.mean([predicted_next_open, predicted_next_low, predicted_next_high])
        short_term_recommendation = 'avoid/sell' if intraday_strength > 0.08 else 'buy'

    # Calculate long-term sell targets, predicted return, and recommendations
    long_term_sell_price = max(round((predicted_avg_15_days * (1 + (0.5 * predicted_volatility_15_days))), 2), 0.01)
    long_term_predicted_return = ((long_term_sell_price / target_buy_price) - 1) if target_buy_price > 0 else 0

    long_term_direction = 'flat'
    if horizon_df['Predicted_Close'].iloc[-1] > target_buy_price: 
        long_term_direction = 'up'
    if predicted_low_15_days < target_buy_price: 
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

    # If a dip within a certain threshold is foreseen in the 15-day horizon, avoid buying
    # Use the second-lowest value to soften responses to a single temporary dip.
    # If the second-lowest is not far below the target buy price, treat the dip as temporary and keep short-term recommendation.
    DIP_TOLERANCE = 0.02  # 2% tolerance; tweak as needed or expose to UI
    if long_term_direction == 'down':
        try:
            # If the second-lowest is significantly below the target buy price (beyond tolerance), cancel short-term buy
            if predicted_second_lowest_15_days < (target_buy_price * (1 - DIP_TOLERANCE)):
                short_term_recommendation = 'avoid/sell'
            else:
                # treat as temporary dip: do not override previously determined short_term_recommendation
                pass
        except Exception:
            short_term_recommendation = 'avoid/sell'

    # If predicted return is very high (greater than 50%), likely too good to be true - avoid
    if predicted_return > 0.50:
        short_term_recommendation = 'avoid/sell'
        long_term_recommendation = 'avoid/sell'

    summary_df = pd.DataFrame({
        'ticker_symbol': [stock_name], 
        'short_term_direction': [short_term_direction], 
        'short_term_recommendation': [short_term_recommendation],
        'target_buy_price': [target_buy_price],
        'target_sell_price': [target_sell_price],
        'short_term_predicted_return_%': [predicted_return * 100],
        'predicted_open': [predicted_next_open],
        'predicted_high': [predicted_next_high],
        'predicted_low': [predicted_next_low],
        'long_term_direction': [long_term_direction],
        'long_term_recommendation': [long_term_recommendation],
        'long_term_sell_price': [long_term_sell_price],
        'long_term_predicted_return_%': [long_term_predicted_return * 100],
        'predicted_high_15_day': [predicted_high_15_days],
        'predicted_low_15_day': [predicted_low_15_days], 
        'predicted_second_lowest_15_day': [predicted_second_lowest_15_days],
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
    st.info("App pre-populates the top 200 most active stocks, but feel free to paste your own tickers - as messy as they may be!")

    if tiingo_api_key:
        default_tickers = get_top_200_active_tickers(tiingo_api_key)
        default_stocks = ", ".join(default_tickers)
    else:
        default_stocks = "AAPL, MSFT, GOOG, AMZN"

    stock_list_str = st.text_area("Paste Stock Tickers Here", default_stocks, height=150, help="Paste a list of tickers. Don't worry about formatting or weeding out supplemental information like recent returns, prices, etc. The app will clean and de-duplicate the list for you.")
    do_not_buy_list_str = st.text_area("Do Not Buy List (Optional)", "APPN, BL, BTG, IOVA", height=100, help="Tickers you do not wish to buy...")

    st.subheader("Forecasting Parameters")
    n_periods = st.slider("Forecast Horizon (days)", 10, 100, 45)
    max_trials = st.slider("Max Optimization Trials", 10, 100, 20)
    st.subheader("Performance Chart Options")
    indices_to_show = st.multiselect("Benchmarks to include", ['DJIA', 'SP500', 'Nasdaq'], default=['DJIA', 'SP500', 'Nasdaq'], help="Choose which benchmark indices to show in the portfolio performance chart")

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
        # --- Account performance vs major indices ---
        try:
            trade_history = get_trade_history()
            perf_df = compute_account_performance(trade_history, tiingo_api_key, initial_cash=cash_balance)
            if perf_df is not None:
                # UI controls for date range, normalization and extra metrics
                # Use safe defaults when perf_df has no rows
                try:
                    min_date = perf_df.index.min().date() if not perf_df.empty else datetime.date.today()
                    max_date = perf_df.index.max().date() if not perf_df.empty else datetime.date.today()
                except Exception:
                    min_date = datetime.date.today()
                    max_date = datetime.date.today()

                date_range = st.date_input("Performance date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
                normalize = st.checkbox("Normalize series to 100 (relative performance)", value=True)
                show_metrics = st.checkbox("Show cumulative returns and drawdown", value=False)

                # Normalize/validate date_range into start_dt/end_dt (always datetimes)
                if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                    start_dt, end_dt = date_range
                else:
                    start_dt = date_range
                    end_dt = date_range
                start_dt = pd.to_datetime(start_dt).normalize()
                end_dt = pd.to_datetime(end_dt).normalize()

                # Build boolean mask in a vectorized way (avoid ambiguous truth values)
                idx = pd.to_datetime(perf_df.index).normalize()
                mask = (idx >= start_dt) & (idx <= end_dt)
                plot_window = perf_df.loc[mask].copy()

                # Allow user to pick benchmarks from sidebar multiselect; fall back to the sidebar variable
                current_indices_selection = st.session_state.get('indices_to_show', indices_to_show) if isinstance(st.session_state.get('indices_to_show', indices_to_show), (list, tuple)) else indices_to_show
                desired_indices = set(current_indices_selection)
                available_indices = [col for col in ['DJIA', 'SP500', 'Nasdaq'] if col in plot_window.columns]
                selected_indices = [col for col in available_indices if col in desired_indices]

                cols_to_plot = ['AccountValue'] + selected_indices
                # If AccountValue not present, try to fallback to first numeric column
                if 'AccountValue' not in plot_window.columns and len(plot_window.columns) > 0:
                    cols_to_plot[0] = plot_window.columns[0]

                # Ensure columns exist before slicing
                plot_df = plot_window[cols_to_plot].copy() if not plot_window.empty and all(c in plot_window.columns for c in cols_to_plot) else plot_window.copy()
                abs_df = plot_df.copy()

                # Summary metrics (AccountValue) — total return, annualized return, max drawdown
                summary_metrics = {'total_return_pct': None, 'annualized_return_pct': None, 'max_drawdown_pct': None}
                try:
                    acct = abs_df['AccountValue'].dropna()
                    if len(acct) >= 2:
                        start_val = float(acct.iloc[0])
                        end_val = float(acct.iloc[-1])
                        total_return = (end_val / start_val) - 1.0 if start_val != 0 else 0.0
                        # trading days count
                        trading_days = len(acct) - 1
                        annualized = (1 + total_return) ** (252.0 / trading_days) - 1.0 if trading_days > 0 else 0.0
                        running_max = acct.cummax()
                        drawdowns = (acct / running_max) - 1.0
                        max_dd = drawdowns.min()
                        summary_metrics['total_return_pct'] = round(total_return * 100.0, 2)
                        summary_metrics['annualized_return_pct'] = round(annualized * 100.0, 2)
                        summary_metrics['max_drawdown_pct'] = round(max_dd * 100.0, 2)
                except Exception:
                    pass

                # Show summary metrics
                try:
                    mcol1, mcol2, mcol3 = st.columns(3)
                    mcol1.metric("Total Return", f"{summary_metrics['total_return_pct'] if summary_metrics['total_return_pct'] is not None else 'N/A'}%")
                    mcol2.metric("Annualized Return", f"{summary_metrics['annualized_return_pct'] if summary_metrics['annualized_return_pct'] is not None else 'N/A'}%")
                    mcol3.metric("Max Drawdown", f"{summary_metrics['max_drawdown_pct'] if summary_metrics['max_drawdown_pct'] is not None else 'N/A'}%")
                except Exception:
                    pass

                if normalize:
                    for col in plot_df.columns:
                        first_valid = plot_df[col].dropna().iloc[0] if not plot_df[col].dropna().empty else np.nan
                        if pd.notna(first_valid) and first_valid != 0:
                            plot_df[col] = (plot_df[col] / first_valid) * 100.0
                        else:
                            plot_df[col] = np.nan

                # Main performance chart
                fig = go.Figure()
                for col in plot_df.columns:
                    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[col], mode='lines', name=col))
                y_label = 'Index (Normalized to 100)' if normalize else 'Value'
                fig.update_layout(title='Account Performance vs Major Indices', xaxis_title='Date', yaxis_title=y_label, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)

                # Download performance as Excel with sheets and optional images
                try:
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as ew:
                        # Performance data
                        abs_df.to_excel(ew, sheet_name='Performance_Data')

                        # Cumulative returns sheet
                        try:
                            cret = abs_df.copy()
                            for col in cret.columns:
                                first = cret[col].dropna().iloc[0] if not cret[col].dropna().empty else np.nan
                                cret[col] = (cret[col] / first - 1) * 100.0 if pd.notna(first) and first != 0 else np.nan
                            cret.to_excel(ew, sheet_name='Cumulative_Returns')
                        except Exception:
                            pass

                        # Drawdown sheet
                        try:
                            dd = abs_df.copy()
                            for col in dd.columns:
                                series = dd[col]
                                running_max = series.cummax()
                                dd[col] = (series / running_max - 1) * 100.0
                            dd.to_excel(ew, sheet_name='Drawdown')
                        except Exception:
                            pass

                        # Insert main chart image if possible (best-effort)
                        try:
                            img_bytes = fig.to_image(format='png', width=1200, height=600)
                            workbook = ew.book
                            worksheet = workbook.add_worksheet('Charts')
                            worksheet.insert_image('B2', 'chart.png', {'image_data': io.BytesIO(img_bytes)})
                        except Exception:
                            pass
                        ew.close()
                    excel_buffer.seek(0)
                    st.download_button('Download Performance as Excel', data=excel_buffer, file_name='performance.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                except Exception:
                    # If Excel export fails, ignore and continue
                    pass

                # Additional metrics: cumulative returns and drawdown
                if show_metrics:
                    # Cumulative returns (%) from start
                    cret = abs_df.copy()
                    for col in cret.columns:
                        first = cret[col].dropna().iloc[0] if not cret[col].dropna().empty else np.nan
                        cret[col] = (cret[col] / first - 1) * 100.0 if pd.notna(first) and first != 0 else np.nan

                    fig_cr = go.Figure()
                    for col in cret.columns:
                        fig_cr.add_trace(go.Scatter(x=cret.index, y=cret[col], mode='lines', name=col))
                    fig_cr.update_layout(title='Cumulative Return (%)', xaxis_title='Date', yaxis_title='Return %', template='plotly_white')
                    st.plotly_chart(fig_cr, use_container_width=True)

                    # Drawdown (%)
                    dd = abs_df.copy()
                    for col in dd.columns:
                        series = dd[col]
                        running_max = series.cummax()
                        dd[col] = (series / running_max - 1) * 100.0

                    fig_dd = go.Figure()
                    for col in dd.columns:
                        fig_dd.add_trace(go.Scatter(x=dd.index, y=dd[col], mode='lines', name=col))
                    fig_dd.update_layout(title='Drawdown (%)', xaxis_title='Date', yaxis_title='Drawdown %', template='plotly_white')
                    st.plotly_chart(fig_dd, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render performance chart: {e}")
    else:
        st.metric("Total Account Value", f"${cash_balance:,.2f}")
        st.info("No open positions. Place a trade to get started!")

with tab2:
    st.header("📈 Forecasting Tool")
    if st.button("🚀 Run Forecast"):
        stock_list = parse_and_clean_tickers(stock_list_str)
        do_not_buy_list = parse_and_clean_tickers(do_not_buy_list_str) if do_not_buy_list_str else []
        do_not_buy_list = [ticker.strip().upper() for ticker in do_not_buy_list if ticker.strip()]
        stock_list = [ticker for ticker in stock_list if ticker not in do_not_buy_list]
        
        if not tiingo_api_key:
            st.error("`TIINGO_API_KEY` is not set. Please configure it in your secrets.")
        elif not stock_list:
            st.error("No valid stock tickers found. Please enter tickers.")
        else:
            st.success(f"Found {len(stock_list)} unique tickers to process: {', '.join(stock_list)}")
            st.success(f"The following {len(do_not_buy_list)} tickers were identified as Do Not Buy: {', '.join(do_not_buy_list)}")

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
                    
                    st.dataframe(summary_df, use_container_width=True)

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
