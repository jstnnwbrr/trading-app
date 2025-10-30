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


def clean_transaction_history(uploaded_file):
    """Read an E*TRADE transaction CSV (either a path or a file-like object) and
    return a cleaned DataFrame ready for DB insertion and an initial cash balance.

    This adapts the user-provided function to accept uploaded files from Streamlit.
    """
    # Read CSV from path or file-like
    if isinstance(uploaded_file, str):
        df = pd.read_csv(uploaded_file, skiprows=3)
    else:
        # streamlit file_uploader returns a BytesIO-like object
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, skiprows=3)

    # Standardize column names and drop unneeded columns if present
    df.rename(columns={'TransactionDate': 'trade_date', 'Symbol': 'ticker', 'Quantity': 'quantity', 'Price': 'price'}, inplace=True)
    for col in ['TransactionType', 'SecurityType', 'Commission', 'Description']:
        if col in df.columns:
            try:
                df.drop(columns=[col], inplace=True)
            except Exception:
                pass

    df['trade_date'] = pd.to_datetime(df['trade_date'], utc=True, errors='coerce')
    df['trade_type'] = df['quantity'].apply(lambda x: 'buy' if x > 0 else 'sell')
    df['quantity'] = df['quantity'].abs()

    # Defensive handling for Amount column and specific tickers/dates used to compute initial cash balance
    amount_sum = df.loc[df['quantity'] == 0, 'Amount'].sum() if 'Amount' in df.columns else 0.0
    sphy_sum_0402 = 0.0
    sphy_sum_0409 = 0.0
    msbnk_qty_sum = 0
    try:
        if 'SPHY' in df['ticker'].values:
            sphy_sum_0402 = (df.loc[(df['ticker'] == 'SPHY') & (df['trade_date'].dt.strftime('%Y-%m-%d') == '2025-04-02'), 'price'] * df.loc[(df['ticker'] == 'SPHY') & (df['trade_date'].dt.strftime('%Y-%m-%d') == '2025-04-02'), 'quantity']).sum()
            sphy_sum_0409 = (df.loc[(df['ticker'] == 'SPHY') & (df['trade_date'].dt.strftime('%Y-%m-%d') == '2025-04-09'), 'price'] * df.loc[(df['ticker'] == 'SPHY') & (df['trade_date'].dt.strftime('%Y-%m-%d') == '2025-04-09'), 'quantity']).sum()
    except Exception:
        pass
    try:
        if 'MSBNK' in df['ticker'].values and 'quantity' in df.columns:
            msbnk_qty_sum = df.loc[df['ticker'] == 'MSBNK', 'quantity'].sum()
    except Exception:
        pass

    initial_cash_balance = sphy_sum_0402 + sphy_sum_0409 + amount_sum + (msbnk_qty_sum if not pd.isna(msbnk_qty_sum) else 0)

    # Continue cleaning: drop Amount if exists, remove unwanted tickers/zero-quantity rows
    if 'Amount' in df.columns:
        try:
            df.drop(columns=['Amount'], inplace=True)
        except Exception:
            pass

    df = df.loc[df['ticker'] != 'SPHY'] if 'ticker' in df.columns else df
    df = df.loc[df['quantity'] != 0]
    df = df.loc[df['ticker'] != 'MSBNK'] if 'ticker' in df.columns else df

    df = df.sort_values(by=['trade_date','ticker'], ascending=True)
    df.reset_index(drop=True, inplace=True)
    df['id'] = df.index + 1
    # keep columns in the expected order for insertion
    keep_cols = [c for c in ['id', 'ticker', 'trade_type', 'quantity', 'price', 'trade_date'] if c in df.columns]
    df = df[keep_cols]

    # Format trade_date to the compact ISO style used previously (microseconds truncated to 5 digits)
    if 'trade_date' in df.columns:
        micro = df['trade_date'].dt.strftime('%f').where(df['trade_date'].notna(), '')
        micro5 = micro.str[:5]
        date_time_no_frac = df['trade_date'].dt.strftime('%Y-%m-%d %H:%M:%S').where(df['trade_date'].notna(), '')
        df['trade_date_str'] = ''
        mask = df['trade_date'].notna()
        df.loc[mask, 'trade_date_str'] = date_time_no_frac.loc[mask] + '.' + micro5.loc[mask] + '+00'
        df['trade_date'] = df['trade_date_str']
        df.drop(columns=['trade_date_str'], inplace=True)

    df = df.loc[df['ticker'].isna() == False]

    return df, float(initial_cash_balance)

def get_portfolio(initial_cash_bal=52014.64):
    history = get_trade_history()
    cash_balance = initial_cash_bal  # Starting capital
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


@st.cache_data(ttl=3600)
def get_price_on_date(symbol, date, tiingo_api_key):
    """Fetch adjClose (or close) for a single symbol on a specific date using Tiingo.
    Returns float or np.nan. Cached for 1 hour.
    """
    try:
        # Normalize date to YYYY-MM-DD string so caching keys are stable
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        throttle_request()
        headers = {'Content-Type': 'application/json', 'Authorization': f'Token {tiingo_api_key}'}
        url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
        params = {'startDate': date_str, 'endDate': date_str, 'resampleFreq': 'daily'}
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return np.nan
        # Tiingo returns list of day dicts
        rec = data[-1]
        for key in ('adjClose', 'close', 'last'):
            if key in rec and rec[key] is not None:
                return float(rec[key])
    except Exception:
        return np.nan
    return np.nan


def impute_spikes(series, multiplier=2.5):
    """Replace any value that is >= multiplier * previous_day with previous_day's value.
    Returns a new Series.
    """
    try:
        s = series.copy().astype(float)
        # previous valid value (use forward-fill then shift to get last valid prior to current)
        prev = s.fillna(method='ffill').shift(1)
        # avoid division by zero warnings; treat prev==0 as spike (will set to prev which is 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = s / prev
        mask = ratio >= float(multiplier)
        s.loc[mask] = prev.loc[mask]
        return s
    except Exception:
        return series


def warm_up_price_cache(tickers, days=10, tiingo_api_key=None):
    """Prefetch recent prices for the supplied tickers to populate the cached get_price_on_date results.
    Limits to `days` business days ending today. This is best-effort and silent on errors.
    """
    if not tickers or tiingo_api_key is None:
        return
    try:
        # use business days
        dates = pd.bdate_range(end=datetime.date.today(), periods=days)
        for sym in list(dict.fromkeys(tickers))[:50]:  # limit to first 50 tickers to avoid huge warming
            for d in dates:
                try:
                    # call cached function to warm
                    _ = get_price_on_date(sym, d.strftime('%Y-%m-%d'), tiingo_api_key)
                except Exception:
                    # ignore individual failures
                    continue
    except Exception:
        return

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
    do_not_buy_list_str = st.text_area("Do Not Buy List (Optional)", "APPN, BL, BTG, GLD, GLDM, ICCM, IOVA, PSLV, QID, QQQU, RDDT, SGOL, SLV, SPXU, TQQQ, TZA, ULTY, VIST", height=100, help="Tickers you do not wish to buy...")

    st.subheader("Forecasting Parameters")
    n_periods = st.slider("Forecast Horizon (days)", 10, 100, 45)
    max_trials = st.slider("Max Optimization Trials", 10, 100, 20)
    st.subheader("Performance Chart Options")
    indices_to_show = st.multiselect("Benchmarks to include", ['DJIA', 'SP500', 'Nasdaq'], default=['DJIA', 'SP500', 'Nasdaq'], help="Choose which benchmark indices to show in the portfolio performance chart")

# --- Main Content Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Portfolio Dashboard", "📈 Forecasting", "📜 Trade History"])

with tab1:
    st.header("📊 Portfolio Dashboard")
    # Use initial cash from session if present (set when a CSV is uploaded and processed),
    # otherwise fall back to the hard-coded default used elsewhere in the app.
    initial_cash = st.session_state.get('initial_cash', 52014.64)

    portfolio_df, cash_balance = get_portfolio(initial_cash)

    if not portfolio_df.empty:
        portfolio_df['Current Price'] = portfolio_df['Ticker'].apply(lambda x: get_current_price(x, tiingo_api_key))
        portfolio_df.dropna(subset=['Current Price'], inplace=True)
        portfolio_df['Market Value'] = portfolio_df['Shares'] * portfolio_df['Current Price']
        portfolio_df['Avg Cost/Share'] = portfolio_df['Total Cost'] / portfolio_df['Shares']
        portfolio_df['Unrealized Gain/Loss'] = portfolio_df['Market Value'] - portfolio_df['Total Cost']
        
        total_market_value = portfolio_df['Market Value'].sum()
        total_account_value = total_market_value + cash_balance
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Account Value", f"${total_account_value:,.2f}")
        col2.metric("Portfolio Value", f"${total_market_value:,.2f}")
        col3.metric("Cash Balance", f"${cash_balance:,.2f}")
        
        st.dataframe(portfolio_df, width='stretch',)
        # --- Account performance vs major indices ---
        try:
            trade_history = get_trade_history()
            perf_df = compute_account_performance(trade_history, tiingo_api_key, initial_cash=cash_balance)
            if perf_df is not None:
                # UI controls for date range, normalization and extra metrics
                # Warm-up price cache once per session for current holdings to speed diagnostics/scenarios
                try:
                    if not st.session_state.get('price_cache_warmed', False):
                        holdings_list = portfolio_df['Ticker'].tolist() if not portfolio_df.empty else []
                        warm_up_price_cache(holdings_list, days=10, tiingo_api_key=tiingo_api_key)
                        st.session_state['price_cache_warmed'] = True
                except Exception:
                    pass
                # Use safe defaults when perf_df has no rows
                try:
                    min_date = perf_df.index.min().date() if not perf_df.empty else datetime.date.today()
                    max_date = perf_df.index.max().date() if not perf_df.empty else datetime.date.today()
                except Exception:
                    min_date = datetime.date.today()
                    max_date = datetime.date.today()

                date_range = st.date_input("Performance date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
                # Always show real-dollar account value by scaling normalized series to the real ending account value
                normalize = True

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
                # Reassign abs_df to use imputed plot_df so all downstream charts/exports use cleaned values
                abs_df = plot_df.copy()

                # Summary metrics (AccountValue) — total return and annualized return (recomputed after imputation)
                summary_metrics = {'total_return_pct': None, 'annualized_return_pct': None}
                try:
                    if 'AccountValue' in abs_df.columns:
                        acct = abs_df['AccountValue'].dropna()
                        if len(acct) >= 2:
                            start_val = float(acct.iloc[0])
                            end_val = float(acct.iloc[-1])
                            total_return = (end_val / start_val) - 1.0 if start_val != 0 else 0.0
                            # trading days count
                            trading_days = len(acct) - 1
                            annualized = (1 + total_return) ** (252.0 / trading_days) - 1.0 if trading_days > 0 else 0.0
                            summary_metrics['total_return_pct'] = round(total_return * 100.0, 2)
                            summary_metrics['annualized_return_pct'] = round(annualized * 100.0, 2)
                except Exception:
                    pass

                # Show summary metrics (use imputed values)
                try:
                    mcol1, mcol2 = st.columns(2)
                    mcol1.metric("Total Return", f"{summary_metrics['total_return_pct'] if summary_metrics['total_return_pct'] is not None else 'N/A'}%")
                    mcol2.metric("Annualized Return", f"{summary_metrics['annualized_return_pct'] if summary_metrics['annualized_return_pct'] is not None else 'N/A'}%")
                except Exception:
                    pass

                # Apply spike imputation to AccountValue before normalization so normalization uses cleaned values
                try:
                    if 'AccountValue' in plot_df.columns:
                        plot_df['AccountValue'] = impute_spikes(plot_df['AccountValue'], multiplier=2.5)
                except Exception:
                    pass

                # Normalize (always on) into a separate DataFrame to preserve raw values
                plot_df_norm = plot_df.copy()
                for col in plot_df_norm.columns:
                    first_valid = plot_df_norm[col].dropna().iloc[0] if not plot_df_norm[col].dropna().empty else np.nan
                    if pd.notna(first_valid) and first_valid != 0:
                        plot_df_norm[col] = (plot_df_norm[col] / first_valid) * 100.0
                    else:
                        plot_df_norm[col] = np.nan

                # Always scale normalized series to a real-dollar anchor so the chart shows approximate account values
                plot_to_show = plot_df_norm.copy()
                y_label = 'Index (Normalized to 100)'

                # Determine anchor (prefer computed total_account_value, otherwise use last AccountValue in abs_df)
                try:
                    anchor_value = float(total_account_value) if 'total_account_value' in locals() else None
                except Exception:
                    anchor_value = None
                try:
                    if anchor_value is None and 'AccountValue' in abs_df.columns and not abs_df['AccountValue'].dropna().empty:
                        anchor_value = float(abs_df['AccountValue'].dropna().iloc[-1])
                except Exception:
                    anchor_value = None

                # abs_df already reflects plot_df (imputed) values
                # Ensure AccountValue exists in plot_df_norm
                if 'AccountValue' not in plot_df_norm.columns and 'AccountValue' in abs_df.columns:
                    plot_df_norm['AccountValue'] = (abs_df['AccountValue'] / abs_df['AccountValue'].dropna().iloc[0]) * 100.0 if not abs_df['AccountValue'].dropna().empty else plot_df_norm.iloc[:, 0]

                if anchor_value is not None and 'AccountValue' in plot_df_norm.columns and not plot_df_norm['AccountValue'].dropna().empty:
                    norm_last = float(plot_df_norm['AccountValue'].dropna().iloc[-1])
                    if norm_last != 0:
                        scaling_factor = anchor_value / norm_last
                        plot_to_show = plot_df_norm * scaling_factor
                        y_label = f'Approx. Account Value (scaled to ${anchor_value:,.0f})'
                    else:
                        st.warning('Cannot scale: normalized AccountValue last value is zero.')
                else:
                    st.warning('Cannot scale to account value: missing anchor/account value.')

                # Main performance chart
                color_map = {
                    'AccountValue': '#6a0dad',  # purple
                    'Account': '#6a0dad',
                    'DJIA': '#1f77b4',
                    '^DJI': '#1f77b4',
                    'SP500': '#d62728',
                    '^GSPC': '#d62728',
                    'Nasdaq': '#2ca02c',
                    '^IXIC': '#2ca02c'
                }

                fig = go.Figure()
                for col in plot_to_show.columns:
                    color = color_map.get(col, None)
                    if color is not None:
                        fig.add_trace(go.Scatter(x=plot_to_show.index, y=plot_to_show[col], mode='lines', name=col, line=dict(color=color)))
                    else:
                        fig.add_trace(go.Scatter(x=plot_to_show.index, y=plot_to_show[col], mode='lines', name=col))
                fig.update_layout(title='Account Performance vs Major Indices', xaxis_title='Date', yaxis_title=y_label, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)

                # --- Diagnostic decomposition UI ---
                with st.expander("🔎 Analysis & Simulation", expanded=True):
                    diag_col1, diag_col2 = st.columns([1, 2])
                    with diag_col1:
                        try:
                            default_diag = plot_window.index.date[-1] if not plot_window.empty else max_date
                        except Exception:
                            default_diag = max_date
                        diag_date = st.date_input('Pick a historical trade date to analyze', value=default_diag)
                        diag_date = pd.to_datetime(diag_date).normalize()
                        exclude_ticker = st.text_input('Exclude ticker for scenario (leave blank for none)', '')
                        run_scenario = st.button('Run Exclude-Ticker Scenario')
                    with diag_col2:
                        st.write('Trades & Holdings on Selected Date')
                        # Build decomposition table for the selected date
                        try:
                            trades_history = get_trade_history()
                            if trades_history is None or trades_history.empty:
                                st.info('No trade history available.')
                            else:
                                th = trades_history.copy()
                                th['Date'] = pd.to_datetime(th['Date']).dt.tz_localize(None)
                                th = th.sort_values('Date')
                                # Display trades that occurred on or around the diagnosis date (±3 days)
                                try:
                                    start_window = diag_date - pd.Timedelta(days=3)
                                    end_window = diag_date + pd.Timedelta(days=3)
                                    trades_near = th[(th['Date'] >= start_window) & (th['Date'] <= end_window)].copy()
                                    if not trades_near.empty:
                                        trades_near_display = trades_near[['Date', 'Ticker', 'Type', 'Quantity', 'Price']].sort_values('Date', ascending=False)
                                        st.write(f"Trades within ±3 days of {diag_date.date()}:")
                                        st.dataframe(trades_near_display)
                                    else:
                                        st.info(f"No trades within ±3 days of {diag_date.date()}")
                                except Exception:
                                    # If diag_date is invalid or other error, just skip the nearby trades display
                                    pass

                                holdings_temp = {}
                                for _, r in th.iterrows():
                                    if r['Date'].date() <= diag_date.date():
                                        sym = r['Ticker']
                                        qty = int(r['Quantity'])
                                        if r['Type'].lower() == 'buy':
                                            holdings_temp[sym] = holdings_temp.get(sym, 0) + qty
                                        else:
                                            holdings_temp[sym] = holdings_temp.get(sym, 0) - qty
                                holdings_list = [{'Ticker': k, 'Shares': v} for k, v in holdings_temp.items() if v != 0]
                                if holdings_list:
                                    contrib_rows = []
                                    for row in holdings_list:
                                        sym = row['Ticker']
                                        shares = row['Shares']
                                        px = get_price_on_date(sym, diag_date, tiingo_api_key)
                                        contrib = shares * (px if not pd.isna(px) else 0.0)
                                        contrib_rows.append({'Ticker': sym, 'Shares': shares, 'Price': px, 'Market Value': contrib})
                                    contrib_df = pd.DataFrame(contrib_rows).sort_values('Market Value', ascending=False)
                                    st.dataframe(contrib_df)
                                else:
                                    st.info('No holdings on that date.')
                                # Diagnostic: show original vs imputed AccountValue for the diag_date and previous day
                                try:
                                    # find matching index in perf_df for diag_date
                                    perf_idx = pd.to_datetime(perf_df.index)
                                    mask_date = perf_idx.normalize() == diag_date
                                    if mask_date.any():
                                        perf_loc = perf_df.index[mask_date][0]
                                        orig_val = perf_df.loc[perf_loc, 'AccountValue'] if 'AccountValue' in perf_df.columns else None
                                    else:
                                        orig_val = None
                                    # imputed value from plot_df (we applied imputation earlier)
                                    plot_idx = pd.to_datetime(plot_df.index)
                                    mask_plot = plot_idx.normalize() == diag_date
                                    if mask_plot.any() and 'AccountValue' in plot_df.columns:
                                        plot_loc = plot_df.index[mask_plot][0]
                                        imputed_val = plot_df.loc[plot_loc, 'AccountValue']
                                    else:
                                        imputed_val = None
                                    # previous day values for context
                                    prev_orig = None
                                    prev_imputed = None
                                    if mask_date.any():
                                        # try get previous row in perf_df
                                        idx_pos = list(perf_df.index).index(perf_loc)
                                        if idx_pos > 0:
                                            prev_idx = perf_df.index[idx_pos - 1]
                                            prev_orig = perf_df.loc[prev_idx, 'AccountValue'] if 'AccountValue' in perf_df.columns else None
                                    if mask_plot.any():
                                        ppos = list(plot_df.index).index(plot_loc)
                                        if ppos > 0:
                                            prev_plot_idx = plot_df.index[ppos - 1]
                                            prev_imputed = plot_df.loc[prev_plot_idx, 'AccountValue']
                                except Exception:
                                    pass
                        except Exception as e:
                            st.warning(f'Could not compute trades & holdings: {e}')

                    # Scenario: re-run a lean simulation excluding one ticker
                    if run_scenario:
                        try:
                            original_trades = get_trade_history()
                            if original_trades is None or original_trades.empty:
                                st.info('No trades to simulate.')
                            else:
                                ex = exclude_ticker.strip().upper()
                                if ex:
                                    filtered = original_trades[original_trades['Ticker'].str.upper() != ex].copy()
                                else:
                                    filtered = original_trades.copy()
                                if filtered.empty:
                                    st.info('No trades remain after excluding that ticker.')
                                else:
                                    scenario_perf = compute_account_performance(filtered, tiingo_api_key, initial_cash=cash_balance)
                                    if scenario_perf is None or scenario_perf.empty:
                                        st.warning('Scenario simulation returned no data.')
                                    else:
                                        # align and normalize scenario for overlay
                                        scen_plot = scenario_perf.copy()
                                        scen_plot = scen_plot.reindex(plot_window.index).ffill().bfill()
                                        if 'AccountValue' in scen_plot.columns and not scen_plot['AccountValue'].dropna().empty:
                                            # Impute spikes in the scenario AccountValue as well
                                            scen_plot['AccountValue'] = impute_spikes(scen_plot['AccountValue'], multiplier=2.5)
                                            # Normalize scenario to 100 then scale using the same scaling_factor as main plot (if available)
                                            scen_norm = scen_plot.copy()
                                            first_val = scen_norm['AccountValue'].dropna().iloc[0]
                                            if first_val != 0:
                                                scen_norm['AccountValue'] = (scen_norm['AccountValue'] / first_val) * 100.0
                                                # Apply main scaling_factor if we computed it above; otherwise compute a local one
                                                try:
                                                    sf = scaling_factor
                                                except Exception:
                                                    norm_last_local = float(scen_norm['AccountValue'].dropna().iloc[-1]) if not scen_norm['AccountValue'].dropna().empty else None
                                                    sf = (anchor_value / norm_last_local) if anchor_value is not None and norm_last_local and norm_last_local != 0 else 1.0
                                                scen_scaled = scen_norm * sf
                                                overlay_fig = fig
                                                acct_color = color_map.get('AccountValue', '#6a0dad')
                                                overlay_fig.add_trace(go.Scatter(x=scen_scaled.index, y=scen_scaled['AccountValue'], mode='lines', name=f'Scenario w/o {ex or "(none)"}', line=dict(color=acct_color, dash='dash')))
                                                # Compute scenario summary metrics (total return, annualized return) using the imputed, un-normalized scen_plot
                                                scen_metrics = {'total_return_pct': None, 'annualized_return_pct': None}
                                                try:
                                                    scen_acct = scen_plot['AccountValue'].dropna()
                                                    if len(scen_acct) >= 2:
                                                        s_start = float(scen_acct.iloc[0])
                                                        s_end = float(scen_acct.iloc[-1])
                                                        s_total_return = (s_end / s_start) - 1.0 if s_start != 0 else 0.0
                                                        s_trading_days = len(scen_acct) - 1
                                                        s_annualized = (1 + s_total_return) ** (252.0 / s_trading_days) - 1.0 if s_trading_days > 0 else 0.0
                                                        scen_metrics['total_return_pct'] = round(s_total_return * 100.0, 2)
                                                        scen_metrics['annualized_return_pct'] = round(s_annualized * 100.0, 2)
                                                except Exception:
                                                    pass

                                                # Display the overlay and scenario metrics (match main dashboard metric style)
                                                st.plotly_chart(overlay_fig, use_container_width=True)
                                                try:
                                                    sm1, sm2 = st.columns(2)
                                                    sm1.metric(f"Scenario Total Return (w/o {ex or '(none)'})", f"{scen_metrics['total_return_pct'] if scen_metrics['total_return_pct'] is not None else 'N/A'}%")
                                                    sm2.metric(f"Scenario Annualized Return (w/o {ex or '(none)'})", f"{scen_metrics['annualized_return_pct'] if scen_metrics['annualized_return_pct'] is not None else 'N/A'}%")
                                                except Exception:
                                                    pass
                                            else:
                                                st.warning('Scenario series has zero first value; cannot normalize for overlay.')
                                        else:
                                            st.warning('Scenario AccountValue series missing; cannot overlay.')
                        except Exception as e:
                            st.warning(f'Error running scenario: {e}')

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

                        # Drawdown sheet removed per simplification request

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
    # CSV upload and DB replace UI
    uploaded_file = st.file_uploader("Upload E*TRADE transaction CSV (export) to replace trade history", type=['csv'])
    if uploaded_file is not None:
        st.info(f"Uploaded file: {getattr(uploaded_file, 'name', 'uploaded CSV')}")
        if st.button("Preview cleaned data"):
            try:
                cleaned_df, initial_cash = clean_transaction_history(uploaded_file)
                # Persist computed initial cash so portfolio calculations use it
                st.session_state['initial_cash'] = float(initial_cash)
                st.dataframe(cleaned_df.head(200), width='stretch', hide_index=True)
            except Exception as e:
                st.error(f"Failed to parse/clean uploaded file: {e}")

        st.markdown("---")
        st.markdown("### Replace trades table with uploaded file")
        confirm = st.checkbox("I confirm I want to DELETE existing trades and replace them with the uploaded file")
        if confirm:
            if st.button("Replace trades table (DELETE & INSERT)"):
                if not conn:
                    st.error("Database connection not available. Cannot replace trades.")
                else:
                    try:
                        cleaned_df, initial_cash = clean_transaction_history(uploaded_file)
                        # Persist computed initial cash so portfolio calculations use it after replace
                        st.session_state['initial_cash'] = float(initial_cash)
                        with conn.cursor() as cur:
                            # truncate and reset serials
                            cur.execute("TRUNCATE TABLE trades RESTART IDENTITY CASCADE;")
                            insert_stmt = "INSERT INTO trades (ticker, trade_type, quantity, price, trade_date) VALUES (%s, %s, %s, %s, %s)"
                            for _, r in cleaned_df.iterrows():
                                ticker = r.get('ticker')
                                trade_type = r.get('trade_type')
                                qty = int(r.get('quantity')) if not pd.isna(r.get('quantity')) else 0
                                price = float(r.get('price')) if not pd.isna(r.get('price')) else 0.0
                                trade_date = r.get('trade_date') if 'trade_date' in r.index else None
                                cur.execute(insert_stmt, (ticker, trade_type, qty, round(price, 2), trade_date))
                            conn.commit()
                        st.success("Trades table replaced successfully.")
                    except Exception as e:
                        st.error(f"Failed to replace trades table: {e}")

    # Show current trade history after optional updates
    trade_history_df = get_trade_history()
    if not trade_history_df.empty:
        st.dataframe(trade_history_df, width='stretch')
    else:
        st.info("No trades have been made yet.")
