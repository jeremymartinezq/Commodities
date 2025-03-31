import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from matplotlib.ticker import MaxNLocator
import requests
import json

# Tradovate API credentials
API_URL = 'https://demo.tradovate.com/v1'
API_KEY = 'your_api_key_here'
API_SECRET = 'your_api_secret_here'

# Risk management parameters
POSITION_SIZE = 1
STOP_LOSS_PERCENTAGE = 0.02
TAKE_PROFIT_PERCENTAGE = 0.05


# Tradovate API authentication
def authenticate():
    response = requests.post(
        f"{API_URL}/auth/accesstoken",
        json={"username": API_KEY, "password": API_SECRET},
        headers={'Content-Type': 'application/json'}
    )
    response.raise_for_status()
    return response.json().get('accessToken')


# Place order on Tradovate with risk management
def place_order(access_token, action, qty, symbol, stop_loss_price=None, take_profit_price=None):
    response = requests.post(
        f"{API_URL}/order",
        json={
            "action": action,
            "quantity": qty,
            "symbol": symbol,
            "orderType": "market",
            "stopLossPrice": stop_loss_price,
            "takeProfitPrice": take_profit_price
        },
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {access_token}'
        }
    )
    response.raise_for_status()
    return response.json()


# List of symbols for commodities, stock indexes, and cryptocurrencies
symbols = {
    'Gold': 'GC=F',
    'Silver': 'SI=F',
    'Crude Oil': 'CL=F',
    'Natural Gas': 'NG=F',
    'S&P 500': '^GSPC',
    'Dow Jones': '^DJI',
    'NASDAQ': '^IXIC',
    'Bitcoin': 'BTC-USD',
    'Ethereum': 'ETH-USD'
}


# Download historical data
def download_data(symbol, start='2023-01-01', end='2024-09-05', interval='1h'):
    data = yf.download(symbol, start=start, end=end, interval=interval)
    data['Datetime'] = data.index.tz_localize(None)
    data.reset_index(drop=True, inplace=True)
    return data


# Prepare and fit models for each symbol
def process_symbol(symbol, data):
    # Rename columns for Prophet
    data.rename(columns={'Datetime': 'ds', 'Close': 'y'}, inplace=True)

    # Fit the Prophet model
    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    model.fit(data[['ds', 'y']])

    # Create future dataframe for projections
    future = model.make_future_dataframe(periods=365 * 24, freq='H')
    future['ds'] = future['ds'].dt.tz_localize(None)

    forecast = model.predict(future)
    forecast_data = forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Projected_Close'})
    forecast_data['SMA_24'] = forecast_data['Projected_Close'].rolling(window=24).mean()

    # Calculate 24-hour SMA for historical data
    data.set_index('ds', inplace=True)
    data['SMA_24'] = data['y'].rolling(window=24).mean()

    # Define buy/sell signals
    generate_signals(data)
    generate_signals(forecast_data, historical=False)

    # Calculate Trend Wave Oscillator (TWO)
    calculate_two(data, column_name='y')
    forecast_data.set_index('Date', inplace=True)
    calculate_two(forecast_data, column_name='Projected_Close')

    return data, forecast_data


# Define buy/sell signals
def generate_signals(df, historical=True):
    df['Signal'] = 0
    if historical:
        df.loc[df['y'] > df['SMA_24'], 'Signal'] = 1
        df.loc[df['y'] < df['SMA_24'], 'Signal'] = -1
    else:
        df['SMA_24'] = df['Projected_Close'].rolling(window=24).mean()
        df.loc[df['Projected_Close'] > df['SMA_24'], 'Signal'] = 1
        df.loc[df['Projected_Close'] < df['SMA_24'], 'Signal'] = -1


# Calculate Trend Wave Oscillator (TWO)
def calculate_two(df, short_window=24, long_window=96, column_name='y'):
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")
    df['TWO'] = df[column_name].rolling(window=short_window).mean() - df[column_name].rolling(window=long_window).mean()
    df['TWO_Buy_Marker'] = np.where(df['TWO'] > 0, df['TWO'], np.nan)
    df['TWO_Sell_Marker'] = np.where(df['TWO'] < 0, df['TWO'], np.nan)


# Plot buy/sell signals
def plot_signals(ax, df, historical=True):
    color_map = {'buy': ('g', 'lime'), 'sell': ('r', 'darkred')}
    suffix = '(Historical)' if historical else '(Projected)'
    price_col = 'y' if historical else 'Projected_Close'
    ax.scatter(df[df['Signal'] == 1].index, df[df['Signal'] == 1][price_col],
               marker='^', color=color_map['buy'][0 if historical else 1], label=f'Buy Signal {suffix}')
    ax.scatter(df[df['Signal'] == -1].index, df[df['Signal'] == -1][price_col],
               marker='v', color=color_map['sell'][0 if historical else 1], label=f'Sell Signal {suffix}')


# Plot data
def plot_data(data, forecast_data, symbol):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 20), gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.5})

    ax1.plot(data.index, data['y'], label=f'Historical {symbol} Prices', color='blue')
    ax1.plot(forecast_data.index, forecast_data['Projected_Close'], label=f'Projected {symbol} Prices', color='red')
    ax1.plot(data.index, data['SMA_24'], label='24-Hour SMA (Historical)', color='green', linestyle='--')
    ax1.plot(forecast_data.index, forecast_data['SMA_24'], label='24-Hour SMA (Projected)', color='orange',
             linestyle='--')

    # Plot signals
    plot_signals(ax1, data, historical=True)
    plot_signals(ax1, forecast_data, historical=False)

    # Projection start marker
    ax1.axvline(x=data.index.max(), color='red', linestyle='--', label='Projection Start')
    ax1.set_title(f'{symbol} Price Projection Until 2025 with Buy/Sell Signals (Hourly Data)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel(f'{symbol} Price (USD)')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Plot TWO
    ax2.plot(data.index, data['TWO'], label=f'Trend Wave Oscillator (Historical)', color='blue')
    ax2.plot(forecast_data.index, forecast_data['TWO'], label=f'Trend Wave Oscillator (Projected)', color='red')

    # Plot TWO markers
    ax2.scatter(data.index, data['TWO_Buy_Marker'], color='green', marker='^', label='Buy Marker (Historical)')
    ax2.scatter(data.index, data['TWO_Sell_Marker'], color='red', marker='v', label='Sell Marker (Historical)')
    ax2.scatter(forecast_data.index, forecast_data['TWO_Buy_Marker'], color='lime', marker='^',
                label='Buy Marker (Projected)')
    ax2.scatter(forecast_data.index, forecast_data['TWO_Sell_Marker'], color='darkred', marker='v',
                label='Sell Marker (Projected)')

    ax2.axhline(0, color='black', linestyle='--', label='Zero Line')
    ax2.set_title('Trend Wave Oscillator with Markers')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('TWO')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=6))

    plt.tight_layout(pad=2.0)
    plt.show()


# Main execution
for name, symbol in symbols.items():
    data = download_data(symbol)
    data, forecast_data = process_symbol(symbol, data)
    plot_data(data, forecast_data, name)
    # Uncomment to execute trades
    # execute_trades(data, forecast_data)
