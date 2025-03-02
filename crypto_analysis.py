# crypto_analysis.py
# This file contains functions for retrieving and analyzing cryptocurrency data
# It also includes forecasting functionality for crypto prices

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import logging
from asset_config import AssetConfig
import warnings

# Suppress statsmodels warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_crypto_data(ticker, start_date, end_date):
    """
    Retrieve historical cryptocurrency data from Yahoo Finance
    
    Parameters:
    - ticker (str): The cryptocurrency ticker symbol (e.g., 'BTC-USD')
    - start_date (datetime): Start date for historical data
    - end_date (datetime): End date for historical data
    
    Returns:
    - pandas.DataFrame: DataFrame containing historical price and volume data
    """
    # Make sure the ticker is in the right format for cryptocurrencies
    if '-USD' not in ticker:
        ticker = f"{ticker}-USD"
    
    # Convert dates to string format required by yfinance
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # Download data from Yahoo Finance
    crypto = yf.Ticker(ticker)
    data = crypto.history(start=start_str, end=end_str)
    
    # Reset timezone information to prevent timezone comparison issues
    data.index = data.index.tz_localize(None)
    
    # If data is empty, raise an exception
    if data.empty:
        raise Exception(f"No data found for crypto ticker {ticker}")
    
    # Add some cryptocurrency-specific indicators
    data = add_crypto_indicators(data)
    
    # Get info about the cryptocurrency if available
    try:
        info = crypto.info
        data.attrs['info'] = {
            'name': info.get('name', ticker.replace('-USD', '')),
            'market_cap': info.get('marketCap', 'N/A'),
            'volume': info.get('volume24Hr', 'N/A'),
            'circulating_supply': info.get('circulatingSupply', 'N/A')
        }
    except:
        data.attrs['info'] = {
            'name': ticker.replace('-USD', ''),
            'market_cap': 'N/A',
            'volume': 'N/A',
            'circulating_supply': 'N/A'
        }
    
    return data

def add_crypto_indicators(data):
    """
    Add technical indicators to cryptocurrency data
    
    Parameters:
    - data (pandas.DataFrame): DataFrame containing price data
    
    Returns:
    - pandas.DataFrame: DataFrame with added indicators
    """
    try:
        # Get crypto indicator settings
        indicators = AssetConfig.get_config('crypto')['indicators']
        
        # Calculate Relative Strength Index (RSI)
        rsi_period = indicators['rsi_period']
        delta = data['Close'].diff()
        gain = delta.mask(delta < 0, 0)
        loss = -delta.mask(delta > 0, 0)
        
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate Moving Average Convergence Divergence (MACD)
        fast_period, slow_period, signal_period = indicators['macd_periods']
        exp1 = data['Close'].ewm(span=fast_period, adjust=False).mean()
        exp2 = data['Close'].ewm(span=slow_period, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal'] = data['MACD'].ewm(span=signal_period, adjust=False).mean()
        
        # Calculate Moving Averages
        for period in indicators['ma_periods']:
            data[f'MA_{period}'] = data['Close'].rolling(window=period).mean()
        
        # Calculate Bollinger Bands
        bb_period = indicators['bb_period']
        data['BB_Middle'] = data['Close'].rolling(window=bb_period).mean()
        data['BB_Std'] = data['Close'].rolling(window=bb_period).std()
        data['BB_Upper'] = data['BB_Middle'] + 2 * data['BB_Std']
        data['BB_Lower'] = data['BB_Middle'] - 2 * data['BB_Std']
        
        # Volume Moving Average
        volume_ma_period = indicators.get('volume_ma_period', 20)
        data['Volume_MA'] = data['Volume'].rolling(window=volume_ma_period).mean()
        
        return data
    except Exception as e:
        logger.error(f"Error adding crypto indicators: {str(e)}")
        return data  # Return original data if there was an error

def analyze_crypto(crypto_data):
    """
    Perform analysis on cryptocurrency data
    
    Parameters:
    - crypto_data (pandas.DataFrame): DataFrame containing historical crypto data
    
    Returns:
    - dict: Dictionary containing various metrics and analysis results
    """
    # Extract the last 30 days of data for calculating short-term metrics
    last_30_days = crypto_data.tail(30)
    
    # Calculate returns over different time periods
    current_price = crypto_data['Close'].iloc[-1]
    
    # Calculate volatility (standard deviation of returns)
    daily_returns = crypto_data['Close'].pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(365)  # Annualized volatility
    
    # Calculate different time period returns
    # 7-day return
    seven_day_price = crypto_data['Close'].iloc[-8] if len(crypto_data) > 7 else crypto_data['Close'].iloc[0]
    seven_day_return = (current_price / seven_day_price - 1) * 100
    
    # 30-day return
    thirty_day_price = crypto_data['Close'].iloc[-31] if len(crypto_data) > 30 else crypto_data['Close'].iloc[0]
    thirty_day_return = (current_price / thirty_day_price - 1) * 100
    
    # YTD return
    today = datetime.now()
    ytd_start = datetime(today.year, 1, 1)
    ytd_data = crypto_data.loc[crypto_data.index >= ytd_start]
    if not ytd_data.empty:
        ytd_return = (current_price / ytd_data['Close'].iloc[0] - 1) * 100
    else:
        ytd_return = 0
    
    # Get the last RSI value
    last_rsi = crypto_data['RSI'].iloc[-1]
    
    # Determine market sentiment based on RSI
    if last_rsi > 70:
        sentiment = "Overbought"
    elif last_rsi < 30:
        sentiment = "Oversold"
    elif last_rsi > 50:
        sentiment = "Bullish"
    else:
        sentiment = "Bearish"
    
    # Get crypto info if available
    info = crypto_data.attrs.get('info', {})
    
    # Calculate market cap if we have the data
    if info.get('circulating_supply') != 'N/A' and not pd.isna(current_price):
        market_cap = info.get('circulating_supply') * current_price
    else:
        market_cap = info.get('market_cap', 'N/A')
    
    # Format market cap for display
    if isinstance(market_cap, (int, float)) and not pd.isna(market_cap):
        if market_cap >= 1e9:
            market_cap_str = f"${market_cap / 1e9:.2f}B"
        elif market_cap >= 1e6:
            market_cap_str = f"${market_cap / 1e6:.2f}M"
        else:
            market_cap_str = f"${market_cap:.2f}"
    else:
        market_cap_str = 'N/A'
    
    # Return metrics as a dictionary
    return {
        'Name': info.get('name', 'Unknown'),
        'Market Cap': market_cap_str,
        '24h Volume': info.get('volume', 'N/A'),
        'Circulating Supply': f"{info.get('circulating_supply', 'N/A'):,}",
        '7D Return': f"{seven_day_return:.2f}%",
        '30D Return': f"{thirty_day_return:.2f}%",
        'YTD Return': f"{ytd_return:.2f}%",
        'Volatility (Annual)': f"{volatility * 100:.2f}%",
        'RSI': f"{last_rsi:.2f}",
        'Sentiment Score': sentiment
    }

def forecast_crypto(crypto_data, days=30, method='ARIMA'):
    """
    Forecast cryptocurrency prices for a specified number of days ahead
    
    Parameters:
    - crypto_data (pandas.DataFrame): DataFrame containing historical crypto data
    - days (int): Number of days to forecast
    - method (str): Forecasting method to use ('ARIMA', 'Prophet', 'Linear Regression', or 'LSTM')
    
    Returns:
    - pandas.DataFrame: DataFrame containing the forecasted prices
    """
    # Extract the closing prices
    close_prices = crypto_data['Close']
    
    # Create a date range for the forecast period
    last_date = crypto_data.index[-1]
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
    
    if method == 'ARIMA':
        # Fit ARIMA model
        try:
            arima_params = AssetConfig.get_config('crypto')['arima_params']
            model = ARIMA(close_prices, order=arima_params)
            model_fit = model.fit()
            
            # Make forecast
            forecast_result = model_fit.forecast(steps=days)
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Forecast': forecast_result,
                'Forecast_Lower': forecast_result * 0.95,  # Approximate lower bound
                'Forecast_Upper': forecast_result * 1.05   # Approximate upper bound
            }).set_index('Date')
        except Exception as e:
            logger.error(f"ARIMA forecast error: {str(e)}")
            # Fallback to simpler method
            last_price = close_prices.iloc[-1]
            avg_change = close_prices.pct_change().dropna().mean()
            
            forecast_result = [last_price]
            for _ in range(days):
                next_price = forecast_result[-1] * (1 + avg_change)
                forecast_result.append(next_price)
            
            forecast_result = forecast_result[1:]  # Remove the first value (which is the last actual price)
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                'Date': forecast_dates,
                'Forecast': forecast_result
            }).set_index('Date')
        
    elif method == 'Prophet':
        # Prepare data for Prophet
        prophet_data = pd.DataFrame({
            'ds': close_prices.index,
            'y': close_prices.values
        })
        
        # Fit Prophet model
        model = Prophet(interval_width=0.95)
        model.fit(prophet_data)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=days)
        
        # Make forecast
        forecast = model.predict(future)
        
        # Filter to only the forecast period
        future_forecast = forecast.iloc[-days:]
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'Date': future_forecast['ds'].values,
            'Forecast': future_forecast['yhat'].values,
            'Forecast_Lower': future_forecast['yhat_lower'].values,
            'Forecast_Upper': future_forecast['yhat_upper'].values
        }).set_index('Date')
        
    elif method == 'Linear Regression':
        # Simple linear regression forecast
        # Create a sequence of numbers from 0 to the length of the data
        X = np.array(range(len(close_prices))).reshape(-1, 1)
        y = close_prices.values
        
        # Fit linear regression
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict future values
        future_X = np.array(range(len(close_prices), len(close_prices) + days)).reshape(-1, 1)
        forecast_result = model.predict(future_X)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': forecast_result
        }).set_index('Date')
        
    elif method == 'LSTM':
        # LSTM requires more complex implementation
        # For simplicity, we'll use a placeholder implementation
        # In a real application, you would implement a proper LSTM model
        
        # Simple extrapolation based on recent trend
        last_price = close_prices.iloc[-1]
        avg_change = close_prices.pct_change().dropna().mean()
        
        forecast_result = [last_price]
        for _ in range(days):
            next_price = forecast_result[-1] * (1 + avg_change)
            forecast_result.append(next_price)
        
        forecast_result = forecast_result[1:]  # Remove the first value (which is the last actual price)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecast': forecast_result
        }).set_index('Date')
    
    else:
        raise ValueError(f"Unknown forecasting method: {method}")
    
    # Combine historical data with forecast
    historical_df = pd.DataFrame({
        'Close': close_prices,
        'Forecast': np.nan
    })
    
    forecast_df = pd.DataFrame({
        'Close': np.nan,
        'Forecast': forecast_df['Forecast']
    }, index=forecast_df.index)
    
    # Add confidence intervals if available
    if 'Forecast_Lower' in forecast_df.columns and 'Forecast_Upper' in forecast_df.columns:
        forecast_df['Forecast_Lower'] = forecast_df['Forecast_Lower']
        forecast_df['Forecast_Upper'] = forecast_df['Forecast_Upper']
    
    combined_df = pd.concat([historical_df, forecast_df])
    
    return combined_df