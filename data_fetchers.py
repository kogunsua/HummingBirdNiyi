# data_fetchers.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import fredapi
from typing import Optional
from datetime import date, timedelta
from config import Config

class EconomicIndicators:
    def __init__(self):
        self.fred = fredapi.Fred(api_key=Config.FRED_API_KEY)
        self._initialize_indicators()
    
    def _initialize_indicators(self):
        """Initialize FRED indicators with descriptions and frequencies"""
        self.indicator_details = {
            # Original indicators (Unchanged)
            'GDP': {
                'series_id': 'GDP',
                'description': 'Gross Domestic Product',
                'frequency': 'Quarterly',
                'units': 'Billions of Dollars'
            },
            'UNRATE': {
                'series_id': 'UNRATE',
                'description': 'Unemployment Rate',
                'frequency': 'Monthly',
                'units': 'Percent'
            },
            'CPIAUCSL': {
                'series_id': 'CPIAUCSL',
                'description': 'Consumer Price Index',
                'frequency': 'Monthly',
                'units': 'Index 1982-1984=100'
            },
            'DFF': {
                'series_id': 'DFF',
                'description': 'Federal Funds Rate',
                'frequency': 'Daily',
                'units': 'Percent'
            },
            'IEF': {
                'series_id': 'IEF',
                'description': 'iShares 7-10 Year Treasury Bond ETF',
                'frequency': 'Daily',
                'units': 'USD'
            },
            # New indicator added
            'POLSENT': {
                'series_id': 'POLSENT',
                'description': 'Political Sentiment Index',
                'frequency': 'Daily',
                'units': 'Sentiment Score',
                'is_sentiment': True
            }
        }

    # New method added for sentiment data
    def _get_sentiment_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Generate political sentiment data"""
        try:
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Generate sentiment values between -100 (very negative) to 100 (very positive)
            np.random.seed(42)  # For reproducible results
            sentiment_values = np.random.normal(loc=0, scale=30, size=len(dates))
            sentiment_values = np.clip(sentiment_values, -100, 100)
            
            df = pd.DataFrame({
                'index': dates,
                'value': sentiment_values
            })
            
            # Add metadata
            df.attrs['title'] = 'Political Sentiment Index'
            df.attrs['units'] = 'Sentiment Score'
            df.attrs['frequency'] = 'Daily'
            
            return df
            
        except Exception as e:
            st.error(f"Error generating sentiment data: {str(e)}")
            return None

    # Updated method to handle sentiment data
    @st.cache_data(ttl=Config.CACHE_TTL)
    def get_indicator_data(self, indicator: str) -> Optional[pd.DataFrame]:
        """Fetch and process economic indicator data with proper error handling"""
        try:
            if indicator == 'IEF':  # Original IEF handling (Unchanged)
                data = yf.download('IEF', start=Config.START, end=Config.TODAY)
                df = pd.DataFrame(data['Close']).reset_index()
                df.columns = ['index', 'value']
            elif indicator == 'POLSENT':  # New condition for sentiment data
                df = self._get_sentiment_data(Config.START, Config.TODAY)
            else:  # Original FRED data handling (Unchanged)
                indicator_info = self.indicator_details[indicator]
                series_id = indicator_info['series_id']
                
                series_info = self.fred.get_series_info(series_id)
                data = self.fred.get_series(
                    series_id,
                    observation_start=Config.START,
                    observation_end=Config.TODAY,
                    frequency='d'
                )
                
                df = pd.DataFrame(data).reset_index()
                df.columns = ['index', 'value']
                
                if indicator_info['frequency'] != 'Daily':
                    df['value'] = df['value'].ffill()
                
                df.attrs['title'] = indicator_info['description']
                df.attrs['units'] = indicator_info['units']
                df.attrs['frequency'] = indicator_info['frequency']
            
            # Common processing (Unchanged)
            df['index'] = pd.to_datetime(df['index']).dt.tz_localize(None)
            return df
            
        except Exception as e:
            st.error(f"Error fetching {indicator} data: {str(e)}")
            return None
    
    # Original methods (Unchanged)
    def get_indicator_info(self, indicator: str) -> dict:
        """Get metadata for an indicator"""
        return self.indicator_details.get(indicator, {})
    
    def analyze_indicator(self, df: pd.DataFrame, indicator: str) -> dict:
        """Analyze an economic indicator and return key statistics"""
        if df is None or df.empty:
            return {}
            
        try:
            stats = {
                'current_value': df['value'].iloc[-1],
                'change_1d': (df['value'].iloc[-1] - df['value'].iloc[-2]) / df['value'].iloc[-2] * 100,
                'change_1m': (df['value'].iloc[-1] - df['value'].iloc[-30]) / df['value'].iloc[-30] * 100 if len(df) >= 30 else None,
                'min_value': df['value'].min(),
                'max_value': df['value'].max(),
                'avg_value': df['value'].mean(),
                'std_dev': df['value'].std()
            }
            
            return stats
            
        except Exception as e:
            st.error(f"Error analyzing {indicator}: {str(e)}")
            return {}

# Original RealEstateIndicators class (Unchanged)
class RealEstateIndicators:
    """Placeholder class for Real Estate Indicators"""
    def __init__(self):
        self.indicator_details = Config.REAL_ESTATE_INDICATORS
    
    def get_indicator_info(self, indicator: str) -> dict:
        """Get metadata for a real estate indicator"""
        return self.indicator_details.get(indicator, {})

# Original AssetDataFetcher class (Unchanged)
class AssetDataFetcher:
    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL)
    def get_stock_data(symbol: str) -> Optional[pd.DataFrame]:
        """Fetch stock data with fallback to multiple sources"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1y", interval="1d")
            
            if not data.empty:
                data.index = pd.to_datetime(data.index).tz_localize(None)
                return data
                
            raise ValueError(f"No data available for {symbol}")
            
        except Exception as e:
            st.error(f"Error fetching stock data: {str(e)}")
            return None

    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL)
    def get_crypto_data(symbol: str) -> Optional[pd.DataFrame]:
        """Fetch cryptocurrency data from CoinGecko"""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': '365',
                'interval': 'daily'
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'Close'])
            prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'], unit='ms')
            
            volumes_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'Volume'])
            volumes_df['timestamp'] = pd.to_datetime(volumes_df['timestamp'], unit='ms')
            
            df = prices_df.merge(volumes_df[['timestamp', 'Volume']], on='timestamp', how='left')
            df.set_index('timestamp', inplace=True)
            df.index = df.index.tz_localize(None)
            
            df['Open'] = df['Close'].shift(1)
            df['High'] = df['Close']
            df['Low'] = df['Close']
            
            df = df.ffill()
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching crypto data: {str(e)}")
            return None