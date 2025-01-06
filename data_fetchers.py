# data_fetchers.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import fredapi
from typing import Optional, Dict, Any
from datetime import date, timedelta
from config import Config

class EconomicIndicators:
    def __init__(self):
        self.fred = fredapi.Fred(api_key=Config.FRED_API_KEY)
        self._initialize_indicators()
    
    def _initialize_indicators(self):
        """Initialize FRED indicators with descriptions and frequencies"""
        self.indicator_details = {
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
            }
        }
    
    @st.cache_data(ttl=Config.CACHE_TTL)
    def get_indicator_data(self, indicator: str) -> Optional[pd.DataFrame]:
        """Fetch and process economic indicator data with proper error handling"""
        try:
            if indicator == 'IEF':
                data = yf.download('IEF', start=Config.START, end=Config.TODAY)
                df = pd.DataFrame(data['Close']).reset_index()
                df.columns = ['index', 'value']
            else:
                indicator_info = self.indicator_details[indicator]
                series_id = indicator_info['series_id']
                
                # Get series information
                series_info = self.fred.get_series_info(series_id)
                
                # Fetch the data
                data = self.fred.get_series(
                    series_id,
                    observation_start=Config.START,
                    observation_end=Config.TODAY,
                    frequency='d'  # Convert to daily frequency
                )
                
                df = pd.DataFrame(data).reset_index()
                df.columns = ['index', 'value']
                
                # Forward fill missing values for non-daily series
                if indicator_info['frequency'] != 'Daily':
                    df['value'] = df['value'].ffill()
                
                # Add metadata
                df.attrs['title'] = indicator_info['description']
                df.attrs['units'] = indicator_info['units']
                df.attrs['frequency'] = indicator_info['frequency']
            
            # Remove timezone information
            df['index'] = pd.to_datetime(df['index']).dt.tz_localize(None)
            return df
            
        except Exception as e:
            st.error(f"Error fetching {indicator} data: {str(e)}")
            return None
    
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

class RealEstateIndicators:
    """Placeholder class for Real Estate Indicators"""
    def __init__(self):
        self.indicator_details = Config.REAL_ESTATE_INDICATORS
    
    def get_indicator_info(self, indicator: str) -> dict:
        """Get metadata for a real estate indicator"""
        return self.indicator_details.get(indicator, {})

class AssetDataFetcher:
    @staticmethod
    def _format_crypto_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Format cryptocurrency dataframe to ensure consistent structure"""
        df = df.copy()
        
        # Ensure all required columns exist
        if 'Close' not in df.columns:
            raise ValueError("DataFrame must contain 'Close' column")
            
        # Add missing columns if needed
        if 'Open' not in df.columns:
            df['Open'] = df['Close'].shift(1)
        if 'High' not in df.columns:
            df['High'] = df['Close']
        if 'Low' not in df.columns:
            df['Low'] = df['Close']
        if 'Volume' not in df.columns:
            df['Volume'] = 0
            
        # Forward fill missing values
        df = df.ffill()
        
        # Ensure correct column order
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        return df

    @staticmethod
    def _get_coingecko_data(symbol: str) -> Optional[pd.DataFrame]:
        """Fetch cryptocurrency data from CoinGecko"""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': '365',
                'interval': 'daily'
            }
            response = requests.get(url, params=params, timeout=10)
            
            # Check for rate limit
            if response.status_code == 429:
                return None
                
            response.raise_for_status()
            data = response.json()
            
            # Process price data
            prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'Close'])
            prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'], unit='ms')
            
            # Process volume data
            volumes_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'Volume'])
            volumes_df['timestamp'] = pd.to_datetime(volumes_df['timestamp'], unit='ms')
            
            # Merge data
            df = prices_df.merge(volumes_df[['timestamp', 'Volume']], on='timestamp', how='left')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            st.warning(f"CoinGecko error: {str(e)}")
            return None

    @staticmethod
    def _get_polygon_crypto_data(symbol: str) -> Optional[pd.DataFrame]:
        """Fetch cryptocurrency data from Polygon.io as backup"""
        try:
            crypto_mapping = Config.CRYPTO_MAPPINGS.get(symbol.lower(), {})
            if not crypto_mapping:
                return None
                
            polygon_symbol = crypto_mapping.get('polygon')
            if not polygon_symbol:
                return None
            
            url = f"https://api.polygon.io/v2/aggs/ticker/{polygon_symbol}/range/1/day/{Config.START}/{Config.TODAY}"
            params = {
                'apiKey': Config.POLYGON_API_KEY,
                'limit': 365
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('resultsCount', 0) == 0:
                return None
                
            df = pd.DataFrame(data['results'])
            df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Rename columns
            df = df.rename(columns={
                'o': 'Open',
                'h': 'High',
                'l': 'Low',
                'c': 'Close',
                'v': 'Volume'
            })
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            st.warning(f"Polygon.io error: {str(e)}")
            return None

    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL)
    def get_stock_data(symbol: str) -> Optional[pd.DataFrame]:
        """Fetch stock data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1y", interval="1d")
            
            if not data.empty:
                # Remove timezone information
                data.index = pd.to_datetime(data.index).tz_localize(None)
                return data
                
            raise ValueError(f"No data available for {symbol}")
            
        except Exception as e:
            st.error(f"Error fetching stock data: {str(e)}")
            return None

    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL)
    def get_crypto_data(symbol: str) -> Optional[pd.DataFrame]:
        """Fetch cryptocurrency data with fallback to multiple sources"""
        try:
            # Try CoinGecko first
            coingecko_data = AssetDataFetcher._get_coingecko_data(symbol)
            if coingecko_data is not None:
                return AssetDataFetcher._format_crypto_dataframe(coingecko_data)
                
            # Fallback to Polygon.io if CoinGecko fails
            st.info("Falling back to Polygon.io for cryptocurrency data...")
            polygon_data = AssetDataFetcher._get_polygon_crypto_data(symbol)
            if polygon_data is not None:
                return AssetDataFetcher._format_crypto_dataframe(polygon_data)
            
            raise ValueError(f"Could not fetch data for {symbol} from any source")
            
        except Exception as e:
            st.error(f"Error fetching crypto data: {str(e)}")
            return None