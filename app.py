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
from alpha_vantage.timeseries import TimeSeries
from pycoingecko import CoinGeckoAPI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataSourceManager:
    """Manages multiple data source connections and API keys"""
    def __init__(self):
        self.fred = fredapi.Fred(api_key=Config.FRED_API_KEY)
        self.alpha_vantage = TimeSeries(key=Config.ALPHA_VANTAGE_API_KEY)
        self.cg = CoinGeckoAPI()
        self.polygon_headers = {"Authorization": f"Bearer {Config.POLYGON_API_KEY}"}

    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL)
    def fetch_polygon_data(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch data from Polygon.io"""
        try:
            headers = {"Authorization": f"Bearer {Config.POLYGON_API_KEY}"}
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                if 'results' in data:
                    df = pd.DataFrame(data['results'])
                    df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'})
                    return df
            return None
        except Exception as e:
            logger.warning(f"Polygon.io fetch failed: {str(e)}")
            return None

    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL)
    def fetch_alpha_vantage_data(symbol: str) -> Optional[pd.DataFrame]:
        """Fetch data from Alpha Vantage"""
        try:
            ts = TimeSeries(key=Config.ALPHA_VANTAGE_API_KEY)
            data, _ = ts.get_daily(symbol=symbol, outputsize='full')
            df = pd.DataFrame(data).rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            })
            return df
        except Exception as e:
            logger.warning(f"Alpha Vantage fetch failed: {str(e)}")
            return None

class EconomicIndicators:
    def __init__(self):
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

    def get_indicator_data(self, indicator: str) -> Optional[pd.DataFrame]:
        """Public wrapper for fetching indicator data"""
        return self._fetch_indicator_data(indicator)

    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL)
    def _fetch_indicator_data(indicator: str) -> Optional[pd.DataFrame]:
        """Internal method to fetch and cache indicator data"""
        try:
            if indicator == 'IEF':
                df = EconomicIndicators._get_ief_data()
            else:
                fred = fredapi.Fred(api_key=Config.FRED_API_KEY)
                indicator_details = EconomicIndicators._get_indicator_details()
                indicator_info = indicator_details[indicator]
                df = EconomicIndicators._get_fred_data(fred, indicator_info)
            
            if df is not None:
                df['index'] = pd.to_datetime(df['index']).dt.tz_localize(None)
                return df
            
            raise ValueError(f"No data available for {indicator}")
            
        except Exception as e:
            st.error(f"Error fetching {indicator} data: {str(e)}")
            return None

    @staticmethod
    def _get_indicator_details():
        """Get indicator details dictionary"""
        return {
            'GDP': {'series_id': 'GDP', 'description': 'Gross Domestic Product', 'frequency': 'Quarterly'},
            'UNRATE': {'series_id': 'UNRATE', 'description': 'Unemployment Rate', 'frequency': 'Monthly'},
            'CPIAUCSL': {'series_id': 'CPIAUCSL', 'description': 'Consumer Price Index', 'frequency': 'Monthly'},
            'DFF': {'series_id': 'DFF', 'description': 'Federal Funds Rate', 'frequency': 'Daily'},
            'IEF': {'series_id': 'IEF', 'description': 'iShares 7-10 Year Treasury Bond ETF', 'frequency': 'Daily'}
        }

    @staticmethod
    def _get_ief_data() -> Optional[pd.DataFrame]:
        """Get IEF data with multiple source fallback"""
        try:
            # Try Polygon first
            df = DataSourceManager.fetch_polygon_data('IEF', Config.START, Config.TODAY)
            if df is not None:
                return df[['Close']].reset_index().rename(columns={'timestamp': 'index'})

            # Try Alpha Vantage
            df = DataSourceManager.fetch_alpha_vantage_data('IEF')
            if df is not None:
                return df[['Close']].reset_index().rename(columns={'date': 'index'})

            # Fallback to Yahoo Finance
            data = yf.download('IEF', start=Config.START, end=Config.TODAY)
            if not data.empty:
                return pd.DataFrame(data['Close']).reset_index()

            return None
        except Exception as e:
            logger.warning(f"Error fetching IEF data: {str(e)}")
            return None

    @staticmethod
    def _get_fred_data(fred: fredapi.Fred, indicator_info: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Get data from FRED"""
        series_id = indicator_info['series_id']
        data = fred.get_series(
            series_id,
            observation_start=Config.START,
            observation_end=Config.TODAY,
            frequency='d'
        )
        
        df = pd.DataFrame(data).reset_index()
        df.columns = ['index', 'value']
        
        if indicator_info['frequency'] != 'Daily':
            df['value'] = df['value'].ffill()
            
        return df

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

class AssetDataFetcher:
    def get_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Public wrapper for fetching stock data"""
        return self._fetch_stock_data(symbol)

    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL)
    def _fetch_stock_data(symbol: str) -> Optional[pd.DataFrame]:
        """Internal method to fetch and cache stock data"""
        try:
            # Try Polygon first
            df = DataSourceManager.fetch_polygon_data(symbol, Config.START, Config.TODAY)
            if df is not None:
                return df

            # Try Alpha Vantage
            df = DataSourceManager.fetch_alpha_vantage_data(symbol)
            if df is not None:
                return df

            # Fallback to Yahoo Finance
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1y", interval="1d")
            
            if not data.empty:
                data.index = pd.to_datetime(data.index).tz_localize(None)
                return data
                
            raise ValueError(f"No data available for {symbol}")
            
        except Exception as e:
            st.error(f"Error fetching stock data: {str(e)}")
            return None

    def get_crypto_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Public wrapper for fetching crypto data"""
        return self._fetch_crypto_data(symbol)

    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL)
    def _fetch_crypto_data(symbol: str) -> Optional[pd.DataFrame]:
        """Internal method to fetch and cache crypto data"""
        try:
            cg = CoinGeckoAPI()
            # Ensure the correct ID is used for 'xrp'
            coin_id = 'ripple' if symbol.lower() == 'xrp' else symbol
            data = cg.get_coin_market_chart_by_id(
                id=coin_id,
                vs_currency='usd',
                days=365,
                interval='daily'
            )
            
            if data:
                prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'Close'])
                volumes_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'Volume'])
                
                df = prices_df.merge(volumes_df[['timestamp', 'Volume']], on='timestamp', how='left')
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df.index = df.index.tz_localize(None)
                
                df['Open'] = df['Close'].shift(1)
                df['High'] = df['Close']
                df['Low'] = df['Close']
                df = df.ffill()
                
                return df
            
            return None
            
        except Exception as e:
            st.error(f"Error fetching crypto data: {str(e)}")
            return None

class RealEstateIndicators:
    """Placeholder class for Real Estate Indicators"""
    def __init__(self):
        self.indicator_details = Config.REAL_ESTATE_INDICATORS
    
    def get_indicator_info(self, indicator: str) -> dict:
        """Get metadata for a real estate indicator"""
        return self.indicator_details.get(indicator, {})