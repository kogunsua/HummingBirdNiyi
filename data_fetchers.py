#data_fetchers.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import fredapi
from typing import Optional, Dict, Any
from datetime import date, timedelta
from config import Config
from pycoingecko import CoinGeckoAPI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataSourceManager:
    """Manages multiple data source connections and API keys"""
    def __init__(self):
        self.fred = fredapi.Fred(api_key=Config.FRED_API_KEY)
        self.cg = CoinGeckoAPI()
        self.polygon_headers = {"Authorization": f"Bearer {Config.POLYGON_API_KEY}"}

    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL)
    def fetch_yahoo_data(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch data from Yahoo Finance"""
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if not data.empty:
                data.index = pd.to_datetime(data.index).tz_localize(None)
                return data
            return None
        except Exception as e:
            logger.warning(f"Yahoo Finance fetch failed: {str(e)}")
            return None

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
                    df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Close', 'v': 'Volume'})
                    return df
            return None
        except Exception as e:
            logger.warning(f"Polygon.io fetch failed: {str(e)}")
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