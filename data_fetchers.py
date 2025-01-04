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
                df = pd.