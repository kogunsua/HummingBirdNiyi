# data_fetchers.py
#
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
            'DFF': {'series_id': 'DFF', 'description': 'Federal Funds Rate', 'frequency': 'Daily'},
            'IEF': {'series_id': 'IEF', 'description': 'iShares 7-10 Year Treasury Bond ETF', 'frequency': 'Daily'}
        }

    @staticmethod
    def _get_ief_data() -> Optional[pd.DataFrame]:
        """Get IEF data with multiple source fallback"""
        try:
            # Try Yahoo Finance first
            data = DataSourceManager.fetch_yahoo_data('IEF', Config.START, Config.TODAY)
            if data is not None:
                return pd.DataFrame(data['Close']).reset_index()
            
            # Try Alpha Vantage
            df = DataSourceManager.fetch_alpha_vantage_data('IEF')
            if df is not None:
                return df[['Close']].reset_index().rename(columns={'date': 'index'})
            
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
    # Crypto symbol mappings with proper CoinGecko IDs
    CRYPTO_MAPPINGS = {
        'XRP': {'coingecko': 'xrp', 'polygon': 'X:XRPUSD', 'yahoo': 'XRP-USD'},
        'BTC': {'coingecko': 'bitcoin', 'polygon': 'X:BTCUSD', 'yahoo': 'BTC-USD'},
        'ETH': {'coingecko': 'ethereum', 'polygon': 'X:ETHUSD', 'yahoo': 'ETH-USD'},
        'DOGE': {'coingecko': 'dogecoin', 'polygon': 'X:DOGEUSD', 'yahoo': 'DOGE-USD'},
        'ADA': {'coingecko': 'cardano', 'polygon': 'X:ADAUSD', 'yahoo': 'ADA-USD'},
        'DOT': {'coingecko': 'polkadot', 'polygon': 'X:DOTUSD', 'yahoo': 'DOT-USD'},
        'LINK': {'coingecko': 'chainlink', 'polygon': 'X:LINKUSD', 'yahoo': 'LINK-USD'},
        'UNI': {'coingecko': 'uniswap', 'polygon': 'X:UNIUSD', 'yahoo': 'UNI-USD'},
        'MATIC': {'coingecko': 'matic-network', 'polygon': 'X:MATICUSD', 'yahoo': 'MATIC-USD'},
        'SOL': {'coingecko': 'solana', 'polygon': 'X:SOLUSD', 'yahoo': 'SOL-USD'}
    }

    def get_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Public wrapper for fetching stock data"""
        try:
            logger.info(f"Fetching stock data for symbol: {symbol}")
            
            # Get data from Yahoo Finance
            data = DataSourceManager.fetch_yahoo_data(symbol, Config.START, Config.TODAY)
            
            if data is not None and not data.empty:
                # Ensure data has correct columns
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in required_columns:
                    if col not in data.columns:
                        raise ValueError(f"Missing required column: {col}")
                
                # Convert index to datetime and remove timezone
                data.index = pd.to_datetime(data.index).tz_localize(None)
                
                logger.info(f"Successfully fetched stock data for {symbol}")
                logger.info(f"Data shape: {data.shape}")
                logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
                
                return data
            
            raise ValueError(f"No data available for {symbol}")
        
        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}")
            st.error(f"Could not fetch data for {symbol}. Please verify the symbol.")
            return None

    def get_crypto_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Public wrapper for fetching crypto data"""
        return self._fetch_crypto_data(symbol)

    @staticmethod
    def _get_crypto_mappings(symbol: str) -> Dict[str, str]:
        """Get crypto mappings with fallbacks"""
        symbol = symbol.upper()
        COINGECKO_SPECIAL_CASES = {
            'XRP': 'xrp', 'BTC': 'bitcoin', 'ETH': 'ethereum', 'DOGE': 'dogecoin', 'ADA': 'cardano',
            'DOT': 'polkadot', 'LINK': 'chainlink', 'UNI': 'uniswap', 'MATIC': 'matic-network', 'SOL': 'solana'
        }
        default_mappings = {
            'coingecko': COINGECKO_SPECIAL_CASES.get(symbol, symbol.lower()),
            'polygon': f'X:{symbol}USD',
            'yahoo': f'{symbol}-USD'
        }
        return AssetDataFetcher.CRYPTO_MAPPINGS.get(symbol, default_mappings)

    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL)
    def _fetch_crypto_data(symbol: str) -> Optional[pd.DataFrame]:
        """Internal method to fetch and cache crypto data with multiple source fallback"""
        symbol = symbol.upper()
        logger.info(f"Fetching crypto data for symbol: {symbol}")
        mappings = AssetDataFetcher._get_crypto_mappings(symbol)
        logger.info(f"Using mappings: {mappings}")
        error_messages = []
        
        try:
            logger.info(f"Attempting to fetch {symbol} from CoinGecko...")
            cg = CoinGeckoAPI()
            data = cg.get_coin_market_chart_by_id(id=mappings['coingecko'], vs_currency='usd', days=365, interval='daily')
            
            if data and 'prices' in data:
                logger.info(f"Successfully fetched {symbol} from CoinGecko")
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
            error_messages.append("CoinGecko: No data returned")
        except Exception as e:
            error_messages.append(f"CoinGecko: {str(e)}")
            logger.warning(f"CoinGecko fetch failed for {symbol}: {str(e)}")
        
        # Fallback to Polygon.io
        try:
            logger.info(f"Attempting to fetch {symbol} from Polygon.io...")
            data = DataSourceManager.fetch_polygon_data(mappings['polygon'], Config.START, Config.TODAY)
            if data is not None:
                logger.info(f"Successfully fetched {symbol} from Polygon.io")
                return data
            error_messages.append("Polygon.io: No data returned")
        except Exception as e:
            error_messages.append(f"Polygon.io: {str(e)}")
            logger.warning(f"Polygon.io fetch failed for {symbol}: {str(e)}")
        
        # Final fallback to Yahoo Finance
        try:
            logger.info(f"Attempting to fetch {symbol} from Yahoo Finance...")
            data = DataSourceManager.fetch_yahoo_data(mappings['yahoo'], Config.START, Config.TODAY)
            if data is not None:
                logger.info(f"Successfully fetched {symbol} from Yahoo Finance")
                return data
            error_messages.append("Yahoo Finance: No data returned")
        except Exception as e:
            error_messages.append(f"Yahoo Finance: {str(e)}")
            logger.warning(f"Yahoo Finance fetch failed for {symbol}: {str(e)}")
        
        error_msg = f"Failed to fetch {symbol} data from all sources:\n" + "\n".join(error_messages)
        st.error(error_msg)
        return None

class RealEstateIndicators:
    """Class for handling real estate market indicators"""
    def __init__(self):
        self.indicator_details = Config.REAL_ESTATE_INDICATORS
    
    def get_indicator_info(self, indicator: str) -> dict:
        """Get metadata for a real estate indicator"""
        return self.indicator_details.get(indicator, {})

    def get_indicator_data(self, indicator: str) -> Optional[pd.DataFrame]:
        """Fetch real estate indicator data"""
        try:
            # Currently returns None as this is a placeholder
            # To be implemented with actual data sources
            return None
        except Exception as e:
            logger.error(f"Error fetching real estate data: {str(e)}")
            return None
    
    def analyze_indicator(self, df: pd.DataFrame, indicator: str) -> dict:
        """Analyze a real estate indicator and return key statistics"""
        if df is None or df.empty:
            return {}
        
        try:
            # Currently returns empty dict as this is a placeholder
            # To be implemented with actual analysis logic
            return {}
        except Exception as e:
            logger.error(f"Error analyzing real estate indicator: {str(e)}")
            return {}
