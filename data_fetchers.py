# data_fetchers.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import fredapi
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
from config import Config
import json
import warnings
warnings.filterwarnings('ignore')

class AssetDataFetcher:
    """Handle asset data fetching for both stocks and cryptocurrencies"""
    
    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL)
    def get_stock_data(symbol: str) -> Optional[pd.DataFrame]:
        """Fetch stock data with error handling"""
        try:
            with st.spinner(f'Fetching stock data for {symbol}...'):
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1y", interval="1d")
                
                if data.empty:
                    raise ValueError(f"No data available for {symbol}")
                
                data.index = pd.to_datetime(data.index).tz_localize(None)
                return data
                
        except Exception as e:
            st.error(f"Error fetching stock data: {str(e)}")
            return None

    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL)
    def get_crypto_data(symbol: str) -> Optional[pd.DataFrame]:
        """Fetch cryptocurrency data with fallback support"""
        try:
            with st.spinner(f'Fetching crypto data for {symbol}...'):
                # Try CoinGecko first
                url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
                params = {
                    'vs_currency': 'usd',
                    'days': '365',
                    'interval': 'daily'
                }
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 429:
                    st.info("CoinGecko rate limit reached, trying Polygon.io...")
                    return AssetDataFetcher._get_polygon_crypto_data(symbol)
                
                response.raise_for_status()
                data = response.json()
                
                # Process price data
                prices_df = pd.DataFrame(data['prices'], columns=['Date', 'Close'])
                prices_df['Date'] = pd.to_datetime(prices_df['Date'], unit='ms')
                
                # Process volume data
                volumes_df = pd.DataFrame(data['total_volumes'], columns=['Date', 'Volume'])
                volumes_df['Date'] = pd.to_datetime(volumes_df['Date'], unit='ms')
                
                # Merge data
                df = prices_df.merge(volumes_df[['Date', 'Volume']], on='Date', how='left')
                df.set_index('Date', inplace=True)
                df.index = df.index.tz_localize(None)
                
                # Add required columns
                df['Open'] = df['Close'].shift(1)
                df['High'] = df['Close']
                df['Low'] = df['Close']
                
                return df.ffill()
                
        except Exception as e:
            st.info("CoinGecko error, trying Polygon.io...")
            return AssetDataFetcher._get_polygon_crypto_data(symbol)

    @staticmethod
    def _get_polygon_crypto_data(symbol: str) -> Optional[pd.DataFrame]:
        """Fetch crypto data from Polygon.io as fallback"""
        try:
            crypto_mapping = {
                'bitcoin': 'X:BTCUSD',
                'ethereum': 'X:ETHUSD',
                'ripple': 'X:XRPUSD',
                'dogecoin': 'X:DOGEUSD',
                'cardano': 'X:ADAUSD',
                'solana': 'X:SOLUSD',
                'polkadot': 'X:DOTUSD',
                'litecoin': 'X:LTCUSD',
                'chainlink': 'X:LINKUSD',
                'stellar': 'X:XLMUSD'
            }
            
            polygon_symbol = crypto_mapping.get(symbol.lower(), f'X:{symbol.upper()}USD')
            
            url = f"https://api.polygon.io/v2/aggs/ticker/{polygon_symbol}/range/1/day/{Config.START}/{Config.TODAY}"
            params = {
                'apiKey': Config.POLYGON_API_KEY,
                'limit': 365
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('resultsCount', 0) == 0:
                raise ValueError(f"No data available for {symbol}")
                
            df = pd.DataFrame(data['results'])
            df['Date'] = pd.to_datetime(df['t'], unit='ms')
            
            # Rename columns to match our format
            df = df.rename(columns={
                'o': 'Open',
                'h': 'High',
                'l': 'Low',
                'c': 'Close',
                'v': 'Volume'
            })
            
            df.set_index('Date', inplace=True)
            df.index = df.index.tz_localize(None)
            
            return df.sort_index()
            
        except Exception as e:
            st.error(f"Error fetching crypto data: {str(e)}")
            return None

class EconomicIndicators:
    """Handle economic indicators"""
    def __init__(self):
        self.fred = fredapi.Fred(api_key=Config.FRED_API_KEY)
        self._initialize_indicators()
    
    def _initialize_indicators(self):
        """Initialize indicators with metadata"""
        self.indicator_details = Config.INDICATORS

    @st.cache_data(ttl=Config.CACHE_TTL)
    def get_indicator_data(self, indicator: str) -> Optional[pd.DataFrame]:
        """Fetch indicator data"""
        try:
            if indicator == 'IEF':
                data = yf.download('IEF', start=Config.START, end=Config.TODAY)
                df = pd.DataFrame(data['Close']).reset_index()
                df.columns = ['date', 'value']
            else:
                data = self.fred.get_series(
                    indicator,
                    observation_start=Config.START,
                    observation_end=Config.TODAY
                )
                df = pd.DataFrame(data).reset_index()
                df.columns = ['date', 'value']
            
            df['date'] = pd.to_datetime(df['date'])
            return df
            
        except Exception as e:
            st.error(f"Error fetching {indicator}: {str(e)}")
            return None

    def get_indicator_info(self, indicator: str) -> dict:
        """Get metadata for an indicator"""
        return self.indicator_details.get(indicator, {})
    
    def analyze_indicator(self, df: pd.DataFrame) -> dict:
        """Analyze indicator and return statistics"""
        if df is None or df.empty:
            return {}
            
        try:
            return {
                'current_value': df['value'].iloc[-1],
                'change_1d': (df['value'].iloc[-1] - df['value'].iloc[-2]) / df['value'].iloc[-2] * 100,
                'change_1m': (df['value'].iloc[-1] - df['value'].iloc[-30]) / df['value'].iloc[-30] * 100 if len(df) >= 30 else None,
                'min_value': df['value'].min(),
                'max_value': df['value'].max(),
                'avg_value': df['value'].mean(),
                'std_dev': df['value'].std(),
                'trend': 'Upward' if df['value'].iloc[-1] > df['value'].iloc[-2] else 'Downward'
            }
        except Exception as e:
            st.error(f"Error analyzing indicator: {str(e)}")
            return {}

class RealEstateIndicators:
    """Handle Real Estate market indicators"""
    def __init__(self):
        self.fred = fredapi.Fred(api_key=Config.FRED_API_KEY)
        self.indicator_details = Config.REAL_ESTATE_INDICATORS
    
    def get_indicator_info(self, indicator: str) -> dict:
        """Get metadata for a real estate indicator"""
        try:
            return self.indicator_details.get(indicator, {})
        except Exception as e:
            st.error(f"Error fetching real estate indicator info: {str(e)}")
            return {}

    @st.cache_data(ttl=Config.CACHE_TTL)
    def get_indicator_data(self, indicator: str) -> Optional[pd.DataFrame]:
        """Fetch real estate indicator data"""
        try:
            info = self.indicator_details.get(indicator)
            if not info:
                return None

            if info['source'] == 'FRED':
                data = self.fred.get_series(
                    indicator,
                    observation_start=Config.START,
                    observation_end=Config.TODAY
                )
                
                df = pd.DataFrame(data).reset_index()
                df.columns = ['date', 'value']
                df['date'] = pd.to_datetime(df['date'])
                
                return df
            
            return None
            
        except Exception as e:
            st.error(f"Error fetching real estate data: {str(e)}")
            return None

    def analyze_indicator(self, df: pd.DataFrame) -> dict:
        """Analyze real estate indicator and return statistics"""
        if df is None or df.empty:
            return {}
            
        try:
            return {
                'current_value': df['value'].iloc[-1],
                'change_1m': (df['value'].iloc[-1] - df['value'].iloc[-30]) / df['value'].iloc[-30] * 100 if len(df) >= 30 else None,
                'min_value': df['value'].min(),
                'max_value': df['value'].max(),
                'avg_value': df['value'].mean(),
                'trend': 'Upward' if df['value'].iloc[-1] > df['value'].iloc[-2] else 'Downward'
            }
        except Exception as e:
            st.error(f"Error analyzing real estate indicator: {str(e)}")
            return {}

class GDELTDataFetcher:
    """Handle GDELT data collection and analysis"""
    def __init__(self):
        self.config = Config.GDELT_CONFIG
        
    def fetch_gkg_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch GDELT GKG data for given date range"""
        all_data = []
        current_date = start_date

        with st.spinner('Fetching market sentiment data...'):
            while current_date <= end_date:
                try:
                    filename = f"{current_date.strftime('%Y%m%d%H%M%S')}.gkg.csv.zip"
                    url = f"{self.config['gkg_base_url']}{filename}"
                    
                    df = pd.read_csv(url, compression='zip',
                                   names=self._get_gkg_columns(),
                                   sep='\t',
                                   usecols=self.config['required_columns']['gkg'])
                    all_data.append(df)
                    
                except Exception as e:
                    pass  # Silently handle missing data points
                
                current_date += timedelta(minutes=15)

        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    def process_gkg_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process GKG data and calculate market sentiment"""
        if df.empty:
            return pd.DataFrame()

        processed = pd.DataFrame()
        processed['date'] = pd.to_datetime(df['DATE'])
        processed['themes'] = df['THEMES'].apply(self._extract_themes)
        processed['tone'] = df['TONE'].apply(self._process_tone)
        processed['articles'] = pd.to_numeric(df['NUMARTS'], errors='coerce')
        
        # Calculate sentiment scores
        processed['market_sentiment'] = processed.apply(
            self._calculate_sentiment_score, axis=1
        )
        
        # Calculate additional metrics
        processed['volume_impact'] = np.log1p(processed['articles'])
        processed['theme_relevance'] = processed['themes'].apply(len)
        
        # Aggregate by date
        daily_data = processed.groupby(processed['date'].dt.date).agg({
            'market_sentiment': 'mean',
            'volume_impact': 'sum',
            'theme_relevance': 'mean',
            'articles': 'sum'
        }).reset_index()
        
        # Calculate moving averages
        daily_data['sentiment_ma5'] = daily_data['market_sentiment'].rolling(window=5).mean()
        daily_data['sentiment_ma20'] = daily_data['market_sentiment'].rolling(window=20).mean()
        
        return daily_data

    def _extract_themes(self, themes_str: str) -> List[str]:
        """Extract market-relevant themes from GDELT themes string"""
        if pd.isna(themes_str):
            return []
        
        themes = themes_str.split(';')
        return [theme for theme in themes 
                if any(t in theme for t in self.config['themes_of_interest'])]

    def _process_tone(self, tone_str: str) -> Dict[str, float]:
        """Process GDELT tone information"""
        if pd.isna(tone_str):
            return {'positive': 0, 'negative': 0, 'polarity': 0}
        
        try:
            values = list(map(float, tone_str.split(',')))
            return {
                'positive': values[0],
                'negative': values[1],
                'polarity': values[2] if len(values) > 2 else 0
            }
        except:
            return {'positive': 0, 'negative': 0, 'polarity': 0}

    def _calculate_sentiment_score(self, row) -> float:
        """Calculate market sentiment score"""
        weights = self.config['sentiment_weights']
        
        # Base sentiment from tone
        tone_score = row['tone']['polarity']
        
        # Theme relevance
        theme_score = min(len(row['themes']) * 0.2, 1)
        
        # Volume impact
        volume_score = np.log1p(row['articles']) / 10
        
        # Weighted combination
        sentiment = (weights['tone'] * tone_score +
                    weights['themes'] * theme_score +
                    weights['volume'] * volume_score)
        
        # Normalize to [-1, 1]
        return max(min(sentiment, 1), -1)

    @staticmethod
    def _get_gkg_columns() -> List[str]:
        """Get required GKG columns"""
        return ['DATE', 'NUMARTS', 'COUNTS', 'THEMES', 'LOCATIONS', 'PERSONS',
                'ORGANIZATIONS', 'TONE', 'CAMEOEVENTIDS', 'SOURCES', 'SOURCEURLS']

class IntegratedDataFetcher:
    """Handle integrated data fetching from all sources"""
    def __init__(self):
        self.gdelt_fetcher = GDELTDataFetcher()
        self.economic_indicators = EconomicIndicators()
        self.asset_fetcher = AssetDataFetcher()

    def fetch_all_data(self,
                      symbol: str,
                      asset_type: str,
                      include_sentiment: bool = True,
                      include_economic: bool = True) -> Dict[str, pd.DataFrame]:
        """Fetch and integrate data from multiple sources"""
        try:
            data = {}
            
            # Fetch asset price data
            data['price'] = (self.asset_fetcher.get_stock_data(symbol) 
                           if asset_type == "Stocks"
                           else self.asset_fetcher.get_crypto_data(symbol))
            
            if data['price'] is not None:
                start_date = data['price'].index.min()
                end_date = data['price'].index.max()
                
                # Fetch GDELT sentiment if requested
                if include_sentiment:
                    with st.spinner('Fetching market sentiment data...'):
                        gkg_data = self.gdelt_fetcher.fetch_gkg_data(
                            start_date.to_pydatetime(),
                            end_date.to_pydatetime()
                        )
                        sentiment_data = self.gdelt_fetcher.process_gkg_data(gkg_data)
                        if not sentiment_data.empty:
                            data['sentiment'] = sentiment_data
                
                # Fetch economic indicators if requested
                if include_economic:
                    with st.spinner('Fetching economic indicators...'):
                        for indicator in ['GDP', 'UNRATE', 'CPIAUCSL']:
                            indicator_data = self.economic_indicators.get_indicator_data(indicator)
                            if indicator_data is not None:
                                data[f'economic_{indicator}'] = indicator_data
            
            # Align dates across all data sources
            return self._align_dates(data)
            
        except Exception as e:
            st.error(f"Error fetching integrated data: {str(e)}")
            return {}

    def prepare_prophet_data(self,
                           price_data: pd.DataFrame,
                           sentiment_data: Optional[pd.DataFrame] = None,
                           economic_data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """Prepare data for Prophet model"""
        df = price_data.reset_index()
        df = df[['Date', 'Close']]
        df.columns = ['ds', 'y']
        
        # Add sentiment features if available
        if sentiment_data is not None:
            sentiment_df = sentiment_data.copy()
            sentiment_df.columns = ['ds', 'sentiment']
            df = df.merge(sentiment_df, on='ds', how='left')
            df['sentiment'].fillna(method='ffill', inplace=True)
        
        # Add economic features if available
        if economic_data:
            for indicator, data in economic_data.items():
                if data is not None:
                    indicator_df = data.copy()
                    indicator_df.columns = ['ds', indicator]
                    df = df.merge(indicator_df, on='ds', how='left')
                    df[indicator].fillna(method='ffill', inplace=True)
        
        return df

    def _align_dates(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Align all data sources to the same date range"""
        if not data_dict or 'price' not in data_dict:
            return {}
            
        price_data = data_dict['price']
        start_date = price_data.index.min()
        end_date = price_data.index.max()
        
        aligned_data = {'price': price_data}
        
        # Align sentiment data
        if 'sentiment' in data_dict:
            sentiment_df = data_dict['sentiment']
            sentiment_df.set_index('date', inplace=True)
            aligned_data['sentiment'] = sentiment_df[start_date:end_date]
        
        # Align economic indicators
        for key, df in data_dict.items():
            if key.startswith('economic_'):
                df_copy = df.copy()
                df_copy.set_index('date', inplace=True)
                aligned_data[key] = df_copy[start_date:end_date]
        
        return aligned_data

    def get_market_context(self, 
                          symbol: str, 
                          asset_type: str) -> Dict[str, any]:
        """Get market context data"""
        context = {}
        
        try:
            if asset_type == "Stocks":
                # Get S&P 500 data for comparison
                spy_data = self.asset_fetcher.get_stock_data("SPY")
                if spy_data is not None:
                    context['market_correlation'] = self._calculate_correlation(
                        spy_data['Close'],
                        self.asset_fetcher.get_stock_data(symbol)['Close']
                    )
                    context['market_beta'] = self._calculate_beta(
                        spy_data['Close'],
                        self.asset_fetcher.get_stock_data(symbol)['Close']
                    )
            
            elif asset_type == "Cryptocurrency":
                # Get Bitcoin data for comparison
                btc_data = self.asset_fetcher.get_crypto_data("bitcoin")
                if btc_data is not None and symbol.lower() != "bitcoin":
                    context['crypto_correlation'] = self._calculate_correlation(
                        btc_data['Close'],
                        self.asset_fetcher.get_crypto_data(symbol)['Close']
                    )
            
            return context
            
        except Exception as e:
            st.error(f"Error getting market context: {str(e)}")
            return {}

    @staticmethod
    def _calculate_correlation(series1: pd.Series, series2: pd.Series) -> float:
        """Calculate correlation between two price series"""
        return series1.pct_change().corr(series2.pct_change())

    @staticmethod
    def _calculate_beta(market_returns: pd.Series, asset_returns: pd.Series) -> float:
        """Calculate beta of an asset against the market"""
        market_change = market_returns.pct_change()
        asset_change = asset_returns.pct_change()
        
        covariance = market_change.cov(asset_change)
        market_variance = market_change.var()
        
        return covariance / market_variance if market_variance != 0 else 0