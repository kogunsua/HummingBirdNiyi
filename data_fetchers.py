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

class GDELTDataFetcher:
    """Handle GDELT 2.0 GKG data collection and analysis"""
    def __init__(self):
        self.config = Config.GDELT_CONFIG
        
    def fetch_gkg_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch GDELT GKG data for given date range"""
        all_data = []
        current_date = start_date

        with st.spinner('Fetching GDELT market sentiment data...'):
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

        if not all_data:
            st.warning("No GDELT data available for the specified period")
            return pd.DataFrame()
            
        return pd.concat(all_data, ignore_index=True)

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
        
        # Calculate moving averages for smoothing
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
        """Calculate market sentiment score with multiple factors"""
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
                
                # Ensure timezone-naive datetime index
                data.index = pd.to_datetime(data.index).tz_localize(None)
                
                # Add technical indicators
                data = AssetDataFetcher._add_technical_indicators(data)
                
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
                
                # Add additional price columns
                df['Open'] = df['Close'].shift(1)
                df['High'] = df['Close']
                df['Low'] = df['Close']
                
                # Add technical indicators
                df = AssetDataFetcher._add_technical_indicators(df)
                
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
            
            # Add technical indicators
            df = AssetDataFetcher._add_technical_indicators(df)
            
            return df.sort_index()
            
        except Exception as e:
            st.error(f"Error fetching crypto data: {str(e)}")
            return None

    @staticmethod
    def _add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to price data"""
        try:
            # Moving averages
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA50'] = df['Close'].rolling(window=50).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (2 * bb_std)
            df['BB_Lower'] = df['BB_Middle'] - (2 * bb_std)
            
            # Volatility
            df['Volatility'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
            
            # MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            return df
            
        except Exception as e:
            st.warning(f"Error calculating technical indicators: {str(e)}")
            return df

class EconomicIndicators:
    """Handle economic indicators including GDELT sentiment"""
    def __init__(self):
        self.fred = fredapi.Fred(api_key=Config.FRED_API_KEY)
        self.gdelt_fetcher = GDELTDataFetcher()
        self._initialize_indicators()
    
    def _initialize_indicators(self):
        """Initialize indicators with metadata"""
        self.indicator_details = Config.INDICATORS

    @st.cache_data(ttl=Config.CACHE_TTL)
    def get_indicator_data(self, indicator: str) -> Optional[pd.DataFrame]:
        """Fetch indicator data including GDELT sentiment"""
        try:
            if indicator in ['POLSENT', 'MARKETSENT']:
                return self._get_gdelt_sentiment(indicator)
            
            if indicator == 'IEF':
                data = yf.download('IEF', start=Config.START, end=Config.TODAY)
                df = pd.DataFrame(data['Close']).reset_index()
                df.columns = ['date', 'value']
            else:
                data = self.fred.get_series(
                    self.indicator_details[indicator]['series_id'],
                    observation_start=Config.START,
                    observation_end=Config.TODAY
                )
                df = pd.DataFrame(data).reset_index()
                df.columns = ['date', 'value']
            
            df['date'] = pd.to_datetime(df['date'])
            
            # Add trend indicators
            df = self._add_trend_indicators(df)
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching {indicator}: {str(e)}")
            return None

    def _get_gdelt_sentiment(self, indicator: str) -> Optional[pd.DataFrame]:
        """Fetch GDELT sentiment data"""
        try:
            start_date = datetime.strptime(Config.START, "%Y-%m-%d")
            end_date = datetime.strptime(Config.TODAY, "%Y-%m-%d")
            
            if indicator == 'MARKETSENT':
                gkg_data = self.gdelt_fetcher.fetch_gkg_data(start_date, end_date)
                sentiment_data = self.gdelt_fetcher.process_gkg_data(gkg_data)
                
                return pd.DataFrame({
                    'date': sentiment_data['date'],
                    'value': sentiment_data['market_sentiment']
                })
            
            return None
            
        except Exception as e:
            st.error(f"Error fetching GDELT sentiment: {str(e)}")
            return None

    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators to economic data"""
        try:
            df = df.copy()
            
            # Calculate moving averages
            df['MA5'] = df['value'].rolling(window=5).mean()
            df['MA20'] = df['value'].rolling(window=20).mean()
            
            # Calculate trend direction
            df['Trend'] = np.where(df['MA5'] > df['MA5'].shift(1), 'Up',
                                 np.where(df['MA5'] < df['MA5'].shift(1), 'Down', 'Neutral'))
            
            # Calculate momentum
            df['Momentum'] = df['value'].pct_change(periods=5)
            
            return df
            
        except Exception as e:
            st.warning(f"Error calculating trend indicators: {str(e)}")
            return df

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
                'std_dev': df['value'].std(),
                'trend': df['Trend'].iloc[-1],
                'momentum': df['Momentum'].iloc[-1]
            }
            
            # Add sentiment-specific stats
            if indicator in ['POLSENT', 'MARKETSENT']:
                stats.update({
                    'sentiment_direction': 'Positive' if stats['current_value'] > 0.2 else 
                                         'Negative' if stats['current_value'] < -0.2 else 'Neutral',
                    'sentiment_trend': 'Improving' if stats['change_1m'] > 0 else 
                                     'Declining' if stats['change_1m'] < 0 else 'Stable',
                    'sentiment_momentum': 'Accelerating' if stats['momentum'] > 0 else 'Decelerating'
                })
            
            return stats
            
        except Exception as e:
            st.error(f"Error analyzing {indicator}: {str(e)}")
            return {}

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
        """Fetch all required data from multiple sources"""
        try:
            data = {}
            
            # Fetch asset price data
            data['price'] = (self.asset_fetcher.get_stock_data(symbol) 
                           if asset_type == "Stocks"
                           else self.asset_fetcher.get_crypto_data(symbol))
            
            if data['price'] is None:
                return {}
            
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

    @staticmethod
    def _align_dates(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
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