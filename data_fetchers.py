To resolve the issue with the `fetch_gkg_data` method in the `GDELTDataFetcher` class, we need to remove the `@st.cache_data` decorator from methods that have non-hashable arguments or refactor the method to avoid the hashing error.

Here is the updated `data_fetchers.py` file with the necessary changes:

```python
"""
Data fetching utilities for market data and indicators
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import requests
from config import Config

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
                    st.error(f"No data available for {symbol}")
                    return None
                
                data.index = pd.to_datetime(data.index).tz_localize(None)
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
                params = {'vs_currency': 'usd', 'days': '365', 'interval': 'daily'}
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 429:
                    st.warning("CoinGecko rate limit reached, using fallback...")
                    return AssetDataFetcher._get_polygon_crypto_data(symbol)
                
                response.raise_for_status()
                data = response.json()
                
                # Process data
                df = pd.DataFrame(data['prices'], columns=['Date', 'Close'])
                df['Date'] = pd.to_datetime(df['Date'], unit='ms')
                df.set_index('Date', inplace=True)
                
                # Add technical indicators
                df = AssetDataFetcher._add_technical_indicators(df)
                return df
                
        except Exception as e:
            st.warning(f"CoinGecko error: {str(e)}, using fallback...")
            return AssetDataFetcher._get_polygon_crypto_data(symbol)

    @staticmethod
    def _get_polygon_crypto_data(symbol: str) -> Optional[pd.DataFrame]:
        """Fetch crypto data from Polygon.io as fallback"""
        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/X:{symbol}USD/range/1/day/{Config.START}/{Config.TODAY}"
            params = {'apiKey': Config.POLYGON_API_KEY, 'limit': 365}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('resultsCount', 0) == 0:
                raise ValueError(f"No data available for {symbol}")
            
            df = pd.DataFrame(data['results'])
            df['Date'] = pd.to_datetime(df['t'], unit='ms')
            df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'})
            df.set_index('Date', inplace=True)
            
            df = AssetDataFetcher._add_technical_indicators(df)
            return df
            
        except Exception as e:
            st.error(f"Error fetching crypto data: {str(e)}")
            return None

    @staticmethod
    def _add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to price data"""
        try:
            df_tech = df.copy()
            
            # RSI
            delta = df_tech['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_tech['RSI'] = 100 - (100 / (1 + rs))
            
            # Moving Averages
            df_tech['MA20'] = df_tech['Close'].rolling(window=20).mean()
            df_tech['MA50'] = df_tech['Close'].rolling(window=50).mean()
            
            # Bollinger Bands
            df_tech['BB_Middle'] = df_tech['Close'].rolling(window=20).mean()
            bb_std = df_tech['Close'].rolling(window=20).std()
            df_tech['BB_Upper'] = df_tech['BB_Middle'] + (2 * bb_std)
            df_tech['BB_Lower'] = df_tech['BB_Middle'] - (2 * bb_std)
            
            # MACD
            exp1 = df_tech['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df_tech['Close'].ewm(span=26, adjust=False).mean()
            df_tech['MACD'] = exp1 - exp2
            df_tech['Signal_Line'] = df_tech['MACD'].ewm(span=9, adjust=False).mean()
            
            return df_tech
            
        except Exception as e:
            st.warning(f"Error calculating technical indicators: {str(e)}")
            return df

class EconomicIndicators:
    """Handle economic indicators data"""
    
    def __init__(self):
        self.indicators_info = Config.INDICATORS
    
    @st.cache_data(ttl=Config.CACHE_TTL)
    def get_indicators(self) -> Optional[pd.DataFrame]:
        """Get economic indicator data"""
        try:
            indicators_data = {}
            for indicator in Config.ECONOMIC_CONFIG['indicators']:
                data = self._generate_indicator_data(indicator)
                if data is not None:
                    indicators_data[indicator] = data
            
            return self._combine_indicators(indicators_data)
            
        except Exception as e:
            st.error(f"Error getting economic indicators: {str(e)}")
            return None
    
    def _generate_indicator_data(self, indicator: str) -> pd.DataFrame:
        """Generate indicator data"""
        freq_map = {'Daily': 'D', 'Monthly': 'M', 'Quarterly': 'Q'}
        freq = self.indicators_info[indicator]['frequency']
        periods = 30 if freq == 'Daily' else 12 if freq == 'Monthly' else 4
        
        dates = pd.date_range(
            end=datetime.now(),
            periods=periods,
            freq=freq_map[freq]
        )
        
        values = np.random.normal(
            2.5 if indicator == 'GDP' else 2.0,
            0.5 if indicator == 'GDP' else 0.3,
            len(dates)
        )
        
        return pd.DataFrame({
            'date': dates,
            'value': values,
            'indicator': indicator
        })
    
    def _combine_indicators(self, indicators_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine multiple indicators"""
        combined = pd.concat(indicators_data.values(), axis=0)
        combined.sort_values('date', inplace=True)
        return combined
    
    def get_indicator_info(self, indicator: str) -> Dict[str, str]:
        """Get indicator information"""
        return self.indicators_info.get(indicator, {})
    
    def analyze_indicator(self, data: pd.DataFrame) -> Dict[str, any]:
        """Analyze an economic indicator"""
        if data is None or data.empty:
            return {}
            
        try:
            values = data['value']
            stats = {
                'current': values.iloc[-1],
                'mean': values.mean(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'trend': self._calculate_trend(values),
                'momentum': self._calculate_momentum(values),
                'volatility': self._calculate_volatility(values)
            }
            return stats
            
        except Exception as e:
            st.error(f"Error analyzing indicator: {str(e)}")
            return {}
    
    def _calculate_trend(self, series: pd.Series) -> str:
        """Calculate trend direction"""
        if len(series) < 2:
            return 'Insufficient Data'
        
        slope = np.polyfit(range(len(series)), series.values, 1)[0]
        return 'Upward' if slope > 0.01 else 'Downward' if slope < -0.01 else 'Stable'
    
    def _calculate_momentum(self, series: pd.Series) -> float:
        """Calculate momentum"""
        if len(series) < 2:
            return 0.0
        return (series.iloc[-1] - series.iloc[-2]) / series.iloc[-2] * 100
    
    def _calculate_volatility(self, series: pd.Series) -> float:
        """Calculate volatility"""
        return series.pct_change().std() * np.sqrt(252)  # Annualized volatility

class GDELTDataFetcher:
    """Handle GDELT data collection and analysis"""
    
    def __init__(self):
        self.config = Config.GDELT_CONFIG
        self.cache = {}
    
    def get_sentiment_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get sentiment data for a symbol with caching"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            gkg_data = self._fetch_gkg_data(start_date, end_date, symbol)
            
            if gkg_data is not None and not gkg_data.empty:
                sentiment_data = self._process_sentiment_data(gkg_data, symbol)
                return sentiment_data
            else:
                st.warning(f"No sentiment data available for {symbol}")
                return None
                
        except Exception as e:
            st.error(f"Error fetching sentiment data: {str(e)}")
            return None
    
    def _fetch_gkg_data(self, start_date: datetime, end_date: datetime, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch GKG data for a specific symbol"""
        try:
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            np.random.seed(42)  # For reproducibility
            data = pd.DataFrame({
                'date': dates,
                'market_sentiment': np.random.normal(0.2, 0.3, len(dates)),
                'volume_impact': np.random.uniform(0, 1, len(dates)),
                'theme_impact': np.random.uniform(0, 1, len(dates)),
                'news_count': np.random.randint(10, 100, len(dates))
            })
            
            data.set_index('date', inplace=True)
            return data
            
        except Exception as e:
            st.error(f"Error fetching GKG data: {str(e)}")
            return None
    
    def _process_sentiment_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Process raw sentiment data"""
        try:
            processed = data.copy()
            
            # Calculate moving averages
            processed['sentiment_ma5'] = processed['market_sentiment'].rolling(window=5).mean()
            processed['sentiment_ma20'] = processed['market_sentiment'].rolling(window=20).mean()
            
            # Calculate momentum
            processed['sentiment_momentum'] = processed['market_sentiment'].diff(
                periods=5
            ).fillna(0)
            
            # Calculate volatility
            processed['sentiment_volatility'] = processed['market_sentiment'].rolling(
                window=20
            ).std().fillna(0)
            
            # Calculate trend strength
            processed['trend_strength'] = np.abs(
                processed['sentiment_ma5'] - processed['sentiment_ma20']
            ) / processed['sentiment_volatility'].replace(0, np.nan)
            
            # Handle infinities and NaNs
            processed = processed.replace([np.inf, -np.inf], np.nan)
            processed = processed.fillna(method='ffill').fillna(method='bfill')
            
            return processed
            
        except Exception as e:
            st.error(f"Error processing sentiment data: {str(e)}")
            return data
    
    @staticmethod
    def _calculate_sentiment_impact(tone_score: float, volume: float, theme_relevance: float) -> float:
        """Calculate overall sentiment impact score"""
        # Normalize inputs
        normalized_tone = max(min(tone_score, 1), -1)
        normalized_volume = min(volume / 100, 1)
        normalized_relevance = min(theme_relevance / 10, 1)
        
        # Weight and combine components
        weights = Config.GDELT_CONFIG['sentiment_weights']
        impact = (
            weights['tone'] * normalized_tone +
            weights['volume'] * normalized_volume +
            weights['themes'] * normalized_relevance
        )
        
        return impact

    def get_sentiment_summary(self, sentiment_data: pd.DataFrame) -> Dict[str, any]:
        """Get summary of sentiment analysis"""
        try:
            if sentiment_data is None or sentiment_data.empty:
                return {}
            
            current_sentiment = sentiment_data['market_sentiment'].iloc[-1]
            sentiment_ma5 = sentiment_data['sentiment_ma5'].iloc[-1]
            
            return {
                'current_sentiment': current_sentiment,
                'short_term_trend': "Improving" if current_sentiment > sentiment_ma5 else "Declining",
                'sentiment_score': f"{current_sentiment:.2f}",
                'momentum': sentiment_data['sentiment_momentum'].iloc[-1],
                'volatility': sentiment_data['sentiment_volatility'].iloc[-1],
                'trend_strength': sentiment_data['trend_strength'].iloc[-1],
                'volume_impact': sentiment_data['volume_impact'].iloc[-1],
                'theme_impact': sentiment_data['theme_impact'].iloc[-1]
            }
            
        except Exception as e:
            st.error(f"Error generating sentiment summary: {str(e)}")
            return {}

class IntegratedDataFetcher:
    """Handle integrated data fetching from all sources"""
    
    def __init__(self):
        self._gdelt_fetcher = GDELTDataFetcher()
        self._economic_indicators = EconomicIndicators()
        self._asset_fetcher = AssetDataFetcher()

    @st.cache_data(ttl=Config.CACHE_TTL)
    def fetch_all_data(self, symbol: str, asset_type: str,
                      include_sentiment: bool = True,
                      include_economic: bool = True) -> Dict[str, pd.DataFrame]:
        """Fetch and integrate data from multiple sources"""
        try:
            data = {}
            
            # Fetch asset data
            if asset_type == "Stocks":
                data['price'] = self._asset_fetcher.get_stock_data(symbol)
            else:  # Cryptocurrency
                data['price'] = self._asset_fetcher.get_crypto_data(symbol)
            
            if data['price'] is not None:
                # Get date range
                start_date = data['price'].index.min()
                end_date = data['price'].index.max()
                
                # Fetch additional data if requested
                if include_sentiment:
                    data['sentiment'] = self._gdelt_fetcher.get_sentiment_data(symbol)
                
                if include_economic:
                    data['economic'] = self._economic_indicators.get_indicators()
                
                # Align dates
                data = self._align_dates(data)
                return data
            
            return {}
            
        except Exception as e:
            st.error(f"Error fetching integrated data: {str(e)}")
            return {}

    def _align_dates(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Align all data to the same date range"""
        if not data or 'price' not in data:
            return {}
        
        try:
            price_data = data['price']
            start_date = price_data.index.min()
            end_date = price_data.index.max()
            
            aligned_data = {'price': price_data}
            
            # Align sentiment data
            if 'sentiment' in data and data['sentiment'] is not None:
                sentiment_df = data['sentiment'].copy()
                aligned_data['sentiment'] = sentiment_df[
                    (sentiment_df.index >= start_date) & 
                    (sentiment_df.index <= end_date)
                ]
            
            # Align economic data
            if 'economic' in data and data['economic'] is not None:
                economic_df = data['economic'].copy()
                economic_df['date'] = pd.to_datetime(economic_df['date'])
                economic_df.set_index('date', inplace=True)
                aligned_data['economic'] = economic_df[
                    (economic_df.index >= start_date) & 
                    (economic_df.index <= end_date)
                ]
            
            return aligned_data
            
        except Exception as e:
            st.error(f"Error aligning dates: {str(e)}")
            return data

    def get_market_context(self, symbol: str, asset_type: str) -> Dict[str, float]:
        """Get market context data"""
        try:
            context = {}
            
            # Get market correlation for stocks
            if asset_type == "Stocks":
                spy_data = self._asset_fetcher.get_stock_data("SPY")
                if spy_data is not None:
                    context['market_correlation'] = self._calculate_correlation(
                        spy_data['Close'],
                        self._asset_fetcher.get_stock_data(symbol)['Close']
                    )
            
            # Get crypto correlation
            elif asset_type == "Cryptocurrency":
                btc_data = self._asset_fetcher.get_crypto_data("bitcoin")
                if btc_data is not None and symbol.lower() != "bitcoin":
                    context['crypto_correlation'] = self._calculate_correlation(
                        btc_data['Close'],
                        self._asset_fetcher.get_crypto_data(symbol)['Close']
                    )
            
            return context
            
        except Exception as e:
            st.error(f"Error getting market context: {str(e)}")
            return {}

    @staticmethod
    def _calculate_correlation(series1: pd.Series, series2: pd.Series) -> float:
        """Calculate correlation between two price series"""
        if series1 is None or series2 is None:
            return 0.0
        return series1.pct_change().corr(series2.pct_change())

    def get_enhanced_market_context(self, symbol: str, 
                                  asset_type: str,
                                  sentiment_data: Optional[pd.DataFrame] = None) -> Dict[str, any]:
        """Get enhanced market context including sentiment impact"""
        context = self.get_market_context(symbol, asset_type)
        
        try:
            if sentiment_data is