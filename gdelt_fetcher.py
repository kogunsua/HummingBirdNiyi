"""
GDELT data fetching and processing functionality
"""

import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from config import Config


class GDELTDataFetcher:
    """Handle GDELT data collection and analysis"""
    
    def __init__(self):
        """Initialize the GDELT fetcher"""
        self.config = Config.GDELT_CONFIG
        self.cache = {}
    
    @st.cache_data(ttl=Config.CACHE_TTL)
    def get_sentiment_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get sentiment data for a symbol with caching"""
        try:
            # Generate dates for fetching
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            # Fetch GKG data
            gkg_data = self._fetch_gkg_data(start_date, end_date, symbol)
            
            if gkg_data is not None and not gkg_data.empty:
                # Process sentiment data
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
            # For demo purposes, generate synthetic data
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Generate synthetic sentiment data
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