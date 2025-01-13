#gdelt_fetchers.py
import pandas as pd
import requests
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, Any
import streamlit as st

logger = logging.getLogger(__name__)

class GDELTFetcher:
    """Handle GDELT 2.0 GKG API interactions and data processing"""
    
    def __init__(self):
        self.base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
        self.theme_filters = [
            'ECON_BANKRUPTCY', 'ECON_COST', 'ECON_DEBT', 'ECON_REFORM',
            'BUS_MARKET_CLOSE', 'BUS_MARKET_CRASH', 'BUS_MARKET_DOWN',
            'BUS_MARKET_UP', 'BUS_STOCK_MARKET', 'POLITICS'
        ]
        
    @st.cache_data(ttl=3600)
    def fetch_sentiment_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch and process GDELT sentiment data for market analysis
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
        """
        try:
            logger.info(f"Fetching GDELT data from {start_date} to {end_date}")
            
            # Construct query for financial and political themes
            theme_query = ' OR '.join(f'theme:{theme}' for theme in self.theme_filters)
            query = f"({theme_query}) AND sourcelang:eng"
            
            # Add date range to query
            date_format = "%Y%m%d%H%M%S"
            start = datetime.strptime(start_date, "%Y-%m-%d").strftime(date_format)
            end = datetime.strptime(end_date, "%Y-%m-%d").strftime(date_format)
            
            # Construct final URL
            url = f"{self.base_url}?query={query}&START={start}&END={end}&format=csv&maxrecords=250000"
            
            # Fetch data
            response = requests.get(url)
            if response.status_code != 200:
                raise Exception(f"GDELT API request failed with status {response.status_code}")
            
            # Process the CSV data
            df = pd.read_csv(pd.compat.StringIO(response.text), sep='\t')
            
            # Process and aggregate sentiment data
            sentiment_df = self._process_sentiment_data(df)
            
            return sentiment_df
            
        except Exception as e:
            logger.error(f"Error fetching GDELT data: {str(e)}")
            return None
    
    def _process_sentiment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process raw GDELT data into daily sentiment scores"""
        try:
            # Convert date column
            df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d%H%M%S')
            
            # Calculate daily sentiment metrics
            daily_sentiment = df.groupby(df['DATE'].dt.date).agg({
                'Tone': ['mean', 'count', 'std'],
                'PositiveTone': 'mean',
                'NegativeTone': 'mean',
                'Polarity': 'mean'
            }).reset_index()
            
            # Flatten column names
            daily_sentiment.columns = ['ds', 'tone_avg', 'article_count', 'tone_std', 
                                     'positive_tone', 'negative_tone', 'polarity']
            
            # Convert date to datetime
            daily_sentiment['ds'] = pd.to_datetime(daily_sentiment['ds'])
            
            # Calculate sentiment score (normalized)
            daily_sentiment['sentiment_score'] = (
                (daily_sentiment['tone_avg'] + 100) / 200 * 
                daily_sentiment['article_count'] / daily_sentiment['article_count'].max()
            )
            
            # Sort by date
            daily_sentiment = daily_sentiment.sort_values('ds')
            
            return daily_sentiment
            
        except Exception as e:
            logger.error(f"Error processing GDELT sentiment data: {str(e)}")
            raise e

    def get_latest_sentiment_stats(self) -> Dict[str, Any]:
        """Get latest sentiment statistics for display"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            sentiment_data = self.fetch_sentiment_data(
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )
            
            if sentiment_data is None or sentiment_data.empty:
                return {}
            
            latest_stats = {
                'current_sentiment': sentiment_data['sentiment_score'].iloc[-1],
                'sentiment_change': (
                    sentiment_data['sentiment_score'].iloc[-1] - 
                    sentiment_data['sentiment_score'].iloc[-2]
                ),
                'article_count': sentiment_data['article_count'].iloc[-1],
                'average_tone': sentiment_data['tone_avg'].iloc[-1],
                'volatility': sentiment_data['tone_std'].iloc[-1]
            }
            
            return latest_stats
            
        except Exception as e:
            logger.error(f"Error getting latest sentiment stats: {str(e)}")
            return {}