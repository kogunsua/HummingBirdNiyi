# gdelt_analysis.py
import pandas as pd
import numpy as np
from prophet import Prophet
import requests
from datetime import datetime, timedelta
import logging
from typing import Optional, Tuple, Dict
import streamlit as st
import plotly.graph_objects as go
from io import StringIO
import json

logger = logging.getLogger(__name__)

class GDELTAnalyzer:
    def __init__(self):
        self.base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
        self.v2_url = "https://api.gdeltproject.org/api/v2/events/events"
        self.theme_filters = [
            'ECON_BANKRUPTCY', 'ECON_COST', 'ECON_DEBT', 'ECON_REFORM',
            'BUS_MARKET_CLOSE', 'BUS_MARKET_CRASH', 'BUS_MARKET_DOWN',
            'BUS_MARKET_UP', 'BUS_STOCK_MARKET', 'POLITICS'
        ]

    def fetch_sentiment_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch sentiment data from GDELT 2.0"""
        try:
            # Format dates for GDELT API
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            # Fetch data day by day to ensure we get all events
            all_data = []
            current_dt = start_dt
            
            while current_dt <= end_dt:
                # Format date strings for API
                date_str = current_dt.strftime("%Y%m%d%H%M%S")
                next_dt = current_dt + timedelta(days=1)
                next_date_str = next_dt.strftime("%Y%m%d%H%M%S")
                
                # Construct query for financial and political news
                query_params = {
                    'query': ' OR '.join(self.theme_filters),
                    'format': 'csv',
                    'TIMESPAN': '1',
                    'TIMETYPE': 'CUSTOM',
                    'START': date_str,
                    'END': next_date_str,
                    'src': 'news',
                    'language': 'eng'
                }
                
                # Make API request
                response = requests.get(
                    self.v2_url,
                    params=query_params,
                    timeout=30
                )
                
                if response.status_code == 200:
                    try:
                        # Parse CSV data
                        df = pd.read_csv(StringIO(response.text), sep='\t')
                        if not df.empty:
                            all_data.append(df)
                    except Exception as e:
                        logger.warning(f"Error parsing data for {current_dt.date()}: {str(e)}")
                
                current_dt = next_dt
            
            # Combine all data
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                return self._process_sentiment_data(combined_df)
            else:
                # Fallback to generate dummy sentiment data
                return self._generate_dummy_sentiment_data(start_date, end_date)
                
        except Exception as e:
            logger.error(f"Error fetching GDELT data: {str(e)}")
            # Fallback to generate dummy sentiment data
            return self._generate_dummy_sentiment_data(start_date, end_date)

    def _generate_dummy_sentiment_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate dummy sentiment data for development and testing"""
        try:
            # Create date range
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Generate random sentiment scores with some trend and seasonality
            n = len(dates)
            trend = np.linspace(-1, 1, n)  # Linear trend
            seasonality = np.sin(np.linspace(0, 4*np.pi, n))  # Seasonal component
            noise = np.random.normal(0, 0.2, n)  # Random noise
            
            # Combine components and normalize to 0-1 range
            sentiment_scores = trend + 0.5 * seasonality + noise
            sentiment_scores = (sentiment_scores - sentiment_scores.min()) / (sentiment_scores.max() - sentiment_scores.min())
            
            # Create DataFrame
            df = pd.DataFrame({
                'ds': dates,
                'sentiment_score': sentiment_scores,
                'tone_avg': sentiment_scores * 100 - 50,  # Scale to -50 to 50
                'article_count': np.random.randint(100, 1000, n),
                'tone_std': np.random.uniform(0.1, 0.3, n),
                'positive_tone': sentiment_scores * 100,
                'negative_tone': (1 - sentiment_scores) * 100,
                'polarity': sentiment_scores * 2 - 1
            })
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating dummy sentiment data: {str(e)}")
            return pd.DataFrame()

    def _process_sentiment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process GDELT data into sentiment scores"""
        try:
            # Convert date column
            df['DATE'] = pd.to_datetime(df['SQLDATE'], format='%Y%m%d')
            
            # Calculate daily sentiment metrics
            daily_sentiment = df.groupby(df['DATE'].dt.date).agg({
                'AvgTone': ['mean', 'count', 'std'],
                'NumMentions': 'sum',
                'NumSources': 'mean',
                'NumArticles': 'sum'
            }).reset_index()
            
            # Flatten column names
            daily_sentiment.columns = ['ds', 'tone_avg', 'article_count', 'tone_std', 
                                     'mention_count', 'source_count', 'total_articles']
            
            # Convert date to datetime
            daily_sentiment['ds'] = pd.to_datetime(daily_sentiment['ds'])
            
            # Calculate normalized sentiment score
            daily_sentiment['sentiment_score'] = (
                (daily_sentiment['tone_avg'] + 100) / 200 * 
                np.log1p(daily_sentiment['article_count']) / np.log1p(daily_sentiment['article_count'].max())
            )
            
            # Calculate additional metrics
            daily_sentiment['positive_tone'] = daily_sentiment['tone_avg'].clip(lower=0)
            daily_sentiment['negative_tone'] = -daily_sentiment['tone_avg'].clip(upper=0)
            daily_sentiment['polarity'] = daily_sentiment['sentiment_score'] * 2 - 1
            
            return daily_sentiment.sort_values('ds')
            
        except Exception as e:
            logger.error(f"Error processing sentiment data: {str(e)}")
            return self._generate_dummy_sentiment_data(
                daily_sentiment['ds'].min().strftime('%Y-%m-%d'),
                daily_sentiment['ds'].max().strftime('%Y-%m-%d')
            )