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
                try:
                    response = requests.get(
                        self.v2_url,
                        params=query_params,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        df = pd.read_csv(StringIO(response.text), sep='\t')
                        if not df.empty:
                            all_data.append(df)
                    else:
                        logger.warning(f"Failed to fetch data for {current_dt.date()}: Status code {response.status_code}")
                
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Request failed for {current_dt.date()}: {str(e)}")
                
                current_dt = next_dt
            
            # Combine all data
            if not all_data:
                raise ValueError("No sentiment data available from GDELT API")
                
            combined_df = pd.concat(all_data, ignore_index=True)
            return self._process_sentiment_data(combined_df)
                
        except Exception as e:
            logger.error(f"Error fetching GDELT data: {str(e)}")
            raise ValueError(f"Failed to fetch sentiment data: {str(e)}")

    def _process_sentiment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process GDELT data into sentiment scores"""
        try:
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
            
            daily_sentiment['ds'] = pd.to_datetime(daily_sentiment['ds'])
            
            # Calculate sentiment score (normalized between 0 and 1)
            daily_sentiment['sentiment_score'] = (daily_sentiment['tone_avg'] + 100) / 200
            
            # Ensure the data is properly sorted
            return daily_sentiment.sort_values('ds')
            
        except Exception as e:
            logger.error(f"Error processing sentiment data: {str(e)}")
            raise ValueError(f"Failed to process sentiment data: {str(e)}")

def integrate_sentiment_analysis(sentiment_period: int) -> Optional[pd.DataFrame]:
    """Integrate sentiment analysis into the main application"""
    try:
        # Initialize GDELT analyzer
        analyzer = GDELTAnalyzer()
        
        # Get sentiment data
        try:
            sentiment_data = analyzer.fetch_sentiment_data(
                (datetime.now() - timedelta(days=sentiment_period)).strftime("%Y-%m-%d"),
                datetime.now().strftime("%Y-%m-%d")
            )
        except ValueError as e:
            st.error(str(e))
            st.warning("Unable to incorporate sentiment analysis. Proceeding with price-only forecast.")
            return None
        
        if sentiment_data is None or sentiment_data.empty:
            st.error("No sentiment data available")
            return None
            
        st.markdown("### 📊 Market Sentiment Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_sentiment = sentiment_data['sentiment_score'].iloc[-1]
            sentiment_change = sentiment_data['sentiment_score'].iloc[-1] - sentiment_data['sentiment_score'].iloc[-2]
            st.metric(
                "Current Sentiment",
                f"{current_sentiment:.2f}",
                f"{sentiment_change:+.2f}"
            )
        
        with col2:
            st.metric(
                "Average Sentiment",
                f"{sentiment_data['sentiment_score'].mean():.2f}"
            )
        
        with col3:
            st.metric(
                "Sentiment Volatility",
                f"{sentiment_data['sentiment_score'].std():.2f}"
            )
        
        fig_sentiment = go.Figure()
        fig_sentiment.add_trace(
            go.Scatter(
                x=sentiment_data['ds'],
                y=sentiment_data['sentiment_score'],
                name='Sentiment Score',
                line=dict(color='purple')
            )
        )
        
        fig_sentiment.update_layout(
            title="Market Sentiment Trend",
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
            height=400
        )
        
        st.plotly_chart(fig_sentiment, use_container_width=True)
        
        return sentiment_data
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis integration: {str(e)}")
        st.error(f"Error in sentiment analysis: {str(e)}")
        return None

def update_forecasting_process(price_data: pd.DataFrame, 
                             sentiment_data: Optional[pd.DataFrame] = None,
                             sentiment_weight: float = 0.5) -> Tuple[Optional[pd.DataFrame], Dict]:
    """Updated forecasting process incorporating sentiment analysis with configurable weight"""
    try:
        analyzer = GDELTAnalyzer()
        
        if sentiment_data is not None and not sentiment_data.empty and 'sentiment_score' in sentiment_data.columns:
            # Prepare the combined dataset
            combined_data = pd.DataFrame({
                'ds': price_data.index,
                'y': price_data['Close']
            })
            
            # Merge sentiment data and handle missing values
            combined_data = combined_data.merge(
                sentiment_data[['ds', 'sentiment_score']], 
                on='ds', 
                how='left'
            )
            combined_data['sentiment_score'] = combined_data['sentiment_score'].fillna(method='ffill')
            
            # Apply sentiment weight
            combined_data['sentiment_score'] = combined_data['sentiment_score'] * sentiment_weight
            
            # Generate forecast
            forecast, error = analyzer.enhanced_prophet_forecast(combined_data)
            
            if error:
                st.error(f"Forecasting error: {error}")
                return None, {}
                
            impact_metrics = analyzer.analyze_sentiment_impact(forecast)
            return forecast, impact_metrics
        else:
            # Fall back to regular forecasting if no valid sentiment data
            from forecasting import prophet_forecast
            forecast, error = prophet_forecast(price_data, periods=30)
            if error:
                st.error(f"Forecasting error: {error}")
                return None, {}
            return forecast, {}
            
    except Exception as e:
        logger.error(f"Error in forecasting process: {str(e)}")
        st.error(f"Error in forecasting process: {str(e)}")
        return None, {}