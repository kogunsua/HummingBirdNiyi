"""
Multi-source sentiment analyzer for financial market analysis.
Provides synchronous sentiment analysis from multiple sources.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import requests
import time
import json
from typing import Optional, Dict, List, Tuple, Any
import yfinance as yf
from newsapi import NewsApiClient
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import streamlit as st
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultiSourceSentimentAnalyzer:
    """Enhanced sentiment analyzer with multiple data sources"""
    
    def __init__(self, newsapi_key: str = None, finnhub_key: str = None):
        """Initialize the sentiment analyzer"""
        self.newsapi_key = newsapi_key
        self.finnhub_key = finnhub_key
        self._initialize_nltk()
        self.newsapi_client = self._initialize_newsapi()

    def _initialize_nltk(self) -> None:
        """Initialize NLTK with error handling"""
        try:
            nltk.download('vader_lexicon', quiet=True)
            self.sia = SentimentIntensityAnalyzer()
        except Exception as e:
            logger.error(f"Failed to initialize NLTK: {e}")
            raise RuntimeError("Failed to initialize sentiment analyzer")

    def _initialize_newsapi(self) -> Optional[NewsApiClient]:
        """Initialize News API client"""
        if self.newsapi_key:
            try:
                return NewsApiClient(api_key=self.newsapi_key)
            except Exception as e:
                logger.error(f"Failed to initialize News API client: {e}")
        return None

    def fetch_yahoo_sentiment(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch sentiment from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if not news:
                logger.warning(f"No news found for {symbol}")
                return None
            
            sentiment_data = []
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            for article in news:
                try:
                    # Get article date
                    date = datetime.fromtimestamp(article['providerPublishTime'])
                    
                    # Check if article is within date range
                    if not (start_dt <= date <= end_dt):
                        continue
                    
                    # Process article text
                    title = article.get('title', '').strip()
                    description = article.get('description', '').strip()
                    
                    if not title:  # Skip if no title
                        continue
                    
                    sentiment_data.append(
                        self._process_article(
                            title,
                            description,
                            date,
                            'yahoo'
                        )
                    )
                except Exception as e:
                    logger.debug(f"Error processing Yahoo article: {e}")
                    continue
            
            valid_data = [s for s in sentiment_data if s is not None]
            if valid_data:
                return pd.DataFrame(valid_data)
            return None
            
        except Exception as e:
            logger.error(f"Error in Yahoo Finance sentiment fetch: {e}")
            return None

    def fetch_newsapi_sentiment(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch sentiment from News API"""
        if not self.newsapi_client:
            return None
            
        try:
            articles = self._safe_api_call(
                lambda: self.newsapi_client.get_everything(
                    q=symbol,
                    from_param=start_date,
                    to=end_date,
                    language='en',
                    sort_by='publishedAt'
                ),
                max_retries=3
            )
            
            if not articles or 'articles' not in articles:
                return None
                
            sentiment_data = []
            for article in articles['articles']:
                try:
                    title = article.get('title', '').strip()
                    description = article.get('description', '').strip()
                    published_at = article.get('publishedAt')
                    
                    if not (title and published_at):
                        continue
                        
                    sentiment_data.append(
                        self._process_article(
                            title,
                            description,
                            pd.to_datetime(published_at),
                            'newsapi'
                        )
                    )
                except Exception as e:
                    logger.debug(f"Error processing News API article: {e}")
                    continue
            
            valid_data = [s for s in sentiment_data if s is not None]
            if valid_data:
                return pd.DataFrame(valid_data)
            return None
            
        except Exception as e:
            logger.error(f"Error in News API sentiment fetch: {e}")
            return None

    def _process_article(self, 
                        title: str, 
                        description: str, 
                        date: datetime,
                        source: str) -> Optional[Dict[str, Any]]:
        """Process a single article with sentiment analysis"""
        if not title.strip():
            return None
            
        text = f"{title} {description}".strip()
        
        try:
            vader_scores = self.sia.polarity_scores(text)
            blob_sentiment = TextBlob(text).sentiment
            
            sentiment_score = (vader_scores['compound'] + blob_sentiment.polarity) / 2
            
            return {
                'ds': date,
                'sentiment_score': (sentiment_score + 1) / 2,  # Normalize to 0-1
                'source': source,
                'confidence': abs(blob_sentiment.subjectivity),
                'title': title
            }
        except Exception as e:
            logger.debug(f"Error processing article: {e}")
            return None

    def _process_combined_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process combined sentiment data with additional metrics"""
        try:
            df['ds'] = pd.to_datetime(df['ds'])
            df = df[df['ds'] <= datetime.now()]
            
            daily_sentiment = df.groupby('ds').agg({
                'sentiment_score': lambda x: np.average(x, weights=df.loc[x.index, 'confidence']),
                'confidence': 'mean',
                'title': 'count'
            }).rename(columns={'title': 'article_count'}).reset_index()
            
            # Add rolling statistics
            daily_sentiment['sentiment_ma'] = daily_sentiment['sentiment_score'].rolling(window=3).mean()
            daily_sentiment['sentiment_std'] = daily_sentiment['sentiment_score'].rolling(window=5).std()
            
            # Add momentum indicators
            daily_sentiment['sentiment_momentum'] = daily_sentiment['sentiment_score'].diff()
            daily_sentiment['sentiment_acceleration'] = daily_sentiment['sentiment_momentum'].diff()
            
            return daily_sentiment.sort_values('ds')
            
        except Exception as e:
            logger.error(f"Error processing combined sentiment: {e}")
            return pd.DataFrame()

    def _safe_api_call(self, func: callable, max_retries: int = 3, delay: float = 1.0) -> Any:
        """Execute API calls with retry logic and rate limiting"""
        for attempt in range(max_retries):
            try:
                time.sleep(delay)  # Rate limiting
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(delay * (attempt + 1))  # Exponential backoff

def display_sentiment_impact_analysis(sentiment_period: int, 
                                    sentiment_weight: float,
                                    sentiment_source: str) -> None:
    """Display sentiment impact analysis configuration"""
    st.markdown("### 🎭 Sentiment Impact Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Analysis Period",
            f"{sentiment_period} days",
            help="Historical period used for sentiment analysis"
        )
    
    with col2:
        impact_level = (
            "High" if sentiment_weight > 0.7
            else "Medium" if sentiment_weight > 0.3
            else "Low"
        )
        impact_color = (
            "🔴" if sentiment_weight > 0.7
            else "🟡" if sentiment_weight > 0.3
            else "🟢"
        )
        st.metric(
            "Impact Level",
            f"{impact_level} {impact_color}",
            f"{sentiment_weight:.1%}",
            help="Level of influence sentiment has on forecast"
        )
    
    with col3:
        source_reliability = {
            "Multi-Source": {"level": "High", "confidence": 0.9},
            "GDELT": {"level": "Medium-High", "confidence": 0.8},
            "Yahoo Finance": {"level": "Medium", "confidence": 0.7},
            "News API": {"level": "Medium", "confidence": 0.7}
        }
        
        reliability_info = source_reliability.get(sentiment_source, {"level": "Medium", "confidence": 0.7})
        st.metric(
            "Source Reliability",
            reliability_info['level'],
            f"{reliability_info['confidence']:.0%}",
            help="Reliability of the selected sentiment data source"
        )

def display_sentiment_impact_results(impact_metrics: Dict) -> None:
    """Display sentiment impact analysis results"""
    st.subheader("🎭 Sentiment Impact Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        correlation = impact_metrics.get('sentiment_correlation', 0)
        correlation_color = (
            "🟢" if abs(correlation) > 0.7
            else "🟡" if abs(correlation) > 0.3
            else "🔴"
        )
        st.metric(
            "Price-Sentiment Correlation",
            f"{correlation:.2f} {correlation_color}"
        )
    
    with col2:
        volatility = impact_metrics.get('sentiment_volatility', 0)
        volatility_color = (
            "🔴" if volatility > 0.7
            else "🟡" if volatility > 0.3
            else "🟢"
        )
        st.metric(
            "Sentiment Volatility",
            f"{volatility:.2f} {volatility_color}"
        )
    
    with col3:
        sensitivity = impact_metrics.get('price_sensitivity', 0)
        sensitivity_color = (
            "🟡" if sensitivity > 0.7
            else "🟢" if sensitivity > 0.3
            else "🔴"
        )
        st.metric(
            "Price Sensitivity",
            f"{sensitivity:.2f} {sensitivity_color}"
        )

def get_sentiment_data(analyzer: MultiSourceSentimentAnalyzer,
                      symbol: str,
                      start_date: str,
                      end_date: str,
                      sentiment_source: str = "Multi-Source") -> Optional[pd.DataFrame]:
    """Get sentiment data from specified source"""
    try:
        if sentiment_source == "Yahoo Finance":
            return analyzer.fetch_yahoo_sentiment(symbol, start_date, end_date)
        elif sentiment_source == "News API":
            return analyzer.fetch_newsapi_sentiment(symbol, start_date, end_date)
        else:  # Multi-Source
            # Collect data from all available sources
            sentiment_data = []
            
            # Try Yahoo Finance
            yahoo_data = analyzer.fetch_yahoo_sentiment(symbol, start_date, end_date)
            if yahoo_data is not None:
                sentiment_data.append(yahoo_data)
            
            # Try News API
            news_data = analyzer.fetch_newsapi_sentiment(symbol, start_date, end_date)
            if news_data is not None:
                sentiment_data.append(news_data)
            
            if not sentiment_data:
                logger.warning("No sentiment data available from any source")
                return None
            
            # Combine and process data
            combined_data = pd.concat(sentiment_data, ignore_index=True)
            return analyzer._process_combined_sentiment(combined_data)
    
    except Exception as e:
        logger.error(f"Error getting sentiment data: {str(e)}")
        return None

def integrate_multi_source_sentiment(symbol: str,
                                   sentiment_period: int,
                                   sentiment_weight: float = 0.5) -> Tuple[Optional[pd.DataFrame], Dict]:
    """Integrate sentiment analysis with forecasting"""
    try:
        analyzer = MultiSourceSentimentAnalyzer()
        start_date = (datetime.now() - timedelta(days=sentiment_period)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        sentiment_data = get_sentiment_data(analyzer, symbol, start_date, end_date)
        
        if sentiment_data is not None:
            impact_metrics = {
                'sentiment_correlation': sentiment_data['sentiment_score'].autocorr(),
                'sentiment_volatility': sentiment_data['sentiment_score'].std(),
                'price_sensitivity': sentiment_weight
            }
            return sentiment_data, impact_metrics
        
        return None, {}
    
    except Exception as e:
        logger.error(f"Error in sentiment integration: {str(e)}")
        return None, {}

# Export all required functions and classes
__all__ = [
    'MultiSourceSentimentAnalyzer',
    'integrate_multi_source_sentiment',
    'display_sentiment_impact_analysis',
    'display_sentiment_impact_results',
    'get_sentiment_data'
]