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

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentimentConfig:
    """Configuration class for sentiment analysis parameters"""
    def __init__(self, 
                 newsapi_key: str,
                 finnhub_key: str,
                 sentiment_period: int = 30,
                 sentiment_weight: float = 0.5):
        self.newsapi_key = newsapi_key
        self.finnhub_key = finnhub_key
        self.sentiment_period = sentiment_period
        self.sentiment_weight = sentiment_weight

class MultiSourceSentimentAnalyzer:
    """Enhanced sentiment analyzer with multiple data sources and robust error handling"""
    
    def __init__(self, config: SentimentConfig):
        """Initialize the sentiment analyzer with configuration"""
        self.config = config
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
        """Initialize News API client with error handling"""
        try:
            return NewsApiClient(api_key=self.config.newsapi_key)
        except Exception as e:
            logger.error(f"Failed to initialize News API client: {e}")
            return None

    async def fetch_sentiment(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Main method to fetch sentiment from all sources asynchronously"""
        try:
            sentiment_tasks = [
                self._fetch_yahoo_sentiment(symbol, start_date, end_date),
                self._fetch_newsapi_sentiment(symbol, start_date, end_date),
                self._fetch_finnhub_sentiment(symbol, start_date, end_date)
            ]
            
            results = await asyncio.gather(*sentiment_tasks, return_exceptions=True)
            valid_results = [df for df in results if isinstance(df, pd.DataFrame) and not df.empty]
            
            if not valid_results:
                logger.warning("No valid sentiment data retrieved from any source")
                return None
                
            return self._process_combined_sentiment(pd.concat(valid_results, ignore_index=True))
            
        except Exception as e:
            logger.error(f"Error in fetch_sentiment: {e}")
            return None

    async def _fetch_yahoo_sentiment(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch sentiment from Yahoo Finance with improved error handling"""
        try:
            ticker = yf.Ticker(symbol)
            news = await self._safe_api_call(lambda: ticker.news)
            
            if not news:
                return None
                
            sentiment_data = []
            for article in news:
                try:
                    sentiment_data.append(
                        await self._process_article(
                            article.get('title', ''),
                            article.get('description', ''),
                            datetime.fromtimestamp(article['providerPublishTime']),
                            'yahoo'
                        )
                    )
                except Exception as e:
                    logger.debug(f"Skipping article due to error: {e}")
                    continue
                    
            return pd.DataFrame([s for s in sentiment_data if s is not None])
            
        except Exception as e:
            logger.error(f"Error in Yahoo Finance sentiment fetch: {e}")
            return None

    async def _fetch_newsapi_sentiment(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch sentiment from News API with rate limiting and retries"""
        if not self.newsapi_client:
            return None
            
        try:
            articles = await self._safe_api_call(
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
                    sentiment_data.append(
                        await self._process_article(
                            article.get('title', ''),
                            article.get('description', ''),
                            pd.to_datetime(article['publishedAt']),
                            'newsapi'
                        )
                    )
                except Exception as e:
                    logger.debug(f"Skipping article due to error: {e}")
                    continue
                    
            return pd.DataFrame([s for s in sentiment_data if s is not None])
            
        except Exception as e:
            logger.error(f"Error in News API sentiment fetch: {e}")
            return None

    async def _process_article(self, 
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

    async def _safe_api_call(self, 
                            func: callable, 
                            max_retries: int = 3, 
                            delay: float = 1.0) -> Any:
        """Execute API calls with retry logic and rate limiting"""
        for attempt in range(max_retries):
            try:
                await asyncio.sleep(delay)  # Rate limiting
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                await asyncio.sleep(delay * (attempt + 1))  # Exponential backoff

class SentimentVisualizer:
    """Class for creating sentiment visualizations in Streamlit"""
    
    @staticmethod
    def display_sentiment_metrics(sentiment_data: pd.DataFrame) -> None:
        """Display key sentiment metrics"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_sentiment = sentiment_data['sentiment_score'].iloc[-1]
            sentiment_change = sentiment_data['sentiment_momentum'].iloc[-1]
            st.metric(
                "Current Sentiment",
                f"{current_sentiment:.2f}",
                f"{sentiment_change:+.2f}"
            )
        
        with col2:
            avg_sentiment = sentiment_data['sentiment_score'].mean()
            st.metric("Average Sentiment", f"{avg_sentiment:.2f}")
        
        with col3:
            conf_level = sentiment_data['confidence'].mean()
            st.metric("Confidence Level", f"{conf_level:.2f}")

    @staticmethod
    def create_sentiment_chart(sentiment_data: pd.DataFrame) -> None:
        """Create an interactive sentiment chart"""
        fig = go.Figure()
        
        # Main sentiment line
        fig.add_trace(go.Scatter(
            x=sentiment_data['ds'],
            y=sentiment_data['sentiment_score'],
            name='Sentiment',
            line=dict(color='purple', width=2)
        ))
        
        # Moving average
        fig.add_trace(go.Scatter(
            x=sentiment_data['ds'],
            y=sentiment_data['sentiment_ma'],
            name='3-Day MA',
            line=dict(color='blue', width=1, dash='dot')
        ))
        
        # Confidence bands
        upper_band = sentiment_data['sentiment_score'] + sentiment_data['sentiment_std']
        lower_band = sentiment_data['sentiment_score'] - sentiment_data['sentiment_std']
        
        fig.add_trace(go.Scatter(
            x=sentiment_data['ds'].tolist() + sentiment_data['ds'].tolist()[::-1],
            y=upper_band.tolist() + lower_band.tolist()[::-1],
            fill='toself',
            fillcolor='rgba(128, 0, 128, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Band'
        ))
        
        fig.update_layout(
            title="Market Sentiment Analysis",
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)

def initialize_sentiment_analyzer(newsapi_key: str, 
                                finnhub_key: str,
                                sentiment_period: int = 30,
                                sentiment_weight: float = 0.5) -> MultiSourceSentimentAnalyzer:
    """Factory function to create and initialize the sentiment analyzer"""
    config = SentimentConfig(
        newsapi_key=newsapi_key,
        finnhub_key=finnhub_key,
        sentiment_period=sentiment_period,
        sentiment_weight=sentiment_weight
    )
    return MultiSourceSentimentAnalyzer(config)