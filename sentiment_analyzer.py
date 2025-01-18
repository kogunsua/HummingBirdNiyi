import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import requests
from typing import Optional, Dict, List, Tuple
import yfinance as yf
import json
from newsapi import NewsApiClient
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import streamlit as st
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiSourceSentimentAnalyzer:
    def __init__(self):
        # Initialize sentiment analyzers
        try:
            nltk.download('vader_lexicon', quiet=True)
            self.sia = SentimentIntensityAnalyzer()
        except Exception as e:
            logger.error(f"Error initializing NLTK: {str(e)}")
            self.sia = None

        # API Keys (you should store these securely)
        self.newsapi_key = "pub_65773c625d48ffecc8522ad52fe0fd7199cce"  # Get from https://newsapi.org/
        self.finnhub_key = "cpllsnpr01qn8g1v08hgcpllsnpr01qn8g1v08i0"   # Get from https://finnhub.io/

    def fetch_combined_sentiment(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch sentiment data from multiple sources"""
        try:
            sentiment_data = []
            
            # 1. Try Yahoo Finance News Sentiment
            yahoo_sentiment = self.fetch_yahoo_sentiment(symbol, start_date, end_date)
            if yahoo_sentiment is not None:
                sentiment_data.append(yahoo_sentiment)

            # 2. Try News API
            news_sentiment = self.fetch_newsapi_sentiment(symbol, start_date, end_date)
            if news_sentiment is not None:
                sentiment_data.append(news_sentiment)

            # 3. Try Finnhub Sentiment (if available)
            finnhub_sentiment = self.fetch_finnhub_sentiment(symbol, start_date, end_date)
            if finnhub_sentiment is not None:
                sentiment_data.append(finnhub_sentiment)

            # Combine all sentiment data
            if sentiment_data:
                combined_df = pd.concat(sentiment_data)
                return self._process_combined_sentiment(combined_df)
            else:
                logger.warning("No sentiment data available from any source")
                return None

        except Exception as e:
            logger.error(f"Error fetching combined sentiment: {str(e)}")
            return None

    def fetch_yahoo_sentiment(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch and analyze sentiment from Yahoo Finance news"""
        try:
            # Get stock/asset info
            ticker = yf.Ticker(symbol)
            
            # Fetch news
            news = ticker.news
            if not news:
                return None

            sentiment_data = []
            for article in news:
                try:
                    # Get article date
                    date = datetime.fromtimestamp(article['providerPublishTime'])
                    
                    # Only include articles within the date range
                    if not (start_date <= date.strftime("%Y-%m-%d") <= end_date):
                        continue
                    
                    # Analyze sentiment using both VADER and TextBlob
                    title_body = f"{article['title']} {article.get('description', '')}"
                    
                    # VADER sentiment
                    vader_scores = self.sia.polarity_scores(title_body)
                    
                    # TextBlob sentiment
                    blob_sentiment = TextBlob(title_body).sentiment
                    
                    # Combine sentiment scores
                    sentiment_score = (vader_scores['compound'] + blob_sentiment.polarity) / 2
                    
                    sentiment_data.append({
                        'ds': date,
                        'sentiment_score': (sentiment_score + 1) / 2,  # Normalize to 0-1
                        'source': 'yahoo',
                        'confidence': abs(blob_sentiment.subjectivity)
                    })
                except Exception as e:
                    logger.warning(f"Error processing Yahoo article: {str(e)}")
                    continue

            if sentiment_data:
                return pd.DataFrame(sentiment_data)
            return None

        except Exception as e:
            logger.error(f"Error fetching Yahoo sentiment: {str(e)}")
            return None

    def fetch_newsapi_sentiment(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch and analyze sentiment from News API"""
        try:
            newsapi = NewsApiClient(api_key=self.newsapi_key)
            
            # Get news articles
            articles = newsapi.get_everything(
                q=symbol,
                from_param=start_date,
                to=end_date,
                language='en',
                sort_by='publishedAt'
            )

            if not articles['articles']:
                return None

            sentiment_data = []
            for article in articles['articles']:
                try:
                    # Combine title and description
                    text = f"{article['title']} {article.get('description', '')}"
                    
                    # Get both VADER and TextBlob sentiment
                    vader_scores = self.sia.polarity_scores(text)
                    blob_sentiment = TextBlob(text).sentiment
                    
                    # Combine scores
                    sentiment_score = (vader_scores['compound'] + blob_sentiment.polarity) / 2
                    
                    sentiment_data.append({
                        'ds': pd.to_datetime(article['publishedAt']),
                        'sentiment_score': (sentiment_score + 1) / 2,
                        'source': 'newsapi',
                        'confidence': abs(blob_sentiment.subjectivity)
                    })
                except Exception as e:
                    logger.warning(f"Error processing NewsAPI article: {str(e)}")
                    continue

            if sentiment_data:
                return pd.DataFrame(sentiment_data)
            return None

        except Exception as e:
            logger.error(f"Error fetching NewsAPI sentiment: {str(e)}")
            return None

    def fetch_finnhub_sentiment(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch sentiment data from Finnhub"""
        try:
            base_url = "https://finnhub.io/api/v1"
            headers = {'X-Finnhub-Token': self.finnhub_key}
            
            # Convert dates to timestamps
            start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
            end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
            
            # Fetch sentiment data
            url = f"{base_url}/news-sentiment?symbol={symbol}"
            response = requests.get(url, headers=headers)
            
            if response.status_code != 200:
                return None

            data = response.json()
            if not data:
                return None

            # Create DataFrame from sentiment data
            sentiment_data = []
            for news in data.get('data', []):
                if start_timestamp <= news['datetime'] <= end_timestamp:
                    sentiment_data.append({
                        'ds': pd.to_datetime(news['datetime'], unit='s'),
                        'sentiment_score': news['sentiment'],
                        'source': 'finnhub',
                        'confidence': news.get('confidence', 0.5)
                    })

            if sentiment_data:
                return pd.DataFrame(sentiment_data)
            return None

        except Exception as e:
            logger.error(f"Error fetching Finnhub sentiment: {str(e)}")
            return None

    def _process_combined_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and combine sentiment data from multiple sources"""
        try:
            # Group by date and calculate weighted average sentiment
            daily_sentiment = df.groupby('ds').agg({
                'sentiment_score': lambda x: np.average(x, weights=df.loc[x.index, 'confidence']),
                'confidence': 'mean'
            }).reset_index()

            # Calculate additional metrics
            daily_sentiment['tone_avg'] = (daily_sentiment['sentiment_score'] - 0.5) * 200
            daily_sentiment['tone_std'] = df.groupby(df['ds'].dt.date)['sentiment_score'].transform('std')
            daily_sentiment['article_count'] = df.groupby(df['ds'].dt.date)['sentiment_score'].transform('count')

            return daily_sentiment.sort_values('ds')

        except Exception as e:
            logger.error(f"Error processing combined sentiment: {str(e)}")
            raise

def integrate_multi_source_sentiment(symbol: str, sentiment_period: int) -> Optional[pd.DataFrame]:
    """Integrate multi-source sentiment analysis"""
    try:
        analyzer = MultiSourceSentimentAnalyzer()
        
        start_date = (datetime.now() - timedelta(days=sentiment_period)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        with st.spinner('Fetching sentiment data from multiple sources...'):
            sentiment_data = analyzer.fetch_combined_sentiment(symbol, start_date, end_date)
        
        if sentiment_data is None or sentiment_data.empty:
            st.warning("No sentiment data available from any source. Using price-only forecast.")
            return None
        
        st.success("Successfully fetched sentiment data!")
        
        # Display sentiment analysis
        st.markdown("### 📊 Multi-Source Market Sentiment Analysis")
        
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
                "Confidence Level",
                f"{sentiment_data['confidence'].mean():.2f}"
            )
        
        # Create sentiment visualization
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=sentiment_data['ds'],
                y=sentiment_data['sentiment_score'],
                name='Sentiment Score',
                line=dict(color='purple')
            )
        )
        
        fig.update_layout(
            title="Market Sentiment Trend (Multi-Source)",
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        return sentiment_data

    except Exception as e:
        logger.error(f"Error in multi-source sentiment integration: {str(e)}")
        st.error(f"Error in sentiment analysis: {str(e)}")
        return None