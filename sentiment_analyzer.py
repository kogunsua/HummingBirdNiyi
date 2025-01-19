import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import requests
import time
import json
from typing import Optional, Dict, List, Tuple
import yfinance as yf
from newsapi import NewsApiClient
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import streamlit as st
import plotly.graph_objects as go

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def display_sentiment_impact_analysis(sentiment_period: int, sentiment_weight: float, sentiment_source: str):
    """Display sentiment impact analysis configuration and explanation"""
    st.markdown("### 🎭 Sentiment Impact Analysis")
    
    # Configure columns for metrics
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
    
    with st.expander("💡 Understanding Sentiment Impact"):
        st.markdown("""
        **How Sentiment Affects the Forecast:**
        
        1. **Analysis Period** (Historical Window)
           - Longer periods provide more stable analysis
           - Shorter periods capture recent market sentiment
           - Optimal period varies by asset volatility
        
        2. **Impact Level** (Weight)
           - High (>70%): Strong sentiment influence
           - Medium (30-70%): Balanced price-sentiment mix
           - Low (<30%): Minimal sentiment adjustment
        
        3. **Source Reliability**
           - Multi-Source: Highest reliability (combined sources)
           - GDELT: Global event impact
           - News/Finance API: Market-specific sentiment
        """)

def display_sentiment_impact_results(impact_metrics: dict):
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
    
    with st.expander("📊 Impact Analysis Interpretation"):
        st.markdown(f"""
        **Current Market Sentiment Analysis:**
        
        1. **Correlation** ({correlation:.2f}):
           - {
            "Strong price-sentiment relationship" if abs(correlation) > 0.7
            else "Moderate price-sentiment relationship" if abs(correlation) > 0.3
            else "Weak price-sentiment relationship"
           }
        
        2. **Volatility** ({volatility:.2f}):
           - {
            "High sentiment volatility - exercise caution" if volatility > 0.7
            else "Moderate sentiment volatility" if volatility > 0.3
            else "Low sentiment volatility - stable sentiment"
           }
        
        3. **Price Sensitivity** ({sensitivity:.2f}):
           - {
            "High price sensitivity to sentiment" if sensitivity > 0.7
            else "Moderate price sensitivity" if sensitivity > 0.3
            else "Low price sensitivity to sentiment"
           }
        """)

def get_sentiment_data(analyzer, symbol: str, start_date: str, end_date: str, sentiment_source: str) -> Optional[pd.DataFrame]:
    """Get sentiment data from specified source"""
    try:
        # Map sentiment sources to method names
        source_method_map = {
            "Yahoo Finance": "yahoo",
            "News API": "newsapi",
            "Finnhub": "finnhub",
            "Multi-Source": "combined_sentiment"  # Changed this to match the method name
        }
        
        # Get the correct method name from the map
        if sentiment_source == "Multi-Source":
            logger.info("Using combined sentiment analysis")
            sentiment_data = analyzer.fetch_combined_sentiment(symbol, start_date, end_date)
            if sentiment_data is None or sentiment_data.empty:
                logger.warning("No data available from combined sources")
                st.warning("No sentiment data available from combined sources")
                return None
            return sentiment_data
            
        method_name = source_method_map.get(sentiment_source.replace(" ", "_").lower())
        if not method_name:
            logger.error(f"Invalid sentiment source: {sentiment_source}")
            st.error(f"Invalid sentiment source: {sentiment_source}")
            return None
            
        # Construct the full method name
        method_name = f"fetch_{method_name}_sentiment"
        logger.info(f"Using sentiment method: {method_name}")
        
        # Get the method and call it
        sentiment_method = getattr(analyzer, method_name, None)
        if sentiment_method is None:
            logger.error(f"Method {method_name} not found in analyzer")
            st.error(f"Method {method_name} not found in analyzer")
            return None
            
        sentiment_data = sentiment_method(symbol, start_date, end_date)
        if sentiment_data is None or sentiment_data.empty:
            logger.warning(f"No sentiment data available from {sentiment_source}")
            st.warning(f"No sentiment data available from {sentiment_source}")
            return None
            
        return sentiment_data
        
    except Exception as e:
        logger.error(f"Error getting sentiment data: {str(e)}")
        st.error(f"Error getting sentiment data from {sentiment_source}: {str(e)}")
        return None
        
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
        self.newsapi_key = "65773c625d48ffecc8522ad52fe0fd7199cce"  # Updated key without pub_ prefix
        self.finnhub_key = "cpllsnpr01qn8g1v08hgcpllsnpr01qn8g1v08i0"

    def fetch_combined_sentiment(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch sentiment data from multiple sources"""
        try:
            logger.info(f"Fetching combined sentiment for {symbol} from {start_date} to {end_date}")
            sentiment_data = []
            source_count = 0
            
            # 1. Try Yahoo Finance News Sentiment with delay
            try:
                time.sleep(1)  # Add delay before API call
                yahoo_sentiment = self.fetch_yahoo_sentiment(symbol, start_date, end_date)
                if yahoo_sentiment is not None and not yahoo_sentiment.empty:
                    sentiment_data.append(yahoo_sentiment)
                    source_count += 1
                    logger.info(f"Successfully fetched Yahoo Finance sentiment data: {len(yahoo_sentiment)} records")
                else:
                    logger.warning("No Yahoo Finance sentiment data available")
            except Exception as e:
                logger.warning(f"Error fetching Yahoo Finance sentiment: {str(e)}")

            # 2. Try News API with delay
            try:
                time.sleep(1)  # Add delay before API call
                news_sentiment = self.fetch_newsapi_sentiment(symbol, start_date, end_date)
                if news_sentiment is not None and not news_sentiment.empty:
                    sentiment_data.append(news_sentiment)
                    source_count += 1
                    logger.info(f"Successfully fetched News API sentiment data: {len(news_sentiment)} records")
                else:
                    logger.warning("No News API sentiment data available")
            except Exception as e:
                logger.warning(f"Error fetching News API sentiment: {str(e)}")

            # 3. Try Finnhub Sentiment with delay
            try:
                time.sleep(1)  # Add delay before API call
                finnhub_sentiment = self.fetch_finnhub_sentiment(symbol, start_date, end_date)
                if finnhub_sentiment is not None and not finnhub_sentiment.empty:
                    sentiment_data.append(finnhub_sentiment)
                    source_count += 1
                    logger.info(f"Successfully fetched Finnhub sentiment data: {len(finnhub_sentiment)} records")
                else:
                    logger.warning("No Finnhub sentiment data available")
            except Exception as e:
                logger.warning(f"Error fetching Finnhub sentiment: {str(e)}")

            # Process combined sentiment data
            if sentiment_data:
                logger.info(f"Successfully fetched data from {source_count} sources")
                try:
                    # Combine all data
                    combined_df = pd.concat(sentiment_data, ignore_index=True)
                    
                    # Sort by date and remove duplicates if any
                    combined_df = combined_df.sort_values('ds').drop_duplicates(subset=['ds'], keep='first')
                    
                    # Process the combined data
                    processed_df = self._process_combined_sentiment(combined_df)
                    
                    if processed_df is not None and not processed_df.empty:
                        logger.info(f"Successfully processed combined sentiment data: {len(processed_df)} records")
                        return processed_df
                    else:
                        logger.warning("Failed to process combined sentiment data")
                        return None
                except Exception as e:
                    logger.error(f"Error processing combined sentiment data: {str(e)}")
                    return None
            else:
                logger.warning("No sentiment data available from any source")
                return None

        except Exception as e:
            logger.error(f"Error in fetch_combined_sentiment: {str(e)}")
            return None
            
def fetch_yahoo_sentiment(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch and analyze sentiment from Yahoo Finance news"""
        try:
            logger.info(f"Fetching Yahoo Finance sentiment for {symbol} from {start_date} to {end_date}")
            
            # Convert dates to datetime objects for comparison
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Get stock/asset info with better error handling
            try:
                ticker = yf.Ticker(symbol)
                # Force an update of the info to ensure we have fresh data
                try:
                    _ = ticker.info
                except:
                    pass  # Continue even if info fetch fails
                    
                news = ticker.news
                
                if not news:
                    logger.warning(f"No news found for {symbol} in Yahoo Finance")
                    return None
                    
            except Exception as e:
                logger.error(f"Error accessing Yahoo Finance API: {str(e)}")
                return None

            # Process news articles
            sentiment_data = []
            for article in news:
                try:
                    # Ensure we have required fields
                    if 'providerPublishTime' not in article or not article.get('title'):
                        continue
                        
                    # Get article date
                    date = datetime.fromtimestamp(article['providerPublishTime'])
                    
                    # Only include articles within the date range
                    if not (start_dt <= date <= end_dt):
                        continue
                    
                    # Combine title and description with better handling
                    title = article.get('title', '').strip()
                    description = article.get('description', '').strip()
                    
                    if not title:  # Skip if no title
                        continue
                        
                    title_body = f"{title} {description}"
                    
                    # VADER sentiment with error handling
                    try:
                        vader_scores = self.sia.polarity_scores(title_body)
                    except Exception as e:
                        logger.warning(f"Error calculating VADER sentiment: {str(e)}")
                        continue
                    
                    # TextBlob sentiment with error handling
                    try:
                        blob_sentiment = TextBlob(title_body).sentiment
                    except Exception as e:
                        logger.warning(f"Error calculating TextBlob sentiment: {str(e)}")
                        continue
                    
                    # Combine sentiment scores
                    sentiment_score = (vader_scores['compound'] + blob_sentiment.polarity) / 2
                    
                    sentiment_data.append({
                        'ds': date,
                        'sentiment_score': (sentiment_score + 1) / 2,  # Normalize to 0-1
                        'source': 'yahoo',
                        'confidence': abs(blob_sentiment.subjectivity),
                        'title': title  # Store title for reference
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing Yahoo article: {str(e)}")
                    continue

            if sentiment_data:
                df = pd.DataFrame(sentiment_data)
                logger.info(f"Successfully processed {len(df)} articles from Yahoo Finance")
                return df
            
            logger.warning("No valid sentiment data found from Yahoo Finance")
            return None

        except Exception as e:
            logger.error(f"Error fetching Yahoo sentiment: {str(e)}")
            return None

def fetch_newsapi_sentiment(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch and analyze sentiment from News API"""
        try:
            logger.info(f"Fetching News API sentiment for {symbol} from {start_date} to {end_date}")
            
            try:
                newsapi = NewsApiClient(api_key=self.newsapi_key)
                # Test the API connection with minimal query
                test_response = newsapi.get_everything(
                    q=symbol,
                    from_param=start_date,
                    to=end_date,
                    language='en',
                    page_size=1
                )
                if not test_response.get('status') == 'ok':
                    logger.error("News API connection test failed")
                    return None
                
            except Exception as e:
                logger.error(f"Error initializing News API client: {str(e)}")
                return None
            
            # Get news articles with better error handling
            try:
                articles = newsapi.get_everything(
                    q=symbol,
                    from_param=start_date,
                    to=end_date,
                    language='en',
                    sort_by='publishedAt'
                )
            except Exception as e:
                logger.error(f"Error fetching articles from News API: {str(e)}")
                return None

            if not articles.get('articles'):
                logger.warning("No articles found from News API")
                return None

            sentiment_data = []
            for article in articles['articles']:
                try:
                    # Ensure we have required fields
                    title = article.get('title', '').strip()
                    description = article.get('description', '').strip()
                    published_at = article.get('publishedAt')
                    
                    if not title or not published_at:  # Skip if missing essential data
                        continue
                    
                    # Combine title and description
                    text = f"{title} {description}"
                    
                    # VADER sentiment with error handling
                    try:
                        vader_scores = self.sia.polarity_scores(text)
                    except Exception as e:
                        logger.warning(f"Error calculating VADER sentiment: {str(e)}")
                        continue
                    
                    # TextBlob sentiment with error handling
                    try:
                        blob_sentiment = TextBlob(text).sentiment
                    except Exception as e:
                        logger.warning(f"Error calculating TextBlob sentiment: {str(e)}")
                        continue
                    
                    # Combine scores
                    sentiment_score = (vader_scores['compound'] + blob_sentiment.polarity) / 2
                    
                    sentiment_data.append({
                        'ds': pd.to_datetime(published_at),
                        'sentiment_score': (sentiment_score + 1) / 2,
                        'source': 'newsapi',
                        'confidence': abs(blob_sentiment.subjectivity),
                        'title': title  # Store title for reference
                    })
                except Exception as e:
                    logger.warning(f"Error processing NewsAPI article: {str(e)}")
                    continue

            if sentiment_data:
                df = pd.DataFrame(sentiment_data)
                logger.info(f"Successfully processed {len(df)} articles from News API")
                return df
            
            return None

        except Exception as e:
            logger.error(f"Error fetching NewsAPI sentiment: {str(e)}")
            return None
            
def fetch_finnhub_sentiment(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch sentiment data from Finnhub"""
        try:
            logger.info(f"Fetching Finnhub sentiment for {symbol} from {start_date} to {end_date}")
            base_url = "https://finnhub.io/api/v1"
            headers = {'X-Finnhub-Token': self.finnhub_key}
            
            # Convert dates to timestamps
            start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
            end_timestamp = int(datetime.strptime(end