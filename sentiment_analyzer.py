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
            "Multi-Source": "combined"
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
            return None
            
        return sentiment_data
        
    except Exception as e:
        logger.error(f"Error getting sentiment data: {str(e)}")
        st.error(f"Error getting sentiment data: {str(e)}")
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
        self.newsapi_key = "pub_65773c625d48ffecc8522ad52fe0fd7199cce"  # Get from https://newsapi.org/
        self.finnhub_key = "cpllsnpr01qn8g1v08hgcpllsnpr01qn8g1v08i0"   # Get from https://finnhub.io/

def fetch_combined_sentiment(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch sentiment data from multiple sources"""
        try:
            logger.info(f"Fetching combined sentiment for {symbol} from {start_date} to {end_date}")
            sentiment_data = []
            source_count = 0
            
            # 1. Try Yahoo Finance News Sentiment
            try:
                yahoo_sentiment = self.fetch_yahoo_sentiment(symbol, start_date, end_date)
                if yahoo_sentiment is not None and not yahoo_sentiment.empty:
                    sentiment_data.append(yahoo_sentiment)
                    source_count += 1
                    logger.info(f"Successfully fetched Yahoo Finance sentiment data: {len(yahoo_sentiment)} records")
                else:
                    logger.warning("No Yahoo Finance sentiment data available")
            except Exception as e:
                logger.warning(f"Error fetching Yahoo Finance sentiment: {str(e)}")

            # 2. Try News API
            try:
                news_sentiment = self.fetch_newsapi_sentiment(symbol, start_date, end_date)
                if news_sentiment is not None and not news_sentiment.empty:
                    sentiment_data.append(news_sentiment)
                    source_count += 1
                    logger.info(f"Successfully fetched News API sentiment data: {len(news_sentiment)} records")
                else:
                    logger.warning("No News API sentiment data available")
            except Exception as e:
                logger.warning(f"Error fetching News API sentiment: {str(e)}")

            # 3. Try Finnhub Sentiment
            try:
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

    def _process_combined_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and combine sentiment data from multiple sources"""
        try:
            logger.info("Processing combined sentiment data")
            
            # Ensure datetime format for ds column
            df['ds'] = pd.to_datetime(df['ds'])
            
            # Group by date and calculate weighted average sentiment
            daily_sentiment = df.groupby('ds').agg({
                'sentiment_score': lambda x: np.average(x, weights=df.loc[x.index, 'confidence']),
                'confidence': 'mean'
            }).reset_index()

            # Calculate additional metrics
            daily_sentiment['tone_avg'] = (daily_sentiment['sentiment_score'] - 0.5) * 200
            daily_sentiment['tone_std'] = df.groupby(df['ds'].dt.date)['sentiment_score'].transform('std')
            daily_sentiment['article_count'] = df.groupby(df['ds'].dt.date)['sentiment_score'].transform('count')

            # Sort by date
            result = daily_sentiment.sort_values('ds')
            
            logger.info(f"Successfully processed sentiment data: {len(result)} records")
            return result

        except Exception as e:
            logger.error(f"Error processing combined sentiment: {str(e)}")
            return pd.DataFrame()

    def fetch_yahoo_sentiment(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch and analyze sentiment from Yahoo Finance news"""
        try:
            logger.info(f"Fetching Yahoo Finance sentiment for {symbol} from {start_date} to {end_date}")
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
            logger.info(f"Fetching News API sentiment for {symbol} from {start_date} to {end_date}")
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
            logger.info(f"Fetching Finnhub sentiment for {symbol} from {start_date} to {end_date}")
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

def integrate_multi_source_sentiment(symbol: str, sentiment_period: int, sentiment_weight: float = 0.5) -> Optional[pd.DataFrame]:
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
        
        # Calculate impact metrics
        impact_metrics = {}
        try:
            # Calculate correlation between sentiment and price changes
            impact_metrics['sentiment_correlation'] = sentiment_data['sentiment_score'].autocorr()
            impact_metrics['sentiment_volatility'] = sentiment_data['sentiment_score'].std()
            impact_metrics['price_sensitivity'] = sentiment_weight
        except Exception as e:
            logger.warning(f"Error calculating impact metrics: {str(e)}")
        
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
        
        # Display sentiment impact results if metrics are available
        if impact_metrics:
            display_sentiment_impact_results(impact_metrics)
        
        return sentiment_data

    except Exception as e:
        logger.error(f"Error in multi-source sentiment integration: {str(e)}")
        st.error(f"Error in sentiment analysis: {str(e)}")
        return None