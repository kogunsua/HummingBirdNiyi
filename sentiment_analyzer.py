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
            end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
            
            # Fetch sentiment data with better error handling
            try:
                url = f"{base_url}/news-sentiment?symbol={symbol}"
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 429:
                    logger.warning("Rate limit exceeded for Finnhub API")
                    return None
                elif response.status_code != 200:
                    logger.warning(f"Failed to fetch data from Finnhub. Status code: {response.status_code}")
                    return None
                    
                data = response.json()
                
            except requests.exceptions.Timeout:
                logger.error("Timeout while fetching data from Finnhub")
                return None
            except requests.exceptions.RequestException as e:
                logger.error(f"Error making request to Finnhub: {str(e)}")
                return None
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding Finnhub response: {str(e)}")
                return None

            if not data or not data.get('data'):
                logger.warning("No data received from Finnhub")
                return None

            # Create DataFrame from sentiment data
            sentiment_data = []
            for news in data.get('data', []):
                try:
                    if start_timestamp <= news['datetime'] <= end_timestamp:
                        sentiment_data.append({
                            'ds': pd.to_datetime(news['datetime'], unit='s'),
                            'sentiment_score': news['sentiment'],
                            'source': 'finnhub',
                            'confidence': news.get('confidence', 0.5),
                            'title': news.get('headline', '')
                        })
                except Exception as e:
                    logger.warning(f"Error processing Finnhub news item: {str(e)}")
                    continue

            if sentiment_data:
                df = pd.DataFrame(sentiment_data)
                logger.info(f"Successfully processed {len(df)} articles from Finnhub")
                return df
            
            logger.warning("No valid sentiment data found in the specified date range")
            return None

        except Exception as e:
            logger.error(f"Error fetching Finnhub sentiment: {str(e)}")
            return None

def _process_combined_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and combine sentiment data from multiple sources"""
        try:
            logger.info("Processing combined sentiment data")
            
            # Ensure datetime format for ds column
            df['ds'] = pd.to_datetime(df['ds'])
            
            # Remove any future dates
            df = df[df['ds'] <= datetime.now()]
            
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

def integrate_multi_source_sentiment(symbol: str, sentiment_period: int, sentiment_weight: float = 0.5) -> Optional[pd.DataFrame]:
    """Integrate multi-source sentiment analysis"""
    try:
        logger.info(f"Starting sentiment analysis for {symbol} over {sentiment_period} days")
        analyzer = MultiSourceSentimentAnalyzer()
        
        start_date = (datetime.now() - timedelta(days=sentiment_period)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        with st.spinner('Fetching sentiment data from multiple sources...'):
            sentiment_data = analyzer.fetch_combined_sentiment(symbol, start_date, end_date)
        
        if sentiment_data is None or sentiment_data.empty:
            logger.warning(f"No sentiment data available for {symbol}")
            st.warning("No sentiment data available from any source. Using price-only forecast.")
            return None
        
        logger.info(f"Successfully fetched sentiment data for {symbol}")
        st.success("Successfully fetched sentiment data!")
        
        # Calculate impact metrics
        impact_metrics = {}
        try:
            # Calculate correlation between sentiment and price changes
            impact_metrics['sentiment_correlation'] = sentiment_data['sentiment_score'].autocorr()
            impact_metrics['sentiment_volatility'] = sentiment_data['sentiment_score'].std()
            impact_metrics['price_sensitivity'] = sentiment_weight
            
            logger.info(f"Calculated sentiment impact metrics: correlation={impact_metrics['sentiment_correlation']:.2f}, "
                       f"volatility={impact_metrics['sentiment_volatility']:.2f}, "
                       f"sensitivity={impact_metrics['price_sensitivity']:.2f}")
            
        except Exception as e:
            logger.warning(f"Error calculating impact metrics: {str(e)}")
            impact_metrics = {
                'sentiment_correlation': 0.0,
                'sentiment_volatility': 0.0,
                'price_sensitivity': sentiment_weight
            }
        
        # Display sentiment analysis
        st.markdown("### 📊 Multi-Source Market Sentiment Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_sentiment = sentiment_data['sentiment_score'].iloc[-1]
            sentiment_change = sentiment_data['sentiment_score'].iloc[-1] - sentiment_data['sentiment_score'].iloc[-2]
            st.metric(
                "Current Sentiment",
                f"{current_sentiment:.2f}",
                f"{sentiment_change:+.2f}",
                help="Latest sentiment score and change from previous period"
            )
        
        with col2:
            avg_sentiment = sentiment_data['sentiment_score'].mean()
            st.metric(
                "Average Sentiment",
                f"{avg_sentiment:.2f}",
                help="Mean sentiment score across the analysis period"
            )
        
        with col3:
            conf_level = sentiment_data['confidence'].mean()
            st.metric(
                "Confidence Level",
                f"{conf_level:.2f}",
                help="Average confidence in sentiment analysis"
            )
        
        # Create sentiment visualization
        fig = go.Figure()
        
        # Add main sentiment line
        fig.add_trace(
            go.Scatter(
                x=sentiment_data['ds'],
                y=sentiment_data['sentiment_score'],
                name='Sentiment Score',
                line=dict(color='purple', width=2)
            )
        )
        
        # Add confidence bands if available
        if 'tone_std' in sentiment_data.columns:
            upper_band = sentiment_data['sentiment_score'] + sentiment_data['tone_std']
            lower_band = sentiment_data['sentiment_score'] - sentiment_data['tone_std']
            
            fig.add_trace(
                go.Scatter(
                    x=sentiment_data['ds'].tolist() + sentiment_data['ds'].tolist()[::-1],
                    y=upper_band.tolist() + lower_band.tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(128, 0, 128, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Band',
                    showlegend=True
                )
            )
        
        fig.update_layout(
            title={
                'text': "Market Sentiment Trend (Multi-Source)",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
            height=400,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.8)'
            ),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Add range slider
        fig.update_xaxes(rangeslider_visible=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display sentiment impact results if metrics are available
        if impact_metrics:
            display_sentiment_impact_results(impact_metrics)
            
            # Add metric details explanation
            with st.expander("📈 Detailed Metrics Information"):
                st.markdown(f"""
                    ### Sentiment Analysis Details
                    
                    #### Coverage Statistics:
                    - Total Days Analyzed: {len(sentiment_data)}
                    - Average Daily Articles: {sentiment_data['article_count'].mean():.1f}
                    - Data Completeness: {(len(sentiment_data) / sentiment_period * 100):.1f}%
                    
                    #### Sentiment Trends:
                    - Highest Sentiment: {sentiment_data['sentiment_score'].max():.3f}
                    - Lowest Sentiment: {sentiment_data['sentiment_score'].min():.3f}
                    - Sentiment Volatility: {sentiment_data['sentiment_score'].std():.3f}
                    
                    #### Reliability Metrics:
                    - Average Confidence: {sentiment_data['confidence'].mean():.3f}
                    - Data Sources Used: Multiple
                    - Update Frequency: Daily
                """)
        
        logger.info(f"Completed sentiment analysis for {symbol}")
        return sentiment_data

    except Exception as e:
        logger.error(f"Error in multi-source sentiment integration: {str(e)}")
        st.error(f"Error in sentiment analysis: {str(e)}")
        return None

# Define exports
__all__ = [
    'MultiSourceSentimentAnalyzer',
    'integrate_multi_source_sentiment',
    'display_sentiment_impact_analysis',
    'display_sentiment_impact_results',
    'get_sentiment_data'
]