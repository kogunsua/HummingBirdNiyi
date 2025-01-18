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

logger = logging.getLogger(__name__)

class GDELTAnalyzer:
    """
    Enhanced GDELT analyzer that combines market and sentiment analysis
    """
    def __init__(self):
        self.base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
        self.theme_filters = [
            'ECON_BANKRUPTCY', 'ECON_COST', 'ECON_DEBT', 'ECON_REFORM',
            'BUS_MARKET_CLOSE', 'BUS_MARKET_CRASH', 'BUS_MARKET_DOWN',
            'BUS_MARKET_UP', 'BUS_STOCK_MARKET', 'POLITICS'
        ]

    def fetch_sentiment_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch GDELT sentiment data"""
        try:
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
        
    def prepare_combined_forecast_data(self, price_data: pd.DataFrame, 
                                     sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare combined dataset for Prophet forecasting"""
        try:
            # Ensure datetime index
            price_data.index = pd.to_datetime(price_data.index)
            sentiment_data['ds'] = pd.to_datetime(sentiment_data['ds'])
            
            # Create Prophet dataframe
            df = pd.DataFrame({
                'ds': price_data.index,
                'y': price_data['Close']
            })
            
            # Merge sentiment data
            df = df.merge(sentiment_data[['ds', 'sentiment_score']], 
                         on='ds', 
                         how='left')
            
            # Fill missing sentiment values
            df['sentiment_score'] = df['sentiment_score'].fillna(method='ffill')
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing combined forecast data: {str(e)}")
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

    def enhanced_prophet_forecast(self, combined_data: pd.DataFrame, 
                                periods: int = 30) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Generate enhanced forecast using both price and sentiment data"""
        try:
            if combined_data is None or combined_data.empty:
                return None, "No data provided for forecasting"

            # Initialize Prophet with custom parameters
            model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                holidays_prior_scale=10.0,
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True
            )
            
            # Add sentiment regressor
            model.add_regressor('sentiment_score')
            
            # Fit model
            model.fit(combined_data)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=periods)
            
            # Add sentiment predictions for future dates
            last_sentiment = combined_data['sentiment_score'].iloc[-1]
            future['sentiment_score'] = last_sentiment
            
            # Make forecast
            forecast = model.predict(future)
            
            # Add actual values
            forecast['actual'] = np.nan
            forecast.loc[forecast['ds'].isin(combined_data['ds']), 'actual'] = combined_data['y'].values
            
            return forecast, None
            
        except Exception as e:
            logger.error(f"Error in enhanced prophet forecast: {str(e)}")
            return None, str(e)

    def analyze_sentiment_impact(self, forecast: pd.DataFrame) -> Dict[str, float]:
        """Analyze the impact of sentiment on price predictions"""
        try:
            sentiment_effect = forecast['sentiment_score'].corr(forecast['yhat'])
            sentiment_volatility = forecast['sentiment_score'].std()
            
            impact_metrics = {
                'sentiment_correlation': sentiment_effect,
                'sentiment_volatility': sentiment_volatility,
                'price_sensitivity': abs(sentiment_effect * sentiment_volatility)
            }
            
            return impact_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment impact: {str(e)}")
            return {}

def integrate_sentiment_analysis(app_instance) -> Optional[pd.DataFrame]:
    """Integrate sentiment analysis into the main application"""
    try:
        # Add sentiment analysis section to sidebar
        st.sidebar.header("🎭 Sentiment Analysis")
        show_sentiment = st.sidebar.checkbox("Include Sentiment Analysis", value=True)
        
        if show_sentiment:
            sentiment_period = st.sidebar.slider(
                "Sentiment Analysis Period (days)",
                min_value=7,
                max_value=90,
                value=30,
                help="Select the period for sentiment analysis"
            )
            
            # Initialize GDELT analyzer
            analyzer = GDELTAnalyzer()
            
            # Get sentiment data
            sentiment_data = analyzer.fetch_sentiment_data(
                (datetime.now() - timedelta(days=sentiment_period)).strftime("%Y-%m-%d"),
                datetime.now().strftime("%Y-%m-%d")
            )
            
            if sentiment_data is not None:
                st.markdown("### 📊 Market Sentiment Analysis")
                
                # Display sentiment metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    current_sentiment = sentiment_data['sentiment_score'].iloc[-1]
                    sentiment_change = (
                        sentiment_data['sentiment_score'].iloc[-1] - 
                        sentiment_data['sentiment_score'].iloc[-2]
                    )
                    st.metric(
                        "Current Sentiment",
                        f"{current_sentiment:.2f}",
                        f"{sentiment_change:+.2f}"
                    )
                
                with col2:
                    avg_sentiment = sentiment_data['sentiment_score'].mean()
                    st.metric(
                        "Average Sentiment",
                        f"{avg_sentiment:.2f}"
                    )
                
                with col3:
                    sentiment_volatility = sentiment_data['sentiment_score'].std()
                    st.metric(
                        "Sentiment Volatility",
                        f"{sentiment_volatility:.2f}"
                    )
                
                # Create sentiment trend visualization
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
                
        return None

    except Exception as e:
        logger.error(f"Error in sentiment analysis integration: {str(e)}")
        return None

def update_forecasting_process(price_data: pd.DataFrame, 
                             sentiment_data: Optional[pd.DataFrame] = None) -> Tuple[Optional[pd.DataFrame], Dict]:
    """Updated forecasting process incorporating sentiment analysis"""
    try:
        analyzer = GDELTAnalyzer()
        
        if sentiment_data is not None:
            # Prepare combined dataset
            combined_data = analyzer.prepare_combined_forecast_data(price_data, sentiment_data)
            
            # Generate enhanced forecast
            forecast, error = analyzer.enhanced_prophet_forecast(combined_data)
            
            if error:
                st.error(f"Forecasting error: {error}")
                return None, {}
                
            # Analyze sentiment impact
            impact_metrics = analyzer.analyze_sentiment_impact(forecast)
            
            return forecast, impact_metrics
            
        else:
            # Fall back to regular forecasting if no sentiment data
            from forecasting import prophet_forecast
            forecast, error = prophet_forecast(price_data, periods=30)
            return forecast, {}

    except Exception as e:
        logger.error(f"Error in forecasting process: {str(e)}")
        return None, {}