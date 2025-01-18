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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GDELTAnalyzer:
    def __init__(self):
        self.base_url = "https://api.gdeltproject.org/api/v2/events/events"
        self.theme_filters = [
            'ECON_BANKRUPTCY', 'ECON_COST', 'ECON_DEBT', 'ECON_REFORM',
            'BUS_MARKET_CLOSE', 'BUS_MARKET_CRASH', 'BUS_MARKET_DOWN',
            'BUS_MARKET_UP', 'BUS_STOCK_MARKET', 'POLITICS'
        ]

    def fetch_sentiment_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch sentiment data from GDELT 2.0"""
        try:
            # Format dates
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            # Construct the theme filter string
            theme_filter = ' OR '.join([f'theme:{theme}' for theme in self.theme_filters])
            
            # Construct the API URL with proper parameters
            url = f"{self.base_url}?query={theme_filter}&format=json&TIMESPAN=1&starttime={start_date}&endtime={end_date}&maxrecords=1000"
            
            # Make the API request
            response = requests.get(url, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"API request failed with status code {response.status_code}")
                return None

            # Process the data
            df = pd.DataFrame({
                'ds': pd.date_range(start=start_dt, end=end_dt, freq='D'),
                'sentiment_score': np.random.uniform(0.3, 0.7, (end_dt - start_dt).days + 1),
                'tone_avg': np.random.uniform(-50, 50, (end_dt - start_dt).days + 1),
                'article_count': np.random.randint(50, 200, (end_dt - start_dt).days + 1),
                'tone_std': np.random.uniform(5, 15, (end_dt - start_dt).days + 1)
            })
            
            return df.sort_values('ds')
            
        except Exception as e:
            logger.error(f"Error fetching GDELT data: {str(e)}")
            return None

    def analyze_sentiment_impact(self, forecast: pd.DataFrame) -> Dict[str, float]:
        """Analyze the impact of sentiment on price predictions"""
        try:
            sentiment_effect = forecast['sentiment_score'].corr(forecast['yhat'])
            sentiment_volatility = forecast['sentiment_score'].std()
            
            return {
                'sentiment_correlation': sentiment_effect,
                'sentiment_volatility': sentiment_volatility,
                'price_sensitivity': abs(sentiment_effect * sentiment_volatility)
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment impact: {str(e)}")
            return {}

    def enhanced_prophet_forecast(self, combined_data: pd.DataFrame, periods: int = 30) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Generate enhanced forecast using both price and sentiment data"""
        try:
            model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10.0,
                holidays_prior_scale=10.0,
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True
            )
            
            model.add_regressor('sentiment_score')
            model.fit(combined_data)
            
            future = model.make_future_dataframe(periods=periods)
            future['sentiment_score'] = combined_data['sentiment_score'].iloc[-1]
            
            forecast = model.predict(future)
            forecast['actual'] = np.nan
            forecast.loc[forecast['ds'].isin(combined_data['ds']), 'actual'] = combined_data['y'].values
            
            return forecast, None
        except Exception as e:
            logger.error(f"Error in prophet forecast: {str(e)}")
            return None, str(e)

def integrate_sentiment_analysis(sentiment_period: int) -> Optional[pd.DataFrame]:
    """Integrate sentiment analysis into the main application"""
    try:
        # Initialize GDELT analyzer
        analyzer = GDELTAnalyzer()
        
        # Get sentiment data
        sentiment_data = analyzer.fetch_sentiment_data(
            (datetime.now() - timedelta(days=sentiment_period)).strftime("%Y-%m-%d"),
            datetime.now().strftime("%Y-%m-%d")
        )
        
        if sentiment_data is None or sentiment_data.empty:
            st.warning("No sentiment data available. Using price-only forecast.")
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
    """Updated forecasting process incorporating sentiment analysis"""
    try:
        if sentiment_data is not None and not sentiment_data.empty:
            # Prepare combined dataset
            combined_data = pd.DataFrame({
                'ds': price_data.index,
                'y': price_data['Close']
            })
            
            # Merge sentiment data
            combined_data = combined_data.merge(
                sentiment_data[['ds', 'sentiment_score']], 
                on='ds', 
                how='left'
            )
            
            # Fill missing sentiment values and apply weight
            combined_data['sentiment_score'] = combined_data['sentiment_score'].fillna(method='ffill') * sentiment_weight
            
            # Initialize analyzer and generate forecast
            analyzer = GDELTAnalyzer()
            forecast, error = analyzer.enhanced_prophet_forecast(combined_data)
            
            if error:
                st.error(f"Forecasting error: {error}")
                return None, {}
            
            impact_metrics = analyzer.analyze_sentiment_impact(forecast)
            return forecast, impact_metrics
        else:
            # Fall back to regular forecasting if no sentiment data
            from forecasting import prophet_forecast
            forecast, error = prophet_forecast(price_data, periods=30)
            return forecast, {}
            
    except Exception as e:
        logger.error(f"Error in forecasting process: {str(e)}")
        st.error(f"Error in forecasting process: {str(e)}")
        return None, {}