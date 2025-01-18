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
    def __init__(self):
        self.base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
        self.theme_filters = [
            'ECON_BANKRUPTCY', 'ECON_COST', 'ECON_DEBT', 'ECON_REFORM',
            'BUS_MARKET_CLOSE', 'BUS_MARKET_CRASH', 'BUS_MARKET_DOWN',
            'BUS_MARKET_UP', 'BUS_STOCK_MARKET', 'POLITICS'
        ]

    def fetch_sentiment_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        try:
            theme_query = ' OR '.join(f'theme:{theme}' for theme in self.theme_filters)
            query = f"({theme_query}) AND sourcelang:eng"
            
            date_format = "%Y%m%d%H%M%S"
            start = datetime.strptime(start_date, "%Y-%m-%d").strftime(date_format)
            end = datetime.strptime(end_date, "%Y-%m-%d").strftime(date_format)
            
            url = f"{self.base_url}?query={query}&START={start}&END={end}&format=csv&maxrecords=250000"
            
            response = requests.get(url)
            if response.status_code != 200:
                logger.error(f"GDELT API request failed with status {response.status_code}")
                return None
            
            df = pd.read_csv(pd.compat.StringIO(response.text), sep='\t')
            return self._process_sentiment_data(df)
        except Exception as e:
            logger.error(f"Error fetching GDELT data: {str(e)}")
            return None

    def _process_sentiment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df['DATE'] = pd.to_datetime(df['DATE'], format='%Y%m%d%H%M%S')
            
            daily_sentiment = df.groupby(df['DATE'].dt.date).agg({
                'Tone': ['mean', 'count', 'std'],
                'PositiveTone': 'mean',
                'NegativeTone': 'mean',
                'Polarity': 'mean'
            }).reset_index()
            
            daily_sentiment.columns = ['ds', 'tone_avg', 'article_count', 'tone_std', 
                                     'positive_tone', 'negative_tone', 'polarity']
            
            daily_sentiment['ds'] = pd.to_datetime(daily_sentiment['ds'])
            daily_sentiment['sentiment_score'] = ((daily_sentiment['tone_avg'] + 100) / 200 * 
                                                daily_sentiment['article_count'] / daily_sentiment['article_count'].max())
            
            return daily_sentiment.sort_values('ds')
        except Exception as e:
            logger.error(f"Error processing sentiment data: {str(e)}")
            return pd.DataFrame()

    def prepare_combined_forecast_data(self, price_data: pd.DataFrame, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        try:
            price_data.index = pd.to_datetime(price_data.index)
            sentiment_data['ds'] = pd.to_datetime(sentiment_data['ds'])
            
            df = pd.DataFrame({
                'ds': price_data.index,
                'y': price_data['Close']
            })
            
            df = df.merge(sentiment_data[['ds', 'sentiment_score']], on='ds', how='left')
            df['sentiment_score'] = df['sentiment_score'].fillna(method='ffill')
            
            return df
        except Exception as e:
            logger.error(f"Error preparing forecast data: {str(e)}")
            return pd.DataFrame()

    def enhanced_prophet_forecast(self, combined_data: pd.DataFrame, periods: int = 30) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        try:
            if combined_data is None or combined_data.empty:
                return None, "No data provided for forecasting"

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

    def analyze_sentiment_impact(self, forecast: pd.DataFrame) -> Dict[str, float]:
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

def integrate_sentiment_analysis(app_instance) -> Optional[pd.DataFrame]:
    try:
        st.sidebar.header("🎭 Sentiment Analysis")
        show_sentiment = st.sidebar.checkbox("Include Sentiment Analysis", value=True)
        
        if not show_sentiment:
            return None
            
        sentiment_period = st.sidebar.slider(
            "Sentiment Analysis Period (days)",
            min_value=7,
            max_value=90,
            value=30,
            help="Select the period for sentiment analysis"
        )
        
        analyzer = GDELTAnalyzer()
        sentiment_data = analyzer.fetch_sentiment_data(
            (datetime.now() - timedelta(days=sentiment_period)).strftime("%Y-%m-%d"),
            datetime.now().strftime("%Y-%m-%d")
        )
        
        if sentiment_data is None or sentiment_data.empty:
            st.warning("No sentiment data available")
            return None
            
        st.markdown("### 📊 Market Sentiment Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_sentiment = sentiment_data['sentiment_score'].iloc[-1]
            sentiment_change = sentiment_data['sentiment_score'].iloc[-1] - sentiment_data['sentiment_score'].iloc[-2]
            st.metric("Current Sentiment", f"{current_sentiment:.2f}", f"{sentiment_change:+.2f}")
        
        with col2:
            st.metric("Average Sentiment", f"{sentiment_data['sentiment_score'].mean():.2f}")
        
        with col3:
            st.metric("Sentiment Volatility", f"{sentiment_data['sentiment_score'].std():.2f}")
        
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
        return None

def update_forecasting_process(price_data: pd.DataFrame, 
                             sentiment_data: Optional[pd.DataFrame] = None) -> Tuple[Optional[pd.DataFrame], Dict]:
    try:
        analyzer = GDELTAnalyzer()
        
        if sentiment_data is not None and not sentiment_data.empty:
            combined_data = analyzer.prepare_combined_forecast_data(price_data, sentiment_data)
            forecast, error = analyzer.enhanced_prophet_forecast(combined_data)
            
            if error:
                st.error(f"Forecasting error: {error}")
                return None, {}
                
            impact_metrics = analyzer.analyze_sentiment_impact(forecast)
            return forecast, impact_metrics
        else:
            from forecasting import prophet_forecast
            forecast, error = prophet_forecast(price_data, periods=30)
            return forecast, {}
    except Exception as e:
        logger.error(f"Error in forecasting process: {str(e)}")
        return None, {}