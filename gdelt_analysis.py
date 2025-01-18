# gdelt_analysis.py
import pandas as pd
import numpy as np
from prophet import Prophet
import requests
from datetime import datetime, timedelta
import logging
from typing import Optional, Tuple, Dict
import streamlit as st

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
        
    def prepare_combined_forecast_data(self, 
                                     price_data: pd.DataFrame, 
                                     sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare combined dataset for Prophet forecasting
        """
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

    def enhanced_prophet_forecast(self, 
                                combined_data: pd.DataFrame, 
                                periods: int = 30) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Generate enhanced forecast using both price and sentiment data
        """
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
        """
        Analyze the impact of sentiment on price predictions
        """
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

def integrate_sentiment_analysis(app_instance):
    """
    Integrate sentiment analysis into the main application
    """
    
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
            
            # Return sentiment data for use in forecasting
            return sentiment_data
            
    return None

def update_forecasting_process(price_data: pd.DataFrame, 
                             sentiment_data: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Updated forecasting process incorporating sentiment analysis
    """
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
        return prophet_forecast(price_data, periods=30)

# Update the main application to use the new functionality
def update_main_app():
    """
    Main application with integrated sentiment analysis
    """
    if st.button("🚀 Generate Forecast"):
        try:
            with st.spinner('Loading data...'):
                # Get price data
                fetcher = AssetDataFetcher()
                price_data = fetcher.get_stock_data(symbol) if asset_type == "Stocks" else fetcher.get_crypto_data(symbol)
                
                # Get sentiment data
                sentiment_data = integrate_sentiment_analysis()
                
                if price_data is not None:
                    with st.spinner('Generating forecast...'):
                        # Generate forecast with sentiment analysis
                        forecast, impact_metrics = update_forecasting_process(price_data, sentiment_data)
                        
                        if forecast is not None:
                            # Add technical indicators
                            price_data = add_technical_indicators(price_data, asset_type)
                            
                            # Display metrics
                            display_metrics(price_data, forecast, asset_type, symbol)
                            
                            # Display sentiment impact if available
                            if impact_metrics:
                                st.subheader("🎭 Sentiment Impact Analysis")
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric(
                                        "Sentiment Correlation",
                                        f"{impact_metrics['sentiment_correlation']:.2f}"
                                    )
                                
                                with col2:
                                    st.metric(
                                        "Sentiment Volatility",
                                        f"{impact_metrics['sentiment_volatility']:.2f}"
                                    )
                                
                                with col3:
                                    st.metric(
                                        "Price Sensitivity",
                                        f"{impact_metrics['price_sensitivity']:.2f}"
                                    )
                            
                            # Create and display forecast plot
                            fig = create_forecast_plot(price_data, forecast, "Enhanced Prophet", symbol)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            with st.expander("View Detailed Forecast Data"):
                                st.dataframe(forecast)
                else:
                    st.error(f"Could not load data for {symbol}. Please verify the symbol.")

        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            st.exception(e)