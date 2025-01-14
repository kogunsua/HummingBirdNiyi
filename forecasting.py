import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from typing import Tuple, Optional, Dict
import logging
from asset_config import AssetConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_data_for_prophet(data: pd.DataFrame) -> pd.DataFrame:
    """[Your existing prepare_data_for_prophet function]"""
    # ... [Keep your existing implementation]

def add_crypto_specific_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """[Your existing add_crypto_specific_indicators function]"""
    # ... [Keep your existing implementation]

def add_technical_indicators(df: pd.DataFrame, asset_type: str = 'stocks') -> pd.DataFrame:
    """[Your existing add_technical_indicators function]"""
    # ... [Keep your existing implementation]

def prophet_forecast(data: pd.DataFrame, periods: int, economic_data: Optional[pd.DataFrame] = None,
                     indicator: Optional[str] = None, asset_type: str = 'stocks') -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """[Your existing prophet_forecast function]"""
    # ... [Keep your existing implementation]

def create_forecast_plot(data: pd.DataFrame, forecast: pd.DataFrame, model_name: str, symbol: str) -> go.Figure:
    """[Your existing create_forecast_plot function]"""
    # ... [Keep your existing implementation]

def display_common_metrics(data: pd.DataFrame, forecast: pd.DataFrame):
    """Display common metrics for both stocks and cryptocurrencies"""
    try:
        st.subheader("📈 Price Metrics")
        
        # Ensure we have the required data
        if 'Close' not in data.columns:
            raise ValueError("Close price data not found in dataset")
            
        # Handle data access safely
        try:
            current_price = float(data['Close'].iloc[-1] if isinstance(data['Close'], pd.Series) else data['Close'][-1])
            price_change_24h = float(data['Close'].pct_change().iloc[-1] * 100)
            price_change_7d = float(data['Close'].pct_change(periods=7).iloc[-1] * 100)
            
            # Price Metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Current Price",
                    f"${current_price:,.2f}",
                    f"{price_change_24h:+.2f}%"
                )
            
            with col2:
                st.metric(
                    "7-Day Change",
                    f"{price_change_7d:+.2f}%"
                )
            
            with col3:
                volatility_30d = float(data['Close'].pct_change().rolling(window=30).std() * np.sqrt(252) * 100)
                st.metric(
                    "30-Day Volatility",
                    f"{volatility_30d:.2f}%"
                )
            
            # Technical Indicators
            st.subheader("📊 Technical Indicators")
            tech_col1, tech_col2, tech_col3 = st.columns(3)
            
            with tech_col1:
                if 'RSI' in data.columns:
                    current_rsi = float(data['RSI'].iloc[-1])
                    rsi_change = float(data['RSI'].diff().iloc[-1])
                    st.metric(
                        "RSI (14)",
                        f"{current_rsi:.2f}",
                        f"{rsi_change:+.2f}"
                    )
            
            with tech_col2:
                if 'MACD' in data.columns and 'Signal_Line' in data.columns:
                    current_macd = float(data['MACD'].iloc[-1])
                    macd_signal = float(data['Signal_Line'].iloc[-1])
                    st.metric(
                        "MACD",
                        f"{current_macd:.2f}",
                        f"Signal: {macd_signal:.2f}"
                    )
            
            with tech_col3:
                if 'MA20' in data.columns:
                    ma20 = float(data['MA20'].iloc[-1])
                    ma20_diff = float(current_price - ma20)
                    st.metric(
                        "20-Day MA",
                        f"${ma20:.2f}",
                        f"{ma20_diff:+.2f} from price"
                    )
            
            # Forecast Metrics
            st.subheader("🔮 Forecast Metrics")
            forecast_col1, forecast_col2, forecast_col3 = st.columns(3)
            
            with forecast_col1:
                final_forecast = float(forecast['yhat'].iloc[-1])
                forecast_change = ((final_forecast / current_price) - 1) * 100
                st.metric(
                    "Forecast End Price",
                    f"${final_forecast:.2f}",
                    f"{forecast_change:+.2f}%"
                )
            
            with forecast_col2:
                confidence_width = float((forecast['yhat_upper'].iloc[-1] - forecast['yhat_lower'].iloc[-1]) / forecast['yhat'].iloc[-1] * 100)
                st.metric(
                    "Forecast Confidence",
                    f"{100 - confidence_width:.1f}%"
                )
            
            with forecast_col3:
                trend_strength = float(abs(forecast_change) / confidence_width * 100)
                st.metric(
                    "Trend Strength",
                    f"{trend_strength:.1f}%"
                )

        except Exception as e:
            logger.error(f"Error processing metrics: {str(e)}")
            st.error("Error processing metrics. Please check your data format.")

    except Exception as e:
        logger.error(f"Error displaying common metrics: {str(e)}")
        st.error(f"Error displaying common metrics: {str(e)}")

def display_confidence_analysis(forecast: pd.DataFrame):
    """Display detailed confidence analysis of the forecast"""
    try:
        st.subheader("📊 Confidence Analysis")

        # Calculate confidence metrics
        confidence_width = (forecast['yhat_upper'] - forecast['yhat_lower']) / forecast['yhat'] * 100
        avg_confidence = 100 - confidence_width.mean()
        
        # Calculate trend metrics
        total_trend = ((forecast['yhat'].iloc[-1] / forecast['yhat'].iloc[0]) - 1) * 100
        trend_consistency = np.sum(np.diff(forecast['yhat']) > 0) / (len(forecast) - 1) * 100

        # Display metrics in columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Average Confidence",
                f"{avg_confidence:.1f}%",
                "Higher is better"
            )

        with col2:
            st.metric(
                "Overall Trend",
                f"{total_trend:+.1f}%",
                f"{'Upward' if total_trend > 0 else 'Downward'} Trend"
            )

        with col3:
            st.metric(
                "Trend Consistency",
                f"{trend_consistency:.1f}%",
                "% of positive daily changes"
            )

        # Display additional analysis
        with st.expander("View Detailed Confidence Analysis"):
            # Calculate confidence bands over time
            confidence_df = pd.DataFrame({
                'Date': forecast['ds'],
                'Confidence Width (%)': confidence_width,
                'Upper Band': forecast['yhat_upper'],
                'Lower Band': forecast['yhat_lower'],
                'Forecast': forecast['yhat']
            })

            # Show confidence statistics
            st.write("**Confidence Statistics:**")
            stats_col1, stats_col2 = st.columns(2)
            
            with stats_col1:
                st.write("Confidence Width Statistics:")
                st.write(f"- Minimum: {confidence_width.min():.1f}%")
                st.write(f"- Maximum: {confidence_width.max():.1f}%")
                st.write(f"- Average: {confidence_width.mean():.1f}%")
            
            with stats_col2:
                st.write("Price Range at End of Forecast:")
                last_idx = -1
                st.write(f"- Upper: ${forecast['yhat_upper'].iloc[last_idx]:.2f}")
                st.write(f"- Forecast: ${forecast['yhat'].iloc[last_idx]:.2f}")
                st.write(f"- Lower: ${forecast['yhat_lower'].iloc[last_idx]:.2f}")

            # Display confidence width trend
            st.write("\n**Confidence Width Over Time:**")
            st.line_chart(confidence_df.set_index('Date')['Confidence Width (%)'])

    except Exception as e:
        logger.error(f"Error displaying confidence analysis: {str(e)}")
        st.error(f"Error displaying confidence analysis: {str(e)}")

def display_metrics(data: pd.DataFrame, forecast: pd.DataFrame, asset_type: str, symbol: str):
    """Display enhanced metrics with confidence analysis based on asset type"""
    try:
        # Display common metrics first
        display_common_metrics(data, forecast)

        # Display asset-specific metrics
        if asset_type.lower() == 'crypto':
            display_crypto_metrics(data, forecast, symbol)

        # Display confidence analysis
        display_confidence_analysis(forecast)

    except Exception as e:
        logger.error(f"Error displaying metrics: {str(e)}")
        st.error(f"Error displaying metrics: {str(e)}")

def display_crypto_metrics(data: pd.DataFrame, forecast: pd.DataFrame, symbol: str):
    """[Your existing display_crypto_metrics function]"""
    # ... [Keep your existing implementation]

def display_economic_indicators(data: pd.DataFrame, indicator: str, economic_indicators: object):
    """[Your existing display_economic_indicators function]"""
    # ... [Keep your existing implementation]

def display_economic_indicators(data: pd.DataFrame, indicator: str, economic_indicators: object):
    """Display economic indicator information and analysis"""
    try:
        st.subheader("📊 Economic Indicator Analysis")

        # Get indicator details
        indicator_info = economic_indicators.get_indicator_info(indicator)

        # Display indicator information
        st.markdown(f"""
            **Indicator:** {indicator_info.get('description', indicator)}  
            **Frequency:** {indicator_info.get('frequency', 'N/A')}  
            **Units:** {indicator_info.get('units', 'N/A')}
        """)

        # Get and display analysis
        analysis = economic_indicators.analyze_indicator(data, indicator)
        if analysis:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Current Value",
                    f"{analysis['current_value']:.2f}",
                    f"{analysis['change_1d']:.2f}% (1d)"
                )

            with col2:
                if analysis.get('change_1m') is not None:
                    st.metric(
                        "30-Day Change",
                        f"{analysis['change_1m']:.2f}%"
                    )
                
            with col3:
                if analysis.get('trend') is not None:
                    st.metric(
                        "Trend",
                        analysis.get('trend', 'Neutral'),
                        f"{analysis.get('trend_strength', '0')}%"
                    )
                
        # Display correlation analysis if available
        if analysis and 'correlation' in analysis:
            st.subheader("Correlation Analysis")
            st.write(f"Correlation with price: {analysis['correlation']:.2f}")
            
    except Exception as e:
        logger.error(f"Error displaying economic indicators: {str(e)}")
        st.error(f"Error displaying economic indicators: {str(e)}")