# forecast_display.py
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

def display_forecast_results(
    price_data: pd.DataFrame,
    forecast: pd.DataFrame,
    impact_metrics: Dict[str, Any],
    forecast_type: str,
    asset_type: str,
    symbol: str
) -> None:
    """Display comprehensive forecast results including price, sentiment, and technical analysis"""
    try:
        st.subheader("📊 Forecast Analysis")

        # Display the main metrics first
        display_common_metrics(price_data, forecast)

        # Create tabs for different analyses
        forecast_tab, technical_tab, metrics_tab = st.tabs([
            "📈 Price Forecast",
            "📊 Technical Analysis",
            "📉 Detailed Metrics"
        ])

        with forecast_tab:
            # Display forecast plot
            fig = create_forecast_plot(price_data, forecast, "Prophet", symbol, asset_type)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

            # Display confidence analysis
            display_confidence_analysis(forecast)

        with technical_tab:
            # Display technical indicators
            if asset_type.lower() == 'stocks':
                display_stock_metrics(price_data, forecast, symbol)
            else:
                display_crypto_metrics(price_data, forecast, symbol)

        with metrics_tab:
            # Display detailed metrics including error metrics
            if 'error_metrics' in impact_metrics:
                st.subheader("Model Performance Metrics")
                error_cols = st.columns(3)
                metrics = impact_metrics['error_metrics']
                
                with error_cols[0]:
                    st.metric("MAE", f"${metrics.get('mae', 0):.2f}")
                with error_cols[1]:
                    st.metric("RMSE", f"${metrics.get('rmse', 0):.2f}")
                with error_cols[2]:
                    st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")

            # Display sentiment impact if available
            if forecast_type == "Price + Market Sentiment" and 'sentiment_impact' in impact_metrics:
                st.subheader("Sentiment Analysis Impact")
                sent_cols = st.columns(2)
                sentiment = impact_metrics['sentiment_impact']
                
                with sent_cols[0]:
                    st.metric("Overall Sentiment", 
                             f"{sentiment.get('overall_score', 0):.2f}",
                             f"{sentiment.get('sentiment_change', 0):+.2f}%")
                with sent_cols[1]:
                    st.metric("Confidence Score",
                             f"{sentiment.get('confidence', 0):.2f}")

    except Exception as e:
        logger.error(f"Error in display_forecast_results: {str(e)}")
        st.error("Error displaying forecast results. Please try again.")

def display_sentiment_impact_analysis(
    sentiment_period: int,
    sentiment_weight: float,
    sentiment_source: str
) -> None:
    """Display the configuration and impact of sentiment analysis"""
    try:
        st.info(
            f"📊 Sentiment Analysis Configuration:\n\n"
            f"• Analysis Period: {sentiment_period} days\n"
            f"• Impact Weight: {sentiment_weight:.1%}\n"
            f"• Data Source: {sentiment_source}"
        )

    except Exception as e:
        logger.error(f"Error in display_sentiment_impact_analysis: {str(e)}")
        st.error("Error displaying sentiment analysis configuration.")

def display_sentiment_impact_results(
    sentiment_data: pd.DataFrame,
    price_data: pd.DataFrame
) -> None:
    """Display the results of sentiment impact analysis"""
    try:
        if sentiment_data is None or sentiment_data.empty:
            st.warning("No sentiment data available for analysis.")
            return

        st.subheader("📊 Sentiment Impact Analysis")

        # Create sentiment trend visualization
        fig = go.Figure()
        
        # Add sentiment score line
        fig.add_trace(
            go.Scatter(
                x=sentiment_data.index,
                y=sentiment_data['sentiment_score'],
                name='Sentiment Score',
                line=dict(color='blue')
            )
        )

        # Add price overlay if available
        if price_data is not None and not price_data.empty:
            # Normalize price data to overlay with sentiment
            normalized_price = (price_data['Close'] - price_data['Close'].min()) / \
                             (price_data['Close'].max() - price_data['Close'].min())
            
            fig.add_trace(
                go.Scatter(
                    x=price_data.index,
                    y=normalized_price,
                    name='Normalized Price',
                    line=dict(color='red', dash='dash'),
                    yaxis='y2'
                )
            )

        fig.update_layout(
            title='Sentiment Score vs Price Trend',
            yaxis=dict(title='Sentiment Score'),
            yaxis2=dict(title='Normalized Price', overlaying='y', side='right'),
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Display summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_sentiment = sentiment_data['sentiment_score'].mean()
            st.metric("Average Sentiment", f"{avg_sentiment:.2f}")
            
        with col2:
            sentiment_volatility = sentiment_data['sentiment_score'].std()
            st.metric("Sentiment Volatility", f"{sentiment_volatility:.2f}")
            
        with col3:
            sentiment_trend = (
                sentiment_data['sentiment_score'].iloc[-1] - 
                sentiment_data['sentiment_score'].iloc[0]
            )
            st.metric("Sentiment Trend", f"{sentiment_trend:+.2f}")

    except Exception as e:
        logger.error(f"Error in display_sentiment_impact_results: {str(e)}")
        st.error("Error displaying sentiment impact results.")