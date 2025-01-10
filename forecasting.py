"""
HummingBird v2 Forecasting Module
Handles all forecasting operations including predictions, visualization, and analysis.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet
from typing import Optional, Tuple, Dict
import numpy as np
from config import Config


def calculate_accuracy(actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
    """Calculate accuracy metrics for the forecast"""
    try:
        # Make sure we only compare overlapping dates
        actual = actual[-len(predicted):]
        
        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        # Root Mean Square Error (RMSE)
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        # R-squared (R²)
        r2 = 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - actual.mean()) ** 2))
        
        return {
            'MAPE (%)': mape,
            'RMSE': rmse,
            'R²': r2
        }
    except Exception as e:
        st.error(f"Error calculating accuracy metrics: {str(e)}")
        return {}


def prophet_forecast(
    data: pd.DataFrame, 
    periods: int, 
    sentiment_data: Optional[pd.DataFrame] = None,
    economic_data: Optional[pd.DataFrame] = None
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Generate forecasts using Prophet with sentiment and economic indicators"""
    try:
        df = data.reset_index()
        df = df[['Date', 'Close']]
        df.columns = ['ds', 'y']
        
        # Ensure no timezone info
        df['ds'] = pd.to_datetime(df['ds']).tz_localize(None)
        
        # Initialize Prophet with configuration
        model = Prophet(**Config.PROPHET_CONFIG)
        
        # Add sentiment regressor if available
        if sentiment_data is not None:
            sentiment_df = sentiment_data.copy()
            sentiment_df.columns = ['ds', 'sentiment']
            sentiment_df['ds'] = pd.to_datetime(sentiment_df['ds']).tz_localize(None)
            model.add_regressor('sentiment', mode='multiplicative')
            df = df.merge(sentiment_df, on='ds', how='left')
            df['sentiment'].fillna(method='ffill', inplace=True)
        
        # Add economic regressor if available
        if economic_data is not None:
            economic_df = economic_data.copy()
            economic_df.columns = ['ds', 'regressor']
            economic_df['ds'] = pd.to_datetime(economic_df['ds']).tz_localize(None)
            model.add_regressor('regressor', mode='multiplicative')
            df = df.merge(economic_df, on='ds', how='left')
            df['regressor'].fillna(method='ffill', inplace=True)
        
        # Fit model
        model.fit(df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods)
        
        # Add regressors to future dataframe
        if sentiment_data is not None:
            future = future.merge(sentiment_df, on='ds', how='left')
            future['sentiment'].fillna(method='ffill', inplace=True)
        
        if economic_data is not None:
            future = future.merge(economic_df, on='ds', how='left')
            future['regressor'].fillna(method='ffill', inplace=True)
        
        # Generate forecast
        forecast = model.predict(future)
        return forecast, None
    
    except Exception as e:
        return None, str(e)


def create_forecast_plot(
    data: pd.DataFrame, 
    forecast: pd.DataFrame, 
    model_name: str, 
    symbol: str,
    sentiment_data: Optional[pd.DataFrame] = None
) -> go.Figure:
    """Create enhanced forecast plot"""
    fig = go.Figure()

    # Add actual price
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        name='Actual Price',
        line=dict(color='blue')
    ))

    # Add forecast
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))

    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        fill=None,
        mode='lines',
        line=dict(color='gray', width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        fill='tonexty',
        mode='lines',
        line=dict(color='gray', width=0),
        name='Confidence Interval'
    ))

    # Add sentiment overlay if available
    if sentiment_data is not None:
        fig.add_trace(go.Scatter(
            x=sentiment_data.index,
            y=sentiment_data['market_sentiment'],
            name='Market Sentiment',
            yaxis='y2',
            line=dict(color='purple', dash='dot')
        ))

    # Update layout
    layout = {
        'title': f'{symbol} Price Forecast ({model_name})',
        'xaxis_title': 'Date',
        'yaxis_title': 'Price (USD)',
        'template': 'plotly_white',
        'hovermode': 'x unified'
    }

    # Add secondary y-axis for sentiment
    if sentiment_data is not None:
        layout['yaxis2'] = {
            'title': 'Sentiment Score',
            'overlaying': 'y',
            'side': 'right',
            'showgrid': False
        }

    fig.update_layout(layout)
    return fig


def display_components(forecast: pd.DataFrame):
    """Display forecast components"""
    st.subheader("📊 Forecast Components")
    
    # Create tabs for different components
    tab1, tab2, tab3 = st.tabs(["Trend", "Seasonality", "Additional Factors"])
    
    with tab1:
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['trend'],
            name='Trend'
        ))
        
        fig_trend.update_layout(
            title="Trend Component",
            xaxis_title="Date",
            yaxis_title="Trend Value",
            template="plotly_white"
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with tab2:
        # Weekly seasonality
        if 'weekly' in forecast.columns:
            fig_weekly = go.Figure()
            fig_weekly.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['weekly'],
                name='Weekly Pattern'
            ))
            fig_weekly.update_layout(
                title="Weekly Seasonality",
                xaxis_title="Date",
                yaxis_title="Weekly Effect",
                template="plotly_white"
            )
            st.plotly_chart(fig_weekly, use_container_width=True)
        
        # Yearly seasonality
        if 'yearly' in forecast.columns:
            fig_yearly = go.Figure()
            fig_yearly.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yearly'],
                name='Yearly Pattern'
            ))
            fig_yearly.update_layout(
                title="Yearly Seasonality",
                xaxis_title="Date",
                yaxis_title="Yearly Effect",
                template="plotly_white"
            )
            st.plotly_chart(fig_yearly, use_container_width=True)
    
    with tab3:
        # Display additional regressors if available
        extra_regressors = [col for col in forecast.columns 
                          if col.endswith('_regressor') 
                          or col in ['sentiment', 'regressor']]
        
        for regressor in extra_regressors:
            if regressor in forecast.columns:
                fig_regressor = go.Figure()
                fig_regressor.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast[regressor],
                    name=regressor.capitalize()
                ))
                fig_regressor.update_layout(
                    title=f"{regressor.capitalize()} Impact",
                    xaxis_title="Date",
                    yaxis_title="Effect",
                    template="plotly_white"
                )
                st.plotly_chart(fig_regressor, use_container_width=True)