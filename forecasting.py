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
    """Calculate accuracy metrics for the forecast."""
    try:
        # Align the actual and predicted series
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
    """Generate forecasts using Prophet with optional regressors."""
    try:
        # Prepare data for Prophet
        df = data.reset_index()
        df = df[['Date', 'Close']]
        df.columns = ['ds', 'y']
        df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)  # Remove timezone info

        # Initialize Prophet model with configuration
        model = Prophet(**Config.PROPHET_CONFIG)

        # Add sentiment data as a regressor if provided
        if sentiment_data is not None:
            sentiment_data = sentiment_data.copy()
            sentiment_data.columns = ['ds', 'sentiment']
            sentiment_data['ds'] = pd.to_datetime(sentiment_data['ds']).dt.tz_localize(None)
            model.add_regressor('sentiment')
            df = df.merge(sentiment_data, on='ds', how='left')
            df['sentiment'].fillna(method='ffill', inplace=True)

        # Add economic data as a regressor if provided
        if economic_data is not None:
            economic_data = economic_data.copy()
            economic_data.columns = ['ds', 'economic']
            economic_data['ds'] = pd.to_datetime(economic_data['ds']).dt.tz_localize(None)
            model.add_regressor('economic')
            df = df.merge(economic_data, on='ds', how='left')
            df['economic'].fillna(method='ffill', inplace=True)

        # Fit the model
        model.fit(df)

        # Create a future dataframe for predictions
        future = model.make_future_dataframe(periods=periods)

        # Include regressors in the future dataframe
        if sentiment_data is not None:
            future = future.merge(sentiment_data, on='ds', how='left')
            future['sentiment'].fillna(method='ffill', inplace=True)
        if economic_data is not None:
            future = future.merge(economic_data, on='ds', how='left')
            future['economic'].fillna(method='ffill', inplace=True)

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
    """Create a detailed forecast plot."""
    fig = go.Figure()

    # Add historical data
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['Close'], 
        name="Actual Prices", 
        line=dict(color="blue")
    ))

    # Add forecasted values
    fig.add_trace(go.Scatter(
        x=forecast['ds'], 
        y=forecast['yhat'], 
        name="Forecast", 
        line=dict(color="red", dash="dash")
    ))

    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast['ds'], 
        y=forecast['yhat_upper'], 
        fill=None, 
        mode='lines', 
        line=dict(color="gray", width=0), 
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'], 
        y=forecast['yhat_lower'], 
        fill='tonexty', 
        mode='lines', 
        line=dict(color="gray", width=0), 
        name="Confidence Interval"
    ))

    # Sentiment overlay
    if sentiment_data is not None:
        fig.add_trace(go.Scatter(
            x=sentiment_data.index, 
            y=sentiment_data['sentiment'], 
            name="Sentiment Score", 
            yaxis="y2", 
            line=dict(color="purple", dash="dot")
        ))

    # Update layout
    fig.update_layout(
        title=f"{symbol} Forecast using {model_name}",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        yaxis2=dict(
            title="Sentiment Score",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        template="plotly_white",
        hovermode="x unified"
    )
    return fig


def display_components(forecast: pd.DataFrame):
    """Display components of the forecast."""
    st.subheader("📊 Forecast Components")
    
    # Tabs for components
    tab1, tab2, tab3 = st.tabs(["Trend", "Seasonality", "Additional Factors"])
    
    with tab1:
        st.plotly_chart(go.Figure(
            go.Scatter(x=forecast['ds'], y=forecast['trend'], name="Trend")
        ), use_container_width=True)

    with tab2:
        if 'weekly' in forecast:
            st.plotly_chart(go.Figure(
                go.Scatter(x=forecast['ds'], y=forecast['weekly'], name="Weekly Seasonality")
            ), use_container_width=True)

        if 'yearly' in forecast:
            st.plotly_chart(go.Figure(
                go.Scatter(x=forecast['ds'], y=forecast['yearly'], name="Yearly Seasonality")
            ), use_container_width=True)

    with tab3:
        for column in forecast.columns:
            if column.endswith('_regressor'):
                st.plotly_chart(go.Figure(
                    go.Scatter(x=forecast['ds'], y=forecast[column], name=column)
                ), use_container_width=True)