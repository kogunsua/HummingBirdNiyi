# forecasting.py
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet
from typing import Optional, Tuple
import numpy as np

def prepare_data_for_prophet(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for Prophet model"""
    # Reset index and ensure it's named correctly
    df = df.reset_index()
    
    # Rename columns to Prophet format
    if 'Date' in df.columns:
        df = df.rename(columns={'Date': 'ds'})
    elif 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'ds'})
    else:
        # If index was unnamed, the reset_index creates a column named 'index'
        df = df.rename(columns={'index': 'ds'})
    
    # Ensure Close price is named 'y' for Prophet
    df = df.rename(columns={'Close': 'y'})
    
    # Ensure datetime is timezone naive
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    
    # Select only required columns
    df = df[['ds', 'y']]
    
    return df

def prophet_forecast(data: pd.DataFrame, periods: int, economic_data: Optional[pd.DataFrame] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Generate forecasts using Prophet with optional economic indicators"""
    try:
        # Prepare data for Prophet
        df = prepare_data_for_prophet(data)
        
        model = Prophet(daily_seasonality=True)
        
        if economic_data is not None:
            economic_df = economic_data.copy()
            economic_df.columns = ['ds', 'regressor']
            economic_df['ds'] = pd.to_datetime(economic_df['ds']).dt.tz_localize(None)
            model.add_regressor('regressor')
            df = df.merge(economic_df, on='ds', how='left')
            df['regressor'].fillna(method='ffill', inplace=True)
        
        model.fit(df)
        future = model.make_future_dataframe(periods=periods)
        
        if economic_data is not None:
            future = future.merge(economic_df, on='ds', how='left')
            future['regressor'].fillna(method='ffill', inplace=True)
        
        forecast = model.predict(future)
        return forecast, None
    
    except Exception as e:
        return None, str(e)

def create_forecast_plot(data: pd.DataFrame, forecast: pd.DataFrame, model_name: str, symbol: str) -> go.Figure:
    """Create an interactive forecast plot using Plotly"""
    fig = go.Figure()

    # Add actual price trace
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        name='Actual Price',
        line=dict(color='blue')
    ))

    # Add forecast trace
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))

    # Add confidence interval
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