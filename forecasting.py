#forecasting.py
###########################################
# Section 1: Imports and Configuration
###########################################

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

# Export all functions
__all__ = [
    'prepare_data_for_prophet',
    'add_crypto_specific_indicators',
    'add_technical_indicators',
    'prophet_forecast',
    'create_forecast_plot',
    'display_common_metrics',
    'display_confidence_analysis',
    'display_stock_metrics',
    'display_crypto_metrics',
    'display_metrics',
    'display_economic_indicators'
]

###########################################
# Section 2: Data Preparation Functions
###########################################

def prepare_data_for_prophet(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for Prophet model"""
    try:
        logger.info("Starting data preparation for Prophet")
        logger.info(f"Input data shape: {data.shape}")
        logger.info(f"Input data columns: {data.columns.tolist()}")

        df = data.copy()

        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df.rename(columns={'index': 'ds'}, inplace=True)

        if 'Close' in df.columns:
            if 'ds' not in df.columns:  
                date_cols = [col for col in df.columns if isinstance(col, str) and col.lower() in ['date', 'timestamp', 'time']]
                if date_cols:
                    df.rename(columns={date_cols[0]: 'ds'}, inplace=True)
                else:
                    df['ds'] = df.index

            df['y'] = df['Close']
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                if 'ds' not in df.columns:
                    df['ds'] = df.index
                df['y'] = df[numeric_cols[0]]

        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = df['y'].astype(float)

        prophet_df = df[['ds', 'y']].sort_values('ds').reset_index(drop=True)
        prophet_df = prophet_df.dropna()

        logger.info(f"Prepared Prophet DataFrame shape: {prophet_df.shape}")
        logger.info(f"Sample of prepared data:\n{prophet_df.head()}")

        return prophet_df

    except Exception as e:
        logger.error(f"Error in prepare_data_for_prophet: {str(e)}")
        raise Exception(f"Failed to prepare data for Prophet: {str(e)}")
        
        ###########################################
# Section 3: Technical Analysis Functions
###########################################

def add_crypto_specific_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cryptocurrency-specific indicators to the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data

    Returns:
        pd.DataFrame: DataFrame with added crypto indicators
    """
    try:
        # Market Volume Analysis
        df['volume_ma'] = df['Volume'].rolling(window=24).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma']
        
        # Volatility Indicators
        df['hourly_volatility'] = df['Close'].pct_change().rolling(window=24).std()
        df['volatility_ratio'] = df['hourly_volatility'] / df['hourly_volatility'].rolling(window=168).mean()
        
        # Market Dominance (placeholder)
        df['market_dominance'] = 0.5
        
        # Network Metrics (placeholder)
        df['network_transactions'] = 0.5
        df['active_addresses'] = 0.5
        
        return df
        
    except Exception as e:
        logger.error(f"Error adding crypto indicators: {str(e)}")
        return df

def add_technical_indicators(df: pd.DataFrame, asset_type: str = 'stocks') -> pd.DataFrame:
    """
    Add technical indicators based on asset type.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        asset_type (str): Type of asset ('stocks' or 'crypto')

    Returns:
        pd.DataFrame: DataFrame with added technical indicators
    """
    try:
        # Get configuration for indicators
        config = AssetConfig.get_config(asset_type)
        indicators = config['indicators']
        
        # Moving Averages
        df['MA5'] = df['Close'].rolling(window=indicators['ma_periods'][0]).mean()
        df['MA20'] = df['Close'].rolling(window=indicators['ma_periods'][1]).mean()
        df['MA50'] = df['Close'].rolling(window=indicators['ma_periods'][2]).mean()
        
        # RSI Calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=indicators['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=indicators['rsi_period']).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD Calculation
        macd_fast, macd_slow, signal = indicators['macd_periods']
        exp1 = df['Close'].ewm(span=macd_fast, adjust=False).mean()
        exp2 = df['Close'].ewm(span=macd_slow, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        
        # Add asset-specific indicators
        if asset_type.lower() == 'crypto':
            df = add_crypto_specific_indicators(df)
            
        return df
        
    except Exception as e:
        logger.error(f"Error in add_technical_indicators: {str(e)}")
        return df

###########################################
# Section 4: Core Forecasting Functions
###########################################

def prophet_forecast(data: pd.DataFrame, 
                    periods: int, 
                    economic_data: Optional[pd.DataFrame] = None,
                    indicator: Optional[str] = None, 
                    asset_type: str = 'stocks') -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Generate forecast using Prophet model with proper scaling for stocks.
    
    Args:
        data (pd.DataFrame): Historical price data
        periods (int): Number of periods to forecast
        economic_data (Optional[pd.DataFrame]): Economic indicator data
        indicator (Optional[str]): Name of economic indicator
        asset_type (str): Type of asset ('stocks' or 'crypto')

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[str]]: (Forecast DataFrame, Error message if any)
    """
    try:
        if data is None or data.empty:
            return None, "No data provided for forecasting"

        # Prepare data for Prophet
        prophet_df = prepare_data_for_prophet(data)
        if prophet_df is None or prophet_df.empty:
            return None, "No valid data for forecasting after preparation"

        # Apply log transformation for stocks to dampen extreme forecasts
        if asset_type.lower() == 'stocks':
            prophet_df['y'] = np.log(prophet_df['y'])

        # Initialize Prophet with asset-specific parameters
        if asset_type.lower() == 'stocks':
            model = Prophet(
                changepoint_prior_scale=0.01,  # More conservative for stocks
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_range=0.8,  # More stable forecasting
                interval_width=0.95  # 95% confidence interval
            )
        else:
            model = Prophet(
                changepoint_prior_scale=0.05,
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False
            )

        # Add monthly seasonality
        model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=5
        )

        # Fit model and generate forecast
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        # Transform forecasts back if using log scale for stocks
        if asset_type.lower() == 'stocks':
            forecast['yhat'] = np.exp(forecast['yhat'])
            forecast['yhat_lower'] = np.exp(forecast['yhat_lower'])
            forecast['yhat_upper'] = np.exp(forecast['yhat_upper'])
            
            # Apply reasonable bounds to avoid extreme values
            current_price = np.exp(prophet_df['y'].iloc[-1])
            max_forecast = current_price * 2  # Max 100% increase
            min_forecast = current_price * 0.5  # Max 50% decrease
            
            forecast['yhat'] = forecast['yhat'].clip(lower=min_forecast, upper=max_forecast)
            forecast['yhat_upper'] = forecast['yhat_upper'].clip(upper=max_forecast * 1.2)
            forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=min_forecast * 0.8)

        # Add actual values
        forecast['actual'] = np.nan
        actual_values = np.exp(prophet_df['y'].values) if asset_type.lower() == 'stocks' else prophet_df['y'].values
        forecast.loc[forecast['ds'].isin(prophet_df['ds']), 'actual'] = actual_values

        return forecast, None

    except Exception as e:
        logger.error(f"Error in prophet_forecast: {str(e)}")
        return None, str(e)