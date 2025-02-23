#forecasting.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from typing import Tuple, Optional, Dict, Any
import logging
from datetime import datetime, timedelta
from asset_config import AssetConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_data_for_prophet(data: pd.DataFrame, asset_type: str = 'stocks') -> pd.DataFrame:
    """Prepare data for Prophet model with asset-specific handling"""
    try:
        logger.info("Starting data preparation for Prophet")
        df = data.copy()
        
        # Handle multi-index columns if present
        if isinstance(df.columns, pd.MultiIndex):
            symbol = df.columns[0][1]  # Get the symbol from first column
            df = pd.DataFrame({
                col[0]: df[col] for col in df.columns if col[1] == symbol
            })
        
        # Handle the index
        if isinstance(df.index, pd.DatetimeIndex):
            df['ds'] = df.index
            prophet_df = df.reset_index(drop=True)
        else:
            prophet_df = df.copy()
            date_cols = [col for col in prophet_df.columns if col.lower() in 
                        ['date', 'time', 'timestamp']]
            if date_cols:
                prophet_df['ds'] = prophet_df[date_cols[0]]
            else:
                raise ValueError("No date column found in dataset")

        # Ensure 'y' column exists
        if 'Close' in prophet_df.columns:
            prophet_df['y'] = prophet_df['Close']
        else:
            numeric_cols = prophet_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                prophet_df['y'] = prophet_df[numeric_cols[0]]
            else:
                raise ValueError("No numeric column found for target variable")

        # Ensure proper datetime format
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        if prophet_df['ds'].dt.tz is not None:
            prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)

        # Convert target to float
        prophet_df['y'] = prophet_df['y'].astype(float)

        # Select only required columns and sort
        prophet_df = prophet_df[['ds', 'y']].sort_values('ds').reset_index(drop=True)
        
        # Drop any NaN values
        prophet_df = prophet_df.dropna()

        return prophet_df

    except Exception as e:
        logger.error(f"Error in prepare_data_for_prophet: {str(e)}")
        raise Exception(f"Failed to prepare data for Prophet: {str(e)}")

def add_technical_indicators(df: pd.DataFrame, asset_type: str = 'stocks') -> pd.DataFrame:
    """Add technical indicators based on asset type"""
    try:
        # Get configuration
        config = AssetConfig.get_config(asset_type)
        indicators = config['indicators']
        
        df = df.copy()  # Create a copy to avoid modifying the original
        
        # Calculate Moving Averages
        for period in indicators['ma_periods']:
            df[f'MA{period}'] = df['Close'].rolling(window=period).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=indicators['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=indicators['rsi_period']).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        fast, slow, signal = indicators['macd_periods']
        exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=indicators['bb_period']).mean()
        std = df['Close'].rolling(window=indicators['bb_period']).std()
        df['BB_upper'] = df['BB_middle'] + (std * 2)
        df['BB_lower'] = df['BB_middle'] - (std * 2)
        
        # Add asset-specific indicators
        if asset_type.lower() == 'crypto':
            df = add_crypto_specific_indicators(df)
        else:
            df = add_stock_specific_indicators(df)
            
        return df
        
    except Exception as e:
        logger.error(f"Error in add_technical_indicators: {str(e)}")
        return df

def prophet_forecast(data: pd.DataFrame, periods: int, economic_data: Optional[pd.DataFrame] = None,
                    indicator: Optional[str] = None, asset_type: str = 'stocks') -> Tuple[Optional[pd.DataFrame], Dict[str, float]]:
    """Generate forecast using Prophet model"""
    try:
        if data is None or data.empty:
            return None, {"error": "No data provided for forecasting"}

        # Prepare data for Prophet
        prophet_df = prepare_data_for_prophet(data, asset_type)
        
        # Get asset-specific configuration
        config = AssetConfig.get_config(asset_type)
        model_config = config['model_config']
        
        # Initialize Prophet with configuration
        model = Prophet(**model_config)

        # Add economic indicators if provided
        if economic_data is not None and indicator is not None:
            model.add_regressor(indicator)
            prophet_df = prophet_df.merge(economic_data, on='ds', how='left')

        # Fit model and make forecast
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=periods)
        
        if economic_data is not None and indicator is not None:
            future = future.merge(economic_data, on='ds', how='left')
            
        forecast = model.predict(future)
        
        # Calculate error metrics
        error_metrics = {}
        if len(prophet_df) > 0:
            actual = prophet_df['y'].values
            predicted = forecast['yhat'][:len(actual)]
            
            error_metrics = {
                'mae': np.mean(np.abs(actual - predicted)),
                'rmse': np.sqrt(np.mean((actual - predicted) ** 2)),
                'mape': np.mean(np.abs((actual - predicted) / actual)) * 100
            }

        return forecast, error_metrics

    except Exception as e:
        logger.error(f"Error in prophet_forecast: {str(e)}")
        return None, {"error": str(e)}

def create_forecast_plot(data: pd.DataFrame, forecast: pd.DataFrame, model_name: str, 
                        symbol: str, asset_type: str = 'stocks') -> go.Figure:
    """Create an interactive plot with historical data and forecast"""
    try:
        config = AssetConfig.get_config(asset_type)
        viz_config = config['viz']
        
        # Create figure with secondary y-axis for volume
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=(
                f'{symbol} Price Forecast',
                'Volume Analysis',
                'Technical Indicators'
            )
        )

        # Add historical data
        historical_dates = pd.to_datetime(data.index) if isinstance(data.index, pd.DatetimeIndex) else data.index
        
        # Price data
        fig.add_trace(
            go.Scatter(
                x=historical_dates,
                y=data['Close'],
                name='Historical',
                line=dict(color=viz_config['line_colors'][0])
            ),
            row=1, col=1
        )

        # Forecast line
        fig.add_trace(
            go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                name=f'{model_name} Forecast',
                line=dict(color=viz_config['line_colors'][1], dash='dash')
            ),
            row=1, col=1
        )

        # Confidence intervals
        fig.add_trace(
            go.Scatter(
                x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
                y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
                fill='toself',
                fillcolor=viz_config['confidence_colors'][0],
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval'
            ),
            row=1, col=1
        )

        # Add Volume
        if 'Volume' in data.columns:
            fig.add_trace(
                go.Bar(
                    x=historical_dates,
                    y=data['Volume'],
                    name='Volume',
                    marker_color='lightgray'
                ),
                row=2, col=1
            )

        # Add technical indicators based on asset type
        _add_technical_traces(fig, data, historical_dates, asset_type)

        # Update layout
        fig.update_layout(
            height=viz_config['plot_height'],
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating forecast plot: {str(e)}")
        return None

def _add_technical_traces(fig: go.Figure, data: pd.DataFrame, dates: pd.Index, asset_type: str):
    """Add technical indicator traces to the plot based on asset type"""
    try:
        config = AssetConfig.get_config(asset_type)
        indicators = config['indicators']
        
        if asset_type.lower() == 'crypto':
            if 'hourly_volatility' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=data['hourly_volatility'],
                        name='Hourly Volatility',
                        line=dict(color='orange')
                    ),
                    row=3, col=1
                )
        else:
            # Add Bollinger Bands
            for band in ['BB_middle', 'BB_upper', 'BB_lower']:
                if band in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=data[band],
                            name=band.replace('_', ' '),
                            line=dict(
                                color='gray',
                                dash='dash' if band == 'BB_middle' else None
                            )
                        ),
                        row=3, col=1
                    )
    except Exception as e:
        logger.error(f"Error adding technical traces: {str(e)}")

# Continue with rest of your display functions...
[The rest of your display functions would go here...]

# Export all required functions
__all__ = [
    'prophet_forecast',
    'create_forecast_plot',
    'display_metrics',
    'display_confidence_analysis',
    'add_technical_indicators',
    'display_forecast_results',
    'prepare_data_for_prophet',
    'add_crypto_specific_indicators',
    'add_stock_specific_indicators',
    'display_stock_metrics',
    'display_common_metrics',
    'display_crypto_metrics',
    'display_economic_indicators'
]