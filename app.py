# app.py
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
    """Prepare data for Prophet model"""
    try:
        logger.info("Starting data preparation for Prophet")
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
        prophet_df = df[['ds', 'y']].sort_values('ds').reset_index(drop=True).dropna()
        return prophet_df

    except Exception as e:
        logger.error(f"Error in prepare_data_for_prophet: {str(e)}")
        raise Exception(f"Failed to prepare data for Prophet: {str(e)}")

def add_technical_indicators(df: pd.DataFrame, asset_type: str = 'stocks') -> pd.DataFrame:
    """Add technical indicators based on asset type"""
    try:
        config = AssetConfig.get_config(asset_type)
        indicators = config['indicators']
        
        # Moving Averages
        df['MA5'] = df['Close'].rolling(window=indicators['ma_periods'][0]).mean()
        df['MA20'] = df['Close'].rolling(window=indicators['ma_periods'][1]).mean()
        df['MA50'] = df['Close'].rolling(window=indicators['ma_periods'][2]).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=indicators['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=indicators['rsi_period']).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        macd_fast, macd_slow, signal = indicators['macd_periods']
        exp1 = df['Close'].ewm(span=macd_fast, adjust=False).mean()
        exp2 = df['Close'].ewm(span=macd_slow, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        
        return df
        
    except Exception as e:
        logger.error(f"Error in add_technical_indicators: {str(e)}")
        return df

def prophet_forecast(data: pd.DataFrame, periods: int, economic_data: Optional[pd.DataFrame] = None,
                     indicator: Optional[str] = None, asset_type: str = 'stocks') -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Generate forecast using Prophet model"""
    try:
        if data is None or data.empty:
            return None, "No data provided for forecasting"

        prophet_df = prepare_data_for_prophet(data)
        if prophet_df is None or prophet_df.empty:
            return None, "No valid data for forecasting"

        model = Prophet(
            changepoint_prior_scale=0.05,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )

        model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=5
        )

        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        forecast['actual'] = np.nan
        forecast.loc[forecast['ds'].isin(prophet_df['ds']), 'actual'] = prophet_df['y'].values

        return forecast, None

    except Exception as e:
        logger.error(f"Error in prophet_forecast: {str(e)}")
        return None, str(e)

def create_forecast_plot(data: pd.DataFrame, forecast: pd.DataFrame, model_name: str, symbol: str) -> go.Figure:
    """Create an interactive plot with historical data and forecast"""
    try:
        fig = make_subplots(rows=1, cols=1, subplot_titles=[f'{symbol} Price Forecast'])

        # Historical data
        historical_dates = pd.to_datetime(data.index) if isinstance(data.index, pd.DatetimeIndex) else pd.to_datetime(data['Date'])
        fig.add_trace(
            go.Scatter(x=historical_dates, y=data['Close'], name='Historical', line=dict(color='blue'))
        )

        # Forecast
        fig.add_trace(
            go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(color='red', dash='dash'))
        )

        # Confidence intervals
        fig.add_trace(
            go.Scatter(
                x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
                y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
                fill='toself',
                fillcolor='rgba(0,100,255,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Interval'
            )
        )

        fig.update_layout(
            title=f'{symbol} Price Forecast',
            yaxis_title='Price ($)',
            height=600,
            showlegend=True
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating forecast plot: {str(e)}")
        return None

def display_metrics(data: pd.DataFrame, forecast: pd.DataFrame, asset_type: str, symbol: str):
    """Display all metrics"""
    try:
        st.subheader("📈 Price Metrics")
        
        current_price = float(data['Close'].iloc[-1])
        price_change = float(data['Close'].pct_change().iloc[-1] * 100)
        forecast_price = float(forecast['yhat'].iloc[-1])
        forecast_change = ((forecast_price / current_price) - 1) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Price", f"${current_price:,.2f}", f"{price_change:+.2f}%")
        with col2:
            st.metric("Forecast Price", f"${forecast_price:,.2f}", f"{forecast_change:+.2f}%")
        
    except Exception as e:
        logger.error(f"Error displaying metrics: {str(e)}")
        st.error(f"Error displaying metrics: {str(e)}")

def display_economic_indicators(data: pd.DataFrame, indicator: str, economic_indicators: object):
    """Display economic indicator information and analysis"""
    try:
        st.subheader("📊 Economic Indicator Analysis")
        
        indicator_info = economic_indicators.get_indicator_info(indicator)
        st.markdown(f"""
            **Indicator:** {indicator_info.get('description', indicator)}  
            **Frequency:** {indicator_info.get('frequency', 'N/A')}  
            **Units:** {indicator_info.get('units', 'N/A')}
        """)
        
        analysis = economic_indicators.analyze_indicator(data, indicator)
        if analysis:
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Current Value",
                    f"{analysis['current_value']:.2f}",
                    f"{analysis.get('change_1d', 0):.2f}% (1d)"
                )
            with col2:
                if 'change_1m' in analysis:
                    st.metric(
                        "30-Day Change",
                        f"{analysis['change_1m']:.2f}%"
                    )
                    
    except Exception as e:
        logger.error(f"Error displaying economic indicators: {str(e)}")
        st.error(f"Error displaying economic indicators: {str(e)}")