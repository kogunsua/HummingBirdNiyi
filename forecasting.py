#forcasting.py
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
        logger.info(f"Input data shape: {data.shape}")
        logger.info(f"Input data columns: {data.columns.tolist()}")

        # Make a copy of the data
        df = data.copy()

        # If the DataFrame has a DatetimeIndex, reset it to a column
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df.rename(columns={'index': 'ds'}, inplace=True)

        # If 'Close' column exists, use it as 'y'
        if 'Close' in df.columns:
            if 'ds' not in df.columns:  # If we haven't set 'ds' from the index
                date_cols = [col for col in df.columns if isinstance(col, str) and col.lower() in ['date', 'timestamp', 'time']]
                if date_cols:
                    df.rename(columns={date_cols[0]: 'ds'}, inplace=True)
                else:
                    df['ds'] = df.index

            df['y'] = df['Close']
        else:
            # If no 'Close' column, use the first numeric column as 'y'
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                if 'ds' not in df.columns:
                    df['ds'] = df.index
                df['y'] = df[numeric_cols[0]]

        # Ensure 'ds' is datetime
        df['ds'] = pd.to_datetime(df['ds'])

        # Ensure 'y' is float
        df['y'] = df['y'].astype(float)

        # Select only required columns and sort by date
        prophet_df = df[['ds', 'y']].sort_values('ds').reset_index(drop=True)
        # Drop NaN values
        prophet_df = prophet_df.dropna()

        return prophet_df

    except Exception as e:
        logger.error(f"Error in prepare_data_for_prophet: {str(e)}")
        raise Exception(f"Failed to prepare data for Prophet: {str(e)}")

def add_crypto_specific_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add cryptocurrency-specific indicators"""
    try:
        # Market Volume Analysis
        df['volume_ma'] = df['Volume'].rolling(window=24).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma']
        
        # Volatility Indicators
        df['hourly_volatility'] = df['Close'].pct_change().rolling(window=24).std()
        df['volatility_ratio'] = df['hourly_volatility'] / df['hourly_volatility'].rolling(window=168).mean()
        
        # Price Momentum
        df['momentum_1h'] = df['Close'].pct_change(periods=1)
        df['momentum_24h'] = df['Close'].pct_change(periods=24)
        df['momentum_ratio'] = df['momentum_1h'] / df['momentum_24h']
        
        # Market Dominance (placeholder - replace with actual data)
        df['market_dominance'] = 0.5
        
        # Network Metrics (placeholder - replace with actual data)
        df['network_transactions'] = 0.5
        df['active_addresses'] = 0.5
        
        return df
        
    except Exception as e:
        logger.error(f"Error adding crypto indicators: {str(e)}")
        return df

def add_technical_indicators(df: pd.DataFrame, asset_type: str = 'stocks') -> pd.DataFrame:
    """Add technical indicators based on asset type"""
    try:
        # Get configuration
        config = AssetConfig.get_config(asset_type)
        indicators = config['indicators']
        
        # Add base technical indicators with configured parameters
        df['MA5'] = df['Close'].rolling(window=indicators['ma_periods'][0]).mean()
        df['MA20'] = df['Close'].rolling(window=indicators['ma_periods'][1]).mean()
        df['MA50'] = df['Close'].rolling(window=indicators['ma_periods'][2]).mean()
        
        # RSI with configured period
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=indicators['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=indicators['rsi_period']).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD with configured periods
        macd_fast, macd_slow, signal = indicators['macd_periods']
        exp1 = df['Close'].ewm(span=macd_fast, adjust=False).mean()
        exp2 = df['Close'].ewm(span=macd_slow, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        
        # Add crypto-specific indicators if needed
        if asset_type.lower() == 'crypto':
            df = add_crypto_specific_indicators(df)
            
        return df
        
    except Exception as e:
        logger.error(f"Error in add_technical_indicators: {str(e)}")
        return df

def create_forecast_plot(data: pd.DataFrame, forecast: pd.DataFrame, model_name: str, symbol: str) -> go.Figure:
    """Create an interactive plot with historical data and forecast"""
    try:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, row_heights=[0.7, 0.3],
                           subplot_titles=(f'{symbol} Price Forecast', 'Confidence Analysis'))

        # Historical Data
        if isinstance(data.index, pd.DatetimeIndex):
            historical_dates = data.index
        else:
            historical_dates = pd.to_datetime(data['Date'] if 'Date' in data.columns else data['timestamp'])
        historical_values = data['Close'] if 'Close' in data.columns else data.iloc[:, 0]

        # Add historical data
        fig.add_trace(
            go.Candlestick(
                x=historical_dates,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=historical_values,
                name='Historical',
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )

        # Add forecast line
        fig.add_trace(
            go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                name=f'{model_name} Forecast',
                line=dict(color='blue', dash='dash'),
                hovertemplate='Date: %{x}<br>Forecast: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Update layout
        fig.update_layout(
            title=f'{symbol} Price Forecast',
            yaxis_title='Price ($)',
            yaxis2_title='Confidence (%)',
            height=800
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating forecast plot: {str(e)}")
        return None

def prophet_forecast(data: pd.DataFrame, periods: int, economic_data: Optional[pd.DataFrame] = None,
                     indicator: Optional[str] = None, asset_type: str = 'stocks') -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Generate forecast using Prophet model"""
    try:
        if data is None or data.empty:
            logger.error("No data provided for forecasting")
            return None, "No data provided for forecasting"

        # Prepare data for Prophet
        try:
            prophet_df = prepare_data_for_prophet(data)
        except Exception as e:
            logger.error(f"Error preparing data for Prophet: {str(e)}")
            return None, f"Error preparing data: {str(e)}"

        if prophet_df is None or prophet_df.empty:
            return None, "No valid data for forecasting after preparation"

        # Initialize Prophet with default parameters
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

        # Fit model and make forecast
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        return forecast, None

    except Exception as e:
        logger.error(f"Error in prophet_forecast: {str(e)}")
        return None, str(e)

def display_common_metrics(data: pd.DataFrame, forecast: pd.DataFrame):
    """Display common metrics for both stocks and cryptocurrencies"""
    try:
        st.subheader("📈 Price Metrics")
        
        # Ensure we have the required data
        if 'Close' not in data.columns:
            raise ValueError("Close price data not found in dataset")
            
        # Handle data access safely
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

    except Exception as e:
        logger.error(f"Error displaying common metrics: {str(e)}")
        st.error(f"Error displaying common metrics: {str(e)}")

def display_confidence_analysis(forecast: pd.DataFrame):
    """Display detailed confidence analysis of the forecast"""
    try:
        st.subheader("📊 Confidence Analysis")

        # Calculate metrics
        confidence_width = (forecast['yhat_upper'] - forecast['yhat_lower']) / forecast['yhat'] * 100
        avg_confidence = 100 - confidence_width.mean()
        total_trend = ((forecast['yhat'].iloc[-1] / forecast['yhat'].iloc[0]) - 1) * 100
        trend_consistency = np.sum(np.diff(forecast['yhat']) > 0) / (len(forecast) - 1) * 100

        # Display metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Average Confidence", f"{avg_confidence:.1f}%")
        with col2:
            st.metric("Overall Trend", f"{total_trend:+.1f}%")
        with col3:
            st.metric("Trend Consistency", f"{trend_consistency:.1f}%")

    except Exception as e:
        logger.error(f"Error displaying confidence analysis: {str(e)}")
        st.error(f"Error displaying confidence analysis: {str(e)}")

def display_crypto_metrics(data: pd.DataFrame, forecast: pd.DataFrame, symbol: str):
    """Display cryptocurrency-specific metrics"""
    try:
        st.subheader("🪙 Cryptocurrency Metrics")

        col1, col2 = st.columns(2)
        with col1:
            volume = float(data['Volume'].iloc[-1])
            volume_change = float(data['Volume'].pct_change().iloc[-1] * 100)
            st.metric(
                "24h Volume",
                f"${volume:,.0f}",
                f"{volume_change:+.2f}%"
            )

        with col2:
            if 'volume_ratio' in data.columns:
                vol_ratio = float(data['volume_ratio'].iloc[-1])
                st.metric(
                    "Volume Ratio",
                    f"{vol_ratio:.2f}",
                    "Above Average" if vol_ratio > 1 else "Below Average"
                )

    except Exception as e:
        logger.error(f"Error displaying crypto metrics: {str(e)}")
        st.error(f"Error displaying crypto metrics: {str(e)}")

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
                if analysis.get('change_1m')
                
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
                
                