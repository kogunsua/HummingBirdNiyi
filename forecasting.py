#forecasting.py
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

        logger.info(f"Prepared Prophet DataFrame shape: {prophet_df.shape}")
        logger.info(f"Prophet DataFrame columns: {prophet_df.columns.tolist()}")
        logger.info(f"Sample of prepared data:\n{prophet_df.head()}")

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
    """Add technical indicators based on asset type"""
    try:
        # Get configuration
        config = AssetConfig.get_config(asset_type)
        indicators = config['indicators']
        
        # Add base technical indicators
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
        
        if asset_type.lower() == 'crypto':
            df = add_crypto_specific_indicators(df)
            
        return df
        
    except Exception as e:
        logger.error(f"Error in add_technical_indicators: {str(e)}")
        return df

def prophet_forecast(data: pd.DataFrame, periods: int, economic_data: Optional[pd.DataFrame] = None,
                     indicator: Optional[str] = None, asset_type: str = 'stocks') -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Generate forecast using Prophet model with proper scaling for stocks"""
    try:
        if data is None or data.empty:
            return None, "No data provided for forecasting"

        # Prepare data for Prophet
        prophet_df = prepare_data_for_prophet(data)
        
        if prophet_df is None or prophet_df.empty:
            return None, "No valid data for forecasting after preparation"

        # Calculate scaling factor based on the price range
        max_price = prophet_df['y'].max()
        scale_factor = 1.0
        if asset_type.lower() == 'stocks':
            # Using log transformation for stock prices to dampen extreme forecasts
            prophet_df['y'] = np.log(prophet_df['y'])
            
        # Initialize Prophet with appropriate parameters based on asset type
        if asset_type.lower() == 'stocks':
            model = Prophet(
                changepoint_prior_scale=0.01,  # Reduced to make forecast more conservative
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_range=0.8,  # Reduced from default 0.9 to make forecast more stable
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

        # Fit model
        model.fit(prophet_df)
        
        # Generate future dates
        future = model.make_future_dataframe(periods=periods)
        
        # Make forecast
        forecast = model.predict(future)
        
        # Transform back if using log scale for stocks
        if asset_type.lower() == 'stocks':
            forecast['yhat'] = np.exp(forecast['yhat'])
            forecast['yhat_lower'] = np.exp(forecast['yhat_lower'])
            forecast['yhat_upper'] = np.exp(forecast['yhat_upper'])
            
            # Apply reasonable bounds to avoid extreme values
            current_price = prophet_df['y'].iloc[-1]
            if asset_type.lower() == 'stocks':
                current_price = np.exp(current_price)
                
            # Limit maximum forecast to 2x current price for stocks
            max_forecast = current_price * 2
            forecast['yhat'] = forecast['yhat'].clip(upper=max_forecast)
            forecast['yhat_upper'] = forecast['yhat_upper'].clip(upper=max_forecast * 1.2)
            
            # Limit minimum forecast to 0.5x current price for stocks
            min_forecast = current_price * 0.5
            forecast['yhat'] = forecast['yhat'].clip(lower=min_forecast)
            forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=min_forecast * 0.8)

        # Add actual values
        forecast['actual'] = np.nan
        if asset_type.lower() == 'stocks':
            actual_values = np.exp(prophet_df['y'].values)
        else:
            actual_values = prophet_df['y'].values
        forecast.loc[forecast['ds'].isin(prophet_df['ds']), 'actual'] = actual_values

        return forecast, None

    except Exception as e:
        logger.error(f"Error in prophet_forecast: {str(e)}")
        return None, str(e)

def display_common_metrics(data: pd.DataFrame, forecast: pd.DataFrame, asset_type: str = 'stocks'):
    """Display common metrics with proper scaling for stocks and cryptocurrencies"""
    try:
        st.subheader("📈 Price Metrics")
        
        # Ensure we have the required data
        if 'Close' not in data.columns:
            raise ValueError("Close price data not found in dataset")
            
        # Ensure data is properly formatted
        if isinstance(data['Close'], pd.Series):
            close_data = data['Close']
        else:
            close_data = data['Close'].iloc[:, 0]  # Take first column if DataFrame
            
        # Current price metrics
        current_price = float(close_data.iloc[-1])
        
        # Calculate price changes
        try:
            price_change_24h = float(close_data.pct_change().iloc[-1] * 100)
        except Exception:
            price_change_24h = 0.0
            
        try:
            price_change_7d = float(close_data.pct_change(periods=7).iloc[-1] * 100)
        except Exception:
            price_change_7d = 0.0
            
        # Calculate volatility
        try:
            volatility_30d = float(close_data.pct_change().rolling(window=30).std() * np.sqrt(252) * 100)
        except Exception:
            volatility_30d = 0.0

        # Get forecasted prices with bounds
        try:
            if forecast is not None:
                next_day_forecast = float(forecast['yhat'].iloc[-1])
                
                # Apply reasonable bounds for stocks
                if asset_type.lower() == 'stocks':
                    # Limit the forecast to a reasonable range
                    max_forecast = current_price * 2  # Maximum 100% increase
                    min_forecast = current_price * 0.5  # Maximum 50% decrease
                    next_day_forecast = np.clip(next_day_forecast, min_forecast, max_forecast)
                
                forecast_change = ((next_day_forecast / current_price) - 1) * 100
                
                # Clip the forecast change to reasonable bounds
                if asset_type.lower() == 'stocks':
                    forecast_change = np.clip(forecast_change, -50, 100)
            else:
                next_day_forecast = current_price
                forecast_change = 0.0
        except Exception as e:
            logger.error(f"Error calculating forecast metrics: {str(e)}")
            next_day_forecast = current_price
            forecast_change = 0.0
        
        # Display metrics in two rows
        col1, col2, col3 = st.columns(3)
        
        # First row - Current metrics
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
            st.metric(
                "30-Day Volatility",
                f"{volatility_30d:.2f}%"
            )

        # Second row - Forecast metrics
        st.subheader("🎯 Forecast Metrics")
        fcol1, fcol2, fcol3 = st.columns(3)
        
        with fcol1:
            st.metric(
                "Forecasted Price",
                f"${next_day_forecast:,.2f}",
                f"{forecast_change:+.2f}%"
            )
        
        with fcol2:
            if forecast is not None:
                upper_bound = min(float(forecast['yhat_upper'].iloc[-1]), 
                                current_price * (2 if asset_type.lower() == 'stocks' else 10))
                st.metric("Upper Bound", f"${upper_bound:,.2f}")
        
        with fcol3:
            if forecast is not None:
                lower_bound = max(float(forecast['yhat_lower'].iloc[-1]), 
                                current_price * (0.5 if asset_type.lower() == 'stocks' else 0.1))
                st.metric("Lower Bound", f"${lower_bound:,.2f}")

        logger.info(f"Metrics calculated successfully: Current=${current_price:.2f}, Forecast=${next_day_forecast:.2f}")

    except Exception as e:
        logger.error(f"Error displaying common metrics: {str(e)}")
        st.error(f"Error displaying common metrics: {str(e)}")
        logger.error(f"Data shape: {data.shape}")
        logger.error(f"Close column type: {type(data['Close'])}")

def display_confidence_analysis(forecast: pd.DataFrame):
    """Display confidence analysis of the forecast"""
    try:
        st.subheader("📊 Forecast Analysis")

        if forecast is not None:
            # Calculate confidence metrics
            confidence_width = (forecast['yhat_upper'] - forecast['yhat_lower']) / forecast['yhat'] * 100
            avg_confidence = 100 - confidence_width.mean()
            total_trend = ((forecast['yhat'].iloc[-1] / forecast['yhat'].iloc[0]) - 1) * 100
            
            # Calculate short-term and long-term predictions
            short_term_change = ((forecast['yhat'].iloc[-1] / forecast['yhat'].iloc[-2]) - 1) * 100
            long_term_trend = "Bullish" if total_trend > 0 else "Bearish"
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Forecast Confidence", f"{avg_confidence:.1f}%")
            with col2:
                st.metric("Short-term Change", f"{short_term_change:+.2f}%")
            with col3:
                st.metric("Long-term Trend", long_term_trend)

    except Exception as e:
        logger.error(f"Error displaying confidence analysis: {str(e)}")
        st.error(f"Error displaying confidence analysis: {str(e)}")

def display_crypto_metrics(data: pd.DataFrame, forecast: pd.DataFrame, symbol: str):
    """Display cryptocurrency-specific metrics with forecast integration"""
    try:
        st.subheader("🪙 Cryptocurrency Metrics")

        if 'Volume' in data.columns:
            col1, col2, col3 = st.columns(3)
            
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
            
            with col3:
                if forecast is not None:
                    price_volatility = ((forecast['yhat_upper'].iloc[-1] - forecast['yhat_lower'].iloc[-1]) 
                                     / forecast['yhat'].iloc[-1] * 100)
                    st.metric(
                        "Forecast Volatility",
                        f"{price_volatility:.2f}%"
                    )

    except Exception as e:
        logger.error(f"Error displaying crypto metrics: {str(e)}")
        st.error(f"Error displaying crypto metrics: {str(e)}")
        
        # Log the shape and types of data for debugging
        logger.error(f"Data shape: {data.shape}")
        logger.error(f"Close column type: {type(data['Close'])}")
        logger.error(f"Close column info: {data['Close'].info()}")

def display_confidence_analysis(forecast: pd.DataFrame):
    """Display confidence analysis of the forecast"""
    try:
        st.subheader("📊 Confidence Analysis")

        # Calculate confidence metrics
        confidence_width = (forecast['yhat_upper'] - forecast['yhat_lower']) / forecast['yhat'] * 100
        avg_confidence = 100 - confidence_width.mean()
        total_trend = ((forecast['yhat'].iloc[-1] / forecast['yhat'].iloc[0]) - 1) * 100
        
        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average Confidence", f"{avg_confidence:.1f}%")
        with col2:
            st.metric("Overall Trend", f"{total_trend:+.1f}%")

    except Exception as e:
        logger.error(f"Error displaying confidence analysis: {str(e)}")
        st.error(f"Error displaying confidence analysis: {str(e)}")

def display_crypto_metrics(data: pd.DataFrame, forecast: pd.DataFrame, symbol: str):
    """Display cryptocurrency-specific metrics"""
    try:
        st.subheader("🪙 Cryptocurrency Metrics")

        if 'Volume' in data.columns:
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
    """Display all metrics"""
    try:
        display_common_metrics(data, forecast)
        if asset_type.lower() == 'crypto':
            display_crypto_metrics(data, forecast, symbol)
        display_confidence_analysis(forecast)

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

    except Exception as e:
        logger.error(f"Error displaying economic indicators: {str(e)}")
        st.error(f"Error displaying economic indicators: {str(e)}")