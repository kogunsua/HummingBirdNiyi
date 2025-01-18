# forecasting.py
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

def prepare_data_for_prophet(data: pd.DataFrame, asset_type: str = 'stocks') -> pd.DataFrame:
    """Prepare data for Prophet model with asset-specific handling"""
    try:
        logger.info("Starting data preparation for Prophet")
        logger.info(f"Input data shape: {data.shape}")
        logger.info(f"Input data columns: {data.columns.tolist()}")

        # Make a copy of the data
        df = data.copy()

        if asset_type.lower() == 'stocks':
            # Stock-specific preprocessing
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                df.rename(columns={'index': 'ds'}, inplace=True)
            elif 'Date' in df.columns:
                df = df.rename(columns={'Date': 'ds'})
            elif 'timestamp' in df.columns:
                df = df.rename(columns={'timestamp': 'ds'})
            
            # Ensure Close price is named 'y' for Prophet
            df = df.rename(columns={'Close': 'y'})
            
            # Ensure datetime is timezone naive
            df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
            
            # Select only required columns
            df = df[['ds', 'y']]
        else:
            # Existing crypto preprocessing logic
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

        # Common preprocessing steps
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = df['y'].astype(float)
        prophet_df = df[['ds', 'y']].sort_values('ds').reset_index(drop=True)
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

def add_stock_specific_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add stock-specific indicators"""
    try:
        # Volume Analysis
        df['volume_ma'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma']
        
        # Price Momentum
        df['momentum'] = df['Close'].pct_change(periods=20)
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + (df['Close'].rolling(window=20).std() * 2)
        df['BB_lower'] = df['BB_middle'] - (df['Close'].rolling(window=20).std() * 2)
        
        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        return df
        
    except Exception as e:
        logger.error(f"Error adding stock indicators: {str(e)}")
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
                     indicator: Optional[str] = None, asset_type: str = 'stocks') -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Generate forecast using Prophet model with asset-specific handling"""
    try:
        if data is None or data.empty:
            return None, "No data provided for forecasting"

        # Prepare data for Prophet with asset-specific handling
        prophet_df = prepare_data_for_prophet(data, asset_type)
        
        if prophet_df is None or prophet_df.empty:
            return None, "No valid data for forecasting after preparation"

        # Initialize Prophet with asset-specific parameters
        if asset_type.lower() == 'stocks':
            model = Prophet(
                changepoint_prior_scale=0.05,
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True
            )
            
            # Add stock-specific seasonality
            model.add_seasonality(
                name='quarterly',
                period=91.25,
                fourier_order=5
            )
        else:
            # Crypto-specific parameters
            model = Prophet(
                changepoint_prior_scale=0.05,
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False
            )
            
            # Add crypto-specific seasonality
            model.add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=5
            )

        # Add economic indicators if provided
        if economic_data is not None and indicator is not None:
            economic_df = economic_data.copy()
            economic_df.columns = ['ds', 'regressor']
            economic_df['ds'] = pd.to_datetime(economic_df['ds']).dt.tz_localize(None)
            model.add_regressor('regressor')
            prophet_df = prophet_df.merge(economic_df, on='ds', how='left')
            prophet_df['regressor'].fillna(method='ffill', inplace=True)

        # Fit model and make forecast
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=periods)
        
        if economic_data is not None and indicator is not None:
            future = future.merge(economic_df, on='ds', how='left')
            future['regressor'].fillna(method='ffill', inplace=True)
            
        forecast = model.predict(future)
        forecast['actual'] = np.nan
        forecast.loc[forecast['ds'].isin(prophet_df['ds']), 'actual'] = prophet_df['y'].values

        return forecast, None

    except Exception as e:
        logger.error(f"Error in prophet_forecast: {str(e)}")
        return None, str(e)
        
def create_forecast_plot(data: pd.DataFrame, forecast: pd.DataFrame, model_name: str, symbol: str, asset_type: str = 'stocks') -> go.Figure:
    """Create an interactive plot with historical data and forecast"""
    try:
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
        if isinstance(data.index, pd.DatetimeIndex):
            historical_dates = data.index
        else:
            historical_dates = pd.to_datetime(data.index)
            
        # Price data
        fig.add_trace(
            go.Scatter(
                x=historical_dates,
                y=data['Close'],
                name='Historical',
                line=dict(color='blue')
            ),
            row=1, col=1
        )

        # Forecast line
        fig.add_trace(
            go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                name=f'{model_name} Forecast',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )

        # Confidence intervals
        fig.add_trace(
            go.Scatter(
                x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
                y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
                fill='toself',
                fillcolor='rgba(0,100,255,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval'
            ),
            row=1, col=1
        )

        # Volume
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

        # Add technical indicators
        if asset_type.lower() == 'stocks':
            # Bollinger Bands
            if 'BB_middle' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=historical_dates,
                        y=data['BB_middle'],
                        name='BB Middle',
                        line=dict(color='gray', dash='dash')
                    ),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=historical_dates,
                        y=data['BB_upper'],
                        name='BB Upper',
                        line=dict(color='lightgray')
                    ),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=historical_dates,
                        y=data['BB_lower'],
                        name='BB Lower',
                        line=dict(color='lightgray')
                    ),
                    row=3, col=1
                )
        else:
            # Crypto-specific indicators
            if 'hourly_volatility' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=historical_dates,
                        y=data['hourly_volatility'],
                        name='Hourly Volatility',
                        line=dict(color='orange')
                    ),
                    row=3, col=1
                )

        # Update layout
        fig.update_layout(
            title=f'{symbol} Price Forecast and Analysis',
            yaxis_title='Price ($)',
            yaxis2_title='Volume',
            yaxis3_title='Indicators',
            height=1000,
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
        
def display_stock_metrics(data: pd.DataFrame, forecast: pd.DataFrame, symbol: str):
    """Display stock-specific metrics"""
    try:
        st.subheader("📈 Stock Market Metrics")

        # Stock-specific metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'ATR' in data.columns:
                atr = float(data['ATR'].iloc[-1])
                atr_change = float(data['ATR'].pct_change().iloc[-1] * 100)
                st.metric(
                    "Average True Range",
                    f"${atr:.2f}",
                    f"{atr_change:+.2f}%"
                )

        with col2:
            if 'BB_middle' in data.columns:
                current_price = float(data['Close'].iloc[-1])
                bb_mid = float(data['BB_middle'].iloc[-1])
                bb_position = ((current_price - bb_mid) / bb_mid) * 100
                st.metric(
                    "BB Position",
                    f"{bb_position:+.2f}%",
                    "Above Middle" if bb_position > 0 else "Below Middle"
                )

        with col3:
            if 'momentum' in data.columns:
                momentum = float(data['momentum'].iloc[-1] * 100)
                st.metric(
                    "Price Momentum",
                    f"{momentum:+.2f}%"
                )

    except Exception as e:
        logger.error(f"Error displaying stock metrics: {str(e)}")
        st.error(f"Error displaying stock metrics: {str(e)}")

def display_common_metrics(data: pd.DataFrame, forecast: pd.DataFrame):
    """Display common metrics for both stocks and cryptocurrencies"""
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

        # Get forecasted prices
        try:
            if forecast is not None:
                next_day_forecast = float(forecast['yhat'].iloc[-1])
                forecast_change = ((next_day_forecast / current_price) - 1) * 100
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
                upper_bound = float(forecast['yhat_upper'].iloc[-1])
                st.metric("Upper Bound", f"${upper_bound:,.2f}")
        
        with fcol3:
            if forecast is not None:
                lower_bound = float(forecast['yhat_lower'].iloc[-1])
                st.metric("Lower Bound", f"${lower_bound:,.2f}")

    except Exception as e:
        logger.error(f"Error displaying common metrics: {str(e)}")
        st.error(f"Error displaying common metrics: {str(e)}")
        
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
    """Display cryptocurrency-specific metrics"""
    try:
        st.subheader("🪙 Cryptocurrency Metrics")

        # Crypto-specific metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'hourly_volatility' in data.columns:
                volatility = float(data['hourly_volatility'].iloc[-1] * 100)
                volatility_change = float(data['hourly_volatility'].pct_change().iloc[-1] * 100)
                st.metric(
                    "Hourly Volatility",
                    f"{volatility:.2f}%",
                    f"{volatility_change:+.2f}%"
                )

        with col2:
            if 'volume_ratio' in data.columns:
                volume_ratio = float(data['volume_ratio'].iloc[-1])
                volume_ratio_change = float(data['volume_ratio'].pct_change().iloc[-1] * 100)
                st.metric(
                    "Volume Ratio",
                    f"{volume_ratio:.2f}",
                    f"{volume_ratio_change:+.2f}%"
                )

        with col3:
            if 'market_dominance' in data.columns:
                dominance = float(data['market_dominance'].iloc[-1] * 100)
                st.metric(
                    "Market Dominance",
                    f"{dominance:.2f}%"
                )

        # Network metrics if available
        if 'network_transactions' in data.columns and 'active_addresses' in data.columns:
            ncol1, ncol2 = st.columns(2)
            
            with ncol1:
                transactions = int(data['network_transactions'].iloc[-1])
                st.metric(
                    "Network Transactions",
                    f"{transactions:,d}"
                )
            
            with ncol2:
                addresses = int(data['active_addresses'].iloc[-1])
                st.metric(
                    "Active Addresses",
                    f"{addresses:,d}"
                )

    except Exception as e:
        logger.error(f"Error displaying crypto metrics: {str(e)}")
        st.error(f"Error displaying crypto metrics: {str(e)}")
        
def display_metrics(data: pd.DataFrame, forecast: pd.DataFrame, asset_type: str, symbol: str):
    """Display all metrics based on asset type"""
    try:
        display_common_metrics(data, forecast)
        if asset_type.lower() == 'stocks':
            display_stock_metrics(data, forecast, symbol)
        else:
            display_crypto_metrics(data, forecast, symbol)
        display_confidence_analysis(forecast)

    except Exception as e:
        logger.error(f"Error displaying metrics: {str(e)}")
        st.error(f"Error displaying metrics: {str(e)}")

def display_economic_indicators(data: pd.DataFrame, indicator: str, economic_indicators: object):
    """Display economic indicator information and analysis"""
    try:
        if data is None or indicator is None:
            return
            
        st.subheader("📊 Economic Indicator Analysis")
        
        indicator_info = economic_indicators.get_indicator_info(indicator)
        if not indicator_info:
            return
            
        # Display indicator metadata
        st.markdown(f"""
            **Indicator:** {indicator_info.get('description', indicator)}  
            **Frequency:** {indicator_info.get('frequency', 'N/A')}  
            **Units:** {indicator_info.get('units', 'N/A')}
        """)
        
        # Calculate and display current statistics
        if not data.empty:
            current_value = data['value'].iloc[-1]
            change_1d = (data['value'].iloc[-1] / data['value'].iloc[-2] - 1) * 100 if len(data) > 1 else 0
            change_30d = (data['value'].iloc[-1] / data['value'].iloc[-30] - 1) * 100 if len(data) >= 30 else None
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Value", f"{current_value:.2f}")
            with col2:
                st.metric("24h Change", f"{change_1d:+.2f}%")
            with col3:
                if change_30d is not None:
                    st.metric("30d Change", f"{change_30d:+.2f}%")

    except Exception as e:
        logger.error(f"Error displaying economic indicators: {str(e)}")
        st.error(f"Error displaying economic indicators: {str(e)}")
        
