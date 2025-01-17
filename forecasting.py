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
    """Prepare data for Prophet model with consistent preprocessing across assets"""
    try:
        df = data.copy()
        
        # Reset index if DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df.rename(columns={'index': 'ds'}, inplace=True)

        # Handle date column
        if 'ds' not in df.columns:
            date_cols = [col for col in df.columns if isinstance(col, str) and col.lower() in ['date', 'timestamp', 'time']]
            if date_cols:
                df.rename(columns={date_cols[0]: 'ds'}, inplace=True)
            else:
                df['ds'] = df.index

        # Set target variable
        df['y'] = df['Close']
        
        # Add common technical indicators for both stocks and crypto
        df['volatility'] = df['y'].pct_change().rolling(window=20).std()
        df['rsi'] = calculate_rsi(df['y'])
        df['ma_20'] = df['y'].rolling(window=20).mean()
        df['ma_50'] = df['y'].rolling(window=50).mean()
        
        if 'Volume' in df.columns:
            df['volume_ma'] = df['Volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_ma']
        
        # Add momentum indicators
        df['mom_1d'] = df['y'].pct_change(periods=1)
        df['mom_5d'] = df['y'].pct_change(periods=5)
        
        # Log transform the target variable for better scaling
        df['y'] = np.log1p(df['y'])
        
        # Ensure proper datetime format
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Drop NaN values and select required columns
        prophet_df = df.dropna(subset=['ds', 'y'])
        
        return prophet_df, df['y'].max()  # Return max_y for inverse transform

    except Exception as e:
        logger.error(f"Error in prepare_data_for_prophet: {str(e)}")
        raise

def calculate_rsi(prices, periods=14):
    """Calculate RSI indicator"""
    deltas = np.diff(prices)
    seed = deltas[:periods+1]
    up = seed[seed >= 0].sum()/periods
    down = -seed[seed < 0].sum()/periods
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:periods] = 100. - 100./(1. + rs)

    for i in range(periods, len(prices)):
        delta = deltas[i - 1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (periods - 1) + upval) / periods
        down = (down * (periods - 1) + downval) / periods
        rs = up/down
        rsi[i] = 100. - 100./(1. + rs)
    
    return rsi

def prophet_forecast(data: pd.DataFrame, periods: int, asset_type: str = 'stocks') -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Generate forecast using Prophet with consistent configuration across assets"""
    try:
        prophet_df, max_y = prepare_data_for_prophet(data, asset_type)
        
        # Common model configuration for both stocks and crypto
        model = Prophet(
            changepoint_prior_scale=0.01,
            seasonality_mode='multiplicative',
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True if asset_type.lower() == 'crypto' else False,
            interval_width=0.95,
            seasonality_prior_scale=0.1,
            changepoint_range=0.9,
            mcmc_samples=0
        )
        
        # Add custom seasonalities
        if asset_type.lower() == 'crypto':
            model.add_seasonality(
                name='hourly',
                period=24,
                fourier_order=5
            )
        else:
            model.add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=5
            )
            model.add_seasonality(
                name='quarterly',
                period=91.25,
                fourier_order=5
            )
        
        # Add common regressors
        if 'volume_ratio' in prophet_df.columns:
            model.add_regressor('volume_ratio', mode='multiplicative')
        if 'volatility' in prophet_df.columns:
            model.add_regressor('volatility', mode='multiplicative')
        if 'rsi' in prophet_df.columns:
            model.add_regressor('rsi', mode='multiplicative')

        # Fit model
        model.fit(prophet_df)
        
        # Generate future dataframe
        future = model.make_future_dataframe(
            periods=periods,
            freq='H' if asset_type.lower() == 'crypto' else 'D'
        )
        
        # Add regressor values to future dataframe
        for regressor in model.extra_regressors:
            future[regressor] = prophet_df[regressor].median()
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Inverse transform the predictions
        for column in ['yhat', 'yhat_lower', 'yhat_upper']:
            forecast[column] = np.expm1(forecast[column])
        
        # Add actual values
        forecast['actual'] = np.nan
        forecast.loc[forecast['ds'].isin(data.index), 'actual'] = data['Close'].values
        
        # Apply reasonable bounds to forecasted values
        current_price = data['Close'].iloc[-1]
        forecast['yhat'] = forecast['yhat'].clip(current_price * 0.1, current_price * 3)
        forecast['yhat_lower'] = forecast['yhat_lower'].clip(current_price * 0.05, current_price * 2)
        forecast['yhat_upper'] = forecast['yhat_upper'].clip(current_price * 0.15, current_price * 4)
        
        return forecast, None

    except Exception as e:
        logger.error(f"Error in prophet_forecast: {str(e)}")
        return None, str(e)

def create_forecast_plot(data: pd.DataFrame, forecast: pd.DataFrame, model_name: str, symbol: str) -> go.Figure:
    """Create an interactive plot with historical data and forecast"""
    try:
        # Create figure with secondary y-axis
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.03,
                           subplot_titles=(f'{symbol} Price Forecast', 'Volume & Indicators'),
                           row_heights=[0.7, 0.3])

        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='OHLC'
            ),
            row=1, col=1
        )

        # Add forecast line
        fig.add_trace(
            go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                name='Forecast',
                line=dict(color='blue', dash='dash')
            ),
            row=1, col=1
        )

        # Add confidence intervals
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

        # Add volume bars
        if 'Volume' in data.columns:
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color='rgba(0,0,255,0.3)'
                ),
                row=2, col=1
            )

        # Update layout
        fig.update_layout(
            title=f'{symbol} Price Forecast',
            yaxis_title='Price',
            yaxis2_title='Volume',
            xaxis_rangeslider_visible=False,
            height=800
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating forecast plot: {str(e)}")
        return None

def display_metrics(data: pd.DataFrame, forecast: pd.DataFrame, asset_type: str, symbol: str):
    """Display metrics with reasonable bounds"""
    try:
        current_price = float(data['Close'].iloc[-1])
        next_day_forecast = float(forecast['yhat'].iloc[-1])
        
        # Calculate bounded percent change
        forecast_change = ((next_day_forecast / current_price) - 1) * 100
        forecast_change = np.clip(forecast_change, -50, 50)  # Limit to ±50%
        
        # Display metrics with bounded values
        st.subheader("📈 Price Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Current Price",
                f"${current_price:,.2f}",
                f"{data['Close'].pct_change().iloc[-1]*100:+.2f}%"
            )
        
        with col2:
            st.metric(
                "Forecasted Price",
                f"${next_day_forecast:,.2f}",
                f"{forecast_change:+.2f}%"
            )
        
        with col3:
            volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
            st.metric("Annual Volatility", f"{volatility:.2f}%")
        
        # Display bounded confidence intervals
        st.subheader("🎯 Forecast Bounds")
        fcol1, fcol2 = st.columns(2)
        
        with fcol1:
            upper_bound = float(forecast['yhat_upper'].iloc[-1])
            upper_bound = min(upper_bound, current_price * 2)
            st.metric("Upper Bound", f"${upper_bound:,.2f}")
        
        with fcol2:
            lower_bound = float(forecast['yhat_lower'].iloc[-1])
            lower_bound = max(lower_bound, current_price * 0.5)
            st.metric("Lower Bound", f"${lower_bound:,.2f}")

        # Display additional metrics based on asset type
        if asset_type.lower() == 'crypto':
            display_crypto_metrics(data, forecast, symbol)
        else:
            display_stock_metrics(data, forecast, symbol)

    except Exception as e:
        logger.error(f"Error displaying metrics: {str(e)}")
        st.error(f"Error displaying metrics: {str(e)}")

def display_crypto_metrics(data: pd.DataFrame, forecast: pd.DataFrame, symbol: str):
    """Display cryptocurrency-specific metrics"""
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
                if 'rsi' in data.columns:
                    rsi_value = float(data['rsi'].iloc[-1])
                    st.metric(
                        "RSI",
                        f"{rsi_value:.1f}",
                        "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
                    )

    except Exception as e:
        logger.error(f"Error displaying crypto metrics: {str(e)}")
        st.error(f"Error displaying crypto metrics: {str(e)}")

def display_stock_metrics(data: pd.DataFrame, forecast: pd.DataFrame, symbol: str):
    """Display stock-specific metrics"""
    try:
        st.subheader("📊 Stock Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'rsi' in data.columns:
                rsi_value = float(data['rsi'].iloc[-1])
                st.metric(
                    "RSI",
                    f"{rsi_value:.1f}",
                    "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral"
                )
        
        with col2:
            if 'ma_20' in data.columns and 'ma_50' in data.columns:
                ma_cross = "Bullish" if data['ma_20'].iloc[-1] > data['ma_50'].iloc[-1] else "Bearish"
                st.metric("MA Cross", ma_cross)
        
        with col3:
            if 'volume_ratio' in data.columns:
                vol_ratio = float(data['volume_ratio'].iloc[-1])
                st.metric(
                    "Volume Ratio",
                    f"{vol_ratio:.2f}",
                    "Above Average" if vol_ratio > 1 else "Below Average"
                )

    except Exception as e:
        logger.error(f"Error displaying stock metrics: {str(e)}")
        st.error(f"Error displaying stock metrics: {str(e)}")

def display_forecast_analysis(forecast: pd.

def display_forecast_analysis(forecast: pd.DataFrame):
    """Display detailed analysis of the forecast"""
    try:
        st.subheader("📊 Forecast Analysis")

        # Calculate trend metrics
        short_term_trend = ((forecast['yhat'].iloc[-1] / forecast['yhat'].iloc[-2]) - 1) * 100
        long_term_trend = ((forecast['yhat'].iloc[-1] / forecast['yhat'].iloc[0]) - 1) * 100
        
        # Calculate confidence metrics
        confidence_width = (forecast['yhat_upper'] - forecast['yhat_lower']) / forecast['yhat'] * 100
        avg_confidence = 100 - confidence_width.mean()
        
        # Display trend analysis
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Short-term Trend",
                f"{short_term_trend:+.2f}%",
                "Upward" if short_term_trend > 0 else "Downward"
            )
        
        with col2:
            st.metric(
                "Long-term Trend",
                f"{long_term_trend:+.2f}%",
                "Upward" if long_term_trend > 0 else "Downward"
            )
        
        with col3:
            st.metric(
                "Forecast Confidence",
                f"{avg_confidence:.1f}%"
            )

    except Exception as e:
        logger.error(f"Error displaying forecast analysis: {str(e)}")
        st.error(f"Error displaying forecast analysis: {str(e)}")

def display_technical_analysis(data: pd.DataFrame):
    """Display technical analysis indicators"""
    try:
        st.subheader("📈 Technical Analysis")
        
        # Calculate additional technical indicators
        rsi = data['rsi'].iloc[-1] if 'rsi' in data.columns else None
        ma_20 = data['ma_20'].iloc[-1] if 'ma_20' in data.columns else None
        ma_50 = data['ma_50'].iloc[-1] if 'ma_50' in data.columns else None
        current_price = data['Close'].iloc[-1]
        
        # Display technical indicators
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if rsi is not None:
                rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                st.metric("RSI", f"{rsi:.1f}", rsi_status)
        
        with col2:
            if ma_20 is not None and ma_50 is not None:
                ma_cross = "Golden Cross" if ma_20 > ma_50 else "Death Cross"
                ma_diff = ((ma_20 / ma_50) - 1) * 100
                st.metric("MA Cross", ma_cross, f"{ma_diff:+.2f}%")
        
        with col3:
            if 'volatility' in data.columns:
                volatility = data['volatility'].iloc[-1] * np.sqrt(252) * 100
                st.metric("Volatility", f"{volatility:.2f}%")

    except Exception as e:
        logger.error(f"Error displaying technical analysis: {str(e)}")
        st.error(f"Error displaying technical analysis: {str(e)}")

def display_volume_analysis(data: pd.DataFrame):
    """Display volume analysis"""
    try:
        if 'Volume' in data.columns:
            st.subheader("📊 Volume Analysis")
            
            # Calculate volume metrics
            current_volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Current Volume",
                    f"{current_volume:,.0f}",
                    f"{((current_volume/data['Volume'].iloc[-2])-1)*100:+.2f}%"
                )
            
            with col2:
                st.metric(
                    "Average Volume (20D)",
                    f"{avg_volume:,.0f}"
                )
            
            with col3:
                st.metric(
                    "Volume Ratio",
                    f"{volume_ratio:.2f}",
                    "Above Average" if volume_ratio > 1 else "Below Average"
                )

    except Exception as e:
        logger.error(f"Error displaying volume analysis: {str(e)}")
        st.error(f"Error displaying volume analysis: {str(e)}")

def generate_trading_signals(data: pd.DataFrame, forecast: pd.DataFrame) -> Dict[str, str]:
    """Generate trading signals based on technical analysis and forecast"""
    try:
        signals = {}
        current_price = data['Close'].iloc[-1]
        forecasted_price = forecast['yhat'].iloc[-1]
        
        # Trend signals
        signals['trend'] = "Bullish" if forecasted_price > current_price else "Bearish"
        
        # RSI signals
        if 'rsi' in data.columns:
            rsi = data['rsi'].iloc[-1]
            if rsi > 70:
                signals['rsi'] = "Overbought"
            elif rsi < 30:
                signals['rsi'] = "Oversold"
            else:
                signals['rsi'] = "Neutral"
        
        # Moving average signals
        if 'ma_20' in data.columns and 'ma_50' in data.columns:
            ma_20 = data['ma_20'].iloc[-1]
            ma_50 = data['ma_50'].iloc[-1]
            signals['ma_cross'] = "Golden Cross" if ma_20 > ma_50 else "Death Cross"
        
        # Volume signals
        if 'volume_ratio' in data.columns:
            vol_ratio = data['volume_ratio'].iloc[-1]
            signals['volume'] = "High" if vol_ratio > 1.5 else "Low" if vol_ratio < 0.5 else "Normal"
        
        return signals

    except Exception as e:
        logger.error(f"Error generating trading signals: {str(e)}")
        return {}

def display_trading_signals(signals: Dict[str, str]):
    """Display trading signals"""
    try:
        st.subheader("🎯 Trading Signals")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Technical Signals")
            for signal_type, value in signals.items():
                if signal_type in ['trend', 'rsi', 'ma_cross']:
                    st.write(f"**{signal_type.replace('_', ' ').title()}:** {value}")
        
        with col2:
            st.markdown("### Volume Signals")
            if 'volume' in signals:
                st.write(f"**Volume Activity:** {signals['volume']}")

    except Exception as e:
        logger.error(f"Error displaying trading signals: {str(e)}")
        st.error(f"Error displaying trading signals: {str(e)}")

def read_economic_indicators():
    """Read and process economic indicators"""
    try:
        # Placeholder for economic indicators integration
        return None
    except Exception as e:
        logger.error(f"Error reading economic indicators: {str(e)}")
        return None

def display_economic_indicators(economic_data: Optional[pd.DataFrame], symbol: str):
    """Display economic indicators"""
    try:
        if economic_data is not None:
            st.subheader("📊 Economic Indicators")
            # Add economic indicators display logic here
            pass
    except Exception as e:
        logger.error(f"Error displaying economic indicators: {str(e)}")
        st.error(f"Error displaying economic indicators: {str(e)}")

# Main function to run the forecasting system
def run_forecasting_system(data: pd.DataFrame, symbol: str, asset_type: str = 'stocks', periods: int = 30):
    """Main function to run the entire forecasting system"""
    try:
        # Generate forecast
        forecast, error = prophet_forecast(data, periods, asset_type)
        
        if error:
            st.error(f"Error generating forecast: {error}")
            return
        
        # Create and display forecast plot
        fig = create_forecast_plot(data, forecast, "Prophet", symbol)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Display metrics and analysis
        display_metrics(data, forecast, asset_type, symbol)
        display_forecast_analysis(forecast)
        display_technical_analysis(data)
        display_volume_analysis(data)
        
        # Generate and display trading signals
        signals = generate_trading_signals(data, forecast)
        display_trading_signals(signals)
        
        # Display economic indicators if available
        economic_data = read_economic_indicators()
        if economic_data is not None:
            display_economic_indicators(economic_data, symbol)

    except Exception as e:
        logger.error(f"Error in forecasting system: {str(e)}")
        st.error(f"Error in forecasting system: {str(e)}")