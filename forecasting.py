#forcasting.py
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
    """Add cryptocurrency-specific indicators"""
    try:
        df['volume_ma'] = df['Volume'].rolling(window=24).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma']
        df['hourly_volatility'] = df['Close'].pct_change().rolling(window=24).std()
        df['volatility_ratio'] = df['hourly_volatility'] / df['hourly_volatility'].rolling(window=168).mean()
        df['market_dominance'] = 0.5
        df['network_transactions'] = 0.5
        df['active_addresses'] = 0.5
        return df
    except Exception as e:
        logger.error(f"Error adding crypto indicators: {str(e)}")
        return df

def add_technical_indicators(df: pd.DataFrame, asset_type: str = 'stocks') -> pd.DataFrame:
    """Add technical indicators based on asset type"""
    try:
        config = AssetConfig.get_config(asset_type)
        indicators = config['indicators']
        df['MA5'] = df['Close'].rolling(window=indicators['ma_periods'][0]).mean()
        df['MA20'] = df['Close'].rolling(window=indicators['ma_periods'][1]).mean()
        df['MA50'] = df['Close'].rolling(window=indicators['ma_periods'][2]).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=indicators['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=indicators['rsi_period']).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
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

###########################################
# Section 4: Core Forecasting Functions
###########################################

def prophet_forecast(data: pd.DataFrame, periods: int, economic_data: Optional[pd.DataFrame] = None,
                     indicator: Optional[str] = None, asset_type: str = 'stocks') -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Generate forecast using Prophet model with proper scaling for stocks"""
    try:
        if data is None or data.empty:
            return None, "No data provided for forecasting"

        prophet_df = prepare_data_for_prophet(data)
        if prophet_df is None or prophet_df.empty:
            return None, "No valid data for forecasting after preparation"

        max_price = prophet_df['y'].max()
        if asset_type.lower() == 'stocks':
            prophet_df['y'] = np.log(prophet_df['y'])

        if asset_type.lower() == 'stocks':
            model = Prophet(
                changepoint_prior_scale=0.01,
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                changepoint_range=0.8,
                interval_width=0.95
            )
        else:
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

        if asset_type.lower() == 'stocks':
            forecast['yhat'] = np.exp(forecast['yhat'])
            forecast['yhat_lower'] = np.exp(forecast['yhat_lower'])
            forecast['yhat_upper'] = np.exp(forecast['yhat_upper'])
            current_price = np.exp(prophet_df['y'].iloc[-1])
            max_forecast = current_price * 2
            min_forecast = current_price * 0.5
            forecast['yhat'] = forecast['yhat'].clip(lower=min_forecast, upper=max_forecast)
            forecast['yhat_upper'] = forecast['yhat_upper'].clip(upper=max_forecast * 1.2)
            forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=min_forecast * 0.8)

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

###########################################
# Section 5: Visualization Functions
###########################################

def create_forecast_plot(data: pd.DataFrame, forecast: pd.DataFrame, model_name: str, symbol: str) -> go.Figure:
    """Create an interactive plot with historical data and forecast"""
    try:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.03, row_heights=[0.7, 0.3],
                           subplot_titles=(f'{symbol} Price Forecast', 'Confidence Analysis'))

        if isinstance(data.index, pd.DatetimeIndex):
            historical_dates = data.index
        else:
            historical_dates = pd.to_datetime(data.index)

        fig.add_trace(
            go.Scatter(
                x=historical_dates,
                y=data['Close'],
                name='Historical',
                line=dict(color='blue')
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                name=f'{model_name} Forecast',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )

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

        daily_returns = data['Close'].pct_change()
        volatility = daily_returns.rolling(window=30).std() * np.sqrt(252) * 100

        fig.add_trace(
            go.Scatter(
                x=historical_dates,
                y=volatility,
                name='30-Day Volatility',
                line=dict(color='orange')
            ),
            row=2, col=1
        )

        fig.update_layout(
            title=f'{symbol} Price Forecast',
            yaxis_title='Price ($)',
            yaxis2_title='Volatility (%)',
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)

        return fig

    except Exception as e:
        logger.error(f"Error creating forecast plot: {str(e)}")
        return None

###########################################
# Section 6: Metrics Display Functions
###########################################

def display_common_metrics(data: pd.DataFrame, forecast: pd.DataFrame, asset_type: str = 'stocks'):
    """Display common metrics with proper scaling for stocks and cryptocurrencies"""
    try:
        st.subheader("📈 Price Metrics")

        if 'Close' not in data.columns:
            raise ValueError("Close price data not found in dataset")

        close_data = data['Close'] if isinstance(data['Close'], pd.Series) else data['Close'].iloc[:, 0]

        try:
            current_price = float(close_data.iloc[-1])
            price_change_24h = float(close_data.pct_change().iloc[-1] * 100)
            price_change_7d = float(close_data.pct_change(periods=7).iloc[-1] * 100)
            volatility_30d = float(close_data.pct_change().rolling(window=30).std() * np.sqrt(252) * 100)
        except Exception:
            current_price = float(close_data.iloc[-1])
            price_change_24h = price_change_7d = volatility_30d = 0.0

        try:
            if forecast is not None:
                next_day_forecast = float(forecast['yhat'].iloc[-1])
                if asset_type.lower() == 'stocks':
                    max_forecast = current_price * 2
                    min_forecast = current_price * 0.5
                    next_day_forecast = np.clip(next_day_forecast, min_forecast, max_forecast)

                forecast_change = ((next_day_forecast / current_price) - 1) * 100
                if asset_type.lower() == 'stocks':
                    forecast_change = np.clip(forecast_change, -50, 100)
            else:
                next_day_forecast = current_price
                forecast_change = 0.0
        except Exception as e:
            logger.error(f"Error calculating forecast metrics: {str(e)}")
            next_day_forecast = current_price
            forecast_change = 0.0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${current_price:,.2f}", f"{price_change_24h:+.2f}%")
        with col2:
            st.metric("7-Day Change", f"{price_change_7d:+.2f}%")
        with col3:
            st.metric("30-Day Volatility", f"{volatility_30d:.2f}%")

        st.subheader("🎯 Forecast Metrics")
        fcol1, fcol2, fcol3 = st.columns(3)

        with fcol1:
            st.metric("Forecasted Price", f"${next_day_forecast:,.2f}", f"{forecast_change:+.2f}%")

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

    except Exception as e:
        logger.error(f"Error displaying common metrics: {str(e)}")
        st.error(f"Error displaying common metrics: {str(e)}")

def display_confidence_analysis(forecast: pd.DataFrame, asset_type: str = 'stocks'):
    """Display confidence analysis of the forecast with proper scaling"""
    try:
        st.subheader("📊 Forecast Analysis")

        if forecast is not None:
            recent_actual = forecast[forecast['actual'].notna()]['actual'].iloc[-1]

            confidence_width = (forecast['yhat_upper'] - forecast['yhat_lower']) / forecast['yhat'] * 100
            if asset_type.lower() == 'stocks':
                confidence_width = confidence_width.clip(upper=100)

            avg_confidence = 100 - min(confidence_width.mean(), 90)

            if asset_type.lower() == 'stocks':
                total_trend = min(((forecast['yhat'].iloc[-1] / forecast['yhat'].iloc[0]) - 1) * 100, 100)
                short_term_change = min(((forecast['yhat'].iloc[-1] / forecast['yhat'].iloc[-2]) - 1) * 100, 50)
            else:
                total_trend = ((forecast['yhat'].iloc[-1] / forecast['yhat'].iloc[0]) - 1) * 100
                short_term_change = ((forecast['yhat'].iloc[-1] / forecast['yhat'].iloc[-2]) - 1) * 100

            trend_strength = abs(total_trend)
            trend_description = "Neutral" if trend_strength < 10 else f"Strong {'Bullish' if total_trend > 0 else 'Bearish'}"

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Forecast Confidence", f"{avg_confidence:.1f}%",
                         help="Higher values indicate more confident predictions")
            with col2:
                st.metric("Short-term Change", f"{short_term_change:+.2f}%",
                         help="Expected price change in the next period")
            with col3:
                st.metric("Trend Strength", trend_description,
                         help="Overall trend direction and strength")
            with col4:
                st.metric("Forecast Volatility", f"{confidence_width.mean():.2f}%",
                         help="Expected price volatility")

            support_level = max(forecast['yhat_lower'].iloc[-1],
                              recent_actual * (0.5 if asset_type.lower() == 'stocks' else 0.1))
            resistance_level = min(forecast['yhat_upper'].iloc[-1],
                                 recent_actual * (2 if asset_type.lower() == 'stocks' else 10))

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Support Level", f"${support_level:,.2f}",
                         help="Predicted price floor")
            with col2:
                st.metric("Resistance Level", f"${resistance_level:,.2f}",
                         help="Predicted price ceiling")

            st.markdown(f"""
            ### Trend Analysis Summary
            - **Overall Trend:** The asset is showing a {trend_description.lower()} trend with a {abs(total_trend):.1f}% total expected move
            - **Confidence Level:** The model's predictions show {avg_confidence:.1f}% confidence
            - **Volatility Expectation:** Expected price volatility of {confidence_width.mean():.2f}%
            - **Price Range:** The price is expected to stay between ${support_level:,.2f} and ${resistance_level:,.2f}
            """)

    except Exception as e:
        logger.error(f"Error displaying confidence analysis: {str(e)}")
        st.error(f"Error displaying confidence analysis: {str(e)}")

def display_stock_metrics(data: pd.DataFrame, forecast: pd.DataFrame, symbol: str):
    """Display stock-specific metrics"""
    try:
        st.subheader("📊 Stock-Specific Metrics")

        try:
            beta = None
            if 'Market' in data.columns:
                beta = data['Close'].pct_change().cov(data['Market'].pct_change()) / data['Market'].pct_change().var()

            ma50 = data['Close'].rolling(window=50).mean().iloc[-1]
            ma200 = data['Close'].rolling(window=200).mean().iloc[-1
            
            def display_stock_metrics(data: pd.DataFrame, forecast: pd.DataFrame, symbol: str):
    """Display stock-specific metrics"""
    try:
        st.subheader("📊 Stock-Specific Metrics")

        try:
            beta = None
            if 'Market' in data.columns:
                beta = data['Close'].pct_change().cov(data['Market'].pct_change()) / data['Market'].pct_change().var()

            ma50 = data['Close'].rolling(window=50).mean().iloc[-1]
            ma200 = data['Close'].rolling(window=200).mean().iloc[-1]
            avg_volume = data['Volume'].mean() if 'Volume' in data.columns else None
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if beta is not None:
                    st.metric("Beta", f"{beta:.2f}",
                             help="Stock's volatility compared to the market")
            with col2:
                st.metric("MA50 vs MA200", "Above" if ma50 > ma200 else "Below",
                         help="50-day MA compared to 200-day MA")
            with col3:
                if avg_volume is not None:
                    st.metric("Avg Daily Volume", f"{avg_volume:,.0f}",
                             help="Average daily trading volume")
                    
        except Exception as e:
            logger.error(f"Error calculating stock metrics: {str(e)}")
            st.warning("Some stock-specific metrics could not be calculated")

    except Exception as e:
        logger.error(f"Error displaying stock metrics: {str(e)}")
        st.error(f"Error displaying stock metrics: {str(e)}")


def display_crypto_metrics(data: pd.DataFrame, forecast: pd.DataFrame, symbol: str):
    """Display cryptocurrency-specific metrics"""
    try:
        st.subheader("🪙 Cryptocurrency Metrics")

        if 'Volume' in data.columns:
            col1, col2, col3 = st.columns(3)
            
            try:
                volume = float(data['Volume'].iloc[-1])
                volume_change = float(data['Volume'].pct_change().iloc[-1] * 100)
                
                with col1:
                    st.metric("24h Volume", f"${volume:,.0f}", f"{volume_change:+.2f}%")
                if 'volume_ratio' in data.columns:
                    with col2:
                        vol_ratio = float(data['volume_ratio'].iloc[-1])
                        st.metric("Volume Ratio", f"{vol_ratio:.2f}",
                                 "Above Average" if vol_ratio > 1 else "Below Average")
                if forecast is not None:
                    with col3:
                        price_volatility = ((forecast['yhat_upper'].iloc[-1] - forecast['yhat_lower'].iloc[-1]) 
                                         / forecast['yhat'].iloc[-1] * 100)
                        st.metric("Forecast Volatility", f"{price_volatility:.2f}%")
            except Exception as e:
                logger.error(f"Error calculating crypto metrics: {str(e)}")
                st.warning("Some cryptocurrency metrics could not be calculated")

    except Exception as e:
        logger.error(f"Error displaying crypto metrics: {str(e)}")
        st.error(f"Error displaying crypto metrics: {str(e)}")


def display_metrics(data: pd.DataFrame, forecast: pd.DataFrame, asset_type: str, symbol: str):
    """Display all metrics with asset-specific handling"""
    try:
        # Validate inputs
        if data is None or data.empty:
            raise ValueError("No data provided for metrics display")
            
        if 'Close' not in data.columns:
            raise ValueError("Close price data not found in dataset")
            
        # Display metrics with proper asset type
        display_common_metrics(data, forecast, asset_type)
        
        # Display asset-specific metrics
        if asset_type.lower() == 'crypto':
            display_crypto_metrics(data, forecast, symbol)
        else:
            display_stock_metrics(data, forecast, symbol)
            
        # Display confidence analysis
        if forecast is not None and not forecast.empty:
            display_confidence_analysis(forecast, asset_type)
        else:
            st.warning("No forecast data available for confidence analysis")

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

# Version information
__version__ = '1.0.0'