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

        # Validate the prepared data
        if prophet_df.empty:
            raise ValueError("Prepared DataFrame is empty")
        if not np.isfinite(prophet_df['y']).all():
            raise ValueError("Data contains non-finite values")
        if prophet_df['ds'].isna().any():
            raise ValueError("Date column contains missing values")

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

def prophet_forecast(data: pd.DataFrame, periods: int, economic_data: Optional[pd.DataFrame] = None,
                     indicator: Optional[str] = None, asset_type: str = 'stocks') -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Generate forecast using Prophet model with economic and sentiment data"""
    try:
        logger.info(f"Starting forecast for {periods} periods")

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

        # Get model configuration based on asset type
        config = AssetConfig.get_config(asset_type)
        model_config = config['model_config']

        try:
            # Initialize Prophet with parameters
            model = Prophet(
                changepoint_prior_scale=model_config['changepoint_prior_scale'],
                n_changepoints=model_config['n_changepoints'],
                seasonality_mode=model_config['seasonality_mode'],
                yearly_seasonality=model_config['yearly_seasonality'],
                weekly_seasonality=model_config['weekly_seasonality'],
                daily_seasonality=model_config['daily_seasonality'],
                interval_width=model_config['interval_width']
            )

            # Add monthly seasonality
            model.add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=5
            )

            # Add economic indicator if available
            if economic_data is not None:
                logger.info("Processing economic indicator data")
                if indicator == 'POLSENT':
                    # Special handling for sentiment data
                    economic_df = economic_data.copy()
                    economic_df.columns = ['ds', 'sentiment']
                    prophet_df = prophet_df.merge(economic_df, on='ds', how='left')
                    prophet_df['sentiment'] = prophet_df['sentiment'].fillna(
                        prophet_df['sentiment'].rolling(window=7, min_periods=1).mean()
                    )
                    model.add_regressor('sentiment', mode='multiplicative')
                else:
                    economic_df = economic_data.copy()
                    economic_df.columns = ['ds', 'economic_indicator']
                    prophet_df = prophet_df.merge(economic_df, on='ds', how='left')
                    prophet_df['economic_indicator'] = prophet_df['economic_indicator'].fillna(
                        method='ffill'
                    ).fillna(method='bfill')
                    model.add_regressor('economic_indicator', mode='multiplicative')

            # Fit the model
            model.fit(prophet_df)

            # Create future DataFrame
            future = model.make_future_dataframe(periods=periods)

            # Add economic indicator to future if available
            if economic_data is not None:
                if indicator == 'POLSENT':
                    future = future.merge(economic_df[['ds', 'sentiment']], on='ds', how='left')
                    future['sentiment'] = future['sentiment'].fillna(prophet_df['sentiment'].mean())
                else:
                    future = future.merge(economic_df[['ds', 'economic_indicator']], on='ds', how='left')
                    future['economic_indicator'] = future['economic_indicator'].fillna(method='ffill').fillna(method='bfill')

            # Generate forecast
            forecast = model.predict(future)

            # Add actual values
            forecast['actual'] = np.nan
            forecast.loc[forecast['ds'].isin(prophet_df['ds']), 'actual'] = prophet_df['y'].values

            logger.info("Forecast completed successfully")
            return forecast, None

        except Exception as e:
            logger.error(f"Error in forecasting process: {str(e)}")
            return None, f"Error in forecasting process: {str(e)}"

    except Exception as e:
        logger.error(f"Error in prophet_forecast: {str(e)}")
        return None, str(e)

def create_forecast_plot(data: pd.DataFrame, forecast: pd.DataFrame, model_name: str, symbol: str) -> go.Figure:
    """Create an interactive plot with historical data and forecast"""
    try:
        # Create figure with secondary y-axis
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, row_heights=[0.7, 0.3],
                           subplot_titles=(f'{symbol} Price Forecast', 'Confidence Analysis'))

        # Add historical data
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

        # Add forecast line
        fig.add_trace(
            go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                name=f'{model_name} Forecast',
                line=dict(color='red', dash='dash')
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

def display_common_metrics(data: pd.DataFrame, forecast: pd.DataFrame):
    """Display common metrics for both stocks and cryptocurrencies"""
    try:
        st.subheader("📈 Price Metrics")
        
        # Ensure we have the required data
        if 'Close' not in data.columns:
            raise ValueError("Close price data not found in dataset")
            
        # Handle data access safely
        try:
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
            
            # Technical Indicators
            st.subheader("📊 Technical Indicators")
            tech_col1, tech_col2, tech_col3 = st.columns(3)
            
            with tech_col1:
                if 'RSI' in data.columns:
                    current_rsi = float(data['RSI'].iloc[-1])
                    rsi_change = float(data['RSI'].diff().iloc[-1])
                    st.metric(
                        "RSI (14)",
                        f"{current_rsi:.2f}",
                        f"{rsi_change:+.2f}"
                    )
            
            with tech_col2:
                if 'MACD' in data.columns and 'Signal_Line' in data.columns:
                    current_macd = float(data['MACD'].iloc[-1])
                    macd_signal = float(data['Signal_Line'].iloc[-1])
                    st.metric(
                        "MACD",
                        f"{current_macd:.2f}",
                        f"Signal: {macd_signal:.2f}"
                    )
            
            with tech_col3:
                if 'MA20' in data.columns:
                    ma20 = float(data['MA20'].iloc[-1])
                    ma20_diff = float(current_price - ma20)
                    st.metric(
                        "20-Day MA",
                        f"${ma20:.2f}",
                        f"{ma20_diff:+.2f} from price"
                    )
            
            # Forecast Metrics
            
            # Forecast Metrics
            st.subheader("🔮 Forecast Metrics")
            forecast_col1, forecast_col2, forecast_col3 = st.columns(3)
            
            with forecast_col1:
                final_forecast = float(forecast['yhat'].iloc[-1])
                forecast_change = ((final_forecast / current_price) - 1) * 100
                st.metric(
                    "Forecast End Price",
                    f"${final_forecast:.2f}",
                    f"{forecast_change:+.2f}%"
                )
            
            with forecast_col2:
                confidence_width = float((forecast['yhat_upper'].iloc[-1] - forecast['yhat_lower'].iloc[-1]) / forecast['yhat'].iloc[-1] * 100)
                st.metric(
                    "Forecast Confidence",
                    f"{100 - confidence_width:.1f}%"
                )
            
            with forecast_col3:
                trend_strength = float(abs(forecast_change) / confidence_width * 100)
                st.metric(
                    "Trend Strength",
                    f"{trend_strength:.1f}%"
                )

        except Exception as e:
            logger.error(f"Error processing metrics: {str(e)}")
            st.error("Error processing metrics. Please check your data format.")

    except Exception as e:
        logger.error(f"Error displaying common metrics: {str(e)}")
        st.error(f"Error displaying common metrics: {str(e)}")

def display_confidence_analysis(forecast: pd.DataFrame):
    """Display detailed confidence analysis of the forecast"""
    try:
        st.subheader("📊 Confidence Analysis")

        # Calculate confidence metrics
        confidence_width = (forecast['yhat_upper'] - forecast['yhat_lower']) / forecast['yhat'] * 100
        avg_confidence = 100 - confidence_width.mean()
        
        # Calculate trend metrics
        total_trend = ((forecast['yhat'].iloc[-1] / forecast['yhat'].iloc[0]) - 1) * 100
        trend_consistency = np.sum(np.diff(forecast['yhat']) > 0) / (len(forecast) - 1) * 100

        # Display metrics in columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Average Confidence",
                f"{avg_confidence:.1f}%",
                "Higher is better"
            )

        with col2:
            st.metric(
                "Overall Trend",
                f"{total_trend:+.1f}%",
                f"{'Upward' if total_trend > 0 else 'Downward'} Trend"
            )

        with col3:
            st.metric(
                "Trend Consistency",
                f"{trend_consistency:.1f}%",
                "% of positive daily changes"
            )

        # Display additional analysis
        with st.expander("View Detailed Confidence Analysis"):
            # Calculate confidence bands over time
            confidence_df = pd.DataFrame({
                'Date': forecast['ds'],
                'Confidence Width (%)': confidence_width,
                'Upper Band': forecast['yhat_upper'],
                'Lower Band': forecast['yhat_lower'],
                'Forecast': forecast['yhat']
            })

            # Show confidence statistics
            st.write("**Confidence Statistics:**")
            stats_col1, stats_col2 = st.columns(2)
            
            with stats_col1:
                st.write("Confidence Width Statistics:")
                st.write(f"- Minimum: {confidence_width.min():.1f}%")
                st.write(f"- Maximum: {confidence_width.max():.1f}%")
                st.write(f"- Average: {confidence_width.mean():.1f}%")
            
            with stats_col2:
                st.write("Price Range at End of Forecast:")
                last_idx = -1
                st.write(f"- Upper: ${forecast['yhat_upper'].iloc[last_idx]:.2f}")
                st.write(f"- Forecast: ${forecast['yhat'].iloc[last_idx]:.2f}")
                st.write(f"- Lower: ${forecast['yhat_lower'].iloc[last_idx]:.2f}")

            # Display confidence width trend
            st.write("\n**Confidence Width Over Time:**")
            st.line_chart(confidence_df.set_index('Date')['Confidence Width (%)'])

    except Exception as e:
        logger.error(f"Error displaying confidence analysis: {str(e)}")
        st.error(f"Error displaying confidence analysis: {str(e)}")
        
        def display_crypto_metrics(data: pd.DataFrame, forecast: pd.DataFrame, symbol: str):
    """Display cryptocurrency-specific metrics"""
    try:
        st.subheader("🪙 Cryptocurrency Metrics")

        # Volume Analysis
        vol_col1, vol_col2 = st.columns(2)
        with vol_col1:
            volume = float(data['Volume'].iloc[-1])
            volume_change = float(data['Volume'].pct_change().iloc[-1] * 100)
            st.metric(
                "24h Volume",
                f"${volume:,.0f}",
                f"{volume_change:+.2f}%"
            )

        with vol_col2:
            if 'volume_ratio' in data.columns:
                vol_ratio = float(data['volume_ratio'].iloc[-1])
                st.metric(
                    "Volume Ratio",
                    f"{vol_ratio:.2f}",
                    "Above Average" if vol_ratio > 1 else "Below Average"
                )

        # Volatility Metrics
        vol_metrics_col1, vol_metrics_col2 = st.columns(2)
        with vol_metrics_col1:
            if 'hourly_volatility' in data.columns:
                hourly_vol = float(data['hourly_volatility'].iloc[-1] * 100)
                st.metric(
                    "Hourly Volatility",
                    f"{hourly_vol:.2f}%"
                )

        with vol_metrics_col2:
            if 'volatility_ratio' in data.columns:
                vol_ratio = float(data['volatility_ratio'].iloc[-1])
                st.metric(
                    "Volatility Trend",
                    "Increasing" if vol_ratio > 1 else "Decreasing"
                )

        # Market Metrics
        if 'market_dominance' in data.columns:
            st.metric(
                "Market Dominance",
                f"{float(data['market_dominance'].iloc[-1] * 100):.2f}%"
            )

        # Network Metrics
        if all(col in data.columns for col in ['network_transactions', 'active_addresses']):
            net_col1, net_col2 = st.columns(2)
            with net_col1:
                st.metric("Network Transactions", f"{int(data['network_transactions'].iloc[-1]):,}")
            with net_col2:
                st.metric("Active Addresses", f"{int(data['active_addresses'].iloc[-1]):,}")

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
        st.error(f"Error displaying economic indicators: {str(e)}"
        
        )