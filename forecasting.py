#forecasting.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from typing import Tuple, Optional, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_data_for_prophet(data: pd.DataFrame, asset_type: str) -> pd.DataFrame:
    """Prepare data for Prophet model"""
    try:
        logger.info("Starting data preparation for Prophet")
        logger.info(f"Input data shape: {data.shape}")
        
        # Make a copy of the data
        df = data.copy()
        
        # Convert index to datetime column
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df.rename(columns={'index': 'ds'}, inplace=True)
        else:
            # Try to find date column
            date_cols = [col for col in df.columns if col.lower() in ['date', 'timestamp']]
            if date_cols:
                df.rename(columns={date_cols[0]: 'ds'}, inplace=True)
            else:
                df['ds'] = df.index

        # Ensure Close price is used for 'y'
        if 'Close' in df.columns:
            df['y'] = df['Close'].astype(float)
        else:
            raise ValueError("No 'Close' price column found")

        # Ensure datetime format for ds
        df['ds'] = pd.to_datetime(df['ds'])
        if df['ds'].dt.tz is not None:
            df['ds'] = df['ds'].dt.tz_localize(None)

        # Sort by date and reset index
        df = df.sort_values('ds')
        df = df.reset_index(drop=True)

        # Select only required columns
        prophet_df = df[['ds', 'y']].copy()

        # Validate data
        if prophet_df.empty:
            raise ValueError("Empty dataframe after preparation")
        if not np.isfinite(prophet_df['y']).all():
            raise ValueError("Data contains non-finite values")
        if prophet_df['ds'].isna().any():
            raise ValueError("Missing dates in data")

        logger.info(f"Prepared dataframe shape: {prophet_df.shape}")
        logger.info(f"Date range: {prophet_df['ds'].min()} to {prophet_df['ds'].max()}")
        
        return prophet_df

    except Exception as e:
        logger.error(f"Error in prepare_data_for_prophet: {str(e)}")
        raise e

def get_prophet_params(asset_type: str) -> dict:
    """Get optimal Prophet parameters based on asset type"""
    if asset_type == 'crypto':
        return {
            'changepoint_prior_scale': 0.05,    # More flexible for crypto volatility
            'n_changepoints': 25,               # More changepoints for crypto
            'seasonality_mode': 'multiplicative',
            'daily_seasonality': True,          # Important for crypto's 24/7 trading
            'weekly_seasonality': True,
            'yearly_seasonality': True
        }
    else:  # stock parameters
        return {
            'changepoint_prior_scale': 0.001,   # More conservative for stocks
            'n_changepoints': 10,
            'seasonality_mode': 'multiplicative',
            'daily_seasonality': False,         # Less important for stocks
            'weekly_seasonality': True,
            'yearly_seasonality': True
        }

def prophet_forecast(data: pd.DataFrame, periods: int, asset_type: str, economic_data: Optional[pd.DataFrame] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Generate forecast using Prophet model with realistic constraints"""
    try:
        logger.info(f"Starting forecast for {periods} periods")

        # Prepare data for Prophet
        prophet_df = prepare_data_for_prophet(data, asset_type)

        if prophet_df is None or prophet_df.empty:
            raise ValueError("No valid data for forecasting")

        # Calculate historical volatility and trends
        current_price = prophet_df['y'].iloc[-1]
        historical_std = prophet_df['y'].std()
        historical_daily_returns = prophet_df['y'].pct_change().dropna()
        max_historical_daily_return = historical_daily_returns.abs().quantile(0.95)  # 95th percentile
        historical_volatility = historical_daily_returns.std()

        logger.info(f"Current price: {current_price}")
        logger.info(f"Historical std: {historical_std}")
        logger.info(f"Historical volatility: {historical_volatility}")

        # Initialize Prophet with asset-specific parameters
        params = get_prophet_params(asset_type)
        model = Prophet(**params)

        # Add monthly seasonality
        model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=5
        )

        # Add economic indicator if available
        if economic_data is not None:
            logger.info("Processing economic indicator data")
            economic_df = economic_data.copy()

            if isinstance(economic_df.index, pd.DatetimeIndex):
                economic_df = economic_df.reset_index()

            economic_df.columns = ['ds', 'economic_indicator']
            economic_df['ds'] = pd.to_datetime(economic_df['ds'])
            economic_df['economic_indicator'] = economic_df['economic_indicator'].astype(float)

            prophet_df = prophet_df.merge(economic_df, on='ds', how='left')
            prophet_df['economic_indicator'] = prophet_df['economic_indicator'].fillna(method='ffill').fillna(method='bfill')
            model.add_regressor('economic_indicator', mode='multiplicative')

        # Fit the model
        model.fit(prophet_df)

        # Create future dataframe
        future = model.make_future_dataframe(periods=periods)

        # Add economic indicator to future if available
        if economic_data is not None:
            future = future.merge(economic_df, on='ds', how='left')
            future['economic_indicator'] = future['economic_indicator'].fillna(method='ffill').fillna(method='bfill')

        # Generate forecast
        forecast = model.predict(future)

        # Apply constraints based on historical volatility
        max_daily_move = min(0.05, historical_volatility * 2)  # Cap at 5%
        last_known_price = current_price
        future_start_idx = len(prophet_df)

        for i in range(future_start_idx, len(forecast)):
            max_up_move = last_known_price * (1 + max_daily_move)
            max_down_move = last_known_price * (1 - max_daily_move)
            forecast.loc[i, 'yhat'] = np.clip(forecast.loc[i, 'yhat'], max_down_move, max_up_move)
            last_known_price = forecast.loc[i, 'yhat']

            # Adjust confidence intervals
            confidence_range = max_daily_move * last_known_price
            forecast.loc[i, 'yhat_lower'] = forecast.loc[i, 'yhat'] - confidence_range
            forecast.loc[i, 'yhat_upper'] = forecast.loc[i, 'yhat'] + confidence_range

        # Add actual values
        forecast['actual'] = np.nan
        forecast.loc[forecast['ds'].isin(prophet_df['ds']), 'actual'] = prophet_df['y'].values

        logger.info("Forecast completed successfully")
        return forecast, None

    except Exception as e:
        logger.error(f"Error in prophet_forecast: {str(e)}")
        st.error(f"Forecasting error details: {str(e)}")
        return None, str(e)

def create_forecast_plot(data: pd.DataFrame, forecast: pd.DataFrame, model_name: str, symbol: str) -> go.Figure:
    """Create an interactive plot with historical data and forecast"""
    try:
        fig = go.Figure()

        # Add historical price trace
        fig.add_trace(go.Scatter(
            x=data.index if isinstance(data.index, pd.DatetimeIndex) else pd.to_datetime(data['Date']),
            y=data['Close'],
            name='Historical',
            line=dict(color='blue', width=2),
            hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Price</b>: $%{y:.2f}<extra></extra>'
        ))

        # Add forecast trace
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            name='Forecast',
            line=dict(color='red', dash='dash', width=2),
            hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Forecast</b>: $%{y:.2f}<extra></extra>'
        ))

        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
            y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(255,0,0,0.1)',
            line=dict(color='rgba(255,0,0,0)'),
            name='95% Confidence Interval',
            showlegend=True,
            hoverinfo='skip'
        ))

        # Update layout
        fig.update_layout(
            title={
                'text': f'{symbol} Price Forecast',
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
            },
            xaxis=dict(
                title='Date',
                gridcolor='lightgrey',
                showgrid=True,
                zeroline=False
            ),
            yaxis=dict(
                title='Price (USD)',
                gridcolor='lightgrey',
                showgrid=True,
                zeroline=False,
                tickprefix='$'
            ),
            hovermode='x unified',
            showlegend=True,
            template='plotly_white',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.8)'
            ),
            margin=dict(l=50, r=50, t=50, b=50)
        )

        # Add range slider
        fig.update_xaxes(rangeslider_visible=True)

        return fig

    except Exception as e:
        logger.error(f"Error creating forecast plot: {str(e)}")
        st.error(f"Error creating plot: {str(e)}")
        return None

def display_metrics(data: pd.DataFrame, forecast: pd.DataFrame, asset_type: str, symbol: str):
    """Display key metrics and statistics"""
    try:
        # Get latest values
        latest_price = float(data['Close'].iloc[-1])
        daily_change = ((latest_price - float(data['Close'].iloc[-2])) / float(data['Close'].iloc[-2])) * 100
        
        forecast_price = float(forecast['yhat'].iloc[-1])
        forecast_change = ((forecast_price - latest_price) / latest_price) * 100
        
        confidence_range = float(forecast['yhat_upper'].iloc[-1] - forecast['yhat_lower'].iloc[-1])
        confidence_percentage = (confidence_range / forecast_price) * 100 / 2

        # Display metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Current Price",
                f"${latest_price:,.2f}",
                f"{daily_change:+.2f}%"
            )

        with col2:
            st.metric(
                f"Forecast Price ({forecast['ds'].iloc[-1].strftime('%Y-%m-%d')})",
                f"${forecast_price:,.2f}",
                f"{forecast_change:+.2f}%"
            )

        with col3:
            st.metric(
                "Forecast Range",
                f"${confidence_range:,.2f}",
                f"±{confidence_percentage:.2f}%"
            )

    except Exception as e:
        logger.error(f"Error in display_metrics: {str(e)}")
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
                        "Monthly Change",
                        f"{analysis['current_value']:.2f}",
                        f"{analysis['change_1m']:.2f}% (1m)"
                    )
            
            with col3:
                st.metric(
                    "Average Value",
                    f"{analysis['avg_value']:.2f}",
                    f"σ: {analysis['std_dev']:.2f}"
                )

    except Exception as e:
        logger.error(f"Error displaying economic indicators: {str(e)}")
        st.error(f"Error displaying economic indicators: {str(e)}")