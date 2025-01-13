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

def prepare_data_for_prophet(data: pd.DataFrame) -> pd.DataFrame:
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

def prophet_forecast(data: pd.DataFrame, periods: int, economic_data: Optional[pd.DataFrame] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Generate forecast using Prophet model with realistic constraints"""
    try:
        logger.info(f"Starting forecast for {periods} periods")
        
        # Prepare data for Prophet
        prophet_df = prepare_data_for_prophet(data)
        
        if prophet_df is None or prophet_df.empty:
            raise ValueError("No valid data for forecasting")
        
        # Calculate historical volatility and trends
        current_price = prophet_df['y'].iloc[-1]
        historical_std = prophet_df['y'].std()
        historical_mean = prophet_df['y'].mean()
        historical_daily_returns = prophet_df['y'].pct_change().dropna()
        max_historical_daily_return = historical_daily_returns.abs().quantile(0.95)  # 95th percentile
        
        logger.info(f"Current price: {current_price}")
        logger.info(f"Historical std: {historical_std}")
        logger.info(f"Max historical daily return: {max_historical_daily_return}")
        
        # Calculate historical volatility
        historical_volatility = historical_daily_returns.std()

        # Initialize Prophet with stricter parameters
        model = Prophet(
            changepoint_prior_scale=0.0005,    # Even more conservative
            n_changepoints=10,                 # Very limited changepoints
            seasonality_mode='additive'        # More stable forecasts
        )

        # Add monthly seasonality
        model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=5
        )

        # Add economic indicator if available
        if economic_data is not None:
            economic_df = economic_data.copy()
            
            if isinstance(economic_df.index, pd.DatetimeIndex):
                economic_df = economic_df.reset_index()
            
            economic_df = pd.DataFrame({
                'ds': pd.to_datetime(economic_df['index'] if 'index' in economic_df.columns else economic_df.iloc[:, 0]),
                'economic_indicator': economic_df['value' if 'value' in economic_df.columns else economic_df.columns[1]].astype(float)
            })
            
            prophet_df = prophet_df.merge(economic_df, on='ds', how='left')
            prophet_df['economic_indicator'] = prophet_df['economic_indicator'].fillna(method='ffill').fillna(method='bfill')
            model.add_regressor('economic_indicator', mode='multiplicative')

        # Fit the model
        model.fit(prophet_df)

        # Generate future dates
        future = model.make_future_dataframe(periods=periods)
        
        if economic_data is not None:
            future = future.merge(economic_df, on='ds', how='left')
            future['economic_indicator'] = future['economic_indicator'].fillna(method='ffill').fillna(method='bfill')

        # Generate initial forecast
        forecast = model.predict(future)

        # Daily movement constraints
        max_daily_move = min(0.05, historical_volatility * 2)  # Cap at 5% daily move

        # Progressive constraints
        last_known_price = current_price
        future_start_idx = len(prophet_df)

        for i in range(future_start_idx, len(forecast)):
            max_up_move = last_known_price * (1 + max_daily_move)
            max_down_move = last_known_price * (1 - max_daily_move)
            forecast.loc[i, 'yhat'] = np.clip(forecast.loc[i, 'yhat'], max_down_move, max_up_move)
            last_known_price = forecast.loc[i, 'yhat']

        # Apply dampening to future values
        recent_trend = prophet_df['y'].iloc[-1] - prophet_df['y'].iloc[-2]
        if abs(recent_trend) > 20:
            trend_dampening = 0.8
            for days_out in range(1, periods + 1):
                dampening_factor = trend_dampening ** (days_out / 10)
                forecast.loc[future_start_idx + days_out - 1, 'yhat'] = current_price + (forecast.loc[future_start_idx + days_out - 1, 'yhat'] - current_price) * dampening_factor

        # Final validation
        max_allowed_forecast = current_price * (1 + min(0.5, historical_volatility * np.sqrt(periods)))
        min_allowed_forecast = current_price * (1 - min(0.5, historical_volatility * np.sqrt(periods)))
        forecast['yhat'] = forecast['yhat'].clip(lower=min_allowed_forecast, upper=max_allowed_forecast)
        forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=min_allowed_forecast, upper=max_allowed_forecast)
        forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=min_allowed_forecast, upper=max_allowed_forecast)

        # Add actual values
        forecast['actual'] = np.nan
        forecast.loc[forecast['ds'].isin(prophet_df['ds']), 'actual'] = prophet_df['y'].values

        logger.info("Forecast completed successfully")
        logger.info(f"Final forecast range: {forecast['yhat'].min():.2f} to {forecast['yhat'].max():.2f}")
        
        return forecast, None

    except Exception as e:
        logger.error(f"Error in prophet_forecast: {str(e)}")
        st.error(f"Forecasting error details: {str(e)}")
        return None, str(e)

def create_forecast_plot(data: pd.DataFrame, forecast: pd.DataFrame, model_name: str, symbol: str) -> go.Figure:
    """Create an interactive forecast plot for both stocks and crypto"""
    try:
        logger.info(f"Creating forecast plot for {symbol}")
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
            hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Price</b>: $%{y:.2f}<extra></extra>'
        ))

        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
            y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(255,0,0,0.1)',
            line=dict(color='rgba(255,0,0,0)'),
            name='Confidence Interval',
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
                'font': dict(size=24)
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

        logger.info("Successfully created forecast plot")
        return fig

    except Exception as e:
        logger.error(f"Error creating forecast plot: {str(e)}")
        logger.error(f"Data shape: {data.shape if data is not None else 'None'}")
        logger.error(f"Forecast shape: {forecast.shape if forecast is not None else 'None'}")
        st.error(f"Error creating plot: {str(e)}")
        return None

def display_metrics(data: pd.DataFrame, forecast: pd.DataFrame, asset_type: str, symbol: str):
    """Display key metrics and statistics"""
    try:
        # Enhanced logging
        logger.info(f"Data shape: {data.shape}")
        logger.info(f"Forecast shape: {forecast.shape}")
        
        # Get latest values ensuring proper column access
        if isinstance(data, pd.DataFrame):
            if 'Close' in data.columns:
                latest_price = float(data['Close'].iloc[-1])
                price_change = float(data['Close'].pct_change().iloc[-1] * 100)
            else:
                latest_price = float(data.iloc[-1, 0])
                price_change = float((data.iloc[-1, 0] / data.iloc[-2, 0] - 1) * 100)
        else:
            latest_price = float(data.iloc[-1])
            price_change = float((data.iloc[-1] / data.iloc[-2] - 1) * 100)

        forecast_price = float(forecast['yhat'].iloc[-1])
        forecast_change = ((forecast_price - latest_price) / latest_price) * 100

        # Create metrics display
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Current Price",
                f"${latest_price:,.2f}",
                f"{price_change:+.2f}%"
            )

        with col2:
            st.metric(
                f"Forecast Price ({forecast['ds'].iloc[-1].strftime('%Y-%m-%d')})",
                f"${forecast_price:,.2f}",
                f"{forecast_change:+.2f}%"
            )

        with col3:
            confidence_range = float(forecast['yhat_upper'].iloc[-1]) - float(forecast['yhat_lower'].iloc[-1])
            st.metric(
                "Forecast Range",
                f"${confidence_range:,.2f}",
                f"±{(confidence_range/forecast_price*100/2):.2f}%"
            )

    except Exception as e:
        logger.error(f"Error displaying metrics: {str(e)}")
        logger.error(f"Data type: {type(data)}")
        logger.error(f"Data columns: {data.columns if isinstance(data, pd.DataFrame) else 'Not a DataFrame'}")
        st.error(f"Error displaying metrics: {str(e)}")


def display_economic_indicators(data: pd.DataFrame, indicator: str, economic_indicators: EconomicIndicators):
    """Display economic indicators and their analysis"""
    try:
        stats = economic_indicators.analyze_indicator(data, indicator)
        if stats:
            st.subheader(f"Economic Indicator: {indicator}")
            st.write(f"**Current Value:** {stats['current_value']}")
            st.write(f"**1-day Change:** {stats['change_1d']:.2f}%")
            if stats['change_1m'] is not None:
                st.write(f"**1-month Change:** {stats['change_1m']:.2f}%")
            st.write(f"**Min Value:** {stats['min_value']}")
            st.write(f"**Max Value:** {stats['max_value']}")
            st.write(f"**Average Value:** {stats['avg_value']}")
            st.write(f"**Standard Deviation:** {stats['std_dev']:.2f}")
        else:
            st.warning(f"No stats available for {indicator}")
    except Exception as e:
        logger.error(f"Error displaying economic indicators: {str(e)}")
        st.error(f"Error displaying economic indicators: {str(e)}")