# forecasting.py
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

def prepare_data_for_prophet(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for Prophet model"""
    try:
        # Make a copy of input data
        data = df.copy()
        
        # If data is a DataFrame with DateTimeIndex
        if isinstance(data.index, pd.DatetimeIndex):
            ds = data.index
            y = data['Close'].values
        else:
            # Try to find date column
            date_cols = [col for col in data.columns if col in ['Date', 'date', 'timestamp']]
            if date_cols:
                ds = data[date_cols[0]]
            else:
                ds = data.index
            y = data['Close'].values
        
        # Create Prophet dataframe
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(ds),
            'y': y
        })
        
        # Ensure timezone naive
        prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
        
        # Sort by date
        prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
        
        # Ensure 'y' is float
        prophet_df['y'] = prophet_df['y'].astype(float)
        
        return prophet_df
    
    except Exception as e:
        logger.error(f"Error in prepare_data_for_prophet: {str(e)}")
        raise

def prophet_forecast(data: pd.DataFrame, periods: int, economic_data: Optional[pd.DataFrame] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Generate forecasts using Prophet with optional economic indicators"""
    try:
        # Prepare data for Prophet
        prophet_df = prepare_data_for_prophet(data)
        
        # Configure Prophet model
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.01,
            interval_width=0.95
        )
        
        # Add economic indicator if provided
        if economic_data is not None:
            economic_df = prepare_data_for_prophet(economic_data)
            economic_df = economic_df.rename(columns={'y': 'regressor'})
            model.add_regressor('regressor')
            prophet_df = prophet_df.merge(economic_df, on='ds', how='left')
            prophet_df['regressor'].fillna(method='ffill', inplace=True)
        
        # Fit model
        model.fit(prophet_df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods)
        
        # Add economic indicator to future if provided
        if economic_data is not None:
            future = future.merge(economic_df, on='ds', how='left')
            future['regressor'].fillna(method='ffill', inplace=True)
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Apply conservative growth constraints
        current_price = prophet_df['y'].iloc[-1]
        historical_volatility = prophet_df['y'].pct_change().std()
        max_daily_change = min(historical_volatility * 2, 0.05)  # Cap at 5%
        
        forecast['yhat'] = forecast['yhat'].clip(
            lower=current_price * (1 - max_daily_change),
            upper=current_price * (1 + max_daily_change)
        )
        forecast['yhat_lower'] = forecast['yhat_lower'].clip(
            lower=current_price * (1 - max_daily_change),
            upper=current_price * (1 + max_daily_change)
        )
        forecast['yhat_upper'] = forecast['yhat_upper'].clip(
            lower=current_price * (1 - max_daily_change),
            upper=current_price * (1 + max_daily_change)
        )
        
        # Add actual values
        forecast['actual'] = np.nan
        forecast.loc[forecast['ds'].isin(prophet_df['ds']), 'actual'] = prophet_df['y']
        
        return forecast, None
        
    except Exception as e:
        logger.error(f"Forecasting error: {str(e)}")
        return None, str(e)

def create_forecast_plot(data: pd.DataFrame, forecast: pd.DataFrame, model_name: str, symbol: str) -> go.Figure:
    """Create an interactive forecast plot using Plotly"""
    fig = go.Figure()

    # Add actual price trace
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        name='Actual Price',
        line=dict(color='blue')
    ))

    # Add forecast trace
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))

    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        fill=None,
        mode='lines',
        line=dict(color='gray', width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        fill='tonexty',
        mode='lines',
        line=dict(color='gray', width=0),
        name='Confidence Interval'
    ))

    # Update layout
    fig.update_layout(
        title=f'{symbol} Price Forecast ({model_name})',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        template='plotly_white'
    )

    return fig

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
                f"\u00b1{(confidence_range/forecast_price*100/2):.2f}%"
            )

    except Exception as e:
        logger.error(f"Error displaying metrics: {str(e)}")
        logger.error(f"Data type: {type(data)}")
        logger.error(f"Data columns: {data.columns if isinstance(data, pd.DataFrame) else 'Not a DataFrame'}")
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