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
    """Prepare data for Prophet model with proper date handling"""
    try:
        # Make a copy to avoid modifying original data
        data = df.copy()

        # Handle DateTimeIndex if present
        if isinstance(data.index, pd.DatetimeIndex):
            prophet_df = pd.DataFrame({
                'ds': data.index,
                'y': data['Close'].squeeze()  # Ensure y is 1-dimensional
            })
        else:
            # Look for date column if not in index
            date_col = None
            for col in data.columns:
                if col in ['Date', 'date', 'timestamp', 'Timestamp']:
                    date_col = col
                    break
            
            if date_col:
                prophet_df = pd.DataFrame({
                    'ds': data[date_col],
                    'y': data['Close'].squeeze()  # Ensure y is 1-dimensional
                })
            else:
                raise ValueError("No date column found in data")

        # Ensure datetime and remove timezone
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds']).dt.tz_localize(None)
        
        # Ensure numeric type for y
        prophet_df['y'] = prophet_df['y'].astype(float)
        
        # Sort by date and reset index
        prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
        
        # Log data info for debugging
        logger.info(f"Prophet DataFrame shape: {prophet_df.shape}")
        logger.info(f"Prophet DataFrame columns: {prophet_df.columns.tolist()}")
        logger.info(f"Prophet DataFrame dtypes:\n{prophet_df.dtypes}")
        logger.info(f"First few rows:\n{prophet_df.head()}")
        
        return prophet_df

    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise

def prophet_forecast(data: pd.DataFrame, periods: int, economic_data: Optional[pd.DataFrame] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Generate forecasts using Prophet with optional economic indicators"""
    try:
        # Prepare data for Prophet
        df = prepare_data_for_prophet(data)
        
        # Configure Prophet
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            interval_width=0.95
        )
        
        # Add economic indicator if provided
        if economic_data is not None:
            economic_df = economic_data.copy()
            economic_df.columns = ['ds', 'regressor']
            economic_df['ds'] = pd.to_datetime(economic_df['ds']).dt.tz_localize(None)
            model.add_regressor('regressor')
            df = df.merge(economic_df, on='ds', how='left')
            df['regressor'].fillna(method='ffill', inplace=True)
        
        # Fit model
        model.fit(df)
        
        # Make future dataframe
        future = model.make_future_dataframe(periods=periods)
        
        # Add economic indicator to future if available
        if economic_data is not None:
            future = future.merge(economic_df, on='ds', how='left')
            future['regressor'].fillna(method='ffill', inplace=True)
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Add actual values
        forecast['actual'] = np.nan
        mask = forecast['ds'].isin(df['ds'])
        forecast.loc[mask, 'actual'] = df['y'].values
        
        return forecast, None
    
    except Exception as e:
        logger.error(f"Forecasting error: {str(e)}")
        st.error(f"Forecasting error details: {str(e)}")
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
        # Get the latest actual price
        latest_price = data['Close'].iloc[-1]
        
        # Get the latest forecast price
        forecast_price = forecast.loc[forecast.index[-1], 'yhat']
        
        # Calculate price change percentage
        price_change = ((forecast_price - latest_price) / latest_price) * 100
        
        # Calculate confidence range
        confidence_range = forecast.loc[forecast.index[-1], 'yhat_upper'] - forecast.loc[forecast.index[-1], 'yhat_lower']
        confidence_percentage = (confidence_range / forecast_price) * 100 / 2
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Current Price",
                f"${latest_price:,.2f}",
                f"{data['Close'].pct_change().iloc[-1]*100:.2f}%"
            )
        
        with col2:
            st.metric(
                f"Forecast Price ({forecast['ds'].iloc[-1].strftime('%Y-%m-%d')})",
                f"${forecast_price:,.2f}",
                f"{price_change:.2f}%"
            )
        
        with col3:
            st.metric(
                "Forecast Range",
                f"${confidence_range:,.2f}",
                f"±{confidence_percentage:.2f}%"
            )
            
    except Exception as e:
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
        st.error(f"Error displaying economic indicators: {str(e)}")