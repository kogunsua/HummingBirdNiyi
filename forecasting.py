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
        # Make a copy to avoid modifying original data
        data = df.copy()
        
        # Convert DateTimeIndex to column if needed
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.reset_index()
            data.rename(columns={'index': 'ds'}, inplace=True)
        else:
            # Try to find date column
            date_cols = [col for col in data.columns if col in ['Date', 'timestamp']]
            if date_cols:
                data.rename(columns={date_cols[0]: 'ds'}, inplace=True)
        
        # Ensure 'y' column exists (from 'Close' price)
        if 'Close' in data.columns:
            data['y'] = data['Close'].astype(float)
        else:
            raise ValueError("No 'Close' price column found in data")
        
        # Convert to datetime and remove timezone info
        data['ds'] = pd.to_datetime(data['ds'])
        if data['ds'].dt.tz is not None:
            data['ds'] = data['ds'].dt.tz_localize(None)
        
        # Select only required columns
        prophet_data = pd.DataFrame({
            'ds': data['ds'],
            'y': data['y']
        })
        
        # Remove any duplicates and sort by date
        prophet_data = prophet_data.drop_duplicates('ds').sort_values('ds')
        
        # Reset index
        prophet_data = prophet_data.reset_index(drop=True)
        
        return prophet_data
        
    except Exception as e:
        logger.error(f"Error in prepare_data_for_prophet: {str(e)}")
        raise e

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
        
        # Generate future dates
        future = model.make_future_dataframe(periods=periods)
        
        # Add economic indicator to future dates if provided
        if economic_data is not None:
            future = future.merge(economic_df, on='ds', how='left')
            future['regressor'].fillna(method='ffill', inplace=True)
        
        # Make prediction
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