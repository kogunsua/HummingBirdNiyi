# forecasting.py
#test
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet
from typing import Optional, Tuple
import numpy as np

def prepare_data_for_prophet(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for Prophet model"""
    # Reset index and ensure it's named correctly
    df = df.reset_index()
    
    # Rename columns to Prophet format
    if 'Date' in df.columns:
        df = df.rename(columns={'Date': 'ds'})
    elif 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'ds'})
    else:
        # If index was unnamed, the reset_index creates a column named 'index'
        df = df.rename(columns={'index': 'ds'})
    
    # Ensure Close price is named 'y' for Prophet
    df = df.rename(columns={'Close': 'y'})
    
    # Ensure datetime is timezone naive
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    
    # Select only required columns
    df = df[['ds', 'y']]
    
    return df

def prophet_forecast(data: pd.DataFrame, periods: int, economic_data: Optional[pd.DataFrame] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Generate forecasts using Prophet with optional economic indicators"""
    try:
        # Prepare data for Prophet
        df = prepare_data_for_prophet(data)
        
        model = Prophet(daily_seasonality=True)
        
        if economic_data is not None:
            economic_df = economic_data.copy()
            economic_df.columns = ['ds', 'regressor']
            economic_df['ds'] = pd.to_datetime(economic_df['ds']).dt.tz_localize(None)
            model.add_regressor('regressor')
            df = df.merge(economic_df, on='ds', how='left')
            df['regressor'].fillna(method='ffill', inplace=True)
        
        model.fit(df)
        future = model.make_future_dataframe(periods=periods)
        
        if economic_data is not None:
            future = future.merge(economic_df, on='ds', how='left')
            future['regressor'].fillna(method='ffill', inplace=True)
        
        forecast = model.predict(future)
        return forecast, None
    
    except Exception as e:
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
    """Display key metrics in the Streamlit interface"""
    st.subheader("📊 Market Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Last Close", f"${data['Close'].iloc[-1]:.2f}")
    
    with col2:
        forecast_price = forecast['yhat'].iloc[-1]
        change = ((forecast_price - data['Close'].iloc[-1]) / data['Close'].iloc[-1]) * 100
        st.metric("Forecasted Price", f"${forecast_price:.2f}", f"{change:.1f}%")
    
    with col3:
        volume = data['Volume'].iloc[-1] if 'Volume' in data.columns else 0
        st.metric("24h Volume", f"{volume:,.0f}")
    
    with col4:
        st.metric("Forecast Period", f"{len(forecast) - len(data)} days")

def display_economic_indicators(economic_data: pd.DataFrame, indicator: str, economic_indicators):
    """Display economic indicator data and analysis"""
    if economic_data is not None:
        indicator_info = economic_indicators.get_indicator_info(indicator)
        stats = economic_indicators.analyze_indicator(economic_data, indicator)
        
        st.subheader(f"📈 {indicator_info['description']}")
        
        # Display indicator metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Frequency:** {indicator_info['frequency']}")
        with col2:
            st.markdown(f"**Units:** {indicator_info['units']}")
        with col3:
            if stats.get('trend'):
                trend_color = 'green' if stats['trend'] == 'Upward' else 'red' if stats['trend'] == 'Downward' else 'gray'
                st.markdown(f"**Trend:** <span style='color:{trend_color}'>{stats['trend']}</span>", unsafe_allow_html=True)
        
        # Display current statistics
        if stats:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Value", f"{stats['current_value']:.2f}")
            with col2:
                st.metric("24h Change", f"{stats['change_1d']:.2f}%")
            with col3:
                if stats['change_1m'] is not None:
                    st.metric("30d Change", f"{stats['change_1m']:.2f}%")
            with col4:
                st.metric("Volatility", f"{stats['std_dev']:.2f}")
        
        # Plot the indicator data
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=economic_data['index'],
            y=economic_data['value'],
            name=indicator_info['description'],
            line=dict(color='blue')
        ))
        
        fig.update_layout(
            title=f"{indicator_info['description']} ({indicator_info['units']})",
            xaxis_title="Date",
            yaxis_title=indicator_info['units'],
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed statistics
        with st.expander("View Detailed Statistics"):
            st.write("Summary Statistics:")
            st.write({
                "Minimum": f"{stats['min_value']:.2f}",
                "Maximum": f"{stats['max_value']:.2f}",
                "Average": f"{stats['avg_value']:.2f}",
                "Standard Deviation": f"{stats['std_dev']:.2f}"
            })