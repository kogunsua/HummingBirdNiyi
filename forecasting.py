# forecasting.py

# Original imports
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet
from typing import Optional, Tuple
import numpy as np

# Original prophet_forecast function (unchanged)
def prophet_forecast(data: pd.DataFrame, periods: int, economic_data: Optional[pd.DataFrame] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Generate forecasts using Prophet with optional economic indicators"""
    try:
        df = data.reset_index()
        df = df[['Date', 'Close']]
        df.columns = ['ds', 'y']
        
        # Ensure no timezone info
        df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
        
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

# Original create_forecast_plot function (unchanged)
def create_forecast_plot(data: pd.DataFrame, forecast: pd.DataFrame, model_name: str, symbol: str) -> go.Figure:
    """Create an interactive forecast plot using Plotly"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        name='Actual Price',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))

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

    fig.update_layout(
        title=f'{symbol} Price Forecast ({model_name})',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        template='plotly_white'
    )

    return fig

# Original display_metrics function (unchanged)
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

# Updated display_economic_indicators function with sentiment support
def display_economic_indicators(economic_data: pd.DataFrame, indicator: str, economic_indicators):
    """Display economic indicator data and analysis"""
    if economic_data is not None:
        indicator_info = economic_indicators.get_indicator_info(indicator)
        stats = economic_indicators.analyze_indicator(economic_data, indicator)
        
        st.subheader(f"📈 {indicator_info.get('description', indicator)}")
        
        # Common metrics display
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"**Frequency:** {indicator_info.get('frequency', 'N/A')}")
        with col2:
            st.markdown(f"**Units:** {indicator_info.get('units', 'N/A')}")
        with col3:
            if stats.get('trend'):
                trend_color = 'green' if stats['trend'] == 'Upward' else 'red' if stats['trend'] == 'Downward' else 'gray'
                st.markdown(f"**Trend:** <span style='color:{trend_color}'>{stats['trend']}</span>", unsafe_allow_html=True)
        
        # Special handling for Political Sentiment
        if indicator == 'POLSENT':
            # Display sentiment-specific metrics
            current_sentiment = economic_data['value'].iloc[-1]
            sentiment_color = 'green' if current_sentiment > 20 else 'red' if current_sentiment < -20 else 'gray'
            sentiment_label = 'Positive' if current_sentiment > 20 else 'Negative' if current_sentiment < -20 else 'Neutral'
            
            # Add sentiment reference lines to plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=economic_data['index'],
                y=economic_data['value'],
                name='Sentiment',
                line=dict(color='blue'),
                fill='tonexty'
            ))
            
            # Add reference lines for sentiment zones
            fig.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Positive Zone")
            fig.add_hline(y=-20, line_dash="dash", line_color="red", annotation_text="Negative Zone")
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            
            fig.update_layout(
                title="Political Sentiment Trend",
                yaxis_title="Sentiment Score",
                yaxis_range=[-100, 100],
                template="plotly_white"
            )
            
            # Add sentiment metrics
            st.metric("Current Sentiment", 
                     f"{current_sentiment:.1f}", 
                     delta=f"{sentiment_label}", 
                     delta_color=sentiment_color)
            
        else:
            # Original plotting for other indicators
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=economic_data['index'],
                y=economic_data['value'],
                name=indicator_info.get('description', indicator),
                line=dict(color='blue')
            ))
            
            fig.update_layout(
                title=f"{indicator_info.get('description', indicator)} ({indicator_info.get('units', '')})",
                xaxis_title="Date",
                yaxis_title=indicator_info.get('units', ''),
                template="plotly_white"
            )
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed statistics (for all indicators)
        with st.expander("View Detailed Statistics"):
            st.write({k: f"{v:.2f}" if isinstance(v, (float, np.floating)) else v 
                     for k, v in stats.items() if v is not None})