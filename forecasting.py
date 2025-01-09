# forecasting.py

# Original imports
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet
from typing import Optional, Tuple
import numpy as np
from config import Config

def prophet_forecast(data: pd.DataFrame, periods: int, economic_data: Optional[pd.DataFrame] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Generate forecasts using Prophet with optional economic indicators"""
    try:
        df = data.reset_index()
        df = df[['Date', 'Close']]
        df.columns = ['ds', 'y']
        
        # Ensure no timezone info
        df['ds'] = pd.to_datetime(df['ds']).tz_localize(None)
        
        model = Prophet(daily_seasonality=True)
        
        if economic_data is not None:
            economic_df = economic_data.copy()
            economic_df.columns = ['ds', 'regressor']
            economic_df['ds'] = pd.to_datetime(economic_df['ds']).tz_localize(None)
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
                st.markdown(f"**Trend:** <span style='color:{trend_color}'>{stats['trend']}</span>", 
                          unsafe_allow_html=True)

        # Create visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=economic_data['date'],
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
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed statistics
        with st.expander("View Detailed Statistics"):
            st.write({k: f"{v:.2f}" if isinstance(v, (float, np.floating)) else v 
                     for k, v in stats.items() if v is not None})

# New Enhanced Functions for GDELT Integration

def enhanced_prophet_forecast(data: pd.DataFrame, 
                            periods: int, 
                            sentiment_data: Optional[pd.DataFrame] = None,
                            economic_data: Optional[pd.DataFrame] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Enhanced Prophet forecast with sentiment and economic data"""
    try:
        df = data.reset_index()
        df = df[['Date', 'Close']]
        df.columns = ['ds', 'y']
        
        # Ensure no timezone info
        df['ds'] = pd.to_datetime(df['ds']).tz_localize(None)
        
        # Create model with custom configurations
        model = Prophet(**Config.PROPHET_CONFIG)
        
        # Add sentiment data if available
        if sentiment_data is not None:
            sentiment_df = sentiment_data.copy()
            sentiment_df.columns = ['ds', 'sentiment']
            sentiment_df['ds'] = pd.to_datetime(sentiment_df['ds']).tz_localize(None)
            model.add_regressor('sentiment', 
                              mode=Config.PROPHET_CONFIG['regressors']['sentiment']['mode'])
            df = df.merge(sentiment_df, on='ds', how='left')
            df['sentiment'].fillna(method='ffill', inplace=True)
        
        # Add economic data if available
        if economic_data is not None:
            economic_df = economic_data.copy()
            economic_df.columns = ['ds', 'economic']
            economic_df['ds'] = pd.to_datetime(economic_df['ds']).tz_localize(None)
            model.add_regressor('economic',
                              mode=Config.PROPHET_CONFIG['regressors']['economic']['mode'])
            df = df.merge(economic_df, on='ds', how='left')
            df['economic'].fillna(method='ffill', inplace=True)
        
        # Fit model
        model.fit(df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods)
        
        # Add regressors to future dataframe
        if sentiment_data is not None:
            future = future.merge(sentiment_df, on='ds', how='left')
            future['sentiment'].fillna(method='ffill', inplace=True)
        
        if economic_data is not None:
            future = future.merge(economic_df, on='ds', how='left')
            future['economic'].fillna(method='ffill', inplace=True)
        
        # Generate forecast
        forecast = model.predict(future)
        
        return forecast, None
    
    except Exception as e:
        return None, str(e)

def create_enhanced_forecast_plot(data: pd.DataFrame,
                               forecast: pd.DataFrame,
                               model_name: str,
                               symbol: str,
                               sentiment_data: Optional[pd.DataFrame] = None) -> go.Figure:
    """Create enhanced interactive forecast plot with sentiment overlay"""
    fig = go.Figure()

    # Original price data
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        name='Actual Price',
        line=dict(color=Config.VISUALIZATION_CONFIG['colors']['primary'])
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        name='Forecast',
        line=dict(color=Config.VISUALIZATION_CONFIG['colors']['secondary'], dash='dash')
    ))

    # Confidence intervals
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

    # Add sentiment overlay if available
    if sentiment_data is not None:
        # Create secondary y-axis for sentiment
        fig.add_trace(go.Scatter(
            x=sentiment_data.index,
            y=sentiment_data['market_sentiment'],
            name='Market Sentiment',
            line=dict(color='purple', dash='dot'),
            yaxis='y2'
        ))

    # Update layout with enhanced settings
    fig.update_layout(
        title=f'{symbol} Price Forecast with Market Sentiment ({model_name})',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        yaxis2=dict(
            title='Sentiment Score',
            overlaying='y',
            side='right',
            showgrid=False
        ) if sentiment_data is not None else None,
        hovermode='x unified',
        template=Config.VISUALIZATION_CONFIG['theme'],
        height=Config.VISUALIZATION_CONFIG['plot_height']
    )

    return fig

def display_enhanced_metrics(data: pd.DataFrame, 
                           forecast: pd.DataFrame,
                           asset_type: str,
                           symbol: str,
                           sentiment_data: Optional[pd.DataFrame] = None):
    """Display enhanced metrics including sentiment analysis"""
    st.subheader("📊 Market Metrics")
    
    # Create first row of metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        last_close = data['Close'].iloc[-1]
        daily_change = ((last_close - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100)
        st.metric(
            "Last Close",
            f"${last_close:.2f}",
            f"{daily_change:.1f}%"
        )
    
    with col2:
        forecast_price = forecast['yhat'].iloc[-1]
        forecast_change = ((forecast_price - last_close) / last_close * 100)
        st.metric(
            "Forecasted Price",
            f"${forecast_price:.2f}",
            f"{forecast_change:.1f}%"
        )
    
    with col3:
        if 'Volume' in data.columns:
            volume = data['Volume'].iloc[-1]
            volume_change = ((volume - data['Volume'].iloc[-2]) / data['Volume'].iloc[-2] * 100)
            st.metric(
                "24h Volume",
                f"{volume:,.0f}",
                f"{volume_change:.1f}%"
            )
        else:
            st.metric("24h Volume", "N/A")
    
    with col4:
        st.metric(
            "Forecast Period",
            f"{len(forecast) - len(data)} days"
        )
    
    # Add sentiment metrics if available
    if sentiment_data is not None:
        st.subheader("🌐 Market Sentiment Analysis")
        sent_col1, sent_col2, sent_col3 = st.columns(3)
        
        current_sentiment = sentiment_data['market_sentiment'].iloc[-1]
        sentiment_ma5 = sentiment_data['market_sentiment'].rolling(5).mean().iloc[-1]
        
        with sent_col1:
            sentiment_color = 'green' if current_sentiment > 0 else 'red'
            st.metric(
                "Current Sentiment",
                f"{current_sentiment:.2f}",
                f"{'Positive' if current_sentiment > 0 else 'Negative'}",
                delta_color=sentiment_color
            )
        
        with sent_col2:
            trend = "Improving" if current_sentiment > sentiment_ma5 else "Declining"
            st.metric(
                "Sentiment Trend",
                trend,
                f"{abs(current_sentiment - sentiment_ma5):.2f}"
            )
        
        with sent_col3:
            sentiment_impact = forecast['yhat'].corr(sentiment_data['market_sentiment'])
            st.metric(
                "Price-Sentiment Correlation",
                f"{sentiment_impact:.2f}"
            )

def display_forecast_components(forecast: pd.DataFrame, model: Prophet):
    """Display individual forecast components"""
    st.subheader("📈 Forecast Components")
    
    # Create figures for each component
    with st.expander("View Forecast Components"):
        # Trend
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['trend'],
            name='Trend'
        ))
        fig_trend.update_layout(title="Trend Component")
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Seasonality
        if 'weekly' in model.seasonalities:
            fig_weekly = go.Figure()
            fig_weekly.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['weekly'],
                name='Weekly Seasonality'
            ))
            fig_weekly.update_layout(title="Weekly Season