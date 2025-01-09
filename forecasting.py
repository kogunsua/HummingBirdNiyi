# forecasting.py

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet
from typing import Optional, Tuple, Dict
import numpy as np
from config import Config

def prophet_forecast(data: pd.DataFrame, 
                    periods: int, 
                    sentiment_data: Optional[pd.DataFrame] = None,
                    economic_data: Optional[pd.DataFrame] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Generate forecasts using Prophet with sentiment and economic indicators"""
    try:
        df = data.reset_index()
        df = df[['Date', 'Close']]
        df.columns = ['ds', 'y']
        
        # Ensure no timezone info
        df['ds'] = pd.to_datetime(df['ds']).tz_localize(None)
        
        # Initialize Prophet with configuration
        model = Prophet(**Config.PROPHET_CONFIG)
        
        # Add sentiment if available
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
            economic_df.columns = ['ds', 'regressor']
            economic_df['ds'] = pd.to_datetime(economic_df['ds']).tz_localize(None)
            model.add_regressor('regressor',
                              mode=Config.PROPHET_CONFIG['regressors']['economic']['mode'])
            df = df.merge(economic_df, on='ds', how='left')
            df['regressor'].fillna(method='ffill', inplace=True)
        
        # Fit model and make future dataframe
        model.fit(df)
        future = model.make_future_dataframe(periods=periods)
        
        # Add regressors to future dataframe
        if sentiment_data is not None:
            future = future.merge(sentiment_df, on='ds', how='left')
            future['sentiment'].fillna(method='ffill', inplace=True)
        
        if economic_data is not None:
            future = future.merge(economic_df, on='ds', how='left')
            future['regressor'].fillna(method='ffill', inplace=True)
        
        # Generate forecast
        forecast = model.predict(future)
        
        return forecast, None
    
    except Exception as e:
        st.error(f"Error generating forecast: {str(e)}")
        return None, str(e)

def create_forecast_plot(data: pd.DataFrame, 
                        forecast: pd.DataFrame, 
                        model_name: str, 
                        symbol: str,
                        sentiment_data: Optional[pd.DataFrame] = None) -> go.Figure:
    """Create interactive forecast plot with optional sentiment overlay"""
    fig = go.Figure()

    # Add price data
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        name='Actual Price',
        line=dict(color='blue')
    ))

    # Add forecast
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))

    # Add confidence intervals
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

    # Add sentiment if available
    if sentiment_data is not None:
        fig.add_trace(go.Scatter(
            x=sentiment_data.index,
            y=sentiment_data['market_sentiment'],
            name='Market Sentiment',
            yaxis='y2',
            line=dict(color='purple', dash='dot')
        ))

    # Update layout
    layout = {
        'title': f'{symbol} Price Forecast ({model_name})',
        'xaxis_title': 'Date',
        'yaxis_title': 'Price (USD)',
        'hovermode': 'x unified',
        'template': 'plotly_white'
    }

    # Add secondary y-axis for sentiment
    if sentiment_data is not None:
        layout['yaxis2'] = {
            'title': 'Sentiment Score',
            'overlaying': 'y',
            'side': 'right',
            'showgrid': False
        }

    fig.update_layout(**layout)
    return fig

def display_metrics(data: pd.DataFrame, 
                   forecast: pd.DataFrame, 
                   asset_type: str, 
                   symbol: str,
                   sentiment_data: Optional[pd.DataFrame] = None):
    """Display comprehensive metrics including sentiment if available"""
    st.subheader("📊 Market Metrics")
    
    # Price metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        last_close = data['Close'].iloc[-1]
        daily_change = ((last_close - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100)
        st.metric(
            "Last Close",
            f"${last_close:.2f}",
            f"{daily_change:.1f}%",
            delta_color='normal' if daily_change >= 0 else 'inverse'
        )
    
    with col2:
        forecast_price = forecast['yhat'].iloc[-1]
        forecast_change = ((forecast_price - last_close) / last_close * 100)
        st.metric(
            "Forecasted Price",
            f"${forecast_price:.2f}",
            f"{forecast_change:.1f}%",
            delta_color='normal' if forecast_change >= 0 else 'inverse'
        )
    
    with col3:
        if 'Volume' in data.columns:
            volume = data['Volume'].iloc[-1]
            volume_change = ((volume - data['Volume'].iloc[-2]) / data['Volume'].iloc[-2] * 100)
            st.metric(
                "24h Volume",
                f"{volume:,.0f}",
                f"{volume_change:.1f}%",
                delta_color='normal' if volume_change >= 0 else 'inverse'
            )
        else:
            st.metric("24h Volume", "N/A")
    
    with col4:
        st.metric("Forecast Period", f"{len(forecast) - len(data)} days")

    # Sentiment metrics if available
    if sentiment_data is not None:
        st.subheader("🌐 Market Sentiment Analysis")
        sent_col1, sent_col2, sent_col3 = st.columns(3)
        
        current_sentiment = sentiment_data['market_sentiment'].iloc[-1]
        sentiment_ma5 = sentiment_data['market_sentiment'].rolling(5).mean().iloc[-1]
        
        with sent_col1:
            st.metric(
                "Current Sentiment",
                f"{current_sentiment:.2f}",
                f"{'Positive' if current_sentiment > 0 else 'Negative'}",
                delta_color='normal' if current_sentiment > 0 else 'inverse'
            )
        
        with sent_col2:
            sentiment_change = current_sentiment - sentiment_ma5
            st.metric(
                "Sentiment Trend",
                "Improving" if sentiment_change > 0 else "Declining",
                f"{abs(sentiment_change):.2f}",
                delta_color='normal' if sentiment_change > 0 else 'inverse'
            )
        
        with sent_col3:
            sentiment_volatility = sentiment_data['market_sentiment'].std()
            st.metric(
                "Sentiment Volatility",
                f"{sentiment_volatility:.2f}"
            )

def display_economic_indicators(economic_data: pd.DataFrame, 
                             indicator: str,
                             economic_indicators,
                             sentiment_data: Optional[pd.DataFrame] = None):
    """Display economic and sentiment indicators with enhanced visualization"""
    if economic_data is not None:
        indicator_info = economic_indicators.get_indicator_info(indicator)
        stats = economic_indicators.analyze_indicator(economic_data, indicator)
        
        st.subheader(f"📈 {indicator_info.get('description', indicator)}")
        
        # Metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"**Frequency:** {indicator_info.get('frequency', 'N/A')}")
        with col2:
            st.markdown(f"**Units:** {indicator_info.get('units', 'N/A')}")
        with col3:
            if stats.get('trend'):
                trend = stats['trend']
                trend_color = 'normal' if trend == 'Upward' else 'inverse'
                st.metric("Trend", trend, delta_color=trend_color)

        # Create visualization
        fig = go.Figure()
        
        # Add economic indicator
        fig.add_trace(go.Scatter(
            x=economic_data['date'],
            y=economic_data['value'],
            name=indicator_info.get('description', indicator),
            line=dict(color='blue')
        ))
        
        # Add sentiment overlay if available
        if sentiment_data is not None and indicator in ['POLSENT', 'MARKETSENT']:
            fig.add_trace(go.Scatter(
                x=sentiment_data.index,
                y=sentiment_data['market_sentiment'],
                name='Market Sentiment',
                yaxis='y2',
                line=dict(color='purple', dash='dot')
            ))
            
            fig.update_layout(
                yaxis2=dict(
                    title='Sentiment Score',
                    overlaying='y',
                    side='right',
                    showgrid=False
                )
            )
        
        fig.update_layout(
            title=f"{indicator_info.get('description', indicator)} ({indicator_info.get('units', '')})",
            xaxis_title="Date",
            yaxis_title=indicator_info.get('units', ''),
            template="plotly_white",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed statistics
        with st.expander("View Detailed Statistics"):
            stats_df = pd.DataFrame({
                'Metric': stats.keys(),
                'Value': [f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v)
                         for v in stats.values()]
            })
            st.dataframe(stats_df)

def display_components(forecast: pd.DataFrame):
    """Display forecast components analysis"""
    st.subheader("Forecast Components")
    
    tab1, tab2 = st.tabs(["Main Components", "Seasonality"])
    
    with tab1:
        # Trend
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['trend'],
            name='Trend',
            line=dict(color='blue')
        ))
        
        fig_trend.update_layout(
            title="Trend Component",
            xaxis_title="Date",
            yaxis_title="Trend Value",
            template="plotly_white"
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with tab2:
        # Seasonality components
        if 'yearly_seasonality' in forecast.columns:
            fig_seasonal = go.Figure()
            fig_seasonal.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yearly_seasonality'],
                name='Yearly Seasonality',
                line=dict(color='green')
            ))
            
            if 'weekly_seasonality' in forecast.columns:
                fig_seasonal.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['weekly_seasonality'],
                    name='Weekly Seasonality',
                    line=dict(color='orange')
                ))
            
            fig_seasonal.update_layout(
                title="Seasonality Components",
                xaxis_title="Date",
                yaxis_title="Seasonal Effect",
                template="plotly_white"
            )
            st.plotly_chart(fig_seasonal, use_container_width=True)

def calculate_accuracy(actual: pd.Series, forecast: pd.Series) -> Dict[str, float]:
    """Calculate forecast accuracy metrics"""
    try:
        # Handle overlapping period only
        common_index = actual.index.intersection(pd.to_datetime(forecast.index))
        actual = actual[common_index]
        forecast = forecast[common_index]
        
        # Calculate metrics
        mape = np.mean(np.abs((actual - forecast) / actual)) * 100
        rmse = np.sqrt(np.mean((actual - forecast) ** 2))
        mae = np.mean(np.abs(actual - forecast))
        
        # Direction accuracy
        actual_direction = np.sign(actual.diff())
        forecast_direction = np.sign(forecast.diff())
        direction_accuracy = np.mean(actual_direction == forecast_direction) * 100
        
        return {
            'MAPE': mape,
            'RMSE': rmse,
            'MAE': mae,
            'Direction Accuracy': direction_accuracy
        }
    except Exception as e:
        st.error(f"Error calculating accuracy metrics: {str(e)}")
        return {}

def display_accuracy_metrics(metrics: Dict[str, float]):
    """Display forecast accuracy metrics"""
    if metrics:
        st.subheader("Forecast Accuracy")
        cols = st.columns(len(metrics))
        
        for col, (metric, value) in zip(cols, metrics.items()):
            col.metric(
                metric,
                f"{value:.2f}" + ("%" if metric in ['MAPE', 'Direction Accuracy'] else "")
            )