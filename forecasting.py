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
    """Prepare data for Prophet model."""
    df = df.reset_index()
    
    if 'Date' in df.columns:
        df = df.rename(columns={'Date': 'ds'})
    elif 'timestamp' in df.columns:
        df = df.rename(columns={'timestamp': 'ds'})
    else:
        df = df.rename(columns={'index': 'ds'})
    
    df = df.rename(columns={'Close': 'y'})
    
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
    
    df = df[['ds', 'y']]
    
    return df

def prophet_forecast(data: pd.DataFrame, periods: int, economic_data: Optional[pd.DataFrame] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Generate forecasts using Prophet with optional economic indicators."""
    try:
        df = prepare_data_for_prophet(data)
        
        model = Prophet(daily_seasonality=False, 
                       weekly_seasonality=True,
                       yearly_seasonality=True,
                       changepoint_prior_scale=0.01,
                       interval_width=0.95)
        
        if economic_data is not None:
            economic_df = economic_data.copy()
            economic_df.columns = ['ds', 'regressor']
            economic_df['ds'] = pd.to_datetime(economic_df['ds']).dt.tz_localize(None)
            model.add_regressor('regressor')
            df = df.merge(economic_df, on='ds', how='left')
            df['regressor'].fillna(method='ffill', inplace=True)
        
        model.fit(df)
        
        last_date = df['ds'].max()
        future = model.make_future_dataframe(periods=periods)
        future = future[future['ds'] > last_date]
        
        if economic_data is not None:
            future = future.merge(economic_df, on='ds', how='left')
            future['regressor'].fillna(method='ffill', inplace=True)
        
        forecast = model.predict(future)
        
        current_price = df['y'].iloc[-1]
        historical_volatility = df['y'].pct_change().std()
        max_daily_change = historical_volatility * 2
        
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
        
        forecast['actual'] = np.nan
        actuals_idx = forecast['ds'].isin(df['ds'])
        forecast.loc[actuals_idx, 'actual'] = df['y'].values
        
        return forecast, None
    
    except Exception as e:
        logger.error(f"Error in prophet_forecast: {str(e)}")
        st.error(f"Forecasting error details: {str(e)}")
        return None, str(e)

def create_forecast_plot(data: pd.DataFrame, forecast: pd.DataFrame, model_name: str, symbol: str) -> go.Figure:
    """Create an interactive plot with historical data and forecast."""
    try:
        fig = go.Figure()

        historical_dates = data.index if isinstance(data.index, pd.DatetimeIndex) else pd.to_datetime(data['Date'])
        historical_values = data['Close']

        fig.add_trace(go.Scatter(
            x=historical_dates,
            y=historical_values,
            name='Historical',
            line=dict(color='blue'),
            hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Price</b>: $%{y:.2f}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            name='Forecast',
            line=dict(color='red', dash='dash'),
            hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Forecast</b>: $%{y:.2f}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
            y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(255,0,0,0.1)',
            line=dict(color='rgba(255,0,0,0)'),
            name='Confidence Interval',
            hoverinfo='skip'
        ))

        fig.update_layout(
            title=dict(
                text=f'{symbol} Price Forecast',
                x=0.5,
                y=0.95
            ),
            xaxis=dict(
                title='Date',
                gridcolor='lightgray',
                showgrid=True
            ),
            yaxis=dict(
                title='Price (USD)',
                gridcolor='lightgray',
                showgrid=True,
                tickprefix='$'
            ),
            hovermode='x unified',
            showlegend=True,
            template='plotly_white',
            height=600
        )

        fig.update_xaxes(rangeslider_visible=True)

        return fig

    except Exception as e:
        logger.error(f"Error creating plot: {str(e)}")
        st.error(f"Error creating plot: {str(e)}")
        return None

def display_metrics(data: pd.DataFrame, forecast: pd.DataFrame, asset_type: str, symbol: str):
    """Display key metrics and statistics."""
    try:
        logger.info(f"Data shape: {data.shape}")
        logger.info(f"Forecast shape: {forecast.shape}")
        
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
    """Display economic indicator information and analysis."""
    try:
        st.subheader("📊 Economic Indicator Analysis")
        
        indicator_info = economic_indicators.get_indicator_info(indicator)
        
        st.markdown(f"""
            **Indicator:** {indicator_info.get('description', indicator)}  
            **Frequency:** {indicator_info.get('frequency', 'N/A')}  
            **Units:** {indicator_info.get('units', 'N/A')}
        """)
        
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