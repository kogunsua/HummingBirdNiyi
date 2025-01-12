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
        # Create a copy of the DataFrame
        data = df.copy()

        # If the index is a datetime index, reset it and keep as 'ds'
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.reset_index()
            data = data.rename(columns={'index': 'ds'})
        else:
            # Try to find the date column
            date_col = [col for col in data.columns if col.lower() in ['date', 'timestamp']][0]
            data = data.rename(columns={date_col: 'ds'})

        # Ensure 'ds' column is datetime type and remove timezone
        data['ds'] = pd.to_datetime(data['ds']).dt.tz_localize(None)

        # Rename Close price column to 'y' for Prophet
        data['y'] = data['Close']

        # Select only required columns and sort
        data = data[['ds', 'y']].copy()
        data = data.sort_values('ds')

        return data

    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise e

def prophet_forecast(data: pd.DataFrame, periods: int, economic_data: Optional[pd.DataFrame] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Generate forecasts using Prophet with optional economic indicators"""
    try:
        # Prepare data for Prophet
        df = prepare_data_for_prophet(data)
        
        # Calculate current stats
        current_price = df['y'].iloc[-1]
        historical_volatility = df['y'].pct_change().std()
        
        # Configure Prophet model
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.01,
            interval_width=0.95,
            seasonality_mode='multiplicative',
            n_changepoints=25
        )
        
        # Add economic indicator if provided
        if economic_data is not None:
            economic_df = economic_data.copy()
            economic_df.columns = ['ds', 'regressor']
            economic_df['ds'] = pd.to_datetime(economic_df['ds']).dt.tz_localize(None)
            model.add_regressor('regressor')
            df = df.merge(economic_df, on='ds', how='left')
            df['regressor'].fillna(method='ffill', inplace=True)
        
        # Fit the model
        model.fit(df)
        
        # Create future dates starting from last actual date
        last_date = df['ds'].max()
        future = model.make_future_dataframe(periods=periods)
        future = future[future['ds'] >= last_date].copy()
        
        # Add economic indicator to future if provided
        if economic_data is not None:
            future = future.merge(economic_df, on='ds', how='left')
            future['regressor'].fillna(method='ffill', inplace=True)
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Apply realistic constraints based on historical volatility
        max_daily_change = historical_volatility * 2  # 2 standard deviations
        
        # Apply constraints progressively
        for i in range(len(forecast)):
            if i == 0:
                # First forecast point
                max_change = max_daily_change
                forecast.loc[forecast.index[i], 'yhat'] = np.clip(
                    forecast.loc[forecast.index[i], 'yhat'],
                    current_price * (1 - max_change),
                    current_price * (1 + max_change)
                )
            else:
                # Subsequent points
                prev_forecast = forecast.loc[forecast.index[i-1], 'yhat']
                max_change = max_daily_change * np.sqrt(i + 1)  # Scale with time
                forecast.loc[forecast.index[i], 'yhat'] = np.clip(
                    forecast.loc[forecast.index[i], 'yhat'],
                    prev_forecast * (1 - max_change),
                    prev_forecast * (1 + max_change)
                )
        
        # Adjust confidence intervals
        forecast['yhat_lower'] = forecast['yhat'] * (1 - max_daily_change)
        forecast['yhat_upper'] = forecast['yhat'] * (1 + max_daily_change)
        
        # Add actual values
        forecast['actual'] = np.nan
        actuals_idx = forecast['ds'].isin(df['ds'])
        forecast.loc[actuals_idx, 'actual'] = df['y'].values
        
        return forecast, None
    
    except Exception as e:
        logger.error(f"Forecasting error: {str(e)}")
        return None, str(e)

def create_forecast_plot(data: pd.DataFrame, forecast: pd.DataFrame, model_name: str, symbol: str) -> go.Figure:
    """Create an interactive forecast plot"""
    try:
        fig = go.Figure()

        # Get the dates for actual data
        historical_dates = data.index if isinstance(data.index, pd.DatetimeIndex) else pd.to_datetime(data['Date'])

        # Add historical data trace
        fig.add_trace(go.Scatter(
            x=historical_dates,
            y=data['Close'],
            name='Historical',
            line=dict(color='blue'),
            hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Price</b>: $%{y:.2f}<extra></extra>'
        ))

        # Add forecast trace
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            name='Forecast',
            line=dict(color='red', dash='dash'),
            hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Forecast</b>: $%{y:.2f}<extra></extra>'
        ))

        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
            y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(255,0,0,0.1)',
            line=dict(color='rgba(255,0,0,0)'),
            name='Confidence Interval',
            hoverinfo='skip'
        ))

        # Update layout
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

        # Add range slider
        fig.update_xaxes(rangeslider_visible=True)

        return fig

    except Exception as e:
        logger.error(f"Error creating plot: {str(e)}")
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