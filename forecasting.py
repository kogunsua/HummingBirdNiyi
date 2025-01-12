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

def prepare_data_for_prophet(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for Prophet model"""
    try:
        logger.info("Starting data preparation for Prophet")
        logger.info(f"Input data shape: {data.shape}")
        logger.info(f"Input data columns: {data.columns.tolist()}")
        
        # Make a copy of the data
        df = data.copy()
        
        # If the dataframe has a DatetimeIndex, reset it to a column
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df.rename(columns={'index': 'ds'}, inplace=True)
        
        # If 'Close' column exists, use it as 'y'
        if 'Close' in df.columns:
            if 'ds' not in df.columns:  # If we haven't set 'ds' from the index
                date_cols = [col for col in df.columns if isinstance(col, str) and col.lower() in ['date', 'timestamp', 'time']]
                if date_cols:
                    df.rename(columns={date_cols[0]: 'ds'}, inplace=True)
                else:
                    df['ds'] = df.index
            
            df['y'] = df['Close']
        else:
            # If no 'Close' column, use the first numeric column as 'y'
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                if 'ds' not in df.columns:
                    df['ds'] = df.index
                df['y'] = df[numeric_cols[0]]
        
        # Ensure 'ds' is datetime
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Ensure 'y' is float
        df['y'] = df['y'].astype(float)
        
        # Select only required columns and sort by date
        prophet_df = df[['ds', 'y']].sort_values('ds').reset_index(drop=True)
        # Drop NaN values
        prophet_df = prophet_df.dropna()
        
        logger.info(f"Prepared Prophet dataframe shape: {prophet_df.shape}")
        logger.info(f"Prophet dataframe columns: {prophet_df.columns.tolist()}")
        logger.info(f"Sample of prepared data:\n{prophet_df.head()}")
        
        # Validate the prepared data
        if prophet_df.empty:
            raise ValueError("Prepared dataframe is empty")
        if not np.isfinite(prophet_df['y']).all():
            raise ValueError("Data contains non-finite values")
        if prophet_df['ds'].isna().any():
            raise ValueError("Date column contains missing values")
            
        return prophet_df
    
    except Exception as e:
        logger.error(f"Error in prepare_data_for_prophet: {str(e)}")
        raise Exception(f"Failed to prepare data for Prophet: {str(e)}")

def prophet_forecast(data: pd.DataFrame, periods: int, economic_data: Optional[pd.DataFrame] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Generate forecast using Prophet model with realistic constraints"""
    try:
        logger.info(f"Starting forecast for {periods} periods")
        
        # Prepare data for Prophet
        prophet_df = prepare_data_for_prophet(data)
        
        if prophet_df is None or prophet_df.empty:
            raise ValueError("No valid data for forecasting")
        
        # Log the shape and type of prophet_df to debug the error
        logger.info(f"Prophet DataFrame shape: {prophet_df.shape}")
        logger.info(f"Prophet DataFrame columns: {prophet_df.columns}")
        logger.info(f"Prophet DataFrame types: {prophet_df.dtypes}")

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
    """Create an interactive plot with historical data and forecast"""
    try:
        fig = go.Figure()

        # Get historical dates and values
        if isinstance(data.index, pd.DatetimeIndex):
            historical_dates = data.index
        else:
            historical_dates = pd.to_datetime(data['Date'] if 'Date' in data.columns else data['timestamp'])
        
        historical_values = data['Close'] if 'Close' in data.columns else data.iloc[:, 0]

        # Add historical data trace
        fig.add_trace(go.Scatter(
            x=historical_dates,
            y=historical_values,
            name='Historical',
            line=dict(color='blue'),
            hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))

        # Add forecast trace
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            name=f'{model_name} Forecast',
            line=dict(color='red', dash='dot'),
            hovertemplate='Date: %{x}<br>Forecast: $%{y:.2f}<extra></extra>'
        ))

        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
            y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(255,0,0,0.1)',
            line=dict(color='rgba(255,0,0,0)'),
            name='95% Confidence Interval',
            hoverinfo='skip'
        ))

        # Update layout
        fig.update_layout(
            title={
                'text': f'{symbol} Price Forecast',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            showlegend=True,
            template='plotly_white',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            margin=dict(l=50, r=50, t=50, b=50)
        )

        # Update axes
        fig.update_xaxes(gridcolor='LightGray', showgrid=True)
        fig.update_yaxes(gridcolor='LightGray', showgrid=True)

        return fig

    except Exception as e:
        logger.error(f"Error creating forecast plot: {str(e)}")
        st.error(f"Error creating plot: {str(e)}")
        return None

def display_metrics(data: pd.DataFrame, forecast: pd.DataFrame, asset_type: str, symbol: str):
    """Display key metrics and statistics"""
    try:
        # Enhanced logging for debugging
        logger.info(f"Data shape: {data.shape}")
        logger.info(f"Data columns: {data.columns.tolist()}")
        logger.info(f"Forecast shape: {forecast.shape}")
        logger.info(f"Forecast columns: {forecast.columns.tolist()}")

        # Validate input data
        if 'Close' not in data.columns:
            raise ValueError("The 'data' DataFrame must contain a 'Close' column.")
        if 'yhat' not in forecast.columns or 'yhat_upper' not in forecast.columns or 'yhat_lower' not in forecast.columns:
            raise ValueError("The 'forecast' DataFrame must contain 'yhat', 'yhat_upper', and 'yhat_lower' columns.")
        if data.empty:
            raise ValueError("The 'data' DataFrame is empty.")
        if forecast.empty:
            raise ValueError("The 'forecast' DataFrame is empty.")

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
        logger.error(f"Error displaying metrics: {str(e)}")
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