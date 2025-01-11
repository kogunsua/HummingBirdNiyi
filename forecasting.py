"""
Handles all forecasting operations including predictions, visualizations, and analysis.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet
from typing import Optional, Tuple, Dict
import numpy as np
from config import Config


class Forecasting:
    """Class to handle all forecasting operations"""
    
    def __init__(self):
        pass

    @staticmethod
    def calculate_accuracy(actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
        """Calculate accuracy metrics for the forecast"""
        try:
            actual = actual[-len(predicted):]
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))
            r2 = 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - actual.mean()) ** 2))
            
            return {
                'MAPE (%)': mape,
                'RMSE': rmse,
                'R²': r2
            }
        except Exception as e:
            st.error(f"Error calculating accuracy metrics: {str(e)}")
            return {}

    @staticmethod
    def prophet_forecast(data: pd.DataFrame, 
                        periods: int, 
                        sentiment_data: Optional[pd.DataFrame] = None,
                        economic_data: Optional[pd.DataFrame] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Generate forecasts using Prophet with sentiment and economic indicators"""
        try:
            # Prepare data for Prophet
            df = data.reset_index()
            df = df[['Date', 'Close']]
            df.columns = ['ds', 'y']
            
            # Ensure no timezone info
            df['ds'] = pd.to_datetime(df['ds']).tz_localize(None)
            
            # Initialize Prophet
            model = Prophet(**Config.PROPHET_CONFIG)
            
            # Add sentiment regressor if available
            if sentiment_data is not None:
                sentiment_df = sentiment_data.copy()
                sentiment_df['ds'] = pd.to_datetime(sentiment_df.index)
                model.add_regressor('sentiment', mode='multiplicative')
                df = df.merge(sentiment_df[['ds', 'market_sentiment']], on='ds', how='left')
                df = df.rename(columns={'market_sentiment': 'sentiment'})
                df['sentiment'].fillna(method='ffill', inplace=True)
            
            # Add economic regressor if available
            if economic_data is not None:
                economic_df = economic_data.copy()
                economic_df.columns = ['ds', 'regressor']
                economic_df['ds'] = pd.to_datetime(economic_df['ds']).tz_localize(None)
                model.add_regressor('regressor', mode='multiplicative')
                df = df.merge(economic_df, on='ds', how='left')
                df['regressor'].fillna(method='ffill', inplace=True)
            
            # Fit model and make future dataframe
            model.fit(df)
            future = model.make_future_dataframe(periods=periods)
            
            # Add regressors to future dataframe
            if sentiment_data is not None:
                future = future.merge(sentiment_df[['ds', 'market_sentiment']], on='ds', how='left')
                future = future.rename(columns={'market_sentiment': 'sentiment'})
                future['sentiment'].fillna(method='ffill', inplace=True)
            
            if economic_data is not None:
                future = future.merge(economic_df, on='ds', how='left')
                future['regressor'].fillna(method='ffill', inplace=True)
            
            # Generate forecast
            forecast = model.predict(future)
            return forecast, None
        
        except Exception as e:
            return None, str(e)

    @staticmethod
    def create_forecast_plot(data: pd.DataFrame, 
                           forecast: pd.DataFrame, 
                           model_name: str, 
                           symbol: str,
                           sentiment_data: Optional[pd.DataFrame] = None) -> go.Figure:
        """Create enhanced forecast plot"""
        fig = go.Figure()

        # Add actual price
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            name='Actual Price',
            line=dict(color='blue')
        ))

        # Add forecast line
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

        # Add sentiment overlay if available
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
            'template': 'plotly_white',
            'hovermode': 'x unified'
        }

        # Add secondary y-axis for sentiment
        if sentiment_data is not None:
            layout['yaxis2'] = {
                'title': 'Sentiment Score',
                'overlaying': 'y',
                'side': 'right',
                'showgrid': False
            }

        fig.update_layout(layout)
        return fig

    @staticmethod
    def display_metrics(data: pd.DataFrame, 
                       forecast: pd.DataFrame, 
                       asset_type: str, 
                       symbol: str,
                       sentiment_data: Optional[pd.DataFrame] = None):
        """Display enhanced metrics including sentiment"""
        st.subheader("📊 Market Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            last_close = data['Close'].iloc[-1]
            daily_change = ((last_close - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100)
            
            if asset_type == "Cryptocurrency" and last_close < 1:
                price_format = "${:,.8f}"
            else:
                price_format = "${:,.2f}"
                
            st.metric(
                "Last Close",
                price_format.format(last_close),
                f"{daily_change:.1f}%",
                delta_color='normal' if daily_change >= 0 else 'inverse'
            )
        
        with col2:
            forecast_price = forecast['yhat'].iloc[-1]
            forecast_change = ((forecast_price - last_close) / last_close * 100)
            st.metric(
                "Forecasted Price",
                price_format.format(forecast_price),
                f"{forecast_change:.1f}%",
                delta_color='normal' if forecast_change >= 0 else 'inverse'
            )
        
        with col3:
            if 'Volume' in data.columns:
                volume = data['Volume'].iloc[-1]
                volume_change = ((volume - data['Volume'].iloc[-2]) / data['Volume'].iloc[-2] * 100)
                volume_format = "{:,.0f} USD" if asset_type == "Cryptocurrency" else "{:,.0f} shares"
                st.metric(
                    "24h Volume",
                    volume_format.format(volume),
                    f"{volume_change:.1f}%",
                    delta_color='normal' if volume_change >= 0 else 'inverse'
                )
            else:
                st.metric("24h Volume", "N/A")
        
        with col4:
            if sentiment_data is not None and not sentiment_data.empty:
                sentiment = sentiment_data['market_sentiment'].iloc[-1]
                sentiment_change = sentiment_data['sentiment_momentum'].iloc[-1]
                st.metric(
                    "Market Sentiment",
                    f"{sentiment:.2f}",
                    f"{'Improving' if sentiment_change > 0 else 'Declining'}",
                    delta_color='normal' if sentiment_change > 0 else 'inverse'
                )
            else:
                st.metric("Forecast Period", f"{len(forecast) - len(data)} days")

    @staticmethod
    def display_components(forecast: pd.DataFrame):
        """Display forecast components"""
        st.subheader("📊 Forecast Components")
        
        tab1, tab2, tab3 = st.tabs(["Trend", "Seasonality", "Additional Factors"])
        
        with tab1:
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['trend'],
                name='Trend'
            ))
            
            fig_trend.update_layout(
                title="Trend Component",
                xaxis_title="Date",
                yaxis_title="Trend Value",
                template="plotly_white"
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with tab2:
            # Weekly seasonality
            if 'weekly' in forecast.columns:
                fig_weekly = go.Figure()
                fig_weekly.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['weekly'],
                    name='Weekly Pattern'
                ))
                fig_weekly.update_layout(
                    title="Weekly Seasonality",
                    xaxis_title="Date",
                    yaxis_title="Weekly Effect",
                    template="plotly_white"
                )
                st.plotly_chart(fig_weekly, use_container_width=True)
            
            # Yearly seasonality
            if 'yearly' in forecast.columns:
                fig_yearly = go.Figure()
                fig_yearly.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yearly'],
                    name='Yearly Pattern'
                ))
                fig_yearly.update_layout(
                    title="Yearly Seasonality",
                    xaxis_title="Date",
                    yaxis_title="Yearly Effect",
                    template="plotly_white"
                )
                st.plotly_chart(fig_yearly, use_container_width=True)
        
        with tab3:
            extra_regressors = [col for col in forecast.columns 
                              if col.endswith('_regressor') 
                              or col in ['sentiment', 'regressor']]
            
            for regressor in extra_regressors:
                if regressor in forecast.columns:
                    fig_regressor = go.Figure()
                    fig_regressor.add_trace(go.Scatter(
                        x=forecast['ds'],
                        y=forecast[regressor],
                        name=regressor.capitalize()
                    ))
                    fig_regressor.update_layout(
                        title=f"{regressor.capitalize()} Impact",
                        xaxis_title="Date",
                        yaxis_title="Effect",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_regressor, use_container_width=True)

    @staticmethod
    def display_economic_indicators(economic_data: pd.DataFrame, 
                                 indicator: str,
                                 economic_indicators,
                                 sentiment_data: Optional[pd.DataFrame] = None):
        """Display economic indicator data and analysis"""
        if economic_data is not None:
            indicator_info = economic_indicators.get_indicator_info(indicator)
            stats = economic_indicators.analyze_indicator(economic_data)
            
            st.subheader(f"📈 {indicator_info.get('description', indicator)}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Frequency:** {indicator_info.get('frequency', 'N/A')}")
            with col2:
                st.markdown(f"**Units:** {indicator_info.get('units', 'N/A')}")
            with col3:
                if stats.get('trend'):
                    st.metric(
                        "Trend", 
                        stats['trend'], 
                        delta_color='normal' if stats['trend'] == 'Upward' else 'inverse'
                    )

            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=economic_data['date'],
                y=economic_data['value'],
                name=indicator_info.get('description', indicator),
                line=dict(color='blue')
            ))
            
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
            
            with st.expander("View Detailed Statistics"):
                stats_df = pd.DataFrame({
                    'Metric': stats.keys(),
                    'Value': [f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v)
                             for v in stats.values()]
                })
                st.dataframe(stats_df)

@staticmethod
    def display_sentiment_analysis(sentiment_data: pd.DataFrame):
        """Display sentiment analysis visualization and metrics"""
        if sentiment_data is not None and not sentiment_data.empty:
            st.subheader("🌐 Market Sentiment Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                current_sentiment = sentiment_data['market_sentiment'].iloc[-1]
                st.metric(
                    "Current Sentiment",
                    f"{current_sentiment:.2f}",
                    f"{'Positive' if current_sentiment > 0 else 'Negative'}",
                    delta_color='normal' if current_sentiment > 0 else 'inverse'
                )
            
            with col2:
                sentiment_ma5 = sentiment_data['sentiment_ma5'].iloc[-1]
                trend = "Improving" if current_sentiment > sentiment_ma5 else "Declining"
                st.metric(
                    "5-Day Trend",
                    trend,
                    f"{abs(current_sentiment - sentiment_ma5):.2f}",
                    delta_color='normal' if current_sentiment > sentiment_ma5 else 'inverse'
                )
            
            with col3:
                if 'sentiment_volatility' in sentiment_data.columns:
                    volatility = sentiment_data['sentiment_volatility'].iloc[-1]
                    st.metric(
                        "Sentiment Volatility",
                        f"{volatility:.2f}",
                        "High" if volatility > 0.2 else "Low"
                    )
            
            # Create sentiment visualization
            fig = go.Figure()
            
            # Add sentiment line
            fig.add_trace(go.Scatter(
                x=sentiment_data.index,
                y=sentiment_data['market_sentiment'],
                name='Market Sentiment',
                line=dict(color='purple')
            ))
            
            # Add moving averages
            fig.add_trace(go.Scatter(
                x=sentiment_data.index,
                y=sentiment_data['sentiment_ma5'],
                name='5-Day MA',
                line=dict(color='blue', dash='dot')
            ))
            
            if 'sentiment_ma20' in sentiment_data.columns:
                fig.add_trace(go.Scatter(
                    x=sentiment_data.index,
                    y=sentiment_data['sentiment_ma20'],
                    name='20-Day MA',
                    line=dict(color='orange', dash='dot')
                ))
            
            # Add reference lines
            fig.add_hline(y=0.2, line_dash="dash", line_color="green", annotation_text="Positive Zone")
            fig.add_hline(y=-0.2, line_dash="dash", line_color="red", annotation_text="Negative Zone")
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            
            # Update layout
            fig.update_layout(
                title="Market Sentiment Trend",
                xaxis_title="Date",
                yaxis_title="Sentiment Score",
                template="plotly_white",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display detailed sentiment metrics
            with st.expander("View Detailed Sentiment Metrics"):
                metrics_df = pd.DataFrame({
                    'Metric': [
                        'Current Sentiment',
                        '5-Day MA',
                        '20-Day MA',
                        'Sentiment Momentum',
                        'Theme Impact',
                        'Volume Impact',
                        'Trend Strength'
                    ],
                    'Value': [
                        f"{sentiment_data['market_sentiment'].iloc[-1]:.2f}",
                        f"{sentiment_data['sentiment_ma5'].iloc[-1]:.2f}",
                        f"{sentiment_data['sentiment_ma20'].iloc[-1]:.2f}",
                        f"{sentiment_data['sentiment_momentum'].iloc[-1]:.2f}",
                        f"{sentiment_data['theme_impact'].iloc[-1]:.2f}",
                        f"{sentiment_data['volume_impact'].iloc[-1]:.2f}",
                        f"{sentiment_data['trend_strength'].iloc[-1]:.2f}"
                    ]
                })
                st.dataframe(metrics_df)
                       except Exception as e:
                       st.error(f"Error    displaying sentiment analysis: {str(e)}")