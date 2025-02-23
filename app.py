#app.py
# app.py
import streamlit as st
from datetime import datetime, timedelta
import logging
import sys
from typing import Optional, Tuple, Dict, Any
import pandas as pd
import traceback
import yfinance as yf
from logging.handlers import RotatingFileHandler
import pytz
from typing_extensions import Literal
from dataclasses import dataclass
from pathlib import Path

# Import local modules
from dividend_analyzer import DividendAnalyzer, show_dividend_education, filter_monthly_dividend_stocks
from config import Config, MODEL_DESCRIPTIONS
from data_fetchers import AssetDataFetcher, EconomicIndicators
from forecasting import (
    prophet_forecast,
    create_forecast_plot,
    display_metrics,
    display_confidence_analysis,
    add_technical_indicators,
    display_forecast_results
)
from sentiment_analyzer import (
    MultiSourceSentimentAnalyzer,
    display_sentiment_impact_analysis,
    display_sentiment_impact_results,
    get_sentiment_data
)
from gdelt_analysis import GDELTAnalyzer, update_forecasting_process
from treasury_interface import display_treasury_dashboard

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type definitions
AssetType = Literal["Stocks", "Crypto"]
ForecastType = Literal["Price Only", "Price + Market Sentiment"]
SentimentSource = Literal["Multi-Source", "GDELT", "Yahoo Finance", "News API"]

@dataclass
class UserInputs:
    """Data class for user inputs"""
    asset_type: AssetType
    symbol: str
    periods: int
    start_date: datetime
    end_date: datetime

class AssetDataFetcher:
    """Class to fetch asset data"""
    @staticmethod
    def get_stock_data(symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            return data if not data.empty else None
        except Exception as e:
            logger.error(f"Error fetching stock data: {e}")
            return None

    @staticmethod
    def get_crypto_data(symbol: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        try:
            data = yf.download(symbol, start=start_date, end=end_date)
            return data if not data.empty else None
        except Exception as e:
            logger.error(f"Error fetching crypto data: {e}")
            return None

def initialize_session_state():
    """Initialize session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.asset_type = "Stocks"
        st.session_state.symbol = ""
        st.session_state.periods = 30
        st.session_state.forecast_type = "Price Only"
        st.session_state.sentiment_source = "Multi-Source"

def display_header():
    """Display the application header"""
    st.markdown(
        """
        <div style='text-align: center;'>
            <h1>🐦 HummingBird v2</h1>
            <p><i>Digital Asset & Stock Forecasting with Economic and Sentiment Indicators</i></p>
            <p>AvaResearch LLC - A Black Collar Production</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def display_footer():
    """Display the application footer"""
    st.markdown(
        """
        <div style='text-align: center; padding: 10px; margin-top: 2rem;'>
            <p>© 2025 AvaResearch LLC. All rights reserved.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def setup_sidebar():
    """Setup and handle sidebar inputs"""
    with st.sidebar:
        st.header("🔮 Model Configuration")
        selected_model = st.selectbox(
            "Select Forecasting Model",
            list(MODEL_DESCRIPTIONS.keys())
        )

        if selected_model in MODEL_DESCRIPTIONS:
            model_info = MODEL_DESCRIPTIONS[selected_model]
            st.markdown(f"### Model Details\n{model_info['description']}")
            
            status_color = 'green' if model_info['development_status'] == 'Active' else 'orange'
            st.markdown(
                f"**Status:** <span style='color:{status_color}'>{model_info['development_status']}</span>", 
                unsafe_allow_html=True
            )
            
            confidence = model_info['confidence_rating']
            color = 'green' if confidence >= 0.8 else 'orange' if confidence >= 0.7 else 'red'
            st.markdown(
                f"**Confidence Rating:** <span style='color:{color}'>{confidence:.0%}</span>", 
                unsafe_allow_html=True
            )

        st.header("📈 Additional Indicators")
        selected_indicator = st.selectbox(
            "Select Economic Indicator",
            ['None'] + list(Config.INDICATORS.keys()),
            format_func=lambda x: Config.INDICATORS.get(x, x) if x != 'None' else x
        )

        return selected_model, selected_indicator

def get_user_inputs() -> Optional[UserInputs]:
    """Get and validate user inputs"""
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            asset_type = st.selectbox(
                "Select Asset Type",
                ["Stocks", "Crypto"]
            )
            
        with col2:
            symbol = st.text_input(
                "Enter Symbol",
                help="Enter stock symbol (e.g., AAPL) or crypto symbol (e.g., BTC-USD)"
            ).upper()

        if not symbol:
            st.info("Please enter a valid symbol")
            return None

        periods = st.slider(
            "Forecast Period (days)",
            min_value=7,
            max_value=365,
            value=30,
            help="Number of days to forecast"
        )

        end_date = datetime.now(pytz.UTC)
        start_date = end_date - timedelta(days=365)

        return UserInputs(
            asset_type=asset_type,
            symbol=symbol,
            periods=periods,
            start_date=start_date,
            end_date=end_date
        )
    
    except Exception as e:
        logger.error(f"Error in get_user_inputs: {str(e)}")
        st.error("Error processing inputs. Please try again.")
        return None

def handle_forecast_tab():
    """Handle the forecast tab functionality"""
    try:
        selected_model, selected_indicator = setup_sidebar()
        user_inputs = get_user_inputs()
        
        if user_inputs:
            st.markdown("### 📊 Select Forecast Type")
            forecast_type: ForecastType = st.radio(
                "Choose forecast type",
                ["Price Only", "Price + Market Sentiment"],
                help="Choose whether to include market sentiment analysis",
                horizontal=True
            )
            
            sentiment_data = None
            if forecast_type == "Price + Market Sentiment":
                sentiment_period = st.slider(
                    "Sentiment Analysis Period (days)",
                    7, 90, 30,
                    help="Historical period for sentiment analysis"
                )
                
                sentiment_weight = st.slider(
                    "Sentiment Impact Weight",
                    0.0, 1.0, 0.5,
                    help="Weight of sentiment analysis in the forecast"
                )
                
                sentiment_source: SentimentSource = st.selectbox(
                    "Sentiment Data Source",
                    ["Multi-Source", "GDELT", "Yahoo Finance", "News API"],
                    help="Choose the source for sentiment analysis"
                )
                
                display_sentiment_impact_analysis(
                    sentiment_period,
                    sentiment_weight,
                    sentiment_source
                )
            
            if st.button("🚀 Generate Forecast"):
                with st.spinner('Loading data...'):
                    fetcher = AssetDataFetcher()
                    price_data = (
                        fetcher.get_stock_data(
                            user_inputs.symbol,
                            user_inputs.start_date,
                            user_inputs.end_date
                        ) if user_inputs.asset_type == "Stocks" 
                        else fetcher.get_crypto_data(
                            user_inputs.symbol,
                            user_inputs.start_date,
                            user_inputs.end_date
                        )
                    )
                    
                    if forecast_type == "Price + Market Sentiment":
                        sentiment_start = user_inputs.end_date - timedelta(days=sentiment_period)
                        analyzer = MultiSourceSentimentAnalyzer()
                        sentiment_data = get_sentiment_data(
                            analyzer,
                            user_inputs.symbol,
                            sentiment_start.strftime('%Y-%m-%d'),
                            user_inputs.end_date.strftime('%Y-%m-%d'),
                            sentiment_source
                        )
                    
                    if price_data is not None:
                        with st.spinner('Generating forecast...'):
                            try:
                                forecast, error_metrics = prophet_forecast(price_data, user_inputs.periods)
                                
                                if forecast is not None:
                                    st.success("Forecast generated successfully!")
                                    price_data = add_technical_indicators(price_data)
                                    display_forecast_results(
                                        price_data,
                                        forecast,
                                        {'error_metrics': error_metrics},
                                        forecast_type,
                                        user_inputs.asset_type,
                                        user_inputs.symbol
                                    )
                                    
                                    if sentiment_data is not None:
                                        display_sentiment_impact_results(
                                            sentiment_data,
                                            price_data
                                        )
                                else:
                                    st.error("Failed to generate forecast")
                            except Exception as e:
                                st.error(f"Error generating forecast: {str(e)}")
                    else:
                        st.error(f"Could not load data for {user_inputs.symbol}")
                        
    except Exception as e:
        logger.error(f"Error in forecast tab: {str(e)}")
        st.error("An error occurred. Please try again.")

def handle_dividend_tab():
    """Handle the dividend tab functionality"""
    try:
        st.title("💰 Dividend Analysis")
        st.info("Dividend analysis feature coming soon!")
        
    except Exception as e:
        logger.error(f"Error in dividend tab: {str(e)}")
        st.error("An error occurred in the dividend analysis tab.")

def handle_treasury_tab():
    """Handle the treasury tab functionality"""
    try:
        display_treasury_dashboard()
        
    except Exception as e:
        logger.error(f"Error in treasury tab: {str(e)}")
        st.error("An error occurred in the treasury analysis tab.")

def main():
    """Main application entry point"""
    try:
        # Set page config
        st.set_page_config(
            page_title="HummingBird v2",
            page_icon="🐦",
            layout="wide"
        )
        
        # Initialize session state
        initialize_session_state()
        
        # Display header
        display_header()
        
        # Create main tabs
        forecast_tab, dividend_tab, treasury_tab = st.tabs([
            "📈 Price Forecast",
            "💰 Dividend Analysis",
            "🏦 Daily Treasury Statement"
        ])
        
        # Handle each tab
        with forecast_tab:
            handle_forecast_tab()
            
        with dividend_tab:
            handle_dividend_tab()
            
        with treasury_tab:
            handle_treasury_tab()
        
        # Display footer
        display_footer()
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}\n{traceback.format_exc()}")
        st.error("An unexpected error occurred. Please try again later.")
        if st.checkbox("Show error details"):
            st.exception(e)

if __name__ == "__main__":
    main()