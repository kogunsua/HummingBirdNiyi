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

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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

def setup_sidebar():
    """Setup and handle sidebar inputs"""
    with st.sidebar:
        st.header("🔮 Model Configuration")
        
        model = st.selectbox(
            "Select Forecasting Model",
            ["Prophet", "LSTM", "XGBoost"],
            help="Choose the forecasting model"
        )
        
        indicator = st.selectbox(
            "Select Economic Indicator",
            ["None", "GDP", "Inflation", "Interest Rates"],
            help="Choose an economic indicator to include"
        )
        
        return model, indicator

def get_user_inputs() -> Optional[UserInputs]:
    """Get and validate user inputs"""
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            asset_type = st.selectbox(
                "Select Asset Type",
                ["Stocks", "Crypto"],
                help="Choose the type of asset to analyze"
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
        
        if not user_inputs:
            return
        
        st.markdown("### 📊 Select Forecast Type")
        forecast_type: ForecastType = st.radio(
            "Choose forecast type",
            ["Price Only", "Price + Market Sentiment"],
            help="Choose whether to include market sentiment analysis",
            horizontal=True
        )
        
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
            
            st.session_state.sentiment_config = {
                'period': sentiment_period,
                'weight': sentiment_weight,
                'source': sentiment_source
            }
        
        if st.button("🚀 Generate Forecast"):
            with st.spinner("Fetching data and generating forecast..."):
                # Placeholder for forecast generation
                st.success("Forecast generated successfully!")
                
    except Exception as e:
        logger.error(f"Error in forecast tab: {str(e)}")
        st.error("An error occurred in the forecast tab. Please try again.")

def handle_dividend_tab():
    """Handle the dividend tab functionality"""
    st.info("Dividend analysis feature coming soon!")

def handle_treasury_tab():
    """Handle the treasury tab functionality"""
    st.info("Treasury statement analysis feature coming soon!")

def main():
    """Main application function"""
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
        
        # Create tabs
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
        st.markdown(
            """
            <div style='text-align: center; padding: 10px; margin-top: 2rem;'>
                <p>© 2025 AvaResearch LLC. All rights reserved.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}\n{traceback.format_exc()}")
        st.error("An unexpected error occurred. Please try again later.")
        if st.checkbox("Show error details"):
            st.exception(e)

if __name__ == "__main__":
    main()