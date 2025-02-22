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
from treasurydata import TreasuryDataFetcher

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

class AppLogger:
    """Logger configuration for the application"""
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.setup_logging()

    def setup_logging(self) -> None:
        """Configure logging with rotation"""
        log_file = self.log_dir / "app.log"
        handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.addHandler(logging.StreamHandler(sys.stdout))

class HummingBirdUI:
    """Main UI class for the HummingBird application"""
    def __init__(self):
        self.config = Config()
        self.logger = logging.getLogger(__name__)

    def display_header(self) -> None:
        """Display the application header with styling"""
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

    def display_footer(self) -> None:
        """Display the application footer"""
        st.markdown(
            """
            <div style='text-align: center; padding: 10px; margin-top: 2rem;'>
                <p>© 2025 AvaResearch LLC. All rights reserved.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    def setup_sidebar(self) -> Tuple[str, str]:
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

    def get_user_inputs(self) -> UserInputs:
        """Get and validate user inputs"""
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
            st.warning("Please enter a valid symbol")
            st.stop()

        periods = st.slider(
            "Forecast Period (days)",
            min_value=7,
            max_value=365,
            value=30,
            help="Number of days to forecast"
        )

        end_date = datetime.now(pytz.UTC)
        start_date = end_date - timedelta(days=365)  # Default to 1 year of historical data

        return UserInputs(
            asset_type=asset_type,
            symbol=symbol,
            periods=periods,
            start_date=start_date,
            end_date=end_date
        )

    def get_sentiment_data_wrapper(
        self,
        sentiment_source: SentimentSource,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        sentiment_period: int,
        max_retries: int = 3
    ) -> Optional[pd.DataFrame]:
        """Wrapper function for sentiment data fetching with retry logic"""
        for attempt in range(max_retries):
            try:
                if sentiment_source == "GDELT":
                    gdelt_analyzer = GDELTAnalyzer()
                    return gdelt_analyzer.fetch_sentiment_data(
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d")
                    )
                else:
                    analyzer = MultiSourceSentimentAnalyzer()
                    return get_sentiment_data(
                        analyzer,
                        symbol,
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d"),
                        sentiment_source
                    )
            except (ConnectionError, TimeoutError) as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Failed to fetch sentiment data after {max_retries} attempts: {str(e)}")
                    return None
                continue
            except Exception as e:
                self.logger.error(f"Error fetching sentiment data: {str(e)}")
                return None

    def process_forecast(
        self,
        price_data: pd.DataFrame,
        sentiment_data: Optional[pd.DataFrame],
        forecast_type: ForecastType,
        periods: int,
        sentiment_weight: float = 0.5,
        max_data_points: int = 1000
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """Process forecast data with memory management"""
        try:
            # Validate input data
            if price_data.empty:
                raise ValueError("Price data is empty")
            
            if len(price_data) > max_data_points:
                price_data = price_data.tail(max_data_points)
                self.logger.info(f"Trimmed price data to {max_data_points} points")

            if forecast_type == "Price Only":
                forecast, error = prophet_forecast(price_data, periods)
                impact_metrics = {}
            else:
                if sentiment_data is not None and len(sentiment_data) > max_data_points:
                    sentiment_data = sentiment_data.tail(max_data_points)
                    self.logger.info(f"Trimmed sentiment data to {max_data_points} points")
                
                forecast, impact_metrics = update_forecasting_process(
                    price_data, 
                    sentiment_data,
                    sentiment_weight
                )
            
            return forecast, impact_metrics
        
        except (ValueError, TypeError) as e:
            self.logger.error(f"Data validation error: {str(e)}")
            st.error(f"Invalid data: {str(e)}")
            return None, {}
        except MemoryError as e:
            self.logger.error(f"Memory error in forecast processing: {str(e)}")
            st.error("Insufficient memory. Try reducing the data range.")
            return None, {}
        except Exception as e:
            self.logger.error(f"Error in forecast processing: {str(e)}")
            st.error("Failed to process forecast. Please try different parameters.")
            return None, {}

    def run(self) -> None:
        """Main application entry point"""
        try:
            st.set_page_config(
                page_title="HummingBird v2",
                page_icon="🐦",
                layout="wide"
            )

            self.display_header()
            
            forecast_tab, dividend_tab, treasury_tab = st.tabs([
                "📈 Price Forecast", 
                "💰 Dividend Analysis",
                "🏦 Daily Treasury Statement"
            ])
            
            with forecast_tab:
                selected_model, selected_indicator = self.setup_sidebar()
                user_inputs = self.get_user_inputs()
                
                st.markdown("### 📊 Select Forecast Type")
                forecast_type: ForecastType = st.radio(
                    "Choose forecast type",
                    ["Price Only", "Price + Market Sentiment"],
                    help="Choose whether to include market sentiment analysis in the forecast",
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
                    try:
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
                                
                                sentiment_data = self.get_sentiment_data_wrapper(
                                    sentiment_source,
                                    user_inputs.symbol,
                                    sentiment_start,
                                    user_inputs.end_date,
                                    sentiment_period
                                )
                            
                            if price_data is not None:
                                if selected_model != "Prophet":
                                    st.warning(
                                        f"{selected_model} model is currently under development. "
                                        "Using Prophet for forecasting instead."
                                    )
                                
                                with st.spinner('Generating forecast...'):
                                    forecast, impact_metrics = self.process_forecast(
                                        price_data,
                                        sentiment_data,
                                        forecast_type,
                                        user_inputs.periods,
                                        sentiment_weight if forecast_type == "Price + Market Sentiment" else 0.5
                                    )
                                    
                                    if forecast is not None:
                                        st.success("Forecast generated successfully!")
                                        price_data = add_technical_indicators(
                                            price_data,
                                            user_inputs.asset_type
                                        )
                                        display_forecast_results(
                                            price_data,
                                            forecast,
                                            impact_metrics,
                                            forecast_type,
                                            user_inputs.asset_type,
                                            user_inputs.symbol
                                        )
                                    else:
                                        st.error(
                                            "Failed to generate forecast. "
                                            "Please try different parameters."
                                        )
                            else:
                                st.error(
                                    f"Could not load data for {user_inputs.symbol}. "
                                    "Please verify the symbol."
                                )

                    except Exception as e:
                        self.logger.error(
                            f"Error in forecast generation: {str(e)}\n"
                            f"{traceback.format_exc()}"
                        )
                        st.error("An error occurred during forecast generation.")
                        if st.checkbox("Show error details"):
                            st.exception(e)

            # Implementation for dividend_tab and treasury_tab would go here
            with dividend_tab:
                st.info("Dividend analysis feature coming soon!")
            
            with treasury_tab:
                st.info("Treasury statement analysis feature coming soon!")

        except Exception as e:
            self.logger.error(
                f"Application error: {str(e)}\n{traceback.format_exc()}"
            )
            st.error("An unexpected error occurred. Please try again later.")
            if st.checkbox("Show error details"):
                st.exception(e)
        
        finally:
            self.display_footer()

def main():
    """Application entry point"""
    logger = AppLogger()
    app = H
