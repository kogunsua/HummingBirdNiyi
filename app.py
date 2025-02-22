#app.py
import streamlit as st
from datetime import datetime, timedelta
import logging
import sys
from typing import Optional, Tuple, Dict
import pandas as pd
import traceback
import yfinance as yf

# Import local modules
from dividend_analyzer import DividendAnalyzer, show_dividend_education, filter_monthly_dividend_stocks
from config import Config, MODEL_DESCRIPTIONS
from data_fetchers import AssetDataFetcher, EconomicIndicators
from forecasting import (
    prophet_forecast,
    create_forecast_plot,
    display_metrics,
    display_confidence_analysis,
    add_technical_indicators
)
from sentiment_analyzer import (
    MultiSourceSentimentAnalyzer,
    display_sentiment_impact_analysis,
    display_sentiment_impact_results,
    get_sentiment_data
)
from gdelt_analysis import GDELTAnalyzer, update_forecasting_process
from treasurydata import TreasuryDataFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def display_header():
    """Display the application header with styling"""
    st.markdown("""
        <div style='text-align: center;'>
            <h1>🐦 HummingBird v2</h1>
            <p><i>Digital Asset & Stock Forecasting with Economic and Sentiment Indicators</i></p>
            <p>AvaResearch LLC - A Black Collar Production</p>
        </div>
    """, unsafe_allow_html=True)

def display_footer():
    """Display the application footer"""
    st.markdown("""
        <div style='text-align: center; padding: 10px; position: fixed; bottom: 0; width: 100%;'>
            <p style='margin-bottom: 10px;'>© 2025 AvaResearch LLC. All rights reserved.</p>
        </div>
    """, unsafe_allow_html=True)

def setup_sidebar() -> Tuple[str, str, Optional[str]]:
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

        st.header("📊 Data Sources")
        with st.expander("View Data Sources"):
            for source, description in Config.DATA_SOURCES.items():
                st.markdown(f"**{source}**: {description}")

        return selected_model, selected_indicator

def get_user_inputs() -> Tuple[str, str, int]:
    """Get and validate user inputs for asset analysis"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        asset_type = st.selectbox(
            "Select Asset Type",
            Config.ASSET_TYPES,
            help="Choose between Stocks and Cryptocurrency"
        )
    
    with col2:
        if asset_type == "Stocks":
            symbol = st.text_input(
                "Enter Stock Symbol",
                Config.DEFAULT_TICKER,
                help="Enter a valid stock symbol (e.g., AAPL, MSFT)"
            ).upper()
        else:
            symbol = st.text_input(
                "Enter Cryptocurrency ID",
                Config.DEFAULT_CRYPTO,
                help="Enter a valid cryptocurrency ID (e.g., bitcoin, xrp)"
            ).lower()
    
    with col3:
        periods = st.slider(
            "Forecast Period (days)",
            7, 90, Config.DEFAULT_PERIODS,
            help="Select the number of days to forecast"
        )
    
    return asset_type, symbol, periods

def update_treasury_tab(treasury_tab):
    """Handle Treasury tab display and analysis"""
    with treasury_tab:
        st.title("Treasury Statement Analysis")
        
        # Create columns for inputs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            end_date = st.date_input(
                "End Date",
                datetime.now().date(),
                max_value=datetime.now().date()
            )
        
        with col2:
            days_back = st.slider(
                "Analysis Period (days)",
                30, 365, 90,
                help="Select the number of days to analyze"
            )
            
        with col3:
            frequency = st.selectbox(
                "Data Frequency",
                ["daily", "weekly", "seven_day_ma"],
                format_func=lambda x: x.replace("_", " ").title(),
                help="Select data frequency for analysis"
            )
        
        start_date = end_date - timedelta(days=days_back)
        
        # Category selection
        categories = st.multiselect(
            "Select Categories to Analyze (max 9)",
            TreasuryDataFetcher().program_categories.keys(),
            default=['SNAP', 'NIH', 'VA'],
            max_selections=9,
            help="Choose up to 9 categories to analyze"
        )
        
        # Analysis button
        if st.button("📊 Analyze Treasury Data"):
            try:
                with st.spinner("Fetching Treasury data..."):
                    # Initialize TreasuryDataFetcher
                    fetcher = TreasuryDataFetcher()
                    
                    # Perform analysis
                    df, alerts, recommendations = fetcher.analyze_treasury_data(
                        start_date,
                        end_date,
                        categories,
                        frequency
                    )
                    
                    if df is not None:
                        # Display summary metrics
                        st.markdown("### 📈 Key Metrics")
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            st.metric(
                                "Total Records",
                                len(df)
                            )
                        
                        with metric_col2:
                            st.metric(
                                "Date Range",
                                f"{df['record_date'].min().strftime('%Y-%m-%d')} to {df['record_date'].max().strftime('%Y-%m-%d')}"
                            )
                        
                        with metric_col3:
                            if 'current_month_gross_rcpt_amt' in df.columns:
                                total_receipts = df['current_month_gross_rcpt_amt'].sum()
                                st.metric(
                                    "Total Receipts",
                                    f"${total_receipts:,.2f}"
                                )
                        
                        # Display alerts and recommendations
                        if alerts:
                            st.markdown("### ⚠️ Alerts")
                            for alert in alerts:
                                st.markdown(f"- {alert['message']}")
                        
                        if recommendations:
                            st.markdown("### 💡 Recommendations")
                            for rec in recommendations:
                                st.markdown(f"- {rec}")
                        
                        # Display visualizations
                        st.markdown("### 📊 Treasury Data Visualization")
                        
                        # Create tabs for different visualizations
                        receipts_tab, outlays_tab = st.tabs(["Receipts", "Outlays"])
                        
                        with receipts_tab:
                            fig_receipts = fetcher.plot_treasury_visualization(
                                df,
                                frequency=frequency,
                                categories=categories,
                                plot_type='receipts'
                            )
                            st.pyplot(fig_receipts)
                        
                        with outlays_tab:
                            fig_outlays = fetcher.plot_treasury_visualization(
                                df,
                                frequency=frequency,
                                categories=categories,
                                plot_type='outlays'
                            )
                            st.pyplot(fig_outlays)
                        
                        # Display raw data
                        with st.expander("🔍 View Raw Data"):
                            st.dataframe(df)
                            
                            # Add download button
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="Download Treasury Data",
                                data=csv,
                                file_name="treasury_data.csv",
                                mime="text/csv",
                            )
                    
                    else:
                        st.error("Failed to fetch Treasury data. Please try again.")
            
            except Exception as e:
                logger.error(f"Treasury analysis error: {str(e)}")
                st.error("An error occurred during Treasury analysis. Please try again.")
                st.exception(e)

def process_forecast(
    price_data: pd.DataFrame,
    sentiment_data: Optional[pd.DataFrame],
    forecast_type: str,
    periods: int,
    sentiment_weight: float = 0.5
) -> Tuple[Optional[pd.DataFrame], Dict]:
    """Process forecast data with or without sentiment analysis"""
    try:
        if forecast_type == "Price Only":
            forecast, error = prophet_forecast(price_data, periods)
            impact_metrics = {}
        else:
            forecast, impact_metrics = update_forecasting_process(
                price_data, 
                sentiment_data,
                sentiment_weight
            )
        
        return forecast, impact_metrics
    
    except Exception as e:
        logger.error(f"Error in forecast processing: {str(e)}")
        st.error("Failed to process forecast. Please try different parameters.")
        return None, {}

def display_forecast_results(
    price_data: pd.DataFrame,
    forecast: pd.DataFrame,
    impact_metrics: Dict,
    forecast_type: str,
    asset_type: str,
    symbol: str
):
    """Display forecast results and visualizations"""
    try:
        # Display metrics section
        st.markdown("### 📊 Market Metrics & Analysis")
        display_metrics(price_data, forecast, asset_type, symbol)
        
        # Display sentiment impact results if available
        if impact_metrics and forecast_type == "Price + Market Sentiment":
            display_sentiment_impact_results(impact_metrics)
        
        # Display forecast plot
        st.markdown("### 📈 Forecast Visualization")
        fig = create_forecast_plot(
            price_data, 
            forecast,
            "Enhanced Prophet" if forecast_type == "Price + Market Sentiment" else "Prophet",
            symbol
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display confidence analysis
        display_confidence_analysis(forecast)
        
        # Display detailed forecast data
        with st.expander("🔍 View Detailed Forecast Data"):
            st.dataframe(
                forecast.style.highlight_max(['yhat'], color='lightgreen')
                .highlight_min(['yhat'], color='lightpink')
            )
            
            # Add download button for forecast data
            csv = forecast.to_csv(index=False)
            st.download_button(
                label="Download Forecast Data",
                data=csv,
                file_name=f"{symbol}_forecast.csv",
                mime="text/csv",
            )
    
    except Exception as e:
        logger.error(f"Error displaying forecast results: {str(e)}")
        st.error("Failed to display forecast results.")

def main():
    try:
        # Page configuration
        st.set_page_config(
            page_title="HummingBird v2",
            page_icon="🐦",
            layout="wide"
        )

        # Create Config instance at the start
        config = Config()
        
        # Display header
        display_header()
        
        # Create tabs for different analyses
        forecast_tab, dividend_tab, treasury_tab = st.tabs([
            "📈 Price Forecast", 
            "💰 Dividend Analysis",
            "🏦 Daily Treasury Statement"
        ])
        
        with forecast_tab:
            # Get sidebar inputs
            selected_model, selected_indicator = setup_sidebar()
            
            # Get main user inputs
            asset_type, symbol, periods = get_user_inputs()
            
            # Forecast Type Selection
            st.markdown("### 📊 Select Forecast Type")
            forecast_type = st.radio(
                "Choose forecast type",
                ["Price Only", "Price + Market Sentiment"],
                help="Choose whether to include market sentiment analysis in the forecast",
                horizontal=True
            )
            
            # Get sentiment settings if needed
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
                
                sentiment_source = st.selectbox(
                    "Sentiment Data Source",
                    ["Multi-Source", "GDELT", "Yahoo Finance", "News API"],
                    help="Choose the source for sentiment analysis"
                )
                
                display_sentiment_impact_analysis(
                    sentiment_period,
                    sentiment_weight,
                    sentiment_source
                )
            
            # Generate Forecast button
            if st.button("🚀 Generate Forecast"):
                try:
                    with st.spinner('Loading data...'):
                        # Get asset data
                        fetcher = AssetDataFetcher()
                        price_data = (
                            fetcher.get_stock_data(symbol) 
                            if asset_type == "Stocks" 
                            else fetcher.get_crypto_data(symbol)
                        )
                        
                        # Get sentiment data if selected
                        if forecast_type == "Price + Market Sentiment":
                            start_date = (
                                datetime.now() - timedelta(days=sentiment_period)
                            ).strftime("%Y-%m-%d")
                            end_date = datetime.now().strftime("%Y-%m-%d")
                            
                            with st.spinner(f'Fetching sentiment data from {sentiment_source}...'):
                                if sentiment_source ==

if sentiment_source == "GDELT":
                                    gdelt_analyzer = GDELTAnalyzer()
                                    sentiment_data = gdelt_analyzer.fetch_sentiment_data(
                                        start_date,
                                        end_date
                                    )
                                else:
                                    analyzer = MultiSourceSentimentAnalyzer()
                                    sentiment_data = get_sentiment_data(
                                        analyzer,
                                        symbol,
                                        start_date,
                                        end_date,
                                        sentiment_source
                                    )
                        
                        if price_data is not None:
                            if selected_model != "Prophet":
                                st.warning(
                                    f"{selected_model} model is currently under development. "
                                    "Using Prophet for forecasting instead."
                                )
                            
                            # Process forecast
                            with st.spinner('Generating forecast...'):
                                forecast, impact_metrics = process_forecast(
                                    price_data,
                                    sentiment_data,
                                    forecast_type,
                                    periods,
                                    sentiment_weight if forecast_type == "Price + Market Sentiment" else 0.5
                                )
                                
                                if forecast is not None:
                                    st.success("Forecast generated successfully!")
                                    
                                    # Add technical indicators
                                    price_data = add_technical_indicators(price_data, asset_type)
                                    
                                    # Display results
                                    display_forecast_results(
                                        price_data,
                                        forecast,
                                        impact_metrics,
                                        forecast_type,
                                        asset_type,
                                        symbol
                                    )
                                else:
                                    st.error("Failed to generate forecast. Please try different parameters.")
                        else:
                            st.error(f"Could not load data for {symbol}. Please verify the symbol.")

                except Exception as e:
                    logger.error(f"Error in forecast generation: {str(e)}")
                    st.error("An error occurred during forecast generation.")
                    st.exception(e)
        
        with dividend_tab:
            st.title("Monthly Dividend Analysis")
            
            # Show education section if desired
            if st.checkbox("💡 Show Dividend Education", value=True):
                show_dividend_education()
            
            # Get stock inputs
            custom_tickers = st.text_input(
                "Enter Stock Symbol",
                Config.DIVIDEND_DEFAULTS['DEFAULT_DIVIDEND_STOCKS'],
                help="Enter stock symbol (e.g., MAIN)"
            )
            
            # Analyze button
            if st.button("🔍 Analyze Dividends"):
                try:
                    # Initialize DividendAnalyzer with Config
                    analyzer = DividendAnalyzer()
                    analyzer.config = config
                    tickers = [t.strip().upper() for t in custom_tickers.split(',')]
                    analyzer.display_dividend_analysis(tickers)
                except Exception as e:
                    logger.error(f"Dividend analysis error: {str(e)}")
                    st.error("An error occurred during dividend analysis. Please try again.")
                    st.exception(e)

        with treasury_tab:
            update_treasury_tab(treasury_tab)

    except Exception as e:
        logger.error(f"Application error: {str(e)}\n{traceback.format_exc()}")
        st.error("An unexpected error occurred. Please try again later.")
        st.exception(e)
    
    finally:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        display_footer()

if __name__ == "__main__":
    main()
                                