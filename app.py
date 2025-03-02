# app.py
import streamlit as st
from datetime import datetime, timedelta
import logging
import sys
from typing import Optional, Tuple, Dict
import pandas as pd
import traceback
import yfinance as yf
import os
import importlib.util

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

# portfolio imports
from portfolio import (
    load_portfolio_data,
    calculate_portfolio_metrics,
    update_portfolio_prices,
    add_portfolio_position,
    remove_portfolio_position,
    get_income_projection,
    get_portfolio_performance_history
)

# Add these imports at the top of app.py
import plotly.graph_objects as go
from etf_analysis import compare_etfs, get_etf_sector_exposure
from crypto_analysis import get_crypto_data, analyze_crypto, forecast_crypto

# Import local modules
from dividend_analyzer import DividendAnalyzer, show_dividend_education, filter_monthly_dividend_stocks
from config import Config, MODEL_DESCRIPTIONS
from data_fetchers import AssetDataFetcher, EconomicIndicators

# Import forecasting module functions directly
from forecasting import (
    add_technical_indicators,
    prophet_forecast,
    create_forecast_plot,
    display_metrics,
    display_economic_indicators as display_economic_indicator_details,
    display_common_metrics,
    display_confidence_analysis
)

# Try to import from sentiment_analyzer
try:
    from sentiment_analyzer import (
        MultiSourceSentimentAnalyzer,
        get_sentiment_data
    )
except ImportError:
    logger.warning("Could not import from sentiment_analyzer, some functionality may be limited")
    # Create placeholder functions if needed
    def get_sentiment_data(*args, **kwargs):
        st.warning("Sentiment analysis functionality is not available")
        return None
    
    class MultiSourceSentimentAnalyzer:
        def __init__(self):
            pass

# Try to import from gdelt_analysis
try:
    from gdelt_analysis import GDELTAnalyzer, update_forecasting_process
except ImportError:
    logger.warning("Could not import from gdelt_analysis, some functionality may be limited")
    # Create placeholder functions
    class GDELTAnalyzer:
        def __init__(self):
            pass
        
        def fetch_sentiment_data(self, *args, **kwargs):
            return None
    
    def update_forecasting_process(*args, **kwargs):
        return None, {}

# Import Treasury modules
from treasury_interface import display_treasury_dashboard

# Try to import from forecast_display if it exists
try:
    from forecast_display import (
        display_forecast_results,
        display_sentiment_impact_analysis,
        display_sentiment_impact_results
    )
except ImportError:
    logger.warning("Could not import from forecast_display, using built-in functions")
    
    # Define placeholder functions that use the imported forecasting functions
    def display_forecast_results(price_data, forecast, impact_metrics, forecast_type, asset_type, symbol):
        """Display forecast results using built-in functions"""
        try:
            # Create and display plot
            fig = create_forecast_plot(price_data, forecast, "Prophet", symbol, asset_type)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display metrics
            display_metrics(price_data, forecast, asset_type, symbol)
            
        except Exception as e:
            logger.error(f"Error displaying forecast results: {str(e)}")
            st.error(f"Error displaying forecast results: {str(e)}")
    
    def display_sentiment_impact_analysis(sentiment_period, sentiment_weight, sentiment_source):
        """Placeholder for sentiment impact analysis display"""
        st.subheader("🔍 Sentiment Analysis Configuration")
        st.info(f"Analyzing sentiment data from {sentiment_source} for the past {sentiment_period} days with a weight of {sentiment_weight:.2f}")
    
    def display_sentiment_impact_results(sentiment_data, impact_metrics):
        """Placeholder for sentiment impact results display"""
        if sentiment_data is not None:
            st.subheader("📊 Sentiment Impact Analysis")
            st.info("Sentiment analysis has been incorporated into the forecast")

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

def setup_sidebar() -> Tuple[str, str]:
    """Setup and handle sidebar inputs"""
    with st.sidebar:
        st.header("🔮 Model Configuration")
        selected_model = st.selectbox(
            "Select Forecasting Model",
            list(MODEL_DESCRIPTIONS.keys()),
            key="model_selection"  # Add unique key
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
            format_func=lambda x: Config.INDICATORS.get(x, x) if x != 'None' else x,
            key="economic_indicator_selection"  # Add unique key
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
            help="Choose between Stocks and Cryptocurrency",
            key="asset_type_selection"  # Add unique key
        )
    
    with col2:
        if asset_type == "Stocks":
            symbol = st.text_input(
                "Enter Stock Symbol",
                Config.DEFAULT_TICKER,
                help="Enter a valid stock symbol (e.g., AAPL, MSFT)",
                key="stock_symbol_input"  # Add unique key
            ).upper()
        else:
            symbol = st.text_input(
                "Enter Cryptocurrency ID",
                Config.DEFAULT_CRYPTO,
                help="Enter a valid cryptocurrency ID (e.g., bitcoin, xrp)",
                key="crypto_symbol_input"  # Add unique key
            ).lower()
    
    with col3:
        periods = st.slider(
            "Forecast Period (days)",
            7, 90, Config.DEFAULT_PERIODS,
            help="Select the number of days to forecast",
            key="forecast_periods_slider"  # Add unique key
        )
    
    return asset_type, symbol, periods

def get_sentiment_settings() -> Tuple[int, float, str]:
    """Get user settings for sentiment analysis"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sentiment_period = st.slider(
            "Sentiment Analysis Period (days)",
            7, 90, 30,
            help="Historical period for sentiment analysis",
            key="sentiment_analysis_period"  # Add unique key
        )
    
    with col2:
        sentiment_weight = st.slider(
            "Sentiment Impact Weight",
            0.0, 1.0, 0.5,
            help="Weight of sentiment analysis in the forecast (0 = none, 1 = maximum)",
            key="sentiment_impact_weight"  # Add unique key
        )
    
    with col3:
        sentiment_source = st.selectbox(
            "Sentiment Data Source",
            ["Multi-Source", "GDELT", "Yahoo Finance", "News API"],
            help="Choose the source for sentiment analysis",
            key="sentiment_source_selection"  # Add unique key
        )
    
    return sentiment_period, sentiment_weight, sentiment_source

def process_forecast(
    price_data: pd.DataFrame,
    sentiment_data: Optional[pd.DataFrame],
    forecast_type: str,
    periods: int,
    sentiment_weight: float = 0.5,
    economic_data: Optional[pd.DataFrame] = None,
    economic_indicator: Optional[str] = None,
    asset_type: str = 'stocks'
) -> Tuple[Optional[pd.DataFrame], Dict]:
    """Process forecast data with or without sentiment analysis"""
    try:
        if forecast_type == "Price Only":
            # Use the imported prophet_forecast function
            forecast, error = prophet_forecast(
                price_data, 
                periods, 
                economic_data=economic_data,
                indicator=economic_indicator,
                asset_type=asset_type.lower()
            )
            
            if error:
                st.error(f"Forecast error: {error}")
                return None, {}
                
            impact_metrics = {"error": error}
            
        else:
            # Use sentiment-enhanced forecasting if available
            try:
                forecast, impact_metrics = update_forecasting_process(
                    price_data, 
                    sentiment_data,
                    sentiment_weight,
                    economic_data=economic_data,
                    indicator=economic_indicator
                )
            except Exception as e:
                logger.error(f"Sentiment forecasting failed, falling back to basic forecast: {str(e)}")
                st.warning("Sentiment analysis integration failed. Falling back to basic forecast.")
                
                # Fallback to basic forecast
                forecast, error = prophet_forecast(
                    price_data, 
                    periods,
                    economic_data=economic_data,
                    indicator=economic_indicator,
                    asset_type=asset_type.lower()
                )
                
                if error:
                    st.error(f"Forecast error: {error}")
                    return None, {}
                    
                impact_metrics = {"error": error}
        
        return forecast, impact_metrics
    
    except Exception as e:
        logger.error(f"Error in forecast processing: {str(e)}")
        st.error("Failed to process forecast. Please try different parameters.")
        return None, {}

def display_economic_indicators_ui(economic_data, selected_indicator, economic_indicators):
    """Display economic indicators data"""
    st.markdown("### 📉 Economic Indicators")
    st.write(f"Selected indicator: {Config.INDICATORS.get(selected_indicator, selected_indicator)}")
    
    # Create visualization for the economic indicator
    fig = economic_indicators.create_indicator_plot(economic_data, selected_indicator)
    st.plotly_chart(fig, use_container_width=True)
    
    # Use the imported display function for detailed analysis
    display_economic_indicator_details(economic_data, selected_indicator, economic_indicators)

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
        forecast_tab, dividend_tab, portfolio_tab, etf_tab, crypto_tab, treasury_tab = st.tabs([
            "📈 Price Forecast", 
            "💰 Dividend Analysis",
            "💼 Portfolio Management", 
            "🏢 ETF Analysis", 
            "🪙 Cryptocurrency", 
            "🏦 Treasury Analysis"
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
                sentiment_period, sentiment_weight, sentiment_source = get_sentiment_settings()
                display_sentiment_impact_analysis(sentiment_period, sentiment_weight, sentiment_source)
            
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
                        
                        # Get economic indicator data if selected
                        economic_data = None
                        if selected_indicator != 'None':
                            economic_indicators = EconomicIndicators()
                            economic_data = economic_indicators.get_indicator_data(selected_indicator)
                            if economic_data is not None:
                                display_economic_indicators_ui(economic_data, selected_indicator, economic_indicators)
                        
                        # Get sentiment data if selected
                        if forecast_type == "Price + Market Sentiment":
                            start_date = (datetime.now() - timedelta(days=sentiment_period)).strftime("%Y-%m-%d")
                            end_date = datetime.now().strftime("%Y-%m-%d")
                            
                            with st.spinner(f'Fetching sentiment data from {sentiment_source}...'):
                                if sentiment_source == "GDELT":
                                    gdelt_analyzer = GDELTAnalyzer()
                                    sentiment_data = gdelt_analyzer.fetch_sentiment_data(start_date, end_date)
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
                                st.warning(f"{selected_model} model is currently under development. Using Prophet for forecasting instead.")
                            
                            # Add technical indicators
                            with st.spinner('Applying technical indicators...'):
                                price_data = add_technical_indicators(price_data, asset_type.lower())
                            
                            # Process forecast
                            with st.spinner('Generating forecast...'):
                                forecast, impact_metrics = process_forecast(
                                    price_data,
                                    sentiment_data,
                                    forecast_type,
                                    periods,
                                    sentiment_weight if forecast_type == "Price + Market Sentiment" else 0.5,
                                    economic_data,
                                    selected_indicator,
                                    asset_type
                                )

                                if forecast is not None:
                                    st.success("Forecast generated successfully!")
                                    
                                    # Display results
                                    display_forecast_results(
                                        price_data,
                                        forecast,
                                        impact_metrics,
                                        forecast_type,
                                        asset_type.lower(),
                                        symbol
                                    )
                                    
                                    # Display sentiment impact if available
                                    if forecast_type == "Price + Market Sentiment" and sentiment_data is not None:
                                        display_sentiment_impact_results(sentiment_data, impact_metrics)
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
                help="Enter stock symbol (e.g. MAIN)"
            )
            
            # Analyze button
            if st.button("🔍 Analyze Dividends"):
                try:
                    # Initialize DividendAnalyzer with Config
                    analyzer = DividendAnalyzer()
                    analyzer.config = config  # Add this line
                    tickers = [t.strip().upper() for t in custom_tickers.split(',')]
                    analyzer.display_dividend_analysis(tickers)
                except Exception as e:
                    logger.error(f"Dividend analysis error: {str(e)}")
                    st.error("An error occurred during dividend analysis. Please try again.")
                    st.exception(e)
        
        with portfolio_tab:
            st.title("Portfolio Management")
            
            # Load current portfolio
            portfolio_data = load_portfolio_data()
            
            # Display portfolio metrics
            st.subheader("Portfolio Overview")
            metrics = calculate_portfolio_metrics(portfolio_data)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Value", f"${metrics['total_value']:,.2f}")
            with col2:
                st.metric("Monthly Income", f"${metrics['monthly_income']:,.2f}")
            with col3:
                st.metric("Avg Dividend Yield", f"{metrics['avg_yield']:.2f}%")
            
            # Portfolio management actions
            st.subheader("Portfolio Actions")
            action = st.selectbox("Select Action", [
                "Add Position", 
                "Remove Position", 
                "Update Prices", 
                "Income Projection",
                "Performance History"
            ])
            
            if action == "Add Position":
                symbol = st.text_input("Enter Stock Symbol").upper()
                quantity = st.number_input("Enter Quantity", min_value=0.0, step=0.001)
                price = st.number_input("Enter Price (Optional)", min_value=0.0, step=0.01, value=0.0)
                
                if st.button("Add Position"):
                    try:
                        price_input = price if price > 0 else None
                        updated_portfolio = add_portfolio_position(portfolio_data, symbol, quantity, price_input)
                        st.success(f"Added {quantity} shares of {symbol}")
                        portfolio_data = updated_portfolio
                    except Exception as e:
                        st.error(f"Error adding position: {str(e)}")
            
            elif action == "Remove Position":
                symbol = st.text_input("Enter Stock Symbol").upper()
                quantity = st.number_input("Enter Quantity to Remove (Leave 0 to remove entire position)", min_value=0.0, step=0.001)
                
                if st.button("Remove Position"):
                    try:
                        quantity_input = quantity if quantity > 0 else None
                        updated_portfolio = remove_portfolio_position(portfolio_data, symbol, quantity_input)
                        st.success(f"Removed {'entire position' if quantity_input is None else f'{quantity} shares'} of {symbol}")
                        portfolio_data = updated_portfolio
                    except Exception as e:
                        st.error(f"Error removing position: {str(e)}")
            
            elif action == "Update Prices":
                if st.button("Refresh Portfolio Prices"):
                    try:
                        updated_portfolio = update_portfolio_prices(portfolio_data)
                        st.success("Portfolio prices updated")
                        portfolio_data = updated_portfolio
                    except Exception as e:
                        st.error(f"Error updating prices: {str(e)}")
            
            elif action == "Income Projection":
                months = st.slider("Projection Period (Months)", 3, 24, 12)
                if st.button("Generate Income Projection"):
                    try:
                        income_projection = get_income_projection(portfolio_data, months)
                        st.subheader("Monthly Dividend Income Projection")
                        st.line_chart(income_projection.set_index('Month')['Income'])
                        st.dataframe(income_projection)
                    except Exception as e:
                        st.error(f"Error generating income projection: {str(e)}")
            
            elif action == "Performance History":
                st.subheader("Portfolio Performance")
                start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
                end_date = st.date_input("End Date", datetime.now())
                
                if st.button("Get Performance"):
                    try:
                        performance = get_portfolio_performance_history(portfolio_data, start_date, end_date)
                        st.line_chart(performance['Value'])
                        st.subheader("Detailed Performance Metrics")
                        st.dataframe(performance)
                    except Exception as e:
                        st.error(f"Error retrieving performance history: {str(e)}")
        
        with etf_tab:
            st.title("ETF Analysis")
            
            # ETF Comparison
            st.subheader("ETF Comparison")
            default_etfs = ['VOO', 'SPY', 'VTI']
            comparison_etfs = st.multiselect(
                "Select ETFs to Compare", 
                default_etfs, 
                default=default_etfs
            )
            
            start_date = st.date_input("Start Date for Comparison", datetime.now() - timedelta(days=365))
            end_date = st.date_input("End Date for Comparison", datetime.now())
            
            if st.button("Compare ETFs"):
                try:
                    comparison_results = compare_etfs(comparison_etfs, start_date, end_date)
                    st.dataframe(comparison_results)
                except Exception as e:
                    st.error(f"ETF comparison error: {str(e)}")
            
            # Sector Exposure
            st.subheader("ETF Sector Exposure")
            sector_etf = st.text_input("Enter ETF Symbol for Sector Analysis", "VOO")
            
            if st.button("Get Sector Exposure"):
                try:
                    sector_exposure = get_etf_sector_exposure(sector_etf)
                    if sector_exposure:
                        # Convert to DataFrame for better display
                        sector_df = pd.DataFrame.from_dict(sector_exposure, orient='index', columns=['Percentage'])
                        sector_df.index.name = 'Sector'
                        sector_df.reset_index(inplace=True)
                        sector_df.sort_values('Percentage', ascending=False, inplace=True)
                        
                        # Display as table
                        st.dataframe(sector_df.style.format({'Percentage': '{:.2f}%'}))
                        
                        # Display pie chart
                        fig = go.Figure(data=[go.Pie(
                            labels=sector_df['Sector'], 
                            values=sector_df['Percentage'], 
                            hole=.3
                        )])
                        fig.update_layout(title=f"Sector Exposure for {sector_etf}")
                        st.plotly_chart(fig)
                    else:
                        st.warning("No sector exposure data found.")
                except Exception as e:
                    st.error(f"Sector exposure error: {str(e)}")
        
        with crypto_tab:
            st.title("Cryptocurrency Analysis")
            
            # Crypto selection
            crypto_symbol = st.text_input("Enter Cryptocurrency Symbol", "BTC")
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
            end_date = st.date_input("End Date", datetime.now())
            
            # Forecast method selection
            forecast_method = st.selectbox(
                "Select Forecast Method", 
                ["ARIMA", "Prophet", "Linear Regression"]
            )
            
            if st.button("Analyze Cryptocurrency"):
                try:
                    # Get crypto data
                    crypto_data = get_crypto_data(crypto_symbol, start_date, end_date)
                    
                    # Analyze data
                    analysis = analyze_crypto(crypto_data)
                    
                    # Display basic metrics
                    st.subheader(f"{analysis['Name']} Cryptocurrency Analysis")
                    
                    # Display metrics in columns
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Market Cap", analysis['Market Cap'])
                        st.metric("24h Volume", analysis['24h Volume'])
                    
                    with col2:
                        st.metric("7D Return", analysis['7D Return'])
                        st.metric("30D Return", analysis['30D Return'])
                    
                    with col3:
                        st.metric("YTD Return", analysis['YTD Return'])
                        st.metric("Volatility", analysis['Volatility (Annual)'])
                    
                    # Additional metrics
                    st.markdown(f"**Sentiment Score:** {analysis['Sentiment Score']}")
                    st.markdown(f"**RSI:** {analysis['RSI']}")
                    
                    # Forecast
                    st.subheader("Price Forecast")
                    forecast_data = forecast_crypto(crypto_data, method=forecast_method)
                    
                    # Plot forecast
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=crypto_data.index, 
                        y=crypto_data['Close'], 
                        mode='lines', 
                        name='Historical Price'
                    ))
                    fig.add_trace(go.Scatter(
                        x=forecast_data.index, 
                        y=forecast_data['Forecast'], 
                        mode='lines', 
                        name='Forecast'
                    ))
                    fig.update_layout(
                        title=f"{crypto_symbol.upper()} Price Forecast",
                        xaxis_title="Date",
                        yaxis_title="Price (USD)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Cryptocurrency analysis error: {str(e)}")
        
        with treasury_tab:
            # Call the imported treasury dashboard function directly
            display_treasury_dashboard()

    except Exception as e:
        logger.error(f"Application error: {str(e)}\n{traceback.format_exc()}")
        st.error("An unexpected error occurred. Please try again later.")
        st.exception(e)
    
    finally:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        display_footer()

if __name__ == "__main__":
    main()