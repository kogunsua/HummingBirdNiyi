#app.py
#dev
import streamlit as st
from datetime import datetime, timedelta
import logging
import sys
from typing import Optional, Tuple, Dict
import pandas as pd
import traceback
import yfinance as yf

# Import local modules
from config import Config, MODEL_DESCRIPTIONS
from data_fetchers import AssetDataFetcher, EconomicIndicators, RealEstateIndicators
from forecasting import (
    prophet_forecast,
    create_forecast_plot,
    display_metrics,
    display_confidence_analysis,
    display_common_metrics,
    display_crypto_metrics,
    display_economic_indicators,
    add_technical_indicators
)
from sentiment_analyzer import (
    MultiSourceSentimentAnalyzer,
    integrate_multi_source_sentiment,
    display_sentiment_impact_analysis,
    display_sentiment_impact_results,
    get_sentiment_data
)
from gdelt_analysis import GDELTAnalyzer, update_forecasting_process

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

# Dividend Analysis Functions
def get_stock_data(tickers):
    stock_data = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if info.get('dividendYield') and info.get('dividendYield') > 0:
                stock_data.append({
                    'Ticker': ticker,
                    'Monthly Dividend': info.get('lastDividendValue', 0),
                    'Current Price': info.get('currentPrice', 0),
                    'Dividend Yield (%)': info.get('dividendYield', 0) * 100,
                    'Market Cap': info.get('marketCap', 0),
                    'Dividend Frequency': info.get('dividendFrequency', 'Unknown')
                })
        except Exception as e:
            st.warning(f"Could not fetch data for {ticker}: {str(e)}")
    
    return pd.DataFrame(stock_data)

def filter_monthly_dividend_stocks(data):
    return data[data['Dividend Frequency'] == 'Monthly']

def format_market_cap(value):
    if value >= 1e12:
        return f"${value/1e12:.2f}T"
    elif value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.2f}M"
    else:
        return f"${value:,.2f}"

def display_dividend_analysis(tickers=None):
    if tickers is None:
        tickers = ['O', 'MAIN', 'STAG', 'GOOD', 'AGNC']
    
    with st.spinner("Analyzing monthly dividend stocks..."):
        stock_data = get_stock_data(tickers)
        
        if stock_data.empty:
            st.warning("No stocks found with dividend information.")
            return
        
        # Create three columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Stocks Analyzed",
                len(stock_data),
                help="Total number of stocks analyzed for dividend payments"
            )
        
        with col2:
            avg_yield = stock_data['Dividend Yield (%)'].mean()
            st.metric(
                "Average Dividend Yield",
                f"{avg_yield:.2f}%",
                help="Average dividend yield across all analyzed stocks"
            )
        
        with col3:
            monthly_count = len(filter_monthly_dividend_stocks(stock_data))
            st.metric(
                "Monthly Dividend Stocks",
                monthly_count,
                help="Number of stocks that pay monthly dividends"
            )
        
        # Display all stocks data
        st.subheader("📊 All Dividend Stocks")
        formatted_data = stock_data.copy()
        formatted_data['Market Cap'] = formatted_data['Market Cap'].apply(format_market_cap)
        formatted_data['Current Price'] = formatted_data['Current Price'].apply(lambda x: f"${x:,.2f}")
        formatted_data['Monthly Dividend'] = formatted_data['Monthly Dividend'].apply(lambda x: f"${x:.4f}")
        st.dataframe(formatted_data)
        
        # Filter and display monthly dividend stocks
        monthly_stocks = filter_monthly_dividend_stocks(stock_data)
        if not monthly_stocks.empty:
            st.subheader("🎯 Top Monthly Dividend Stocks")
            sorted_stocks = monthly_stocks.sort_values(by='Dividend Yield (%)', ascending=False)
            top_stocks = sorted_stocks.head(3)
            
            # Display top stocks in cards
            for _, stock in top_stocks.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div style='padding: 10px; border: 1px solid #ddd; border-radius: 5px; margin: 10px 0;'>
                        <h3 style='margin: 0;'>{stock['Ticker']} - {stock['Dividend Yield (%)']:.2f}% Yield</h3>
                        <p>Monthly Dividend: ${stock['Monthly Dividend']:.4f}</p>
                        <p>Current Price: ${stock['Current Price']:,.2f}</p>
                        <p>Market Cap: {format_market_cap(stock['Market Cap'])}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No monthly dividend stocks found in the analyzed set.")

def display_header():
    """Display the application header with styling"""
    st.markdown("""
        <div style='text-align: center;'>
            <h1>🐦 HummingBird v2m</h1>
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
                help="Enter a valid cryptocurrency ID (e.g., bitcoin, ethereum)"
            ).lower()
    
    with col3:
        periods = st.slider(
            "Forecast Period (days)",
            7, 90, Config.DEFAULT_PERIODS,
            help="Select the number of days to forecast"
        )
    
    return asset_type, symbol, periods

def get_dividend_inputs() -> list:
    """Get user inputs for dividend analysis"""
    st.markdown("### 💰 Monthly Dividend Stock Analysis")
    
    # Allow users to input custom tickers
    custom_tickers = st.text_input(
        "Enter Stock Symbols (comma-separated)",
        "O,MAIN,STAG,GOOD,AGNC",
        help="Enter stock symbols separated by commas (e.g., O,MAIN,STAG)"
    )
    
    # Convert input string to list and clean up
    tickers = [ticker.strip().upper() for ticker in custom_tickers.split(',')]
    return tickers

def get_sentiment_settings() -> Tuple[int, float, str]:
    """Get user settings for sentiment analysis"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sentiment_period = st.slider(
            "Sentiment Analysis Period (days)",
            7, 90, 30,
            help="Historical period for sentiment analysis"
        )
    
    with col2:
        sentiment_weight = st.slider(
            "Sentiment Impact Weight",
            0.0, 1.0, 0.5,
            help="Weight of sentiment analysis in the forecast (0 = none, 1 = maximum)"
        )
    
    with col3:
        sentiment_source = st.selectbox(
            "Sentiment Data Source",
            ["Multi-Source", "GDELT", "Yahoo Finance", "News API"],
            help="Choose the source for sentiment analysis"
        )
    
    return sentiment_period, sentiment_weight, sentiment_source

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
    """Main application function"""
    try:
        # Setup page configuration
        st.set_page_config(
            page_title="HummingBird v2-m",
            page_icon="🐦",
            layout="wide"
        )
        
        # Display header
        display_header()
        
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
                    if selected_indicator != 'None':
                        economic_indicators = EconomicIndicators()
                        economic_data = economic_indicators.get_indicator_data(selected_indicator)
                        if economic_data is not None:
                            display_economic_indicators(economic_data, selected_indicator, economic_indicators)
                    
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
                                st.error("Failed to generate forecast. Please try again with different parameters.")
                    else:
                        st.error(f"Could not load data for {symbol}. Please verify the symbol.")

            except Exception as e:
                logger.error(f"Error in forecast generation: {str(e)}")
                st.error("An error occurred during forecast generation.")
                st.exception(e)

        # Add a separator
        st.markdown("---")
        
        # Monthly Dividend Analysis Section
        st.markdown("## 💰 Monthly Dividend Analysis")
        st.markdown("Analyze stocks that pay monthly dividends")
        
        # Get dividend analysis inputs
        dividend_tickers = get_dividend_inputs()
        
        # Add Monthly Dividend Analysis button
        if st.button("📊 Analyze Monthly Dividends"):
            try:
                display_dividend_analysis(dividend_tickers)
            except Exception as e:
                logger.error(f"Error in dividend analysis: {str(e)}")
                st.error("An error occurred during dividend analysis. Please try again.")
                st.exception(e)

    except Exception as e:
        logger.error(f"Application error: {str(e)}\n{traceback.format_exc()}")
        st.error("An unexpected error occurred. Please try again later.")
        st.exception(e)
    
    finally:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        display_footer()

if __name__ == "__main__":
    main()