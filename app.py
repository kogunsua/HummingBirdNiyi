# app.py
import streamlit as st
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
from sentiment_analyzer import MultiSourceSentimentAnalyzer, integrate_multi_source_sentiment
from gdelt_analysis import GDELTAnalyzer, update_forecasting_process

def display_footer():
    """Display the application footer"""
    st.markdown("""
        <div style='text-align: center; padding: 10px; position: fixed; bottom: 0; width: 100%;'>
            <p style='margin-bottom: 10px;'>© 2025 AvaResearch LLC. All rights reserved.</p>
        </div>
    """, unsafe_allow_html=True)

def main():
    try:
        st.set_page_config(
            page_title="HummingBird v2-m",
            page_icon="🐦",
            layout="wide"
        )
        
        st.markdown("""
            <div style='text-align: center;'>
                <h1>🐦 HummingBird v2m</h1>
                <p><i>Digital Asset Stock Forecasting with Economic and Sentiment Indicators</i></p>
                <p>AvaResearch LLC - A Black Collar Production</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Input Section - First Row
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

        # Forecast Type Selection
        st.markdown("### 📊 Select Forecast Type")
        forecast_type = st.radio(
            "Choose forecast type",
            ["Price Only", "Price + Market Sentiment"],
            help="Choose whether to include market sentiment analysis in the forecast",
            horizontal=True
        )

        # Analysis Options
        if forecast_type == "Price + Market Sentiment":
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
                    ["Multi-Source", "GDELT", "News API", "Yahoo Finance"],
                    help="Choose the source for sentiment analysis"
                )
        
        # Sidebar Content
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
                st.markdown(f"**Status:** <span style='color:{status_color}'>{model_info['development_status']}</span>", 
                           unsafe_allow_html=True)
                
                confidence = model_info['confidence_rating']
                color = 'green' if confidence >= 0.8 else 'orange' if confidence >= 0.7 else 'red'
                st.markdown(f"**Confidence Rating:** <span style='color:{color}'>{confidence:.0%}</span>", 
                           unsafe_allow_html=True)

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

        # Generate Forecast button
        if st.button("🚀 Generate Forecast"):
            try:
                with st.spinner('Loading data...'):
                    # Get asset data
                    fetcher = AssetDataFetcher()
                    price_data = fetcher.get_stock_data(symbol) if asset_type == "Stocks" else fetcher.get_crypto_data(symbol)
                    
                    # Get economic indicator data if selected
                    economic_data = None
                    if selected_indicator != 'None':
                        economic_indicators = EconomicIndicators()
                        economic_data = economic_indicators.get_indicator_data(selected_indicator)
                        if economic_data is not None:
                            display_economic_indicators(economic_data, selected_indicator, economic_indicators)
                    
                    # Get sentiment data if selected
                    sentiment_data = None
                    if forecast_type == "Price + Market Sentiment":
                        if sentiment_source == "Multi-Source":
                            sentiment_data = integrate_multi_source_sentiment(symbol, sentiment_period)
                        elif sentiment_source == "GDELT":
                            gdelt_analyzer = GDELTAnalyzer()
                            sentiment_data = gdelt_analyzer.fetch_sentiment_data(
                                (datetime.now() - timedelta(days=sentiment_period)).strftime("%Y-%m-%d"),
                                datetime.now().strftime("%Y-%m-%d")
                            )
                        else:
                            # Use specific source from multi-source analyzer
                            analyzer = MultiSourceSentimentAnalyzer()
                            sentiment_data = getattr(analyzer, f'fetch_{sentiment_source.lower().replace(" ", "_")}_sentiment')(
                                symbol, 
                                (datetime.now() - timedelta(days=sentiment_period)).strftime("%Y-%m-%d"),
                                datetime.now().strftime("%Y-%m-%d")
                            )
                    
                    if price_data is not None:
                        if selected_model != "Prophet":
                            st.warning(f"{selected_model} model is currently under development. Using Prophet for forecasting instead.")
                        
                        with st.spinner('Generating forecast...'):
                            if forecast_type == "Price Only":
                                # Use regular price forecasting
                                forecast, error = prophet_forecast(price_data, periods)
                                impact_metrics = {}
                            else:
                                # Use sentiment-enhanced forecasting
                                forecast, impact_metrics = update_forecasting_process(
                                    price_data, 
                                    sentiment_data,
                                    sentiment_weight if 'sentiment_weight' in locals() else 0.5
                                )
                            
                            if forecast is not None:
                                # Display results
                                price_data = add_technical_indicators(price_data, asset_type)
                                display_metrics(price_data, forecast, asset_type, symbol)
                                
                                if impact_metrics and forecast_type == "Price + Market Sentiment":
                                    st.subheader("🎭 Sentiment Impact Analysis")
                                    col1, col2, col3 = st.columns(3)
                                    
                                    with col1:
                                        st.metric(
                                            "Sentiment Correlation",
                                            f"{impact_metrics['sentiment_correlation']:.2f}"
                                        )
                                    
                                    with col2:
                                        st.metric(
                                            "Sentiment Volatility",
                                            f"{impact_metrics['sentiment_volatility']:.2f}"
                                        )
                                    
                                    with col3:
                                        st.metric(
                                            "Price Sensitivity",
                                            f"{impact_metrics['price_sensitivity']:.2f}"
                                        )
                                
                                fig = create_forecast_plot(price_data, forecast, 
                                                         "Enhanced Prophet" if forecast_type == "Price + Market Sentiment" else "Prophet", 
                                                         symbol)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                with st.expander("View Detailed Forecast Data"):
                                    st.dataframe(forecast)
                    else:
                        st.error(f"Could not load data for {symbol}. Please verify the symbol.")

            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                st.exception(e)

            finally:
                st.markdown("<br><br><br>", unsafe_allow_html=True)
                display_footer()

    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.exception(e)
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        display_footer()

if __name__ == "__main__":
    main()