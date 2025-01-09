# app.py

import streamlit as st
from config import Config, MODEL_DESCRIPTIONS
from data_fetchers import (
    AssetDataFetcher, 
    EconomicIndicators, 
    RealEstateIndicators,
    GDELTDataFetcher,
    IntegratedDataFetcher
)
from forecasting import (
    prophet_forecast,
    create_forecast_plot,
    display_metrics,
    display_economic_indicators,
    display_sentiment_analysis,
    display_components,
    calculate_accuracy
)

def display_footer():
    """Display the application footer"""
    st.markdown("""
        <div style='text-align: center; padding: 10px;'>
            <p>© 2025 AvaResearch LLC. All rights reserved.</p>
        </div>
    """, unsafe_allow_html=True)

def handle_error(error: Exception, context: str = ""):
    """Enhanced error handling with context"""
    error_type = type(error).__name__
    if context:
        st.error(f"Error in {context}: {str(error)}")
    else:
        st.error(f"{error_type}: {str(error)}")
    
    if st.checkbox("Show detailed error information"):
        st.exception(error)

def validate_inputs(symbol: str, periods: int, asset_type: str) -> bool:
    """Validate user inputs"""
    if not symbol:
        st.error("Please enter a symbol.")
        return False
    
    if periods < 7 or periods > 90:
        st.error("Forecast period must be between 7 and 90 days.")
        return False
    
    if asset_type not in Config.ASSET_TYPES:
        st.error("Invalid asset type selected.")
        return False
    
    return True

def initialize_session_state():
    """Initialize or reset session state variables"""
    if 'economic_indicators' not in st.session_state:
        st.session_state.economic_indicators = EconomicIndicators()
    if 'real_estate_indicators' not in st.session_state:
        st.session_state.real_estate_indicators = RealEstateIndicators()
    if 'integrated_data_fetcher' not in st.session_state:
        st.session_state.integrated_data_fetcher = IntegratedDataFetcher()

def main():
    try:
        # Page configuration
        st.set_page_config(
            page_title="HummingBird v2",
            page_icon="🐦",
            layout="wide"
        )
        
        # Initialize session state
        initialize_session_state()
        
        # Display branding
        st.markdown("""
            <div style='text-align: center;'>
                <h1>🐦 HummingBird v2</h1>
                <p><i>Digital Asset Stock Forecasting with Economic and Market Sentiment Indicators</i></p>
                <p>AvaResearch LLC - A Black Collar Production</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Sidebar - Model Selection
        st.sidebar.header("🔮 Select Forecasting Model")
        selected_model = st.sidebar.selectbox(
            "Available Models",
            list(MODEL_DESCRIPTIONS.keys())
        )

        # Display model information
        if selected_model in MODEL_DESCRIPTIONS:
            model_info = MODEL_DESCRIPTIONS[selected_model]
            st.sidebar.markdown(f"### Model Details\n{model_info['description']}")
            
            # Status indicator
            status_color = 'green' if model_info['development_status'] == 'Active' else 'orange'
            st.sidebar.markdown(f"**Status:** <span style='color:{status_color}'>{model_info['development_status']}</span>", 
                              unsafe_allow_html=True)
            
            # Confidence rating
            confidence = model_info['confidence_rating']
            color = 'green' if confidence >= 0.8 else 'orange' if confidence >= 0.7 else 'red'
            st.sidebar.markdown(f"**Confidence Rating:** <span style='color:{color}'>{confidence:.0%}</span>", 
                              unsafe_allow_html=True)
            
            # Technical details
            st.sidebar.markdown(f"**Technical Level:** {model_info.get('technical_level', 'N/A')}")
            st.sidebar.markdown(f"**Computation Speed:** {model_info.get('computation_speed', 'N/A')}")

            if model_info['development_status'] != 'Active':
                st.sidebar.warning("⚠️ This model is under development. Prophet will be used instead.")

        # Sidebar - Data Sources
        st.sidebar.header("📊 Data Sources")
        with st.sidebar.expander("Available Data Sources", expanded=False):
            for source, description in Config.DATA_SOURCES.items():
                st.markdown(f"**{source}**: {description}")
        
        # Sidebar - Market Sentiment
        st.sidebar.header("🌐 Market Sentiment")
        include_sentiment = st.sidebar.checkbox("Include Market Sentiment Analysis", value=True)
        
        # Sidebar - Economic Indicators
        st.sidebar.header("📈 Economic Indicators")
        selected_indicator = st.sidebar.selectbox(
            "Select Economic Indicator",
            ['None'] + list(Config.INDICATORS.keys()),
            format_func=lambda x: Config.INDICATORS.get(x, x) if x != 'None' else x
        )

        # Sidebar - Real Estate Indicators
        st.sidebar.header("🏠 Real Estate Indicators")
        try:
            selected_re_indicator = st.sidebar.selectbox(
                "Select Real Estate Indicator",
                ['None'] + list(Config.REAL_ESTATE_INDICATORS.keys()),
                format_func=lambda x: Config.REAL_ESTATE_INDICATORS[x]['description'] 
                                    if x != 'None' and x in Config.REAL_ESTATE_INDICATORS else x
            )

            if selected_re_indicator != 'None':
                re_info = Config.REAL_ESTATE_INDICATORS[selected_re_indicator]
                st.sidebar.markdown(f"""
                    **Description:** {re_info['description']}  
                    **Status:** ⚠️ {re_info['status']}
                """)
        except Exception as e:
            handle_error(e, "Real Estate Indicators")
            selected_re_indicator = 'None'

        # Input Section
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
        
        # Generate Forecast
        if st.button("🚀 Generate Forecast"):
            if not validate_inputs(symbol, periods, asset_type):
                return

            with st.spinner('Loading data...'):
                try:
                    # Use integrated data fetcher
                    integrated_data = st.session_state.integrated_data_fetcher.fetch_all_data(
                        symbol=symbol,
                        asset_type=asset_type,
                        include_sentiment=include_sentiment,
                        include_economic=(selected_indicator != 'None')
                    )
                    
                    if not integrated_data:
                        st.error(f"Could not load data for {symbol}. Please verify the symbol.")
                        return
                    
                    data = integrated_data['price']
                    sentiment_data = integrated_data.get('sentiment')
                    
                    # Display sentiment analysis if available
                    if include_sentiment and sentiment_data is not None:
                        display_sentiment_analysis(sentiment_data)
                    
                    # Get economic indicator data
                    economic_data = None
                    if selected_indicator != 'None':
                        economic_data = st.session_state.economic_indicators.get_indicator_data(selected_indicator)
                        if economic_data is not None:
                            display_economic_indicators(
                                economic_data, 
                                selected_indicator, 
                                st.session_state.economic_indicators,
                                sentiment_data if include_sentiment else None
                            )
                    
                    # Display Real Estate Indicator status
                    if selected_re_indicator != 'None':
                        st.info(f"Real Estate Indicator '{selected_re_indicator}' is currently under development.")
                    
                    # Model selection and forecasting
                    if selected_model != "Prophet":
                        st.warning(f"{selected_model} model is currently under development. Using Prophet for forecasting instead.")
                    
                    with st.spinner('Generating forecast...'):
                        try:
                            # Generate forecast with sentiment
                            forecast, error = prophet_forecast(
                                data=data,
                                periods=periods,
                                sentiment_data=sentiment_data if include_sentiment else None,
                                economic_data=economic_data
                            )
                            
                            if error:
                                st.error(f"Forecasting error: {error}")
                            elif forecast is not None:
                                # Display results
                                display_metrics(
                                    data=data,
                                    forecast=forecast,
                                    asset_type=asset_type,
                                    symbol=symbol,
                                    sentiment_data=sentiment_data if include_sentiment else None
                                )
                                
                                # Create and display forecast plot
                                fig = create_forecast_plot(
                                    data=data,
                                    forecast=forecast,
                                    model_name="Prophet",
                                    symbol=symbol,
                                    sentiment_data=sentiment_data if include_sentiment else None
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Display forecast components
                                display_components(forecast)
                                
                                # Calculate and display accuracy metrics
                                accuracy_metrics = calculate_accuracy(data['Close'], forecast['yhat'])
                                if accuracy_metrics:
                                    st.subheader("Forecast Accuracy")
                                    metrics_cols = st.columns(len(accuracy_metrics))
                                    for col, (metric, value) in zip(metrics_cols, accuracy_metrics.items()):
                                        col.metric(metric, f"{value:.2f}")
                                
                                with st.expander("View Detailed Forecast Data"):
                                    st.dataframe(forecast)
                        
                        except Exception as e:
                            handle_error(e, "Forecast Generation")
                
                except Exception as e:
                    handle_error(e, "Data Loading")
    
    except Exception as e:
        handle_error(e, "Application")
    
    finally:
        display_footer()

if __name__ == "__main__":
    main()