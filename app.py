# app.py
import streamlit as st
from config import Config, MODEL_DESCRIPTIONS
from data_fetchers import AssetDataFetcher, EconomicIndicators
from forecasting import (
    prophet_forecast,
    create_forecast_plot,
    display_metrics,
    display_economic_indicators
)

def display_footer():
    st.markdown("""
        <div style='text-align: center; padding: 10px;'>
            <p>© 2025 AvaResearch LLC. All rights reserved.</p>
        </div>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="HummingBird v2",
        page_icon="🐦",
        layout="wide"
    )
    
    try:
        # Display branding
        st.markdown("""
            <div style='text-align: center;'>
                <h1>🐦 HummingBird v2</h1>
                <p><i>Digital Asset Stock Forecasting with Economic Indicators</i></p>
                <p>AvaResearch LLC - A Black Collar Production</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Sidebar - Model Selection
        st.sidebar.header("🔮 Select Forecasting Model")
        selected_model = st.sidebar.selectbox(
            "Available Models",
            list(MODEL_DESCRIPTIONS.keys())
        )

        # Sidebar configuration
        if selected_model in MODEL_DESCRIPTIONS:
            model_info = MODEL_DESCRIPTIONS[selected_model]
            st.sidebar.markdown(f"### Model Details\n{model_info['description']}")
            
            status_color = 'green' if model_info['development_status'] == 'Active' else 'orange'
            st.sidebar.markdown(f"**Status:** <span style='color:{status_color}'>{model_info['development_status']}</span>", 
                              unsafe_allow_html=True)
            
            confidence = model_info['confidence_rating']
            color = 'green' if confidence >= 0.8 else 'orange' if confidence >= 0.7 else 'red'
            st.sidebar.markdown(f"**Confidence Rating:** <span style='color:{color}'>{confidence:.0%}</span>", 
                              unsafe_allow_html=True)
            
            st.sidebar.markdown("**Best Use Cases:**")
            for use_case in model_info['best_use_cases']:
                st.sidebar.markdown(f"- {use_case}")
            
            st.sidebar.markdown("**Limitations:**")
            for limitation in model_info['limitations']:
                st.sidebar.markdown(f"- {limitation}")

        # Sidebar - Data Sources
        st.sidebar.header("📊 Available Data Sources")
        for source, description in Config.DATA_SOURCES.items():
            st.sidebar.markdown(f"**{source}**: {description}")

        # Sidebar - Economic Indicators
        st.sidebar.header("📈 Economic Indicators")
        selected_indicator = st.sidebar.selectbox(
            "Select Indicator",
            ['None'] + list(Config.INDICATORS.keys()),
            format_func=lambda x: Config.INDICATORS.get(x, x) if x != 'None' else x
        )

        # Main interface
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

        # Generate forecast when button is clicked
        if st.button("🚀 Generate Forecast"):
            with st.spinner('Loading data...'):
                data = AssetDataFetcher.get_stock_data(symbol) if asset_type == "Stocks" else AssetDataFetcher.get_crypto_data(symbol)
                
                economic_data = None
                if selected_indicator != 'None':
                    economic_indicators = EconomicIndicators()
                    economic_data = economic_indicators.get_indicator_data(selected_indicator)
                    if economic_data is not None:
                        display_economic_indicators(economic_data, selected_indicator, economic_indicators)

                if data is not None:
                    if selected_model != "Prophet":
                        st.warning(f"{selected_model} model is currently under development. Using Prophet for forecasting instead.")
                    
                    with st.spinner('Generating forecast...'):
                        forecast, error = prophet_forecast(data, periods, economic_data)
                        
                        if error:
                            st.error(f"Forecasting error: {error}")
                        elif forecast is not None:
                            # Get current price if available
                            current_price = AssetDataFetcher.get_current_price(symbol, asset_type)
                            
                            # Display metrics with optional current price
                            display_metrics(data, forecast, asset_type, symbol, current_price)
                            
                            # Create and display forecast plot
                            fig = create_forecast_plot(data, forecast, "Prophet", symbol)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show detailed forecast data
                            with st.expander("View Detailed Forecast Data"):
                                st.dataframe(forecast)
                else:
                    st.error(f"Could not load data for {symbol}. Please verify the symbol.")
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.exception(e)
    
    finally:
        display_footer()

if __name__ == "__main__":
    main()