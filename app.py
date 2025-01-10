"""
Main application file for HummingBird v2
"""

import streamlit as st
import plotly.graph_objects as go
from config import Config, MODEL_DESCRIPTIONS
from data_fetchers import (
    AssetDataFetcher,
    EconomicIndicators,
    RealEstateIndicators,
    GDELTDataFetcher,
    IntegratedDataFetcher
)
from forecasting import Forecasting


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
    if "economic_indicators" not in st.session_state:
        st.session_state.economic_indicators = EconomicIndicators()
    if "real_estate_indicators" not in st.session_state:
        st.session_state.real_estate_indicators = RealEstateIndicators()
    if "integrated_fetcher" not in st.session_state:
        st.session_state.integrated_fetcher = IntegratedDataFetcher()
    if "gdelt_fetcher" not in st.session_state:
        st.session_state.gdelt_fetcher = GDELTDataFetcher()
    if "forecaster" not in st.session_state:
        st.session_state.forecaster = Forecasting()


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

        # Branding
        st.markdown("""
            <div style='text-align: center;'>
                <h1>🐦 HummingBird v2</h1>
                <p><i>Digital Asset Stock Forecasting with Economic and Market Sentiment Indicators</i></p>
                <p>AvaResearch LLC - A Black Collar Production</p>
            </div>
        """, unsafe_allow_html=True)

        # Sidebar - Model selection
        st.sidebar.header("🔮 Select Forecasting Model")
        selected_model = st.sidebar.selectbox(
            "Choose a forecasting model:",
            options=list(MODEL_DESCRIPTIONS.keys()),
            format_func=lambda x: f"{x}: {MODEL_DESCRIPTIONS[x]}"
        )

        # Sidebar - Forecasting parameters
        st.sidebar.header("🔧 Forecast Parameters")
        symbol = st.sidebar.text_input("Enter the symbol (e.g., AAPL, BTC-USD):")
        asset_type = st.sidebar.selectbox("Asset Type:", Config.ASSET_TYPES)
        forecast_period = st.sidebar.slider("Forecast Period (days):", 7, 90, 30)

        # Additional options
        st.sidebar.header("📊 Additional Indicators")
        include_sentiment = st.sidebar.checkbox("Include Sentiment Analysis", value=True)
        include_economic = st.sidebar.checkbox("Include Economic Indicators", value=True)
        include_market_context = st.sidebar.checkbox("Include Market Context", value=True)

        if validate_inputs(symbol, forecast_period, asset_type):
            # Use IntegratedDataFetcher to get all required data
            data = st.session_state.integrated_fetcher.fetch_all_data(
                symbol,
                asset_type,
                include_sentiment=include_sentiment,
                include_economic=include_economic
            )

            if not data:
                st.error("No data available for the selected symbol.")
                return

            historical_data = data.get('price')
            sentiment_data = data.get('sentiment')
            economic_data = {k: v for k, v in data.items() if k.startswith('economic_')}

            # Display historical data overview
            st.header("📊 Historical Data Overview")
            st.dataframe(historical_data.tail(10))

            # Display market context if enabled
            if include_market_context:
                context = st.session_state.integrated_fetcher.get_enhanced_market_context(
                    symbol,
                    asset_type,
                    sentiment_data
                )
                st.header("🌍 Market Context")
                for key, value in context.items():
                    if isinstance(value, dict):
                        st.write(f"**{key.replace('_', ' ').title()}**")
                        for subkey, subvalue in value.items():
                            st.write(f"- {subkey.replace('_', ' ').title()}: {subvalue}")
                    else:
                        st.write(f"**{key.replace('_', ' ').title()}**: {value}")

            # Prepare data for Prophet
            prophet_data = st.session_state.integrated_fetcher.prepare_prophet_data(
                historical_data,
                sentiment_data,
                economic_data
            )

            # Generate forecast
            forecast_df, error = st.session_state.forecaster.prophet_forecast(
                prophet_data,
                forecast_period,
                sentiment_data=sentiment_data
            )

            if error:
                st.error(f"Error generating forecast: {error}")
                return

            # Display forecast plot
            forecast_plot = st.session_state.forecaster.create_forecast_plot(
                historical_data,
                forecast_df,
                selected_model,
                symbol,
                sentiment_data
            )
            st.plotly_chart(forecast_plot, use_container_width=True)

            # Display metrics and components in tabs
            tab1, tab2, tab3, tab4 = st.tabs(["Metrics", "Components", "Sentiment", "Economic"])

            with tab1:
                st.session_state.forecaster.display_metrics(
                    historical_data,
                    forecast_df,
                    asset_type,
                    symbol,
                    sentiment_data
                )

            with tab2:
                st.session_state.forecaster.display_components(forecast_df)

            with tab3:
                if include_sentiment and sentiment_data is not None:
                    st.session_state.forecaster.display_sentiment_analysis(sentiment_data)
                else:
                    st.info("Enable sentiment analysis in the sidebar to view this section.")

            with tab4:
                if include_economic and economic_data:
                    for indicator in Config.ECONOMIC_CONFIG['indicators']:
                        st.session_state.forecaster.display_economic_indicators(
                            economic_data.get(f'economic_{indicator}'),
                            indicator,
                            st.session_state.economic_indicators,
                            sentiment_data
                        )
                else:
                    st.info("Enable economic indicators in the sidebar to view this section.")

            # Display accuracy metrics
            accuracy_metrics = st.session_state.forecaster.calculate_accuracy(
                historical_data['Close'],
                forecast_df['yhat'][-len(historical_data):]
            )
            
            st.header("📈 Forecast Accuracy")
            metrics_cols = st.columns(len(accuracy_metrics))
            for col, (metric, value) in zip(metrics_cols, accuracy_metrics.items()):
                with col:
                    st.metric(metric, f"{value:.2f}")

    except Exception as e:
        handle_error(e, "main application logic")

    # Footer
    display_footer()


if __name__ == "__main__":
    main()