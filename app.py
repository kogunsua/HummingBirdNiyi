import streamlit as st
from datetime import datetime, timedelta
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
from typing import Optional
import pandas as pd

def get_sentiment_data(analyzer, symbol: str, start_date: str, end_date: str, sentiment_source: str) -> Optional[pd.DataFrame]:
    """Get sentiment data from specified source"""
    try:
        # Map sentiment sources to method names
        source_method_map = {
            "Yahoo Finance": "yahoo",
            "News API": "newsapi",
            "Finnhub": "finnhub",
            "Multi-Source": "combined"
        }
        
        # Get the correct method name from the map
        method_name = source_method_map.get(sentiment_source)
        if not method_name:
            st.error(f"Invalid sentiment source: {sentiment_source}")
            return None
            
        # Construct the full method name
        method_name = f"fetch_{method_name}_sentiment"
        
        # Get the method and call it
        sentiment_method = getattr(analyzer, method_name, None)
        if sentiment_method is None:
            st.error(f"Method {method_name} not found in analyzer")
            return None
            
        return sentiment_method(symbol, start_date, end_date)
        
    except Exception as e:
        st.error(f"Error getting sentiment data: {str(e)}")
        return None

def display_footer():
    """Display the application footer"""
    st.markdown("""
        <div style='text-align: center; padding: 10px; position: fixed; bottom: 0; width: 100%;'>
            <p style='margin-bottom: 10px;'>© 2025 AvaResearch LLC. All rights reserved.</p>
        </div>
    """, unsafe_allow_html=True)

def display_sentiment_impact_analysis(sentiment_period: int, sentiment_weight: float, sentiment_source: str):
    """Display sentiment impact analysis configuration and explanation"""
    st.markdown("### 🎭 Sentiment Impact Analysis")
    
    # Configure columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Analysis Period",
            f"{sentiment_period} days",
            help="Historical period used for sentiment analysis"
        )
    
    with col2:
        impact_level = (
            "High" if sentiment_weight > 0.7
            else "Medium" if sentiment_weight > 0.3
            else "Low"
        )
        impact_color = (
            "🔴" if sentiment_weight > 0.7
            else "🟡" if sentiment_weight > 0.3
            else "🟢"
        )
        st.metric(
            "Impact Level",
            f"{impact_level} {impact_color}",
            f"{sentiment_weight:.1%}",
            help="Level of influence sentiment has on forecast"
        )
    
    with col3:
        source_reliability = {
            "Multi-Source": {"level": "High", "confidence": 0.9},
            "GDELT": {"level": "Medium-High", "confidence": 0.8},
            "Yahoo Finance": {"level": "Medium", "confidence": 0.7},
            "News API": {"level": "Medium", "confidence": 0.7}
        }
        
        reliability_info = source_reliability.get(sentiment_source, {"level": "Medium", "confidence": 0.7})
        st.metric(
            "Source Reliability",
            reliability_info['level'],
            f"{reliability_info['confidence']:.0%}",
            help="Reliability of the selected sentiment data source"
        )
    
    # Display impact explanation
    with st.expander("💡 Understanding Sentiment Impact"):
        st.markdown("""
        **How Sentiment Affects the Forecast:**
        
        1. **Analysis Period** (Historical Window)
           - Longer periods provide more stable analysis
           - Shorter periods capture recent market sentiment
           - Optimal period varies by asset volatility
        
        2. **Impact Level** (Weight)
           - High (>70%): Strong sentiment influence
           - Medium (30-70%): Balanced price-sentiment mix
           - Low (<30%): Minimal sentiment adjustment
        
        3. **Source Reliability**
           - Multi-Source: Highest reliability (combined sources)
           - GDELT: Global event impact
           - News/Finance API: Market-specific sentiment
        """)

def display_sentiment_impact_results(impact_metrics: dict):
    """Display sentiment impact analysis results"""
    st.subheader("🎭 Sentiment Impact Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        correlation = impact_metrics.get('sentiment_correlation', 0)
        correlation_color = (
            "🟢" if abs(correlation) > 0.7
            else "🟡" if abs(correlation) > 0.3
            else "🔴"
        )
        st.metric(
            "Price-Sentiment Correlation",
            f"{correlation:.2f} {correlation_color}"
        )
    
    with col2:
        volatility = impact_metrics.get('sentiment_volatility', 0)
        volatility_color = (
            "🔴" if volatility > 0.7
            else "🟡" if volatility > 0.3
            else "🟢"
        )
        st.metric(
            "Sentiment Volatility",
            f"{volatility:.2f} {volatility_color}"
        )
    
    with col3:
        sensitivity = impact_metrics.get('price_sensitivity', 0)
        sensitivity_color = (
            "🟡" if sensitivity > 0.7
            else "🟢" if sensitivity > 0.3
            else "🔴"
        )
        st.metric(
            "Price Sensitivity",
            f"{sensitivity:.2f} {sensitivity_color}"
        )
    
    # Add impact interpretation
    with st.expander("📊 Impact Analysis Interpretation"):
        st.markdown(f"""
        **Current Market Sentiment Analysis:**
        
        1. **Correlation** ({correlation:.2f}):
           - {
            "Strong price-sentiment relationship" if abs(correlation) > 0.7
            else "Moderate price-sentiment relationship" if abs(correlation) > 0.3
            else "Weak price-sentiment relationship"
           }
        
        2. **Volatility** ({volatility:.2f}):
           - {
            "High sentiment volatility - exercise caution" if volatility > 0.7
            else "Moderate sentiment volatility" if volatility > 0.3
            else "Low sentiment volatility - stable sentiment"
           }
        
        3. **Price Sensitivity** ({sensitivity:.2f}):
           - {
            "High price sensitivity to sentiment" if sensitivity > 0.7
            else "Moderate price sensitivity" if sensitivity > 0.3
            else "Low price sensitivity to sentiment"
           }
        """)
        
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

        # Analysis Options with Enhanced Sentiment Impact
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
                    ["Multi-Source", "GDELT", "Yahoo Finance", "News API"],
                    help="Choose the source for sentiment analysis"
                )
            
            # Display sentiment impact analysis in UI
            display_sentiment_impact_analysis(sentiment_period, sentiment_weight, sentiment_source)
        
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
                        start_date = (datetime.now() - timedelta(days=sentiment_period)).strftime("%Y-%m-%d")
                        end_date = datetime.now().strftime("%Y-%m-%d")
                        
                        # Display sentiment data loading status
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
                            
                            if sentiment_data is None:
                                st.warning(f"Could not fetch sentiment data from {sentiment_source}. Proceeding with price-only forecast.")
                    
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
                                st.success("Forecast generated successfully!")
                                
                                # Add technical indicators
                                price_data = add_technical_indicators(price_data, asset_type)
                                
                                # Display metrics section
                                st.markdown("### 📊 Market Metrics & Analysis")
                                display_metrics(price_data, forecast, asset_type, symbol)
                                
                                # Display sentiment impact results if available
                                if impact_metrics and forecast_type == "Price + Market Sentiment":
                                    display_sentiment_impact_results(impact_metrics)
                                
                                # Display forecast plot
                                st.markdown("### 📈 Forecast Visualization")
                                fig = create_forecast_plot(price_data, forecast, 
                                                         "Enhanced Prophet" if forecast_type == "Price + Market Sentiment" else "Prophet", 
                                                         symbol)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Display confidence analysis
                                display_confidence_analysis(forecast)
                                
                                # Display detailed forecast data
                                with st.expander("🔍 View Detailed Forecast Data"):
                                    st.dataframe(forecast.style.highlight_max(['yhat'], color='lightgreen')
                                               .highlight_min(['yhat'], color='lightpink'))
                                    
                                    # Add download button for forecast data
                                    csv = forecast.to_csv(index=False)
                                    st.download_button(
                                        label="Download Forecast Data",
                                        data=csv,
                                        file_name=f"{symbol}_forecast.csv",
                                        mime="text/csv",
                                    )
                            else:
                                st.error("Failed to generate forecast. Please try again with different parameters.")
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