def display_metrics(data: pd.DataFrame, forecast: pd.DataFrame, asset_type: str, symbol: str):
    """Display key metrics and statistics"""
    try:
        # Get the latest actual price
        latest_price = data['Close'].iloc[-1]
        
        # Get the latest forecast price
        forecast_price = forecast.loc[forecast.index[-1], 'yhat']
        
        # Calculate price change percentage
        price_change = ((forecast_price - latest_price) / latest_price) * 100
        
        # Calculate confidence range
        confidence_range = forecast.loc[forecast.index[-1], 'yhat_upper'] - forecast.loc[forecast.index[-1], 'yhat_lower']
        confidence_percentage = (confidence_range / forecast_price) * 100 / 2
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Current Price",
                f"${latest_price:,.2f}",
                f"{data['Close'].pct_change().iloc[-1]*100:.2f}%"
            )
        
        with col2:
            st.metric(
                f"Forecast Price ({forecast['ds'].iloc[-1].strftime('%Y-%m-%d')})",
                f"${forecast_price:,.2f}",
                f"{price_change:.2f}%"
            )
        
        with col3:
            st.metric(
                "Forecast Range",
                f"${confidence_range:,.2f}",
                f"±{confidence_percentage:.2f}%"
            )
            
    except Exception as e:
        st.error(f"Error displaying metrics: {str(e)}")