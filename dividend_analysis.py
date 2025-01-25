# Dividend Analysis Functions
import requests
from bs4 import BeautifulSoup
import numpy as np
from datetime import datetime, timedelta

def determine_dividend_frequency(ticker, dividend_history):
    """
    Determine dividend frequency by analyzing historical dividend dates
    """
    try:
        if dividend_history.empty:
            return "Unknown"
        
        # Sort dividend history by date
        dividend_history = dividend_history.sort_values('Date', ascending=True)
        
        if len(dividend_history) < 2:
            return "Unknown"
        
        # Calculate days between dividends
        days_between = []
        dates = pd.to_datetime(dividend_history['Date'])
        for i in range(1, len(dates)):
            delta = (dates.iloc[i] - dates.iloc[i-1]).days
            days_between.append(delta)
        
        if not days_between:
            return "Unknown"
        
        # Calculate average days between dividends
        avg_days = np.mean(days_between)
        
        # Determine frequency based on average days
        if 25 <= avg_days <= 35:
            return "Monthly"
        elif 85 <= avg_days <= 95:
            return "Quarterly"
        elif 175 <= avg_days <= 185:
            return "Semi-Annually"
        elif 360 <= avg_days <= 370:
            return "Annually"
        else:
            return "Irregular"
            
    except Exception as e:
        logger.error(f"Error determining dividend frequency for {ticker}: {str(e)}")
        return "Unknown"

def get_nasdaq_dividend_info(ticker):
    """
    Get dividend information from Nasdaq website
    """
    try:
        url = f"https://www.nasdaq.com/market-activity/stocks/{ticker.lower()}/dividend-history"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract dividend frequency text
        frequency_text = soup.find('span', text=lambda t: t and 'Dividend Frequency:' in t)
        if frequency_text:
            frequency = frequency_text.find_next('span').text.strip()
            return {
                'Monthly': 'Monthly',
                'Quarterly': 'Quarterly',
                'Semi-Annual': 'Semi-Annually',
                'Annual': 'Annually'
            }.get(frequency, 'Unknown')
            
    except Exception as e:
        logger.error(f"Error fetching Nasdaq dividend info for {ticker}: {str(e)}")
    return None

def get_stock_data(tickers):
    stock_data = []
    for ticker in tickers:
        try:
            # Get data from Yahoo Finance
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if info.get('dividendYield') and info.get('dividendYield') > 0:
                # Get dividend history
                dividend_history = stock.dividends.reset_index()
                if not dividend_history.empty:
                    dividend_history.columns = ['Date', 'Dividend']
                
                # Determine dividend frequency from multiple sources
                yf_frequency = info.get('dividendFrequency', 'Unknown')
                historical_frequency = determine_dividend_frequency(ticker, dividend_history)
                nasdaq_frequency = get_nasdaq_dividend_info(ticker)
                
                # Combine frequencies from different sources
                frequencies = [f for f in [yf_frequency, historical_frequency, nasdaq_frequency] if f != 'Unknown']
                if frequencies:
                    # Use most common frequency
                    final_frequency = max(set(frequencies), key=frequencies.count)
                else:
                    final_frequency = 'Unknown'
                
                # Calculate annualized dividend
                if not dividend_history.empty:
                    recent_dividends = dividend_history.tail(12)['Dividend'].sum()
                    if final_frequency == 'Monthly':
                        annual_dividend = recent_dividends
                    elif final_frequency == 'Quarterly':
                        annual_dividend = recent_dividends * (12/len(recent_dividends))
                    else:
                        annual_dividend = info.get('lastDividendValue', 0) * 12
                else:
                    annual_dividend = info.get('lastDividendValue', 0) * 12
                
                monthly_dividend = annual_dividend / 12
                
                stock_data.append({
                    'Ticker': ticker,
                    'Monthly Dividend': monthly_dividend,
                    'Annual Dividend': annual_dividend,
                    'Current Price': info.get('currentPrice', 0),
                    'Dividend Yield (%)': info.get('dividendYield', 0) * 100,
                    'Market Cap': info.get('marketCap', 0),
                    'Dividend Frequency': final_frequency,
                    'Payment History': len(dividend_history) if not dividend_history.empty else 0,
                    'Last Payment Date': dividend_history['Date'].max().strftime('%Y-%m-%d') if not dividend_history.empty else 'Unknown'
                })
                
        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")
            st.warning(f"Could not fetch data for {ticker}: {str(e)}")
    
    return pd.DataFrame(stock_data)

def display_dividend_analysis(tickers=None):
    if tickers is None:
        tickers = ['O', 'MAIN', 'STAG', 'GOOD', 'SDIV','CLM']
    
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
        formatted_data['Annual Dividend'] = formatted_data['Annual Dividend'].apply(lambda x: f"${x:.2f}")
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
                        <p>Monthly Dividend: ${stock['Monthly Dividend']}</p>
                        <p>Annual Dividend: ${stock['Annual Dividend']}</p>
                        <p>Current Price: ${stock['Current Price']}</p>
                        <p>Market Cap: {format_market_cap(stock['Market Cap'])}</p>
                        <p>Payment History: {stock['Payment History']} payments</p>
                        <p>Last Payment: {stock['Last Payment Date']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Add a download button for the analysis
            csv = monthly_stocks.to_csv(index=False)
            st.download_button(
                label="Download Monthly Dividend Stocks Analysis",
                data=csv,
                file_name="monthly_dividend_stocks.csv",
                mime="text/csv",
            )
        else:
            st.warning("No monthly dividend stocks found in the analyzed set.")