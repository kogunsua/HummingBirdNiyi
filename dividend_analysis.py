# Dividend Analysis Functions
import requests
from bs4 import BeautifulSoup
import numpy as np
from datetime import datetime, timedelta
import re

def get_seekingalpha_dividend_info(ticker):
    """
    Get dividend information from Seeking Alpha
    """
    try:
        url = f"https://seekingalpha.com/symbol/{ticker}/dividends/history"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Parse dividend frequency information
        frequency_text = soup.find(text=re.compile(r'Dividend Frequency', re.IGNORECASE))
        if frequency_text:
            frequency = frequency_text.find_next('td').text.strip()
            return {
                'Monthly': 'Monthly',
                'Quarterly': 'Quarterly',
                'Semi-Annual': 'Semi-Annually',
                'Annual': 'Annually'
            }.get(frequency.capitalize(), None)
    except Exception as e:
        logger.error(f"Error fetching Seeking Alpha dividend info for {ticker}: {str(e)}")
    return None

def get_marketbeat_dividend_info(ticker):
    """
    Get dividend information from MarketBeat
    """
    try:
        url = f"https://www.marketbeat.com/stocks/NYSE/{ticker}/dividend/"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for dividend frequency information
        frequency_div = soup.find('div', text=re.compile(r'Dividend Frequency', re.IGNORECASE))
        if frequency_div:
            frequency = frequency_div.find_next('div').text.strip()
            if 'month' in frequency.lower():
                return 'Monthly'
            elif 'quarter' in frequency.lower():
                return 'Quarterly'
            elif 'semi' in frequency.lower():
                return 'Semi-Annually'
            elif 'year' in frequency.lower() or 'annual' in frequency.lower():
                return 'Annually'
    except Exception as e:
        logger.error(f"Error fetching MarketBeat dividend info for {ticker}: {str(e)}")
    return None

def determine_dividend_frequency(ticker):
    """
    Determine dividend frequency using multiple sources and historical data
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Get historical dividends
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)  # 2 years of history
        dividend_history = stock.history(start=start_date, end=end_date)['Dividends']
        dividend_history = dividend_history[dividend_history > 0]
        
        if len(dividend_history) == 0:
            return 'No Dividends'
        
        # Calculate average days between dividends
        dividend_dates = dividend_history.index
        days_between = []
        for i in range(1, len(dividend_dates)):
            delta = (dividend_dates[i] - dividend_dates[i-1]).days
            days_between.append(delta)
        
        if len(days_between) == 0:
            return 'Unknown'
        
        avg_days = np.mean(days_between)
        std_days = np.std(days_between)
        
        # More precise frequency determination
        if 25 <= avg_days <= 35 and std_days < 10:
            historical_frequency = 'Monthly'
        elif 85 <= avg_days <= 95 and std_days < 15:
            historical_frequency = 'Quarterly'
        elif 175 <= avg_days <= 185 and std_days < 20:
            historical_frequency = 'Semi-Annually'
        elif 360 <= avg_days <= 370 and std_days < 30:
            historical_frequency = 'Annually'
        else:
            historical_frequency = 'Irregular'
        
        # Get frequency from multiple sources
        yahoo_frequency = stock.info.get('payoutFrequency')
        if yahoo_frequency:
            yahoo_frequency = {
                1: 'Annually',
                2: 'Semi-Annually',
                4: 'Quarterly',
                12: 'Monthly'
            }.get(yahoo_frequency, 'Unknown')
        
        seekingalpha_frequency = get_seekingalpha_dividend_info(ticker)
        marketbeat_frequency = get_marketbeat_dividend_info(ticker)
        
        # Combine all frequencies
        frequencies = [f for f in [
            historical_frequency,
            yahoo_frequency,
            seekingalpha_frequency,
            marketbeat_frequency
        ] if f and f != 'Unknown' and f != 'Irregular']
        
        if not frequencies:
            return historical_frequency
        
        # Return most common frequency
        return max(set(frequencies), key=frequencies.count)
    
    except Exception as e:
        logger.error(f"Error determining dividend frequency for {ticker}: {str(e)}")
        return 'Unknown'

def get_stock_data(tickers):
    stock_data = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if info.get('dividendYield') and info.get('dividendYield') > 0:
                # Get dividend frequency
                dividend_frequency = determine_dividend_frequency(ticker)
                
                # Get historical dividends for accurate calculations
                historical_dividends = stock.history(period='1y')['Dividends']
                historical_dividends = historical_dividends[historical_dividends > 0]
                
                if not historical_dividends.empty:
                    latest_dividend = historical_dividends.iloc[-1]
                    if dividend_frequency == 'Monthly':
                        annual_dividend = latest_dividend * 12
                    elif dividend_frequency == 'Quarterly':
                        annual_dividend = latest_dividend * 4
                    elif dividend_frequency == 'Semi-Annually':
                        annual_dividend = latest_dividend * 2
                    else:
                        annual_dividend = latest_dividend
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
                    'Dividend Frequency': dividend_frequency,
                    'Ex-Dividend Date': info.get('exDividendDate', 'Unknown'),
                    'Dividend Growth Rate (5Y)': info.get('fiveYearAvgDividendYield', 0),
                    'Payout Ratio': info.get('payoutRatio', 0) * 100 if info.get('payoutRatio') else 0,
                    'Years of Growth': len(historical_dividends.index.year.unique())
                })
        
        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")
            st.warning(f"Could not fetch data for {ticker}: {str(e)}")
    
    return pd.DataFrame(stock_data)

def display_dividend_analysis(tickers=None):
    if tickers is None:
        tickers = ['O', 'MAIN', 'STAG', 'GOOD', 'AGNC']
    
    with st.spinner("Analyzing dividend stocks... This may take a minute..."):
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
            monthly_count = len(stock_data[stock_data['Dividend Frequency'] == 'Monthly'])
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
        formatted_data['Payout Ratio'] = formatted_data['Payout Ratio'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(formatted_data)
        
        # Filter and display monthly dividend stocks
        monthly_stocks = stock_data[stock_data['Dividend Frequency'] == 'Monthly']
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
                        <p>Payout Ratio: {stock['Payout Ratio']}</p>
                        <p>Years of Dividend Growth: {stock['Years of Growth']}</p>
                        <p>Dividend Frequency: {stock['Dividend Frequency']}</p>
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