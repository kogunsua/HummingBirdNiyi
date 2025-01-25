# dividend_analysis.py
import yfinance as yf
import pandas as pd
import streamlit as st

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