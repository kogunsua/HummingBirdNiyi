def show_education_section():
    """Display the educational content"""
    # Title and Description
    st.title("📊 Dividend Stock Analysis Tool")
    
    # Understanding Dividend Health Section
    st.markdown("## 💡 Understanding Dividend Health")
    st.info("""
    Understanding how we evaluate dividend health is crucial for making informed investment decisions.
    Below are the key factors we consider in our analysis.
    """)

    # Create three columns for the main metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("""
        ### 🎯 Dividend Yield 
        * **Healthy**: 3-7%
        * **Caution**: >10%
        * Higher yields may indicate higher risk
        """)

    with col2:
        st.success("""
        ### 📊 Payout Ratio
        * **Regular Stocks**: <75%
        * **REITs**: <90%
        * Higher ratios suggest potential risk
        """)

    with col3:
        st.success("""
        ### 📈 Growth History
        * **Strong**: 5+ years
        * **Moderate**: 3-5 years
        * **Weak**: <3 years
        """)

    # Additional context in an expander
    with st.expander("Learn More About Dividend Health Analysis", expanded=True):
        st.markdown("""
        ### Understanding Our Ratings 🎯
        
        | Rating | Criteria | Description |
        |--------|----------|-------------|
        | 🟢 **Buy** | Strong Fundamentals | - Strong dividend history<br>- Sustainable payout ratio<br>- Healthy yield range<br>- Stable market cap |
        | 🟡 **Hold** | Mixed Signals | - Some concerning metrics<br>- Requires monitoring<br>- Further analysis needed |
        | 🔴 **Caution** | Risk Factors | - Sustainability concerns<br>- Multiple warning signs<br>- Higher risk profile |
        
        ### Key Considerations 📈
        * **Market Cap Impact**: Larger companies tend to offer more stable dividends
        * **Industry Context**: Different sectors have different typical payout ratios
        * **Economic Conditions**: Market environment affects dividend sustainability
        * **Growth History**: Past performance helps predict future reliability
        """)

def show_analysis_section():
    """Display the analysis section"""
    # Separator
    st.markdown("---")
    st.subheader("Stock Analysis")

    # Create columns for input and button
    col1, col2 = st.columns([3, 1])
    
    with col1:
        custom_tickers = st.text_input(
            "Enter Stock Symbols (comma-separated)",
            "O,MAIN,STAG,GOOD,AGNC,SDIV,CLM",
            help="Enter stock symbols separated by commas (e.g., O,MAIN,STAG)"
        ).strip()
    
    with col2:
        analyze_button = st.button("🔍 Analyze Stocks")

    if analyze_button:
        if custom_tickers:
            tickers = [ticker.strip().upper() for ticker in custom_tickers.split(',')]
            display_dividend_analysis(tickers)
        else:
            display_dividend_analysis()

def main():
    """Main application function"""
    try:
        # Page configuration
        st.set_page_config(
            page_title="Dividend Analysis",
            page_icon="💰",
            layout="wide"
        )
        
        # Display educational content first
        show_education_section()
        
        # Display analysis section
        show_analysis_section()

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An unexpected error occurred. Please try again later.")

if __name__ == "__main__":
    main()
    
# Dividend Analysis Module
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import logging
from datetime import datetime, timedelta
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dividend_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Helper functions
def format_market_cap(value):
    if value >= 1e12:
        return f"${value/1e12:.2f}T"
    elif value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.2f}M"
    else:
        return f"${value:,.2f}"

def get_seekingalpha_dividend_info(ticker):
    """Get dividend information from Seeking Alpha"""
    try:
        url = f"https://seekingalpha.com/symbol/{ticker}/dividends/history"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
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
    """Get dividend information from MarketBeat"""
    try:
        url = f"https://www.marketbeat.com/stocks/NYSE/{ticker}/dividend/"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
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
    """Determine dividend frequency using multiple sources and historical data"""
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
        
        # Determine frequency based on payment patterns
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
        
        # Get additional sources
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

def evaluate_dividend_health(row, sector_data=None):
    """Evaluate dividend health and provide buy/hold recommendation"""
    dividend_yield = row['Dividend Yield (%)']
    payout_ratio = row['Payout Ratio']
    ticker = row['Ticker']
    
    try:
        stock = yf.Ticker(ticker)
        sector = stock.info.get('sector', '').lower()
        is_reit = 'real estate' in sector or ticker in ['O', 'STAG', 'GOOD']
        
        score = 0
        reasons = []
        
        # Yield Analysis
        if dividend_yield > 10:
            score -= 2
            reasons.append("High yield may be unsustainable")
        elif dividend_yield > 7:
            score -= 1
            reasons.append("Yield is above average")
        elif 3 <= dividend_yield <= 7:
            score += 1
            reasons.append("Healthy yield range")
        
        # Payout Ratio Analysis
        max_payout = 90 if is_reit else 75
        if payout_ratio > max_payout:
            score -= 2
            reasons.append(f"High payout ratio for {'REIT' if is_reit else 'stock'}")
        elif payout_ratio > max_payout * 0.8:
            score -= 1
            reasons.append("Payout ratio nearing upper limit")
        elif 0 < payout_ratio <= max_payout * 0.8:
            score += 1
            reasons.append("Healthy payout ratio")
        
        # Dividend Growth History
        years_of_growth = row['Years of Growth']
        if years_of_growth >= 5:
            score += 2
            reasons.append("Strong dividend growth history")
        elif years_of_growth >= 3:
            score += 1
            reasons.append("Moderate dividend growth history")
        
        # Market Cap Analysis
        market_cap = row['Market Cap']
        if market_cap >= 10e9:  # Large cap (>$10B)
            score += 1
            reasons.append("Large market cap indicates stability")
        elif market_cap < 1e9:  # Small cap (<$1B)
            score -= 1
            reasons.append("Small market cap may indicate higher risk")
        
        # Final Recommendation
        if score >= 2:
            recommendation = "Buy"
            color = "green"
        elif score >= 0:
            recommendation = "Hold"
            color = "orange"
        else:
            recommendation = "Caution"
            color = "red"
        
        return recommendation, reasons, color
    except Exception as e:
        logger.error(f"Error evaluating dividend health for {ticker}: {str(e)}")
        return "Unknown", ["Unable to evaluate"], "gray"

def get_stock_data(tickers):
    """Fetch and process stock dividend data"""
    stock_data = []
    for ticker in tickers:
        try:
            with st.spinner(f'Analyzing {ticker}...'):
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
                    
                    data = {
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
                        'Years of Growth': len(historical_dividends.index.year.unique()),
                        'Sector': info.get('sector', 'Unknown'),
                        'Industry': info.get('industry', 'Unknown'),
                        'Company Name': info.get('longName', ticker)
                    }
                    
                    # Add dividend health evaluation
                    recommendation, reasons, color = evaluate_dividend_health(data)
                    data['Action'] = recommendation
                    data['Analysis Notes'] = '; '.join(reasons)
                    data['Action Color'] = color
                    
                    stock_data.append(data)
        
        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")
            st.warning(f"Could not fetch data for {ticker}: {str(e)}")
    
    return pd.DataFrame(stock_data)

def display_dividend_analysis(tickers=None):
    """Display dividend analysis results"""
    if tickers is None:
        tickers = ['O', 'MAIN', 'STAG', 'GOOD', 'AGNC', 'SDIV', 'CLM']
    
    with st.spinner("📊 Analyzing dividend stocks..."):
        stock_data = get_stock_data(tickers)
        
        if stock_data.empty:
            st.error("No stocks found with dividend information.")
            return
        
        # Create metrics dashboard
        st.subheader("📈 Overview Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
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
            
        with col4:
            avg_payout = stock_data['Payout Ratio'].mean()
            st.metric(
                "Average Payout Ratio",
                f"{avg_payout:.1f}%",
                help="Average payout ratio across all stocks"
            )
        
        # High Yield Warning
        high_yield_stocks = stock_data[stock_data['Dividend Yield (%)'] > 10]
        if not high_yield_stocks.empty:
            st.warning("""
            ⚠️ **High Yield Alert**: The following stocks show yields above 10%:
            {}
            
            While high yields are attractive, they may indicate:
            * Potential dividend sustainability risks
            * Recent stock price decline
            * Market concerns about future performance
            """.format(', '.join(high_yield_stocks['Ticker'].tolist())))
        
        # Main Data Display
        st.subheader("📊 Detailed Stock Analysis")
        
        # Format data for display
        display_data = stock_data.copy()
        display_data['Market Cap'] = display_data['Market Cap'].apply(format_market_cap)
        display_data['Current Price'] = display_data['Current Price'].apply(lambda x: f"${x:,.2f}")
        display_data['Monthly Dividend'] = display_data['Monthly Dividend'].apply(lambda x: f"${x:.4f}")
        display_data['Annual Dividend'] = display_data['Annual Dividend'].apply(lambda x: f"${x:.2f}")
        display_data['Payout Ratio'] = display_data['Payout Ratio'].apply(lambda x: f"{x:.1f}%")
        
        # Color-code the Action column
        def color_action(val):
            color = stock_data.loc[stock_data['Action'] == val, 'Action Color'].iloc[0]
            return f'background-color: {color}; color: white; padding: 2px 6px; border-radius: 3px'
        
        styled_df = display_data.style.apply(lambda x: [color_action(val) if val in ['Buy', 'Hold', 'Caution'] else '' for val in x], 
                                           subset=['Action'])
        
        st.dataframe(styled_df, height=400)
        
        # Monthly Dividend Section
        st.subheader("🎯 Top Monthly Dividend Stocks")
        monthly_stocks = stock_data[stock_data['Dividend Frequency'] == 'Monthly']
        
        if not monthly_stocks.empty:
            sorted_stocks = monthly_stocks.sort_values(by='Dividend Yield (%)', ascending=False)
            top_stocks = sorted_stocks.head(3)
            
            for _, stock in top_stocks.iterrows():
                with st.container():
                    health_color = stock["Action Color"]
                    background_color = {
                        "red": "rgba(255, 235, 235, 0.2)",
                        "orange": "rgba(255, 250, 235, 0.2)",
                        "green": "rgba(235, 255, 235, 0.2)"
                    }.get(health_color, "rgba(255, 255, 255, 0.2)")
                    
                    st.markdown(f"""
                    <div style='padding: 20px; border: 1px solid #ddd; border-radius: 10px; margin: 10px 0; background-color: {background_color};'>
                        <h3 style='margin: 0;'>{stock['Company Name']} ({stock['Ticker']})</h3>
                        <p style='color: {stock["Action Color"]}; font-size: 1.2em; margin: 10px 0;'>
                            <strong>Recommendation: {stock['Action']}</strong>
                        </p>
                        <p style='font-style: italic;'>{stock['Analysis Notes']}</p>
                        
                        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 15px;'>
                            <div>
                                <p>💰 <strong>Monthly Dividend:</strong> ${stock['Monthly Dividend']:.4f}</p>
                                <p>📈 <strong>Annual Dividend:</strong> ${stock['Annual Dividend']:.2f}</p>
                                <p>💵 <strong>Current Price:</strong> ${stock['Current Price']:.2f}</p>
                                <p>🔄 <strong>Dividend Yield:</strong> {stock['Dividend Yield (%)']:.2f}%</p>
                            </div>
                            <div>
                                <p>🏢 <strong>Market Cap:</strong> {format_market_cap(stock['Market Cap'])}</p>
                                <p>📊 <strong>Payout Ratio:</strong> {stock['Payout Ratio']:.1f}%</p>
                                <p>📅 <strong>Years of Growth:</strong> {stock['Years of Growth']}</p>
                                <p>🏭 <strong>Industry:</strong> {stock['Industry']}</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Download button for analysis
            csv = monthly_stocks.to_csv(index=False)
            st.download_button(
                label="📥 Download Monthly Dividend Stocks Analysis",
                data=csv,
                file_name="monthly_dividend_stocks.csv",
                mime="text/csv",
            )
        else:
            st.warning("No monthly dividend stocks found in the analyzed set.")

def show_education_section():
    """Display educational content about dividend analysis"""
    st.title("📊 Dividend Stock Analysis Tool")
    
    # Understanding Dividend Health Section
    st.markdown("## 💡 Understanding Dividend Health")
    st.info("""
    Understanding how we evaluate dividend health is crucial for making informed investment decisions.
    Below are the key factors we consider in our analysis.
    """)

    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("""
        ### 🎯 Dividend Yield 
        * **Healthy**: 3-7%
        * **Caution**: >10%
        * Higher yields may indicate higher risk
        """)

    with col2:
        st.success("""
        ### 📊 Payout Ratio
        * **Regular Stocks**: <75%
        * **REITs**: <90%
        * Higher ratios suggest potential risk
        """)

    with col3:
        st.success("""
        ### 📈 Growth History
        * **Strong**: 5+ years
        * **Moderate**: 3-5 years
        * **Weak**: <3 years
        """)

    # Rating System
    st.subheader("🎯 Rating System")
    rcol1, rcol2, rcol3 = st.columns(3)

    with rcol1:
        st.success("""
        ### 🟢 Buy Rating
        - Strong dividend history
        - Sustainable payout ratio
        - Healthy yield range
        - Stable market position
        """)

    with rcol2:
        st.warning("""
        ### 🟡 Hold Rating
        - Mixed signals in metrics
        - Some areas of concern
        - Needs monitoring
        - Further analysis needed
        """)

    with rcol3:
        st.error("""
        ### 🔴 Caution Rating
        - Multiple risk factors
        - Sustainability concerns
        - Higher risk profile
        - Careful analysis required
        """)

def show_analysis_section():
    """Display stock analysis section"""
    st.markdown("---")
    st.subheader("Stock Analysis")

    col1, col2 = st.columns([3, 1])
    
    with col1:
        custom_tickers = st.text_input(
            "Enter Stock Symbols (comma-separated)",
            "O,MAIN,STAG,GOOD,AGNC,SDIV,CLM",
            help="Enter stock symbols separated by commas (e.g., O,MAIN,STAG)"
        ).strip()
    
    with col2:
        analyze_button = st.button("🔍 Analyze Stocks")

    if analyze_button:
        if custom_tickers:
            tickers = [ticker.strip().upper() for ticker in custom_tickers.split(',')]
            display_dividend_analysis(tickers)
        else:
            display_dividend_analysis()

def main():
    """Main application function"""
    try:
        # Page configuration
        st.set_page_config(
            page_title="Dividend Analysis Tool",
            page_icon="💰",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Display educational content first
        show_education_section()
        
        # Display analysis section
        show_analysis_section()

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An unexpected error occurred. Please try again later.")

if __name__ == "__main__":
    main()
    
