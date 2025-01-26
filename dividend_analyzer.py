# dividend_analyzer.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import logging
from datetime import datetime, timedelta
import re
from typing import List, Dict, Optional, Tuple
from config import Config

logger = logging.getLogger(__name__)

def show_dividend_education():
    """Display dividend education content"""
    st.markdown("## 💡 Understanding Dividend Health")
    st.info("Understanding dividend health is crucial for making informed investment decisions.")

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
        * Higher ratios suggest risk
        """)

    with col3:
        st.success("""
        ### 📈 Growth History
        * **Strong**: 5+ years
        * **Moderate**: 3-5 years
        * **Weak**: <3 years
        """)

    with st.expander("Learn More About Rating System", expanded=False):
        st.markdown("""
        ### Rating System
        | Rating | Description | Criteria |
        |--------|-------------|----------|
        | 🟢 Buy | Strong fundamentals | - Healthy yield (3-7%)<br>- Good payout ratio<br>- Strong history |
        | 🟡 Hold | Mixed signals | - Some concerns<br>- Needs monitoring |
        | 🔴 Caution | Risk factors | - High yield (>10%)<br>- High payout ratio |
        """)

class DividendAnalyzer:
    def __init__(self):
        """Initialize DividendAnalyzer with Config instance"""
        self.config = Config()

    def format_market_cap(self, value: float) -> str:
        """Format market cap value into readable string"""
        if value >= 1e12:
            return f"${value/1e12:.2f}T"
        elif value >= 1e9:
            return f"${value/1e9:.2f}B"
        elif value >= 1e6:
            return f"${value/1e6:.2f}M"
        else:
            return f"${value:,.2f}"
    # Add the Polygon.io method here, right after __init__ and before other analysis methods
    def _get_polygon_etf_data(self, ticker: str) -> Optional[Dict]:
        """Fetch ETF data from Polygon.io as a backup source"""
        try:
            if not hasattr(self.config, 'POLYGON_API_KEY'):
                logger.debug(f"Polygon.io API key not configured for {ticker}")
                return None
                
            headers = {
                'Authorization': f'Bearer {self.config.POLYGON_API_KEY}'
            }
            
            try:
                # Get last dividend
                div_url = f"https://api.polygon.io/v3/reference/dividends/{ticker}?limit=1"
                div_response = requests.get(div_url, headers=headers, timeout=10)
                div_data = div_response.json() if div_response.status_code == 200 else None
                
                # Get current price
                price_url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev"
                price_response = requests.get(price_url, headers=headers, timeout=10)
                price_data = price_response.json() if price_response.status_code == 200 else None
                
                if not div_data or not price_data:
                    logger.debug(f"No complete Polygon.io data available for {ticker}")
                    return None
                    
                latest_div = div_data.get('results', [{}])[0]
                latest_price = price_data.get('results', [{}])[0].get('c', 0)
                
                # Calculate annual dividend and yield
                cash_amount = latest_div.get('cash_amount', 0)
                frequency = latest_div.get('frequency', 'quarterly')
                multiplier = {
                    'monthly': 12,
                    'quarterly': 4,
                    'semi-annual': 2,
                    'annual': 1
                }.get(frequency.lower(), 4)
                
                annual_dividend = cash_amount * multiplier
                div_yield = (annual_dividend / latest_price * 100) if latest_price > 0 else 0
                
                return {
                    'dividendYield': div_yield,
                    'currentPrice': latest_price,
                    'lastDividendValue': cash_amount,
                    'dividendRate': annual_dividend
                }
                
            except requests.exceptions.RequestException as e:
                logger.debug(f"Polygon.io API request failed for {ticker}: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Unexpected error in Polygon.io data fetch for {ticker}: {str(e)}")
            return None 
    
def get_etf_dividend_data(self, ticker: str) -> Optional[Dict]:
    """Get ETF dividend data with fallback to Polygon.io"""
    try:
        # Existing yfinance initialization
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Keep existing ETF type check
        if info.get('quoteType', '').upper() != 'ETF':
            logger.debug(f"{ticker} is not identified as an ETF")
            return None
            
        try:
            # Existing yfinance dividend history logic
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            dividend_history = stock.history(start=start_date, end=end_date)['Dividends']
            dividend_history = dividend_history[dividend_history > 0]
            
            # If yfinance data is empty, try Polygon.io as fallback
            if dividend_history.empty:
                logger.debug(f"No yfinance dividend history for {ticker}, attempting Polygon.io")
                polygon_data = self._get_polygon_etf_data(ticker)
                
                if polygon_data:
                    # Merge Polygon.io data with existing yfinance metadata
                    polygon_data.update({
                        'marketCap': info.get('marketCap', 0),
                        'sector': info.get('sector', 'Unknown'),
                        'industry': info.get('industry', 'Unknown'),
                        'longName': info.get('longName', ticker),
                        'payoutRatio': 1.0,
                        'lastUpdated': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                    })
                    return polygon_data
                
                # Keep existing fallback to minimal yfinance data
                if info.get('regularMarketPrice') and info.get('dividendRate'):
                    logger.debug(f"Using minimal yfinance data for {ticker}")
                    current_price = info.get('regularMarketPrice')
                    div_rate = info.get('dividendRate')
                    return {
                        'dividendYield': (div_rate / current_price * 100) if current_price > 0 else 0,
                        'currentPrice': current_price,
                        'lastDividendValue': div_rate / 12,  # Monthly estimate
                        'marketCap': info.get('marketCap', 0),
                        'sector': info.get('sector', 'Unknown'),
                        'industry': info.get('industry', 'Unknown'),
                        'longName': info.get('longName', ticker),
                        'payoutRatio': 1.0,
                        'dividendRate': div_rate,
                        'lastUpdated': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                    }
                
                logger.debug(f"No dividend data available from any source for {ticker}")
                return None
            
            # Keep existing successful yfinance data path
            total_annual_dividend = dividend_history.sum()
            current_price = info.get('regularMarketPrice', info.get('currentPrice', 0))
            actual_yield = (total_annual_dividend / current_price * 100) if current_price > 0 else 0
            
            return {
                'dividendYield': actual_yield,
                'currentPrice': current_price,
                'lastDividendValue': dividend_history.iloc[-1] if not dividend_history.empty else 0,
                'marketCap': info.get('marketCap', 0),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'longName': info.get('longName', ticker),
                'payoutRatio': 1.0,
                'dividendRate': total_annual_dividend,
                'lastUpdated': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except (KeyError, IndexError) as e:
            logger.error(f"Data structure error for {ticker}: {str(e)}")
            return None
        
    except Exception as e:
        logger.error(f"Failed to analyze ETF {ticker}: {str(e)}")
        return None    

    def get_seeking_alpha_info(self, ticker: str) -> Optional[str]:
        """Get dividend information from Seeking Alpha with multi-exchange support"""
        try:
            # Try base URL first
            url = f"https://seekingalpha.com/symbol/{ticker}/dividends/history"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
            }
            
            for exchange in ['NYSE', 'NASDAQ', 'AMEX']:
                try:
                    exchange_url = f"{url}?exchange={exchange}"
                    response = requests.get(exchange_url, headers=headers, timeout=10)
                    if response.status_code == 200:
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
                    self.logger.debug(f"Exchange {exchange} failed for {ticker}: {str(e)}")
                    continue
        except Exception as e:
            self.logger.error(f"SeekingAlpha error for {ticker}: {str(e)}")
        return None

    def get_marketbeat_info(self, ticker: str) -> Optional[str]:
        """Get dividend information from MarketBeat with multi-exchange support"""
        try:
            exchanges = ['NYSE', 'NASDAQ', 'AMEX']
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            for exchange in exchanges:
                try:
                    url = f"https://www.marketbeat.com/stocks/{exchange}/{ticker}/dividend/"
                    response = requests.get(url, headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        frequency_div = soup.find(text=re.compile(r'Dividend Frequency', re.IGNORECASE))
                        
                        if frequency_div:
                            frequency = frequency_div.find_next('div').text.strip().lower()
                            if 'month' in frequency:
                                return 'Monthly'
                            elif 'quarter' in frequency:
                                return 'Quarterly'
                            elif 'semi' in frequency or 'semi-annual' in frequency:
                                return 'Semi-Annually'
                            elif 'year' in frequency or 'annual' in frequency:
                                return 'Annually'
                except Exception as e:
                    self.logger.debug(f"Exchange {exchange} failed for {ticker}: {str(e)}")
                    continue
        except Exception as e:
            self.logger.error(f"MarketBeat error for {ticker}: {str(e)}")
        return None

    def determine_dividend_frequency(self, ticker: str) -> str:
        """Determine dividend payment frequency using multiple methods"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            is_etf = info.get('quoteType', '').upper() == 'ETF'
            
            # Get dividend history
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365*2)
            
            try:
                dividend_history = stock.history(start=start_date, end=end_date)['Dividends']
                dividend_history = dividend_history[dividend_history > 0]
            except Exception:
                dividend_history = pd.Series()

            if len(dividend_history) == 0:
                if is_etf and info.get('dividendRate'):
                    annual_rate = info.get('dividendRate', 0)
                    latest_price = info.get('regularMarketPrice', 0)
                    if annual_rate > 0 and latest_price > 0:
                        dividend_yield = (annual_rate / latest_price) * 100
                        if dividend_yield > 7:  # High-yield ETF threshold
                            return 'Monthly'
                return 'No Dividends'

            # Calculate time between dividends
            dividend_dates = dividend_history.index
            days_between = [(dividend_dates[i] - dividend_dates[i-1]).days 
                          for i in range(1, len(dividend_dates))]
            
            if not days_between:
                if is_etf:
                    frequency_hints = [
                        self.get_seeking_alpha_info(ticker),
                        self.get_marketbeat_info(ticker)
                    ]
                    valid_frequencies = [f for f in frequency_hints if f]
                    if valid_frequencies:
                        return max(set(valid_frequencies), key=valid_frequencies.count)
                return 'Unknown'

            avg_days = np.mean(days_between)
            std_days = np.std(days_between)

            # More lenient thresholds for ETFs
            if is_etf:
                if 20 <= avg_days <= 40 and std_days < 15:
                    return 'Monthly'
                elif 80 <= avg_days <= 100 and std_days < 20:
                    return 'Quarterly'
                elif 170 <= avg_days <= 190 and std_days < 25:
                    return 'Semi-Annually'
                elif 355 <= avg_days <= 375 and std_days < 35:
                    return 'Annually'
            else:
                if 25 <= avg_days <= 35 and std_days < 10:
                    return 'Monthly'
                elif 85 <= avg_days <= 95 and std_days < 15:
                    return 'Quarterly'
                elif 175 <= avg_days <= 185 and std_days < 20:
                    return 'Semi-Annually'
                elif 360 <= avg_days <= 370 and std_days < 30:
                    return 'Annually'

            return 'Irregular'

        except Exception as e:
            logger.error(f"Frequency determination error for {ticker}: {str(e)}")
            return 'Unknown'

    def evaluate_dividend_health(self, data: Dict) -> Tuple[str, List[str], str]:
        """Evaluate dividend health and return recommendation"""
        try:
            dividend_yield = data['Dividend Yield (%)']
            payout_ratio = data['Payout Ratio']
            ticker = data['Ticker']
            is_etf = data.get('Is ETF', False)
            
            score = 0
            reasons = []
            
            if is_etf:
                # ETF-specific analysis
                if dividend_yield > 15:
                    score -= 2
                    reasons.append("Extremely high yield for an ETF - may indicate risk")
                elif dividend_yield > 10:
                    score -= 1
                    reasons.append("High yield ETF - monitor closely")
                elif 5 <= dividend_yield <= 10:
                    score += 1
                    reasons.append("Reasonable yield range for an ETF")
                
                reasons.append("ETF distributions may vary based on underlying holdings")
            else:
                # Regular stock analysis
                if dividend_yield > self.config.DIVIDEND_DEFAULTS['YIELD_THRESHOLDS']['WARNING']:
                    score -= 2
                    reasons.append("High yield may be unsustainable")
                elif dividend_yield > self.config.DIVIDEND_DEFAULTS['YIELD_THRESHOLDS']['HEALTHY_MAX']:
                    score -= 1
                    reasons.append("Yield is above average")
                elif self.config.DIVIDEND_DEFAULTS['YIELD_THRESHOLDS']['HEALTHY_MIN'] <= dividend_yield <= self.config.DIVIDEND_DEFAULTS['YIELD_THRESHOLDS']['HEALTHY_MAX']:
                    score += 1
                    reasons.append("Healthy yield range")
                
                # Payout Analysis for stocks
                stock = yf.Ticker(ticker)
                sector = stock.info.get('sector', '').lower()
                is_reit = 'real estate' in sector or ticker in self.config.DIVIDEND_DEFAULTS['REIT_TICKERS']
                max_payout = self.config.DIVIDEND_DEFAULTS['PAYOUT_RATIOS']['REIT_MAX' if is_reit else 'NORMAL_MAX']
                
                if payout_ratio > max_payout:
                    score -= 2
                    reasons.append(f"High payout ratio for {'REIT' if is_reit else 'stock'}")
                elif payout_ratio > max_payout * 0.8:
                    score -= 1
                    reasons.append("Payout ratio nearing upper limit")
                elif 0 < payout_ratio <= max_payout * 0.8:
                    score += 1
                    reasons.append("Healthy payout ratio")
            
            # Common analysis for both stocks and ETFs
            years_of_growth = data['Years of Growth']
            if years_of_growth >= 5:
                score += 2
                reasons.append("Strong dividend growth history")
            elif years_of_growth >= 3:
                score += 1
                reasons.append("Moderate dividend growth history")
            
            market_cap = data['Market Cap']
            if market_cap >= 10e9:
                score += 1
                reasons.append("Large market cap indicates stability")
            elif market_cap < 1e9:
                score -= 1
                reasons.append("Small market cap may indicate higher risk")
            
            if score >= 2:
                return "Buy", reasons, "green"
            elif score >= 0:
                return "Hold", reasons, "orange"
            else:
                return "Caution", reasons, "red"
        except Exception as e:
            logger.error(f"Health evaluation error: {str(e)}")
            return "Unknown", ["Unable to evaluate"], "gray"

    def get_stock_data(self, tickers: List[str]) -> pd.DataFrame:
        """Get stock data for analysis"""
        stock_data = []
        for ticker in tickers:
            try:
                with st.spinner(f'Analyzing {ticker}...'):
                    # First try yfinance
                    try:
                        stock = yf.Ticker(ticker)
                        info = stock.info
                        
                        if not info or 'dividendYield' not in info:
                            # Try alternative source (Alpha Vantage)
                            if hasattr(self.config, 'ALPHA_VANTAGE_API_KEY'):
                                dividend_data = self._get_alpha_vantage_data(ticker)
                                if dividend_data:
                                    info = dividend_data
                            else:
                                raise ValueError(f"No dividend data found for {ticker}")

                        if info.get('dividendYield') and info.get('dividendYield') > 0:
                            dividend_frequency = self.determine_dividend_frequency(ticker)
                            historical_dividends = stock.history(period='1y')['Dividends']
                            historical_dividends = historical_dividends[historical_dividends > 0]
                            
                            latest_dividend = (historical_dividends.iloc[-1] if not historical_dividends.empty 
                                             else info.get('lastDividendValue', 0))
                            annual_dividend = latest_dividend * {
                                'Monthly': 12,
                                'Quarterly': 4,
                                'Semi-Annually': 2
                            }.get(dividend_frequency, 1)
                            
                            data = {
                                'Ticker': ticker,
                                'Monthly Dividend': annual_dividend / 12,
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
                            
                            recommendation, reasons, color = self.evaluate_dividend_health(data)
                            data.update({
                                'Action': recommendation,
                                'Analysis Notes': '; '.join(reasons),
                                'Action Color': color
                            })
                            
                            stock_data.append(data)
                    except Exception as e:
                        logger.error(f"Error processing {ticker}: {str(e)}")
                        st.warning(f"Could not fetch data for {ticker} using primary source. Error: {str(e)}")
                        continue
            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
                st.warning(f"Could not fetch data for {ticker}")
                
        return pd.DataFrame(stock_data)

    def _get_alpha_vantage_data(self, ticker: str) -> Optional[Dict]:
        """Fetch stock data from Alpha Vantage as a backup source"""
        try:
            api_key = self.config.ALPHA_VANTAGE_API_KEY
            base_url = "https://www.alphavantage.co/query"
            
            # Get Overview
            overview_params = {
                "function": "OVERVIEW",
                "symbol": ticker,
                "apikey": api_key
            }
            overview_response = requests.get(base_url, params=overview_params)
            overview_data = overview_response.json()
            
            if not overview_data or "Error Message" in overview_data:
                return None
                
            # Get Global Quote
            quote_params = {
                "function": "GLOBAL_QUOTE",
                "symbol": ticker,
                "apikey": api_key
            }
            quote_response = requests.get(base_url, params=quote_params)
            quote_data = quote_response.json()
            
            if not quote_data or "Error Message" in quote_data:
                return None
                
            current_price = float(quote_data.get("Global Quote", {}).get("05. price", 0))
            dividend_amount = float(overview_data.get("DividendPerShare", 0))
            
            return {
                'dividendYield': (dividend_amount / current_price) if current_price > 0 else 0,
                'currentPrice': current_price,
                'lastDividendValue': dividend_amount,
                'marketCap': float(overview_data.get("MarketCapitalization", 0)),
                'sector': overview_data.get("Sector", "Unknown"),
                'industry': overview_data.get("Industry", "Unknown"),
                'longName': overview_data.get("Name", ticker),
                'payoutRatio': float(overview_data.get("PayoutRatio", 0))
            }
            
        except Exception as e:
            logger.error(f"Alpha Vantage API error for {ticker}: {str(e)}")
            return None
     
def display_dividend_analysis(self, tickers: Optional[List[str]] = None):
        """Display dividend analysis results"""
        if not tickers:
            tickers = self.config.DIVIDEND_DEFAULTS['DEFAULT_DIVIDEND_STOCKS']
        
        stock_data = self.get_stock_data(tickers)
        if stock_data.empty:
            st.error("No dividend information found for any of the provided stocks")
            st.info("""
            Possible reasons:
            - Stock symbol may have changed or been delisted
            - Stock may not currently pay dividends
            - Data source may be temporarily unavailable

            Try checking:
            1. Current stock symbol on your broker's platform
            2. Company's investor relations website
            3. Alternative stock symbols or tickers
            """)
            return
        
        self._display_metrics(stock_data)
        self._display_warnings(stock_data)
        self._display_data_table(stock_data)
        self._display_monthly_stocks(stock_data)

def _display_metrics(self, stock_data: pd.DataFrame):
        """Display overview metrics"""
        st.subheader("📈 Overview Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Stocks", len(stock_data))
        with col2:
            avg_yield = stock_data['Dividend Yield (%)'].mean()
            st.metric("Average Yield", f"{avg_yield:.2f}%")
        with col3:
            monthly_count = len(stock_data[stock_data['Dividend Frequency'] == 'Monthly'])
            st.metric("Monthly Payers", monthly_count)
        with col4:
            avg_payout = stock_data['Payout Ratio'].mean()
            st.metric("Average Payout", f"{avg_payout:.1f}%")

def _display_warnings(self, stock_data: pd.DataFrame):
        """Display warning messages for high-yield stocks"""
        high_yield = stock_data[stock_data['Dividend Yield (%)'] > 10]
        if not high_yield.empty:
            st.warning(f"⚠️ High Yield Alert: {', '.join(high_yield['Ticker'])}")

def _display_data_table(self, stock_data: pd.DataFrame):
        """Display detailed analysis table"""
        st.subheader("📊 Detailed Analysis")
        display_data = self._format_display_data(stock_data)
        st.dataframe(display_data)

def _display_monthly_stocks(self, stock_data: pd.DataFrame):
        """Display monthly dividend paying stocks"""
        monthly = stock_data[stock_data['Dividend Frequency'] == 'Monthly']
        if monthly.empty:
            return
        
        st.subheader("🎯 Top Monthly Dividend Stocks")
        top_stocks = monthly.sort_values('Dividend Yield (%)', ascending=False).head(3)
        
        for _, stock in top_stocks.iterrows():
            self._display_stock_card(stock)
        
        csv = monthly.to_csv(index=False)
        st.download_button("📥 Download Analysis", csv, "monthly_dividends.csv")

def _format_display_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Format display data for presentation"""
        formatted = data.copy()
        for col in ['Monthly Dividend', 'Annual Dividend', 'Current Price']:
            formatted[col] = formatted[col].apply(lambda x: f"${x:.2f}")
        formatted['Market Cap'] = formatted['Market Cap'].apply(self.format_market_cap)
        formatted['Payout Ratio'] = formatted['Payout Ratio'].apply(lambda x: f"{x:.1f}%")
        return formatted

def _display_stock_card(self, stock: pd.Series):
        """Display individual stock card with styling"""
        background_colors = {
            "red": "rgba(255, 235, 235, 0.2)",
            "orange": "rgba(255, 250, 235, 0.2)",
            "green": "rgba(235, 255, 235, 0.2)"
        }
        bg_color = background_colors.get(stock["Action Color"], "rgba(255, 255, 255, 0.2)")
        
        st.markdown(f"""
        <div style='padding: 20px; border: 1px solid #ddd; border-radius: 10px; 
                    margin: 10px 0; background-color: {bg_color};'>
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
                    <p>🏢 <strong>Market Cap:</strong> {self.format_market_cap(stock['Market Cap'])}</p>
                    <p>📊 <strong>Payout Ratio:</strong> {stock['Payout Ratio']:.1f}%</p>
                    <p>📅 <strong>Years of Growth:</strong> {stock['Years of Growth']}</p>
                    <p>🏭 <strong>Industry:</strong> {stock['Industry']}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)  

    def generate_csv_report(self, stock_data: pd.DataFrame) -> str:
        """Generate CSV report for download"""
        report_data = stock_data.copy()
        return report_data.to_csv(index=False)


def filter_monthly_dividend_stocks(data: pd.DataFrame) -> pd.DataFrame:
    """Filter and return monthly dividend stocks from the provided data."""
    return data[data['Dividend Frequency'] == 'Monthly']
                                                                
                                                            
                                                            
