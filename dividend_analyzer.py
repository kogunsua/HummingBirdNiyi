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
        self.config = Config()
    
    def format_market_cap(self, value: float) -> str:
        if value >= 1e12:
            return f"${value/1e12:.2f}T"
        elif value >= 1e9:
            return f"${value/1e9:.2f}B"
        elif value >= 1e6:
            return f"${value/1e6:.2f}M"
        else:
            return f"${value:,.2f}"

    def get_seeking_alpha_info(self, ticker: str) -> Optional[str]:
        try:
            url = f"https://seekingalpha.com/symbol/{ticker}/dividends/history"
            headers = {
                'User-Agent': 'Mozilla/5.0',
                'Accept': 'text/html,application/xhtml+xml'
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            frequency_text = soup.find(text=re.compile(r'Dividend Frequency'))
            if frequency_text:
                frequency = frequency_text.find_next('td').text.strip()
                return {
                    'Monthly': 'Monthly',
                    'Quarterly': 'Quarterly',
                    'Semi-Annual': 'Semi-Annually',
                    'Annual': 'Annually'
                }.get(frequency.capitalize(), None)
        except Exception as e:
            logger.error(f"SeekingAlpha error for {ticker}: {str(e)}")
        return None

    def get_marketbeat_info(self, ticker: str) -> Optional[str]:
        try:
            url = f"https://www.marketbeat.com/stocks/NYSE/{ticker}/dividend/"
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            frequency_div = soup.find('div', text=re.compile(r'Dividend Frequency'))
            if frequency_div:
                frequency = frequency_div.find_next('div').text.strip().lower()
                if 'month' in frequency:
                    return 'Monthly'
                elif 'quarter' in frequency:
                    return 'Quarterly'
                elif 'semi' in frequency:
                    return 'Semi-Annually'
                elif 'year' in frequency or 'annual' in frequency:
                    return 'Annually'
        except Exception as e:
            logger.error(f"MarketBeat error for {ticker}: {str(e)}")
        return None

    def determine_dividend_frequency(self, ticker: str) -> str:
        try:
            stock = yf.Ticker(ticker)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365*2)
            dividend_history = stock.history(start=start_date, end=end_date)['Dividends']
            dividend_history = dividend_history[dividend_history > 0]
            
            if len(dividend_history) == 0:
                return 'No Dividends'
            
            dividend_dates = dividend_history.index
            days_between = [(dividend_dates[i] - dividend_dates[i-1]).days 
                          for i in range(1, len(dividend_dates))]
            
            if not days_between:
                return 'Unknown'
            
            avg_days = np.mean(days_between)
            std_days = np.std(days_between)
            
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
            
            yahoo_frequency = stock.info.get('payoutFrequency')
            if yahoo_frequency:
                yahoo_frequency = {1: 'Annually', 2: 'Semi-Annually', 
                                 4: 'Quarterly', 12: 'Monthly'}.get(yahoo_frequency, 'Unknown')
            
            frequencies = [f for f in [
                historical_frequency,
                yahoo_frequency,
                self.get_seeking_alpha_info(ticker),
                self.get_marketbeat_info(ticker)
            ] if f and f != 'Unknown' and f != 'Irregular']
            
            return max(set(frequencies), key=frequencies.count) if frequencies else historical_frequency
        except Exception as e:
            logger.error(f"Frequency determination error: {str(e)}")
            return 'Unknown'

  def evaluate_dividend_health(self, data: Dict) -> Tuple[str, List[str], str]:
    try:
        dividend_yield = data['Dividend Yield (%)']
        payout_ratio = data['Payout Ratio']
        ticker = data['Ticker']
        
        stock = yf.Ticker(ticker)
        sector = stock.info.get('sector', '').lower()
        is_reit = 'real estate' in sector or ticker in self.config.DIVIDEND_DEFAULTS['REIT_TICKERS']  # Use self.config
        
        score = 0
        reasons = []
        
        # Yield Analysis
        if dividend_yield > self.config.DIVIDEND_DEFAULTS['YIELD_THRESHOLDS']['WARNING']:
            score -= 2
                reasons.append("High yield may be unsustainable")
            elif dividend_yield > Config.DIVIDEND_DEFAULTS['YIELD_THRESHOLDS']['HEALTHY_MAX']:
                score -= 1
                reasons.append("Yield is above average")
            elif Config.DIVIDEND_DEFAULTS['YIELD_THRESHOLDS']['HEALTHY_MIN'] <= dividend_yield <= Config.DIVIDEND_DEFAULTS['YIELD_THRESHOLDS']['HEALTHY_MAX']:
                score += 1
                reasons.append("Healthy yield range")
            
            # Payout Analysis
            max_payout = Config.DIVIDEND_DEFAULTS['PAYOUT_RATIOS']['REIT_MAX' if is_reit else 'NORMAL_MAX']
            if payout_ratio > max_payout:
                score -= 2
                reasons.append(f"High payout ratio for {'REIT' if is_reit else 'stock'}")
            elif payout_ratio > max_payout * 0.8:
                score -= 1
                reasons.append("Payout ratio nearing upper limit")
            elif 0 < payout_ratio <= max_payout * 0.8:
                score += 1
                reasons.append("Healthy payout ratio")
            
            # Growth History
            years_of_growth = data['Years of Growth']
            if years_of_growth >= 5:
                score += 2
                reasons.append("Strong dividend growth history")
            elif years_of_growth >= 3:
                score += 1
                reasons.append("Moderate dividend growth history")
            
            # Market Cap
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
        stock_data = []
        for ticker in tickers:
            try:
                with st.spinner(f'Analyzing {ticker}...'):
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
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
                st.warning(f"Could not fetch data for {ticker}")
        
        return pd.DataFrame(stock_data)

    def display_dividend_analysis(self, tickers: Optional[List[str]] = None):
        if not tickers:
            tickers = Config.DIVIDEND_DEFAULTS['DEFAULT_DIVIDEND_STOCKS']
        
        stock_data = self.get_stock_data(tickers)
        if stock_data.empty:
            st.error("No dividend information found")
            return
        
        self._display_metrics(stock_data)
        self._display_warnings(stock_data)
        self._display_data_table(stock_data)
        self._display_monthly_stocks(stock_data)

    def _display_metrics(self, stock_data: pd.DataFrame):
        st.subheader("📈 Overview Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Stocks", len(stock_data))
        with col2:
            st.metric("Average Yield", f"{stock_data['Dividend Yield (%)'].mean():.2f}%")
        with col3:
            st.metric("Monthly Payers", len(stock_data[stock_data['Dividend Frequency'] == 'Monthly']))
        with col4:
            st.metric("Average Payout", f"{stock_data['Payout Ratio'].mean():.1f}%")

    def _display_warnings(self, stock_data: pd.DataFrame):
        high_yield = stock_data[stock_data['Dividend Yield (%)'] > 10]
        if not high_yield.empty:
            st.warning(f"⚠️ High Yield Alert: {', '.join(high_yield['Ticker'])}")

    def _display_data_table(self, stock_data: pd.DataFrame):
        st.subheader("📊 Detailed Analysis")
        display_data = self._format_display_data(stock_data)
        st.dataframe(display_data)

    def _display_monthly_stocks(self, stock_data: pd.DataFrame):
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
        formatted = data.copy()
        for col in ['Monthly Dividend', 'Annual Dividend', 'Current Price']:
            formatted[col] = formatted[col].apply(lambda x: f"${x:.2f}")
        formatted['Market Cap'] = formatted['Market Cap'].apply(self.format_market_cap)
        formatted['Payout Ratio'] = formatted['Payout Ratio'].apply(lambda x: f"{x:.1f}%")
        return formatted
        
    def _display_stock_card(self, stock: pd.Series):
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
        # Add any additional calculations or formatting for the report
        return report_data.to_csv(index=False)

def filter_monthly_dividend_stocks(data: pd.DataFrame) -> pd.DataFrame:
    """Filter and return monthly dividend stocks from the provided data."""
    return data[data['Dividend Frequency'] == 'Monthly']
