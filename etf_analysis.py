# etf_analysis.py
# This file contains functions for retrieving and analyzing ETF data

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import logging
from asset_config import AssetConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_etf_data(ticker, start_date, end_date):
    """
    Retrieve historical ETF data from Yahoo Finance
    
    Parameters:
    - ticker (str): The ETF ticker symbol (e.g., 'VOO', 'SPY')
    - start_date (datetime): Start date for historical data
    - end_date (datetime): End date for historical data
    
    Returns:
    - pandas.DataFrame: DataFrame containing historical price and volume data
    """
    # Convert dates to string format required by yfinance
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # Download data from Yahoo Finance
    etf = yf.Ticker(ticker)
    data = etf.history(start=start_str, end=end_str)
    
    # Reset timezone information to prevent timezone comparison issues
    data.index = data.index.tz_localize(None)
    
    # If data is empty, raise an exception
    if data.empty:
        raise Exception(f"No data found for ETF ticker {ticker}")
    
    # Add ETF-specific indicators
    data = add_etf_indicators(data)
    
    # Get ETF info
    try:
        info = etf.info
        data.attrs['info'] = {
            'name': info.get('shortName', ticker),
            'category': info.get('category', 'Unknown'),
            'asset_class': info.get('assetClass', 'Unknown'),
            'net_assets': info.get('totalAssets', 0),
            'expense_ratio': info.get('annualReportExpenseRatio', 0) * 100 if info.get('annualReportExpenseRatio') else 0,
            'dividend_yield': info.get('yield', 0) * 100 if info.get('yield') else 0,
            'beta': info.get('beta', None)
        }
        
        # Try to get holdings information if available
        try:
            holdings = etf.get_holdings()
            if not holdings.empty:
                data.attrs['holdings'] = holdings
        except:
            pass
    except Exception as e:
        logger.error(f"Error getting ETF info for {ticker}: {str(e)}")
        # If info can't be retrieved, set default values
        data.attrs['info'] = {
            'name': ticker,
            'category': 'Unknown',
            'asset_class': 'Unknown',
            'net_assets': 0,
            'expense_ratio': 0,
            'dividend_yield': 0,
            'beta': None
        }
    
    return data

def add_etf_indicators(data):
    """
    Add technical indicators to ETF data
    
    Parameters:
    - data (pandas.DataFrame): DataFrame containing price data
    
    Returns:
    - pandas.DataFrame: DataFrame with added indicators
    """
    try:
        # Get ETF indicator settings
        indicators = AssetConfig.get_config('etf')['indicators']
        
        # Calculate Moving Averages
        for period in indicators['ma_periods']:
            data[f'MA_{period}'] = data['Close'].rolling(window=period).mean()
        
        # Calculate Relative Strength Index (RSI)
        rsi_period = indicators['rsi_period']
        delta = data['Close'].diff()
        gain = delta.mask(delta < 0, 0)
        loss = -delta.mask(delta > 0, 0)
        
        avg_gain = gain.rolling(window=rsi_period).mean()
        avg_loss = loss.rolling(window=rsi_period).mean()
        
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate Moving Average Convergence Divergence (MACD)
        fast_period, slow_period, signal_period = indicators['macd_periods']
        exp1 = data['Close'].ewm(span=fast_period, adjust=False).mean()
        exp2 = data['Close'].ewm(span=slow_period, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['Signal'] = data['MACD'].ewm(span=signal_period, adjust=False).mean()
        
        # Calculate Bollinger Bands
        bb_period = indicators['bb_period']
        data['BB_Middle'] = data['Close'].rolling(window=bb_period).mean()
        data['BB_Std'] = data['Close'].rolling(window=bb_period).std()
        data['BB_Upper'] = data['BB_Middle'] + 2 * data['BB_Std']
        data['BB_Lower'] = data['BB_Middle'] - 2 * data['BB_Std']
        
        return data
    except Exception as e:
        logger.error(f"Error adding ETF indicators: {str(e)}")
        return data  # Return original data if there was an error

def analyze_etf(etf_data):
    """
    Perform analysis on ETF data
    
    Parameters:
    - etf_data (pandas.DataFrame): DataFrame containing historical ETF data
    
    Returns:
    - dict: Dictionary containing various metrics and analysis results
    """
    # Get current date and relevant past dates
    today = datetime.now()
    ytd_start = datetime(today.year, 1, 1)
    one_year_ago = today - timedelta(days=365)
    three_years_ago = today - timedelta(days=3 * 365)
    five_years_ago = today - timedelta(days=5 * 365)
    
    # Get closing prices
    close_prices = etf_data['Close']
    current_price = close_prices.iloc[-1]
    
    # Calculate returns for different periods
    # YTD return
    ytd_data = etf_data.loc[etf_data.index >= ytd_start]
    if not ytd_data.empty:
        ytd_return = (current_price / ytd_data['Close'].iloc[0] - 1) * 100
    else:
        ytd_return = 0
    
    # 1-year return
    year_data = etf_data.loc[etf_data.index >= one_year_ago]
    if len(year_data) > 0:
        year_return = (current_price / year_data['Close'].iloc[0] - 1) * 100
    else:
        year_return = 0
    
    # 3-year return (annualized)
    three_year_data = etf_data.loc[etf_data.index >= three_years_ago]
    if len(three_year_data) > 0:
        three_year_total_return = (current_price / three_year_data['Close'].iloc[0] - 1)
        three_year_return = ((1 + three_year_total_return) ** (1/3) - 1) * 100
    else:
        three_year_return = 0
    
    # 5-year return (annualized)
    five_year_data = etf_data.loc[etf_data.index >= five_years_ago]
    if len(five_year_data) > 0:
        five_year_total_return = (current_price / five_year_data['Close'].iloc[0] - 1)
        five_year_return = ((1 + five_year_total_return) ** (1/5) - 1) * 100
    else:
        five_year_return = 0
    
    # Calculate volatility (standard deviation of returns)
    daily_returns = etf_data['Close'].pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility
    
    # Calculate Sharpe ratio (assuming risk-free rate of 2%)
    risk_free_rate = AssetConfig.PORTFOLIO_CONFIG['risk_free_rate']
    sharpe_ratio = (daily_returns.mean() * 252 - risk_free_rate) / volatility
    
    # Get ETF info if available
    info = etf_data.attrs.get('info', {})
    
    # Calculate risk metrics
    max_drawdown = calculate_max_drawdown(etf_data['Close'])
    
    # Return metrics as a dictionary
    return {
        'name': info.get('name', 'Unknown'),
        'category': info.get('category', 'Unknown'),
        'asset_class': info.get('asset_class', 'Unknown'),
        'net_assets': info.get('net_assets', 0) / 1e9 if info.get('net_assets', 0) > 0 else 0,  # Convert to billions
        'expense_ratio': info.get('expense_ratio', 0),
        'dividend_yield': info.get('dividend_yield', 0),
        'beta': info.get('beta', None),
        'ytd_return': ytd_return,
        '1y_return': year_return,
        '3y_return': three_year_return,
        '5y_return': five_year_return,
        'volatility': volatility * 100,  # Convert to percentage
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown * 100  # Convert to percentage
    }

def calculate_max_drawdown(prices):
    """
    Calculate the maximum drawdown from a series of prices
    
    Parameters:
    - prices (pandas.Series): Series of prices
    
    Returns:
    - float: Maximum drawdown as a decimal (e.g., 0.25 for 25%)
    """
    # Calculate running maximum
    running_max = prices.cummax()
    
    # Calculate drawdown
    drawdown = (prices - running_max) / running_max
    
    # Calculate maximum drawdown
    max_drawdown = drawdown.min()
    
    return max_drawdown

def compare_etfs(tickers, start_date, end_date):
    """
    Compare multiple ETFs based on key metrics
    
    Parameters:
    - tickers (list): List of ETF ticker symbols
    - start_date (datetime): Start date for analysis
    - end_date (datetime): End date for analysis
    
    Returns:
    - pandas.DataFrame: DataFrame with comparison metrics for each ETF
    """
    results = []
    
    for ticker in tickers:
        try:
            # Get ETF data
            etf_data = get_etf_data(ticker, start_date, end_date)
            
            # Analyze ETF
            analysis = analyze_etf(etf_data)
            
            # Add to results
            results.append({
                'Ticker': ticker,
                'Name': analysis['name'],
                'Category': analysis['category'],
                'Expense Ratio': analysis['expense_ratio'],
                'Dividend Yield': analysis['dividend_yield'],
                'YTD Return': analysis['ytd_return'],
                '1Y Return': analysis['1y_return'],
                '3Y Return': analysis['3y_return'],
                '5Y Return': analysis['5y_return'],
                'Volatility': analysis['volatility'],
                'Sharpe Ratio': analysis['sharpe_ratio'],
                'Max Drawdown': analysis['max_drawdown']
            })
        except Exception as e:
            logger.error(f"Error analyzing ETF {ticker}: {str(e)}")
    
    return pd.DataFrame(results)

def get_etf_sector_exposure(ticker):
    """
    Get sector exposure for an ETF
    
    Parameters:
    - ticker (str): ETF ticker symbol
    
    Returns:
    - dict: Dictionary with sector names as keys and percentages as values
    """
    try:
        etf = yf.Ticker(ticker)
        
        # Try different methods of getting sector data
        
        # Method 1: Direct sector holdings (if available in newer yfinance versions)
        try:
            if hasattr(etf, 'get_sector_holdings'):
                sectors = etf.get_sector_holdings()
                return sectors.to_dict() if not sectors.empty else {}
        except:
            pass
        
        # Method 2: Try to extract from holdings data
        try:
            holdings = etf.get_holdings() if hasattr(etf, 'get_holdings') else None
            
            if holdings is not None and not holdings.empty and 'sector' in holdings.columns:
                # Group by sector and sum percentages
                sector_exposure = holdings.groupby('sector')['% of portfolio'].sum()
                return sector_exposure.to_dict()
        except:
            pass
        
        # Method A: Extract from major holdings and their sectors
        try:
            holdings = pd.DataFrame()
            sector_data = {}
            
            # Get top holdings
            top_holdings = [h for h in etf.info.get('holdings', []) if isinstance(h, dict)]
            
            if top_holdings:
                # Get sector for each holding
                for holding in top_holdings:
                    symbol = holding.get('symbol')
                    weight = holding.get('holdingPercent', 0) * 100 if holding.get('holdingPercent') else 0
                    
                    if symbol and weight > 0:
                        try:
                            stock = yf.Ticker(symbol)
                            sector = stock.info.get('sector', 'Unknown')
                            
                            if sector in sector_data:
                                sector_data[sector] += weight
                            else:
                                sector_data[sector] = weight
                        except:
                            pass
                
                return sector_data
        except:
            pass
        
        # Last resort: Use a sample/dummy data for demonstration
        if ticker in ['SPY', 'VOO', 'IVV']:  # S&P 500 ETFs
            return {
                'Information Technology': 28.2,
                'Health Care': 13.3,
                'Financials': 12.5,
                'Consumer Discretionary': 10.2,
                'Communication Services': 8.3,
                'Industrials': 8.2,
                'Consumer Staples': 6.3,
                'Energy': 4.1,
                'Utilities': 2.5,
                'Real Estate': 2.4,
                'Materials': 2.1
            }
        elif ticker in ['QQQ']:  # Nasdaq ETF
            return {
                'Information Technology': 48.5,
                'Communication Services': 15.2,
                'Consumer Discretionary': 17.3,
                'Health Care': 7.1,
                'Industrials': 4.8,
                'Consumer Staples': 4.2,
                'Utilities': 1.2,
                'Financials': 0.9,
                'Real Estate': 0.8
            }
        
        # Return empty dict if no data
        return {}
        
    except Exception as e:
        logger.error(f"Error getting sector exposure for {ticker}: {str(e)}")
        return {}