# portfolio.py
# This file contains functions for managing and analyzing portfolio data

import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf

def load_portfolio_data():
    """
    Load the predefined portfolio data
    
    Returns:
    - pandas.DataFrame: DataFrame containing portfolio data
    """
    # Create a DataFrame with the predefined portfolio data
    data = {
        'Symbol': [
            'SDIV', 'MSFT', 'CLM', 'MAIN', 'PSEC', 'JEPI', 'VOO', 'KO', 'MO', 'AAPL',
            'JNJ', 'SBUX', 'V', 'HD', 'PG', 'NVO', 'NKE', 'LLY', 'PFE', 'TSM', 'COST', 'NVDA'
        ],
        'Description': [
            'GLOBAL X FDS SUPERDIVIDEND', 'MICROSOFT CORP', 'CORNERSTONE STRATEGIC INVESTMENT FUND, INC. COMMON SHARES',
            'MAIN STR CAP CORP COM', 'PROSPECT CAP CORP COM', 'J P MORGAN EXCHANGE TRADED FD EQUITY PREMIUM',
            'VANGUARD INDEX FUNDS S&P 500 ETF USD', 'COCA-COLA CO', 'ALTRIA GROUP INC', 'APPLE INC',
            'JOHNSON &JOHNSON COM', 'STARBUCKS CORP COM USD0.001', 'VISA INC', 'HOME DEPOT INC',
            'PROCTER AND GAMBLE CO COM', 'NOVO NORDISK A/S ADR-EACH CNV INTO 1 CLASS B DKK1',
            'NIKE INC CLASS B COM NPV', 'ELI LILLY &CO COM', 'PFIZER INC', 'TAIWAN SEMICONDUCTOR MANUFACTURING SPON ADS EACH REP 5 ORD TWD10',
            'COSTCO WHOLESALE CORP COM', 'NVIDIA CORPORATION COM'
        ],
        'Quantity': [
            1019.084, 502.15, 635.263, 58.026, 223.207, 18.714, 4.171, 9.224, 4.346, 10.409,
            2.049, 4.099, 4.027, 1.018, 2.048, 5.039, 4.048, 1.002, 3, 1.984, 0.381, 20.005
        ],
        'Last Price': [
            21.20, 396.99, 7.90, 60.80, 4.40, 59.08, 546.33, 71.21, 55.85, 241.84,
            165.02, 115.81, 362.71, 396.60, 173.84, 90.65, 79.03, 920.63, 26.43, 180.53, 1048.61, 124.92
        ],
        'Last Price Change': [
            0.06, 4.46, 0.19, 1.86, 0.09, 0.72, 8.36, 0.34, 0.77, 4.54,
            1.29, 1.32, 6.97, 6.33, 1.87, 1.43, -0.59, 15.47, 0.33, -0.56, 26.90, 4.77
        ],
        'Current Value': [
            21604.58, 199348.52, 5018.57, 3527.98, 982.11, 1105.66, 2278.74, 656.84, 242.72, 2517.31,
            338.12, 474.70, 1460.63, 403.73, 356.02, 456.78, 319.91, 922.47, 79.29, 358.17, 399.52, 2499.02
        ],
        'Percent Of Account': [
            8.52, 78.59, 1.98, 1.39, 0.39, 0.44, 0.90, 0.26, 0.10, 0.99,
            0.13, 0.19, 0.58, 0.16, 0.14, 0.18, 0.13, 0.36, 0.03, 0.14, 0.16, 0.99
        ],
        'Ex-Date': [
            '2025-02-05', '2025-02-20', '2025-03-14', '2025-03-07', '2025-03-27', '2025-03-03', '2024-12-23',
            '2025-03-14', '2025-03-25', '2025-02-10', '2025-02-18', '2025-02-14', '2025-02-11', '2025-03-13',
            '2025-01-24', '2025-03-31', '2025-03-03', '2025-02-14', '2025-01-24', '2025-03-18', '2025-02-07', '2025-03-12'
        ],
        'Amount Per Share': [
            0.19, 0.83, 0.12, 0.25, 0.05, 0.33, 1.74, 0.51, 1.02, 0.25,
            1.24, 0.61, 0.59, 2.30, 1.01, 1.10, 0.40, 1.50, 0.43, 0.69, 1.16, 0.01
        ],
        'Pay-Date': [
            '2025-02-12', '2025-03-13', '2025-03-31', '2025-03-14', '2025-04-17', '2025-03-05', '2024-12-26',
            '2025-04-01', '2025-04-30', '2025-02-13', '2025-03-04', '2025-02-28', '2025-03-03', '2025-03-27',
            '2025-02-18', '2025-04-08', '2025-04-01', '2025-03-10', '2025-03-07', '2025-04-10', '2025-02-21', '2025-04-02'
        ],
        'Dist. Yield': [
            10.97, 0.85, 17.44, 5.09, 12.53, 7.23, 1.25, 2.88, 7.41, 0.42,
            3.03, 2.13, 0.66, 2.36, 2.34, 1.81, 2.00, 0.66, 6.59, 1.36, 0.45, 0.03
        ],
        'Distribution yield as of': [
            None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None, None
        ],
        'SEC Yield': [
            9.76, None, None, None, None, 7.12, 1.19, None, None, None,
            None, None, None, None, None, None, None, None, None, None, None, None
        ],
        'SEC yield as of': [
            '2025-01-31', None, None, None, None, '2025-01-31', '2025-01-31', None, None, None,
            None, None, None, None, None, None, None, None, None, None, None, None
        ],
        'Est. Annual Income': [
            2363.25, 1667.13, 854.17, 174.07, 120.53, 79.37, 27.96, 18.81, 17.73, 10.40,
            10.16, 10.00, 9.50, 9.36, 8.24, 8.14, 6.47, 6.01, 5.16, 4.89, 1.76, 0.80
        ]
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Convert date columns to datetime
    date_columns = ['Ex-Date', 'Pay-Date', 'SEC yield as of']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Convert numeric columns to appropriate types
    numeric_columns = ['Quantity', 'Last Price', 'Last Price Change', 'Current Value', 
                       'Percent Of Account', 'Amount Per Share', 'Dist. Yield', 'SEC Yield', 
                       'Est. Annual Income']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def calculate_portfolio_metrics(portfolio_data):
    """
    Calculate key metrics for the entire portfolio
    
    Parameters:
    - portfolio_data (pandas.DataFrame): DataFrame containing portfolio data
    
    Returns:
    - dict: Dictionary containing portfolio metrics
    """
    # Calculate total portfolio value
    total_value = portfolio_data['Current Value'].sum()
    
    # Calculate total annual income
    annual_income = portfolio_data['Est. Annual Income'].sum()
    
    # Calculate monthly income
    monthly_income = annual_income / 12
    
    # Calculate average dividend yield
    # Use a weighted average based on position value
    dividend_stocks = portfolio_data[portfolio_data['Dist. Yield'] > 0]
    weighted_yield = (dividend_stocks['Dist. Yield'] * dividend_stocks['Current Value']).sum() / dividend_stocks['Current Value'].sum()
    
    # Calculate asset allocation
    asset_allocation = portfolio_data.groupby('Symbol')['Current Value'].sum() / total_value * 100
    
    # Performance metrics
    largest_holding = portfolio_data.loc[portfolio_data['Current Value'].idxmax()]
    smallest_holding = portfolio_data.loc[portfolio_data['Current Value'].idxmin()]
    
    # Sector/Category exposure (basic implementation)
    exposure_groups = portfolio_data.groupby(
        portfolio_data['Description'].apply(lambda x: x.split()[0])  # Simple categorization
    )['Current Value'].sum() / total_value * 100
    
    # Return the metrics
    return {
        'total_value': total_value,
        'annual_income': annual_income,
        'monthly_income': monthly_income,
        'avg_yield': weighted_yield,
        'asset_allocation': asset_allocation,
        'sector_exposure': exposure_groups,
        'largest_holding': {
            'symbol': largest_holding['Symbol'],
            'name': largest_holding['Description'],
            'value': largest_holding['Current Value'],
            'percent': largest_holding['Percent Of Account']
        },
        'smallest_holding': {
            'symbol': smallest_holding['Symbol'],
            'name': smallest_holding['Description'],
            'value': smallest_holding['Current Value'],
            'percent': smallest_holding['Percent Of Account']
        }
    }

def update_portfolio_prices(portfolio_data):
    """
    Update the portfolio with current market prices
    
    Parameters:
    - portfolio_data (pandas.DataFrame): Current portfolio data
    
    Returns:
    - pandas.DataFrame: Updated portfolio data
    """
    # Create a copy of the portfolio data
    updated_portfolio = portfolio_data.copy()
    
    # Get unique symbols
    symbols = updated_portfolio['Symbol'].unique()
    
    # Create a dictionary to store updated prices
    updated_prices = {}
    
    # Fetch current prices for each symbol
    for symbol in symbols:
        try:
            # Get ticker data
            ticker = yf.Ticker(symbol)
            
            # Get the latest price
            data = ticker.history(period='1d')
            
            # Reset timezone information
            data.index = data.index.tz_localize(None)
            
            if not data.empty:
                latest_price = data['Close'].iloc[-1]
                previous_price = portfolio_data.loc[portfolio_data['Symbol'] == symbol, 'Last Price'].iloc[0]
                
                updated_prices[symbol] = {
                    'price': latest_price,
                    'change': latest_price - previous_price
                }
        except Exception as e:
            print(f"Error updating price for {symbol}: {str(e)}")
    
    # Update the portfolio dataframe
    for symbol, price_info in updated_prices.items():
        # Update price and change
        updated_portfolio.loc[updated_portfolio['Symbol'] == symbol, 'Last Price'] = price_info['price']
        updated_portfolio.loc[updated_portfolio['Symbol'] == symbol, 'Last Price Change'] = price_info['change']
        
        # Recalculate current value
        updated_portfolio.loc[updated_portfolio['Symbol'] == symbol, 'Current Value'] = (
            updated_portfolio.loc[updated_portfolio['Symbol'] == symbol, 'Quantity'] * 
            updated_portfolio.loc[updated_portfolio['Symbol'] == symbol, 'Last Price']
        )
    
    # Recalculate percent of account
    total_value = updated_portfolio['Current Value'].sum()
    updated_portfolio['Percent Of Account'] = updated_portfolio['Current Value'] / total_value * 100
    
    return updated_portfolio

def add_portfolio_position(portfolio_data, symbol, quantity, price=None):
    """
    Add a new position to the portfolio or update an existing one
    
    Parameters:
    - portfolio_data (pandas.DataFrame): Current portfolio data
    - symbol (str): Stock/ETF symbol
    - quantity (float): Number of shares to add
    - price (float, optional): Purchase price. If None, current market price is used.
    
    Returns:
    - pandas.DataFrame: Updated portfolio data
    """
    # Create a copy of the portfolio data
    updated_portfolio = portfolio_data.copy()
    
    # Check if symbol already exists in portfolio
    if symbol in updated_portfolio['Symbol'].values:
        # Update existing position
        position_index = updated_portfolio[updated_portfolio['Symbol'] == symbol].index[0]
        
        # Get current position details
        current_quantity = updated_portfolio.loc[position_index, 'Quantity']
        current_value = updated_portfolio.loc[position_index, 'Current Value']
        
        # If price not provided, use current price
        if price is None:
            current_price = updated_portfolio.loc[position_index, 'Last Price']
        else:
            current_price = price
        
        # Calculate new quantity and value
        new_quantity = current_quantity + quantity
        new_value = new_quantity * current_price
        
        # Update the position
        updated_portfolio.loc[position_index, 'Quantity'] = new_quantity
        updated_portfolio.loc[position_index, 'Current Value'] = new_value
    else:
        # Add new position
        try:
            # Get ticker data
            ticker = yf.Ticker(symbol)
            
            # Get ticker information
            info = ticker.info
            description = info.get('shortName', symbol)
            
            # Get the latest price if not provided
            if price is None:
                data = ticker.history(period='1d')
                # Reset timezone information
                data.index = data.index.tz_localize(None)
                if not data.empty:
                    price = data['Close'].iloc[-1]
                else:
                    raise ValueError(f"Could not get price for {symbol}")
            
            # Calculate current value
            current_value = quantity * price
            
            # Get dividend data if available
            dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
            annual_dividend = info.get('dividendRate', 0) if info.get('dividendRate') else 0
            estimated_annual_income = annual_dividend * quantity
            
            # Create new row with available data
            new_position = {
                'Symbol': symbol,
                'Description': description,
                'Quantity': quantity,
                'Last Price': price,
                'Last Price Change': 0,  # Assume no change for new position
                'Current Value': current_value,
                'Percent Of Account': 0,  # Will be recalculated
                'Ex-Date': pd.NaT,  # Not available for new position
                'Amount Per Share': annual_dividend,
                'Pay-Date': pd.NaT,  # Not available for new position
                'Dist. Yield': dividend_yield,
                'Distribution yield as of': None,
                'SEC Yield': None,
                'SEC yield as of': None,
                'Est. Annual Income': estimated_annual_income
            }
            
            # Add the new position to the portfolio
            updated_portfolio = pd.concat([updated_portfolio, pd.DataFrame([new_position])], ignore_index=True)
        
        except Exception as e:
            raise Exception(f"Error adding position for {symbol}: {str(e)}")
    
    # Recalculate percent of account
    total_value = updated_portfolio['Current Value'].sum()
    updated_portfolio['Percent Of Account'] = updated_portfolio['Current Value'] / total_value * 100
    
    return updated_portfolio

def remove_portfolio_position(portfolio_data, symbol, quantity=None):
    """
    Remove a position from the portfolio or reduce its quantity
    
    Parameters:
    - portfolio_data (pandas.DataFrame): Current portfolio data
    - symbol (str): Stock/ETF symbol to remove
    - quantity (float, optional): Number of shares to remove. If None, entire position is removed.
    
    Returns:
    - pandas.DataFrame: Updated portfolio data
    """
    # Create a copy of the portfolio data
    updated_portfolio = portfolio_data.copy()
    
    # Check if symbol exists in portfolio
    if symbol not in updated_portfolio['Symbol'].values:
        raise ValueError(f"Symbol {symbol} not found in portfolio")
    
    # Get position index
    position_index = updated_portfolio[updated_portfolio['Symbol'] == symbol].index[0]
    
    # If quantity is None or greater than or equal to current quantity, remove entire position
    current_quantity = updated_portfolio.loc[position_index, 'Quantity']
    
    if quantity is None or quantity >= current_quantity:
        # Remove the entire position
        updated_portfolio = updated_portfolio[updated_portfolio['Symbol'] != symbol]
    else:
        # Reduce the position quantity
        new_quantity = current_quantity - quantity
        current_price = updated_portfolio.loc[position_index, 'Last Price']
        new_value = new_quantity * current_price
        
        # Update the position
        updated_portfolio.loc[position_index, 'Quantity'] = new_quantity
        updated_portfolio.loc[position_index, 'Current Value'] = new_value
        
        # Recalculate estimated annual income
        amount_per_share = updated_portfolio.loc[position_index, 'Amount Per Share']
        updated_portfolio.loc[position_index, 'Est. Annual Income'] = amount_per_share * new_quantity
    
    # Recalculate percent of account
    if not updated_portfolio.empty:
        total_value = updated_portfolio['Current Value'].sum()
        updated_portfolio['Percent Of Account'] = updated_portfolio['Current Value'] / total_value * 100
    
    return updated_portfolio

def analyze_portfolio_risk(portfolio_data):
    """
    Perform a comprehensive risk analysis of the portfolio
    
    Parameters:
    - portfolio_data (pandas.DataFrame): DataFrame containing portfolio data
    
    Returns:
    - dict: Dictionary containing various risk metrics
    """
    # Check if portfolio is empty
    if portfolio_data.empty:
        return {
            'total_stocks': 0,
            'concentration_risk': {},
            'dividend_risk': {},
            'volatility_analysis': {}
        }
    
    # Calculate total portfolio value
    total_value = portfolio_data['Current Value'].sum()
    
    # Concentration Risk Analysis
    concentration_risk = {}
    
    # Individual stock concentration
    stock_concentration = portfolio_data.groupby('Symbol')['Percent Of Account'].sum()
    concentration_risk['stock_concentration'] = stock_concentration.to_dict()
    
    # Large positions risk (more than 10% of portfolio)
    large_positions = stock_concentration[stock_concentration > 10]
    concentration_risk['large_positions'] = large_positions.to_dict()
    
    # Dividend Risk Analysis
    dividend_risk = {}
    
    # Dividend yield variability
    dividend_stocks = portfolio_data[portfolio_data['Dist. Yield'] > 0]
    if not dividend_stocks.empty:
        dividend_risk = {
            'avg_yield': dividend_stocks['Dist. Yield'].mean(),
            'median_yield': dividend_stocks['Dist. Yield'].median(),
            'yield_std_dev': dividend_stocks['Dist. Yield'].std(),
            'yield_range': {
                'min': dividend_stocks['Dist. Yield'].min(),
                'max': dividend_stocks['Dist. Yield'].max()
            }
        }
    
    # Volatility Analysis
    volatility_analysis = {}
    try:
        # Fetch historical price data for each stock and calculate volatility
        volatilities = {}
        for symbol in portfolio_data['Symbol']:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period='1y')
                
                # Calculate annualized volatility
                daily_returns = data['Close'].pct_change()
                volatility = daily_returns.std() * np.sqrt(252)  # Annualize
                
                volatilities[symbol] = volatility
            except Exception as e:
                print(f"Could not fetch volatility for {symbol}: {e}")
        
        volatility_analysis = {
            'individual_stock_volatility': volatilities,
            'portfolio_volatility': np.mean(list(volatilities.values()))
        }
    except Exception as e:
        print(f"Error in volatility analysis: {e}")
    
    return {
        'total_stocks': len(portfolio_data),
        'concentration_risk': concentration_risk,
        'dividend_risk': dividend_risk,
        'volatility_analysis': volatility_analysis
    }

def get_income_projection(portfolio_data, months=12):
    """
    Project dividend income for a specified number of months
    
    Parameters:
    - portfolio_data (pandas.DataFrame): Portfolio data
    - months (int): Number of months to project
    
    Returns:
    - pandas.DataFrame: Monthly income projection
    """
    # Get current date
    today = datetime.now()
    
    # Create a dataframe for months
    date_range = pd.date_range(start=today, periods=months, freq='M')
    income_df = pd.DataFrame({'Month': date_range, 'Income': 0.0})
    
    # Filter to dividend-paying positions
    dividend_positions = portfolio_data[portfolio_data['Amount Per Share'] > 0].copy()
    
    if dividend_positions.empty:
        return income_df
    
    # For each position, estimate dividend months and amounts
    for _, position in dividend_positions.iterrows():
        symbol = position['Symbol']
        quantity = position['Quantity']
        amount_per_share = position['Amount Per Share']
        payment_date = position['Pay-Date']
        
        # Simple approach: distribute annual income evenly across months
        monthly_income = (amount_per_share * quantity) / 4  # Assuming quarterly payments
        
        # Try to determine dividend frequency and months
        try:
            # If we have a payment date, use it to estimate the payment pattern
            if pd.notna(payment_date):
                payment_month = payment_date.month
                
                # Assume quarterly payments (most common)
                payment_months = [(payment_month + i * 3) % 12 for i in range(4)]
                payment_months = [m if m != 0 else 12 for m in payment_months]  # Replace 0 with 12 for December
                
                # For each projected month, add income if it's a payment month
                for i, month in enumerate(income_df['Month']):
                    if month.month in payment_months:
                        income_df.loc[i, 'Income'] += amount_per_share * quantity
            else:
                # Distribute evenly if we don't have payment date info
                for i in range(months):
                    if i % 3 == 0:  # Assuming quarterly payments
                        income_df.loc[i, 'Income'] += amount_per_share * quantity
        except:
            # If there's an error, just distribute evenly
            for i in range(months):
                if i % 3 == 0:  # Assuming quarterly payments
                    income_df.loc[i, 'Income'] += amount_per_share * quantity
    
    return income_df

def get_portfolio_performance_history(portfolio_data, start_date, end_date):
    """
    Calculate the historical performance of the portfolio
    
    Parameters:
    - portfolio_data (pandas.DataFrame): Portfolio data
    - start_date (datetime): Start date for historical data
    - end_date (datetime): End date for historical data
    
    Returns:
    - pandas.DataFrame: Historical portfolio value
    """
    # Convert dates to string format required by yfinance
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    # Get unique symbols
    symbols = portfolio_data['Symbol'].unique().tolist()
    
    # Get historical data for each symbol
    historical_data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_str, end=end_str)
            # Reset timezone information
            data.index = data.index.tz_localize(None)
            if not data.empty:
                historical_data[symbol] = data['Close']
        except Exception as e:
            print(f"Error getting historical data for {symbol}: {str(e)}")
    
    # Combine all historical price data
    price_history = pd.DataFrame(historical_data)
    
    # Fill any missing values using forward fill
    price_history = price_history.fillna(method='ffill')
    
    # Calculate portfolio value for each day
    portfolio_value = pd.DataFrame(index=price_history.index)
    portfolio_value['Value'] = 0
    
    for symbol in symbols:
        if symbol in price_history.columns:
            # Get quantity
            quantity = portfolio_data.loc[portfolio_data['Symbol'] == symbol, 'Quantity'].iloc[0]
            
            # Calculate value
            portfolio_value['Value'] += price_history[symbol] * quantity
    
    # Calculate daily returns
    portfolio_value['Daily Return'] = portfolio_value['Value'].pct_change()
    
    # Calculate cumulative returns
    portfolio_value['Cumulative Return'] = (1 + portfolio_value['Daily Return']).cumprod() - 1
    
    # Calculate drawdowns
    portfolio_value['Peak'] = portfolio_value['Value'].cummax()
    portfolio_value['Drawdown'] = (portfolio_value['Value'] - portfolio_value['Peak']) / portfolio_value['Peak']
    
    return portfolio_value