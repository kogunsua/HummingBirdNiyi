# config.py
from datetime import date, timedelta

class Config:
    # Original Configuration (Unchanged)
    END_DATE = date.today()
    START_DATE = END_DATE - timedelta(days=365)
    TODAY = END_DATE.strftime("%Y-%m-%d")
    START = START_DATE.strftime("%Y-%m-%d")
    CACHE_TTL = 3600
    DEFAULT_TICKER = "AAPL"
    DEFAULT_CRYPTO = "bitcoin"
    DEFAULT_PERIODS = 30
    ASSET_TYPES = ["Stocks", "Cryptocurrency"]
    
    # Original API Keys (Unchanged)
    FRED_API_KEY = "a81e9c33d8dbac1cc1309e51527e0d53"
    ALPHA_VANTAGE_API_KEY = "E3R1QOXBCPW9924S"
    POLYGON_API_KEY = "9rP1CLlxuoRWPvkEiOMxxIwNyffjUEb4"
    
    # Original Data Sources (Unchanged)
    DATA_SOURCES = {
        "Polygon.io": "Real-time and historical stock, forex, and cryptocurrency data",
        "CoinGecko": "Cryptocurrency data aggregator for price, volume, and historical data",
        "FRED API": "Federal Reserve Economic Data for GDP and macroeconomic indicators",
        "Yahoo Finance": "Historical price data for stocks and ETFs",
        "Quandl": "Mortgage rate data, treasury yield data, and housing market trends",
        "U.S. Census Bureau API": "Datasets for housing permits, construction, and other indicators",
        "Alpha Vantage": "Macroeconomic data, including interest rates and financial market data"
    }
    
    # Updated Economic Indicators (Added Political Sentiment)
    INDICATORS = {
        # Original indicators
        'GDP': 'Gross Domestic Product',
        'UNRATE': 'Unemployment Rate',
        'CPIAUCSL': 'Consumer Price Index',
        'DFF': 'Federal Funds Rate',
        'IEF': 'iShares 7-10 Year Treasury Bond ETF',
        # New indicator
        'POLSENT': 'Political Sentiment Index'
    }
    
    # Original Real Estate Indicators (Unchanged)
    REAL_ESTATE_INDICATORS = {
        'Treasury Yields': {
            'description': 'Treasury yield curves and rates',
            'status': 'Under Development'
        },
        'Federal Reserve Policies': {
            'description': 'Federal Reserve monetary policy impacts',
            'status': 'Under Development'
        },
        'Inflation': {
            'description': 'Inflation rates and trends',
            'status': 'Under Development'
        },
        'Economic Growth': {
            'description': 'GDP and economic growth metrics',
            'status': 'Under Development'
        },
        'Housing Market Trends': {
            'description': 'Housing market indicators and trends',
            'status': 'Under Development'
        },
        'Credit Market Conditions': {
            'description': 'Credit market health and trends',
            'status': 'Under Development'
        },
        'Global Economic Events': {
            'description': 'Major global economic indicators',
            'status': 'Under Development'
        }
    }

    # Original CRYPTO_MAPPINGS (Unchanged)
    CRYPTO_MAPPINGS = {
        'bitcoin': {
            'coingecko': 'bitcoin',
            'polygon': 'X:BTCUSD',
            'description': 'Bitcoin'
        },
        'ethereum': {
            'coingecko': 'ethereum',
            'polygon': 'X:ETHUSD',
            'description': 'Ethereum'
        },
        # ... rest of crypto mappings ...
    }

# Original MODEL_DESCRIPTIONS (Unchanged)
MODEL_DESCRIPTIONS = {
    "Prophet": {
        "description": """
        Facebook's Prophet model is designed for forecasting time series data. It's particularly good at:
        - Handling daily observations with strong seasonal effects
        - Missing data and outliers
        - Shifts in the trend
        - Different types of seasonality
        """,
        "confidence_rating": 0.85,
        "best_use_cases": [
            "Stock price forecasting",
            "Cryptocurrency price prediction",
            "Economic indicator analysis"
        ],
        "limitations": [
            "May not capture sudden market shocks",
            "Requires sufficient historical data"
        ],
        "development_status": "Active"
    },
    # ... rest of model descriptions ...
}