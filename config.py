# config.py
from datetime import date, timedelta

class Config:
    END_DATE = date.today()
    START_DATE = END_DATE - timedelta(days=365)
    TODAY = END_DATE.strftime("%Y-%m-%d")
    START = START_DATE.strftime("%Y-%m-%d")
    CACHE_TTL = 3600
    DEFAULT_TICKER = "MSFT"
    DEFAULT_CRYPTO = "ripple"
    DEFAULT_PERIODS = 30
    ASSET_TYPES = ["Stocks", "Cryptocurrency"]
    
    # API Keys
    FRED_API_KEY = "a81e9c33d8dbac1cc1309e51527e0d53"
    ALPHA_VANTAGE_API_KEY = "E3R1QOXBCPW9924S"
    POLYGON_API_KEY = "9rP1CLlxuoRWPvkEiOMxxIwNyffjUEb4"
    
    # Data Sources
    DATA_SOURCES = {
        "Polygon.io": "Real-time and historical stock, forex, and cryptocurrency data",
        "CoinGecko": "Cryptocurrency data aggregator for price, volume, and historical data",
        "FRED API": "Federal Reserve Economic Data for GDP and macroeconomic indicators",
        "Yahoo Finance": "Historical price data for stocks and ETFs"
    }
    
    # Economic Indicators
    INDICATORS = {
        'GDP': 'Gross Domestic Product',
        'UNRATE': 'Unemployment Rate',
        'CPIAUCSL': 'Consumer Price Index',
        'DFF': 'Federal Funds Rate',
        'IEF': 'iShares 7-10 Year Treasury Bond ETF',
        'Political' :'Sentiment'
    }

# Model descriptions
MODEL_DESCRIPTIONS = {
    "Prophet": {
        "description": """
        Prophet model is designed for forecasting time series data. It's particularly good at:
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
    "XGBoost": {
        "description": "Gradient boosting model for feature-rich prediction",
        "confidence_rating": 0.88,
        "best_use_cases": ["Feature-rich prediction"],
        "limitations": ["Under development"],
        "development_status": "Under Development"
    },
    "Random Forest": {
        "description": "Ensemble tree-based model for stable prediction",
        "confidence_rating": 0.82,
        "best_use_cases": ["Stable prediction"],
        "limitations": ["Under development"],
        "development_status": "Under Development"
    },
    "Linear Regression": {
        "description": "Classic statistical model for trend following",
        "confidence_rating": 0.75,
        "best_use_cases": ["Trend following"],
        "limitations": ["Under development"],
        "development_status": "Under Development"
    },
    "Markov Chain": {
        "description": "Probabilistic state model for regime analysis",
        "confidence_rating": 0.80,
        "best_use_cases": ["Regime analysis"],
        "limitations": ["Under development"],
        "development_status": "Under Development"
    }
}