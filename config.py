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
    
    # API Keys (Unchanged)
    FRED_API_KEY = "a81e9c33d8dbac1cc1309e51527e0d53"
    ALPHA_VANTAGE_API_KEY = "E3R1QOXBCW9924S"
    POLYGON_API_KEY = "9rP1CLlxuoRWPvkEiOMxxIwNyffjUEb4"
    
    # Data Sources
    DATA_SOURCES = {
        "Polygon.io": "Real-time and historical stock, forex, and cryptocurrency data",
        "CoinGecko": "Cryptocurrency data aggregator for price, volume, and historical data",
        "FRED API": "Federal Reserve Economic Data for GDP and macroeconomic indicators",
        "Yahoo Finance": "Historical price data for stocks and ETFs",
        "Quandl": "Mortgage rate data, treasury yield data, and housing market trends",
        "U.S. Census Bureau API": "Datasets for housing permits, construction, and other indicators",
        "Alpha Vantage": "Macroeconomic data, including interest rates and financial market data",
        "GDELT": "Global sentiment and event analysis data"
    }

    # Indicators (Unchanged)
    INDICATORS = {
        'GDP': 'Gross Domestic Product',
        'UNRATE': 'Unemployment Rate',
        'CPIAUCSL': 'Consumer Price Index',
        'DFF': 'Federal Funds Rate',
        'IEF': 'iShares 7-10 Year Treasury Bond ETF',
        'POLSENT': 'Political Sentiment Index'
    }

# Enhanced Model Descriptions
MODEL_DESCRIPTIONS = {
    'Prophet': {
        'name': 'Facebook Prophet',
        'description': """
        A powerful forecasting model developed by Facebook, designed for business-oriented time series data.
        
        Key Strengths:
        • Handles seasonal patterns (daily, weekly, yearly)
        • Robust to missing data and outliers
        • Automatically detects trend changes
        • Great for stocks with clear seasonal patterns
        
        Best Used For:
        • Long-term forecasting (weeks to months)
        • Stocks with seasonal patterns
        • Assets with clear trend components
        • Retail and consumer stocks
        
        Limitations:
        • May not capture sudden market changes
        • Less effective for high-frequency trading
        • Requires substantial historical data
        """,
        'technical_level': 'Medium',
        'computation_speed': 'Medium',
        'confidence_rating': 0.85,
        'development_status': 'Active'
    },
    
    'LSTM': {
        'name': 'Long Short-Term Memory Neural Network',
        'description': """
        A deep learning model specialized in remembering long-term patterns in sequential data.
        
        Key Strengths:
        • Captures complex non-linear patterns
        • Remembers long-term dependencies
        • Adapts to changing market conditions
        • Excellent for pattern recognition
        
        Best Used For:
        • High-frequency trading patterns
        • Volatile markets
        • Complex market behaviors
        • Tech and growth stocks
        """,
        'technical_level': 'High',
        'computation_speed': 'Slow',
        'confidence_rating': 0.80,
        'development_status': 'Under Development'
    },
    
    'XGBoost': {
        'name': 'XGBoost (Extreme Gradient Boosting)',
        'description': """
        A machine learning model that excels at combining multiple weaker predictions into a strong forecast.
        
        Key Strengths:
        • Handles multiple input features
        • Excellent at finding feature importance
        • Robust to outliers
        • Fast prediction speed
        """,
        'technical_level': 'Medium-High',
        'computation_speed': 'Fast',
        'confidence_rating': 0.88,
        'development_status': 'Under Development'
    },
    
    'Random Forest': {
        'name': 'Random Forest Ensemble',
        'description': """
        An ensemble learning model that combines multiple decision trees for robust predictions.
        
        Key Strengths:
        • Highly stable predictions
        • Handles non-linear relationships
        • Resistant to overfitting
        • Good with mixed data types
        """,
        'technical_level': 'Medium',
        'computation_speed': 'Fast',
        'confidence_rating': 0.82,
        'development_status': 'Under Development'
    },
    
    'Linear Regression': {
        'name': 'Linear Regression with Technical Indicators',
        'description': """
        A straightforward model that finds linear relationships between various market factors.
        
        Key Strengths:
        • Simple and interpretable
        • Fast computation
        • Good for trend following
        • Clear confidence intervals
        """,
        'technical_level': 'Low',
        'computation_speed': 'Very Fast',
        'confidence_rating': 0.75,
        'development_status': 'Under Development'
    },
    
    'Markov Chain': {
        'name': 'Markov Chain with Market States',
        'description': """
        A probabilistic model that analyzes market states and transition probabilities.
        
        Key Strengths:
        • Excellent for regime detection
        • Provides state transition probabilities
        • Good for risk assessment
        • Captures market regime changes
        """,
        'technical_level': 'Medium-High',
        'computation_speed': 'Medium',
        'confidence_rating': 0.80,
        'development_status': 'Under Development'
    }
}