# config.py
from datetime import date, timedelta

class Config:
    # Time configurations
    END_DATE = date.today()
    START_DATE = END_DATE - timedelta(days=365)
    TODAY = END_DATE.strftime("%Y-%m-%d")
    START = START_DATE.strftime("%Y-%m-%d")
    CACHE_TTL = 3600

    # Default values
    DEFAULT_TICKER = "MSFT"
    DEFAULT_CRYPTO = "xrp"
    DEFAULT_PERIODS = 30
    ASSET_TYPES = ["Stocks", "Cryptocurrency"]

    # API Keys (Original)
    FRED_API_KEY = "a81e9c33d8dbac1cc1309e51527e0d53"
    ALPHA_VANTAGE_API_KEY = "E3R1QOXBCW9924S"
    POLYGON_API_KEY = "9rP1CLlxuoRWPvkEiOMxxIwNyffjUEb4"

    # GDELT Configurations (New)
    GDELT_CONFIG = {
        "update_frequency": "15 minutes",
        "base_url": "http://data.gdeltproject.org/gdeltv2/",
        "gkg_base_url": "http://data.gdeltproject.org/gkg/",
        "themes_of_interest": [
            "ECON_", "BUS_", "FIN_", "MARKET", "INVEST",
            "CRYPTO", "BLOCKCHAIN", "STOCK_MARKET",
            "BANKING", "MONETARY", "TRADE", "REGULATION"
        ],
        "sentiment_weights": {
            "tone": 0.4,
            "themes": 0.3,
            "volume": 0.3
        },
        "required_columns": {
            "events": ["SQLDATE", "EventCode", "GoldsteinScale", "AvgTone"],
            "gkg": ["DATE", "THEMES", "TONE", "NUMARTS"]
        },
        "sentiment_scale": {
            "strongly_positive": 0.8,
            "positive": 0.3,
            "neutral": 0.0,
            "negative": -0.3,
            "strongly_negative": -0.8
        }
    }

    # Original Data Sources plus new ones
    DATA_SOURCES = {
        # Original sources
        "Polygon.io": "Real-time and historical stock, forex, and cryptocurrency data",
        "CoinGecko": "Cryptocurrency data aggregator for price, volume, and historical data",
        "FRED API": "Federal Reserve Economic Data for GDP and macroeconomic indicators",
        "Yahoo Finance": "Historical price data for stocks and ETFs",
        # New sources
        "GDELT Project": "Global political event and sentiment data (GDELT 2.0)",
        "Quandl": "Mortgage rate data, treasury yield data, and housing market trends",
        "U.S. Census Bureau API": "Housing permits, construction, and other indicators",
        "Alpha Vantage": "Macroeconomic data and financial market data"
    }

    # Economic Indicators (Original + New)
    INDICATORS = {
        # Original indicators
        'GDP': 'Gross Domestic Product',
        'UNRATE': 'Unemployment Rate',
        'CPIAUCSL': 'Consumer Price Index',
        'DFF': 'Federal Funds Rate',
        'IEF': 'iShares 7-10 Year Treasury Bond ETF',
        # New indicators
        'POLSENT': 'Political Sentiment Index (GDELT)',
        'MARKETSENT': 'Market Sentiment Index (GDELT GKG)',
        'GLOBALTONE': 'Global Tone Index (GDELT)',
        'T10YFF': '10-Year Treasury Constant Maturity Minus Federal Funds Rate',
        'REALGDP': 'Real Gross Domestic Product',
        'PCE': 'Personal Consumption Expenditures',
        'INDPRO': 'Industrial Production Index'
    }

    # Real Estate Indicators (Original + New)
    REAL_ESTATE_INDICATORS = {
        # Original indicators
        'HOME_PRICE_INDEX': {
            'description': 'Case-Shiller Home Price Index',
            'status': 'Beta',
            'source': 'FRED'
        },
        'MORTGAGE_RATES': {
            'description': '30-Year Fixed Mortgage Rates',
            'status': 'Beta',
            'source': 'FRED'
        },
        # New indicators
        'PERMIT': {
            'description': 'New Private Housing Units Authorized by Building Permits',
            'status': 'Beta',
            'source': 'FRED'
        },
        'HOUST': {
            'description': 'Housing Starts: Total: New Privately Owned Housing Units Started',
            'status': 'Development',
            'source': 'FRED'
        },
        'RRVRUSQ156N': {
            'description': 'Rental Vacancy Rate',
            'status': 'Development',
            'source': 'FRED'
        }
    }

    # Model Configurations (Enhanced)
    PROPHET_CONFIG = {
        'daily_seasonality': True,
        'weekly_seasonality': True,
        'yearly_seasonality': True,
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10,
        'holidays_prior_scale': 10,
        'mcmc_samples': 0,
        'interval_width': 0.95,
        'growth': 'linear'
    }

    # Technical Analysis Parameters (New)
    TECHNICAL_PARAMS = {
        'rsi_period': 14,
        'ma_short': 20,
        'ma_long': 50,
        'volatility_window': 20,
        'bollinger_window': 20,
        'bollinger_std': 2
    }

# Model descriptions (Enhanced)
MODEL_DESCRIPTIONS = {
    'Prophet': {
        'name': 'Prophet with GDELT Integration',
        'description': """
        Enhanced Facebook Prophet model integrating GDELT 2.0 GKG sentiment analysis.
        
        Key Features:
        • Advanced sentiment integration from GDELT GKG
        • Multiple data source handling (price, sentiment, economic)
        • Automatic seasonality detection
        • Robust to missing data and outliers
        
        Best Used For:
        • Medium to long-term forecasting
        • Assets affected by global events
        • Market sentiment analysis
        • Stocks and cryptocurrencies with clear patterns
        """,
        'technical_level': 'Advanced',
        'computation_speed': 'Medium',
        'confidence_rating': 0.88,
        'development_status': 'Active'
    },
    'LSTM': {
        'name': 'LSTM with Sentiment Integration',
        'description': """
        Deep learning model optimized for time series prediction with integrated sentiment analysis.
        
        Key Features:
        • Long-term pattern recognition
        • Multi-feature input processing
        • Advanced sequence modeling
        
        Best Used For:
        • Complex pattern recognition
        • High-frequency trading signals
        • Sentiment-driven assets
        """,
        'technical_level': 'Expert',
        'computation_speed': 'Slow',
        'confidence_rating': 0.82,
        'development_status': 'Under Development'
    },
    'XGBoost': {
        'name': 'XGBoost with Multi-Source Data',
        'description': """
        Gradient boosting model optimized for financial forecasting.
        
        Key Features:
        • Feature importance analysis
        • Robust to outliers
        • Handles mixed data types
        
        Best Used For:
        • Feature-rich predictions
        • Short to medium-term forecasting
        • Market regime detection
        """,
        'technical_level': 'Advanced',
        'computation_speed': 'Fast',
        'confidence_rating': 0.85,
        'development_status': 'Under Development'
    },
    'Linear Regression': {
        'name': 'Linear Regression with Technical Indicators',
        'description': """
        Enhanced linear model with technical and sentiment features.
        
        Key Features:
        • Simple yet effective
        • Easily interpretable results
        • Fast computation
        
        Best Used For:
        • Quick market analysis
        • Baseline predictions
        • Technical indicator testing
        """,
        'technical_level': 'Basic',
        'computation_speed': 'Very Fast',
        'confidence_rating': 0.75,
        'development_status': 'Active'
    },
    'Random Forest': {
        'name': 'Random Forest Ensemble',
        'description': """
        Ensemble learning model combining multiple decision trees.
        
        Key Features:
        • Handles non-linear relationships
        • Feature importance ranking
        • Robust predictions
        
        Best Used For:
        • Multi-factor analysis
        • Risk assessment
        • Pattern classification
        """,
        'technical_level': 'Intermediate',
        'computation_speed': 'Medium',
        'confidence_rating': 0.80,
        'development_status': 'Under Development'
    }
}