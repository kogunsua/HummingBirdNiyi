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
    
    # API Keys
    FRED_API_KEY = "a81e9c33d8dbac1cc1309e51527e0d53"
    ALPHA_VANTAGE_API_KEY = "E3R1QOXBCW9924S"
    POLYGON_API_KEY = "9rP1CLlxuoRWPvkEiOMxxIwNyffjUEb4"
    
    # Prophet Configuration
    PROPHET_CONFIG = {
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': False,
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10.0,
        'holidays_prior_scale': 10.0,
        'regressors': {
            'sentiment': {
                'mode': 'multiplicative',
                'prior_scale': 10.0
            },
            'economic': {
                'mode': 'multiplicative',
                'prior_scale': 10.0
            }
        }
    }
    # Data Sources
    DATA_SOURCES = {
        # Original sources
        "Polygon.io": "Real-time and historical stock, forex, and cryptocurrency data",
        "CoinGecko": "Cryptocurrency data aggregator for price, volume, and historical data",
        "FRED API": "Federal Reserve Economic Data for GDP and macroeconomic indicators",
        "Yahoo Finance": "Historical price data for stocks and ETFs",
        # New sources
        "GDELT Project": "Global sentiment and event analysis data",
        "Quandl": "Mortgage rate data, treasury yield data, and housing market trends",
        "U.S. Census Bureau API": "Housing permits, construction, and other indicators",
        "Alpha Vantage": "Macroeconomic data and financial market data"
    }

    # Economic Indicators
    INDICATORS = {
        # Original indicators
        'GDP': {
            'series_id': 'GDP',
            'description': 'Gross Domestic Product',
            'frequency': 'Quarterly',
            'units': 'Billions of USD'
        },
        'UNRATE': {
            'series_id': 'UNRATE',
            'description': 'Unemployment Rate',
            'frequency': 'Monthly',
            'units': 'Percent'
        },
        'CPIAUCSL': {
            'series_id': 'CPIAUCSL',
            'description': 'Consumer Price Index',
            'frequency': 'Monthly',
            'units': 'Index 1982-1984=100'
        },
        'DFF': {
            'series_id': 'DFF',
            'description': 'Federal Funds Rate',
            'frequency': 'Daily',
            'units': 'Percent'
        },
        'IEF': {
            'description': 'iShares 7-10 Year Treasury Bond ETF',
            'frequency': 'Daily',
            'units': 'USD'
        },
        # New GDELT indicators
        'POLSENT': {
            'description': 'Political Sentiment Index',
            'frequency': 'Daily',
            'units': 'Sentiment Score'
        },
        'MARKETSENT': {
            'description': 'Market Sentiment Index',
            'frequency': 'Daily',
            'units': 'Sentiment Score'
        },
        'GLOBALTONE': {
            'description': 'Global Tone Index',
            'frequency': 'Daily',
            'units': 'Tone Score'
        }
    }
    # Real Estate Indicators
    REAL_ESTATE_INDICATORS = {
        'MORTGAGE30US': {
            'series_id': 'MORTGAGE30US',
            'description': '30-Year Fixed Rate Mortgage Average',
            'frequency': 'Weekly',
            'units': 'Percent',
            'source': 'FRED',
            'status': 'Active'
        },
        'HOUST': {
            'series_id': 'HOUST',
            'description': 'Housing Starts',
            'frequency': 'Monthly',
            'units': 'Thousands of Units',
            'source': 'FRED',
            'status': 'Active'
        },
        'CSUSHPISA': {
            'series_id': 'CSUSHPISA',
            'description': 'Case-Shiller Home Price Index',
            'frequency': 'Monthly',
            'units': 'Index',
            'source': 'FRED',
            'status': 'Active'
        },
        'RRVRUSQ156N': {
            'series_id': 'RRVRUSQ156N',
            'description': 'Rental Vacancy Rate',
            'frequency': 'Quarterly',
            'units': 'Percent',
            'source': 'FRED',
            'status': 'Active'
        },
        'MSACSR': {
            'series_id': 'MSACSR',
            'description': 'Monthly Supply of New Houses',
            'frequency': 'Monthly',
            'units': 'Months Supply',
            'source': 'FRED',
            'status': 'Active'
        }
    }

    # GDELT Configuration
    GDELT_CONFIG = {
        "update_frequency": "15 minutes",
        "base_url": "http://data.gdeltproject.org/gdeltv2/",
        "gkg_base_url": "http://data.gdeltproject.org/gkg/",
        "themes_of_interest": [
            "ECON_", "BUS_", "FIN_", "MARKET", "INVEST",
            "CRYPTO", "BLOCKCHAIN", "STOCK_MARKET"
        ],
        "sentiment_weights": {
            "tone": 0.4,
            "themes": 0.3,
            "volume": 0.3
        },
        "required_columns": {
            "events": ["SQLDATE", "EventCode", "GoldsteinScale", "AvgTone"],
            "gkg": ["DATE", "THEMES", "TONE", "NUMARTS"]
        }
    }
    # Technical Analysis Parameters
    TECHNICAL_PARAMS = {
        'rsi_period': 14,
        'ma_short': 20,
        'ma_long': 50,
        'volatility_window': 20,
        'bollinger_window': 20,
        'bollinger_std': 2
    }

# Model Descriptions
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
        • Now includes GDELT sentiment analysis
        
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
        """,
        'technical_level': 'Medium-High',
        'computation_speed': 'Fast',
        'confidence_rating': 0.88,
        'development_status': 'Under Development'
    }
}
    