# config.py
import streamlit as st
from datetime import date, timedelta

class Config:
    END_DATE = date.today()
    START_DATE = END_DATE - timedelta(days=365)
    TODAY = END_DATE.strftime("%Y-%m-%d")
    START = START_DATE.strftime("%Y-%m-%d")
    CACHE_TTL = 3600
    DEFAULT_TICKER = "MSFT"
    DEFAULT_CRYPTO = "xrp"
    DEFAULT_PERIODS = 30
    ASSET_TYPES = ["Stocks", "Cryptocurrency"]
    
    # API Keys from Streamlit secrets
    @property
    def FRED_API_KEY(self):
        return st.secrets["api_keys"]["fred"]
    
    @property
    def ALPHA_VANTAGE_API_KEY(self):
        return st.secrets["api_keys"]["alpha_vantage"]
    
    @property
    def POLYGON_API_KEY(self):
        return st.secrets["api_keys"]["polygon"]
    
    @property
    def NEWS_API_KEY(self):
        return st.secrets["api_keys"]["news"]
    
    # Data Sources
    DATA_SOURCES = {
        "Polygon.io": "Real-time and historical stock, forex, and cryptocurrency data",
        "CoinGecko": "Cryptocurrency data aggregator for price, volume, and historical data",
        "FRED API": "Federal Reserve Economic Data for GDP and macroeconomic indicators",
        "Yahoo Finance": "Historical price data for stocks and ETFs",
        "Quandl": "Mortgage rate data, treasury yield data, and housing market trends",
        "U.S. Census Bureau API": "Datasets for housing permits, construction, and other indicators",
        "Alpha Vantage": "Macroeconomic data, including interest rates and financial market data",
        "Political Data": "Political sentiment and market impact analysis"
    }

    # Dividend Analysis Settings
    DIVIDEND_DEFAULTS = {
        'DEFAULT_DIVIDEND_STOCKS': ['O', 'MAIN', 'STAG', 'GOOD', 'AGNC', 'SDIV', 'CLM'],
        'REIT_TICKERS': ['O', 'STAG', 'GOOD'],
        'YIELD_THRESHOLDS': {
            'HEALTHY_MIN': 3,
            'HEALTHY_MAX': 7,
            'WARNING': 10
        },
        'PAYOUT_RATIOS': {
            'NORMAL_MAX': 75,
            'REIT_MAX': 90
        }
    }
    
    # Economic Indicators
    INDICATORS = {
        'CPIAUCSL': 'Consumer Price Index',
        'DFF': 'Federal Funds Rate',
        'GDP': 'Gross Domestic Product',
        'IEF': 'iShares 7-10 Year Treasury Bond ETF',
        'POLSENT': 'Political Sentiment',
        'UNRATE': 'Unemployment Rate'
    }
    
    # Detailed Indicator Information
    INDICATOR_DETAILS = {
        'POLSENT': {
            'description': 'Political Sentiment Index',
            'frequency': 'Daily',
            'source': 'Multi-source analysis',
            'units': 'Sentiment Score (-1 to 1)',
            'impact': 'Measures political climate impact on markets',
            'methodology': 'Combines news sentiment, policy analysis, and market reaction',
            'update_frequency': 'Real-time'
        },
        'CPIAUCSL': {
            'description': 'Consumer Price Index',
            'frequency': 'Monthly',
            'source': 'FRED',
            'units': 'Index 1982-1984=100',
            'impact': 'Measures inflation and purchasing power'
        },
        'DFF': {
            'description': 'Federal Funds Rate',
            'frequency': 'Daily',
            'source': 'FRED',
            'units': 'Percent',
            'impact': 'Key interest rate affecting markets'
        },
        'GDP': {
            'description': 'Gross Domestic Product',
            'frequency': 'Quarterly',
            'source': 'FRED',
            'units': 'Billions of Dollars',
            'impact': 'Overall economic health indicator'
        },
        'IEF': {
            'description': 'iShares 7-10 Year Treasury Bond ETF',
            'frequency': 'Daily',
            'source': 'Yahoo Finance',
            'units': 'USD',
            'impact': 'Treasury yield and bond market indicator'
        },
        'UNRATE': {
            'description': 'Unemployment Rate',
            'frequency': 'Monthly',
            'source': 'FRED',
            'units': 'Percent',
            'impact': 'Labor market and economic health indicator'
        }
    }
    
    # Real Estate Indicators
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

    # Political Sentiment Configuration
    POLITICAL_SENTIMENT_CONFIG = {
        'data_sources': ['News API', 'Social Media', 'Market Analysis'],
        'update_frequency': 'Daily',
        'analysis_methods': ['NLP', 'Sentiment Analysis', 'Topic Modeling'],
        'impact_metrics': ['Market Correlation', 'Volatility Impact', 'Trend Analysis'],
        'lookback_period': 365,
        'confidence_threshold': 0.6
    }

# Model descriptions moved outside the Config class since they're static
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
            "Economic indicator analysis",
            "Political sentiment integration"
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
