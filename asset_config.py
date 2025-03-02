# asset_config.py
class AssetConfig:
    # Prophet Model Configurations
    STOCK_CONFIG = {
        'changepoint_prior_scale': 0.001,
        'n_changepoints': 25,
        'seasonality_mode': 'multiplicative',
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': False,
        'interval_width': 0.95
    }
    
    CRYPTO_CONFIG = {
        'changepoint_prior_scale': 0.05,
        'n_changepoints': 35,
        'seasonality_mode': 'multiplicative',
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': True,
        'interval_width': 0.95
    }
    
    ETF_CONFIG = {
        'changepoint_prior_scale': 0.001,
        'n_changepoints': 25,
        'seasonality_mode': 'multiplicative',
        'yearly_seasonality': True,
        'weekly_seasonality': True,
        'daily_seasonality': False,
        'interval_width': 0.95
    }
    
    # Technical Indicator Parameters
    STOCK_INDICATORS = {
        'ma_periods': [5, 20, 50],
        'rsi_period': 14,
        'macd_periods': (12, 26, 9),
        'bb_period': 20,
        'atr_period': 14,
        'roc_period': 12
    }
    
    CRYPTO_INDICATORS = {
        'ma_periods': [7, 25, 99],
        'rsi_period': 21,
        'macd_periods': (12, 26, 9),
        'bb_period': 24,
        'atr_period': 21,
        'roc_period': 24,
        'volume_ma_period': 24,
        'dominance_period': 7
    }
    
    ETF_INDICATORS = {
        'ma_periods': [10, 30, 60],
        'rsi_period': 14,
        'macd_periods': (12, 26, 9),
        'bb_period': 20,
        'atr_period': 14,
        'roc_period': 12
    }

    # Visualization Settings
    STOCK_VIZ = {
        'confidence_colors': ['rgba(0,100,255,0.1)', 'rgba(0,100,255,0.2)'],
        'line_colors': ['green', 'red'],
        'plot_height': 800
    }
    
    CRYPTO_VIZ = {
        'confidence_colors': ['rgba(255,165,0,0.1)', 'rgba(255,165,0,0.2)'],
        'line_colors': ['orange', 'red'],
        'plot_height': 900
    }
    
    ETF_VIZ = {
        'confidence_colors': ['rgba(100,100,255,0.1)', 'rgba(100,100,255,0.2)'],
        'line_colors': ['blue', 'purple'],
        'plot_height': 800
    }
    
    PORTFOLIO_VIZ = {
        'pie_colors': ['blue', 'green', 'red', 'orange', 'purple', 'yellow', 'pink', 'teal'],
        'line_colors': ['blue', 'green'],
        'plot_height': 600
    }

    # ARIMA model configurations
    ARIMA_CONFIGS = {
        'stocks': (5, 1, 0),
        'crypto': (5, 1, 0),
        'etf': (5, 1, 0)
    }

    # Portfolio Settings
    PORTFOLIO_CONFIG = {
        'default_timeframe': '1Y',
        'timeframes': ['1M', '3M', '6M', 'YTD', '1Y', '3Y', '5Y'],
        'risk_free_rate': 0.02,
        'income_projection_months': 12
    }

    @staticmethod
    def get_config(asset_type: str = 'stocks'):
        """Get configuration based on asset type"""
        asset_type = asset_type.lower()
        
        if asset_type == 'crypto' or asset_type == 'cryptocurrency':
            return {
                'model_config': AssetConfig.CRYPTO_CONFIG,
                'indicators': AssetConfig.CRYPTO_INDICATORS,
                'viz': AssetConfig.CRYPTO_VIZ,
                'arima_params': AssetConfig.ARIMA_CONFIGS['crypto']
            }
        elif asset_type == 'etf' or asset_type == 'etfs':
            return {
                'model_config': AssetConfig.ETF_CONFIG,
                'indicators': AssetConfig.ETF_INDICATORS,
                'viz': AssetConfig.ETF_VIZ,
                'arima_params': AssetConfig.ARIMA_CONFIGS['etf']
            }
        else:  # default to stocks
            return {
                'model_config': AssetConfig.STOCK_CONFIG,
                'indicators': AssetConfig.STOCK_INDICATORS,
                'viz': AssetConfig.STOCK_VIZ,
                'arima_params': AssetConfig.ARIMA_CONFIGS['stocks']
            }