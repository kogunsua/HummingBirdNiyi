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

    @staticmethod
    def get_config(asset_type: str = 'stocks'):
        """Get configuration based on asset type"""
        is_crypto = asset_type.lower() == 'crypto'
        return {
            'model_config': AssetConfig.CRYPTO_CONFIG if is_crypto else AssetConfig.STOCK_CONFIG,
            'indicators': AssetConfig.CRYPTO_INDICATORS if is_crypto else AssetConfig.STOCK_INDICATORS,
            'viz': AssetConfig.CRYPTO_VIZ if is_crypto else AssetConfig.STOCK_VIZ
        }