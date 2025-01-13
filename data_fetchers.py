# data_fetchers.py

class AssetDataFetcher:
    def get_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Public wrapper for fetching stock data"""
        try:
            logger.info(f"Fetching stock data for symbol: {symbol}")
            
            # Get data from Yahoo Finance
            data = DataSourceManager.fetch_yahoo_data(symbol, Config.START, Config.TODAY)
            
            if data is not None and not data.empty:
                # Ensure data has correct columns
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in required_columns:
                    if col not in data.columns:
                        raise ValueError(f"Missing required column: {col}")
                
                # Convert index to datetime and remove timezone
                data.index = pd.to_datetime(data.index).tz_localize(None)
                
                logger.info(f"Successfully fetched stock data for {symbol}")
                logger.info(f"Data shape: {data.shape}")
                logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
                
                return data
            
            raise ValueError(f"No data available for {symbol}")
        
        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}")
            st.error(f"Could not fetch data for {symbol}. Please verify the symbol.")
            return None
