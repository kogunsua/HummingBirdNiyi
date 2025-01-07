class AssetDataFetcher:
    @staticmethod
    def _get_coingecko_data(symbol: str) -> Optional[pd.DataFrame]:
        """Fetch cryptocurrency data from CoinGecko"""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': '365',
                'interval': 'daily'
            }
            response = requests.get(url, params=params, timeout=10)
            
            # Check for rate limit or errors
            if response.status_code == 429:
                st.warning("CoinGecko rate limit reached, falling back to Polygon.io...")
                return None
            
            response.raise_for_status()
            data = response.json()
            
            # Process price data
            prices_df = pd.DataFrame(data['prices'], columns=['Date', 'Close'])
            prices_df['Date'] = pd.to_datetime(prices_df['Date'], unit='ms')
            
            # Process volume data
            volumes_df = pd.DataFrame(data['total_volumes'], columns=['Date', 'Volume'])
            volumes_df['Date'] = pd.to_datetime(volumes_df['Date'], unit='ms')
            
            # Merge data
            df = prices_df.merge(volumes_df[['Date', 'Volume']], on='Date', how='left')
            
            # Add other required columns
            df['Open'] = df['Close'].shift(1)
            df['High'] = df['Close']
            df['Low'] = df['Close']
            
            # Set index and ensure timezone-naive
            df.set_index('Date', inplace=True)
            df.index = df.index.tz_localize(None)
            
            # Forward fill any missing values
            df = df.ffill()
            
            return df
            
        except Exception as e:
            st.warning(f"CoinGecko error: {str(e)}, attempting fallback...")
            return None

    @staticmethod
    def _get_polygon_crypto_data(symbol: str) -> Optional[pd.DataFrame]:
        """Fetch cryptocurrency data from Polygon.io"""
        try:
            # Convert common crypto symbols to Polygon format
            crypto_mapping = {
                'bitcoin': 'X:BTCUSD',
                'ethereum': 'X:ETHUSD',
                'ripple': 'X:XRPUSD',
                'dogecoin': 'X:DOGEUSD',
                'cardano': 'X:ADAUSD',
                'solana': 'X:SOLUSD',
                'polkadot': 'X:DOTUSD',
                'litecoin': 'X:LTCUSD',
                'chainlink': 'X:LINKUSD',
                'stellar': 'X:XLMUSD'
            }
            
            polygon_symbol = crypto_mapping.get(symbol.lower(), f'X:{symbol.upper()}USD')
            end_date = date.today()
            start_date = end_date - timedelta(days=365)
            
            url = f"https://api.polygon.io/v2/aggs/ticker/{polygon_symbol}/range/1/day/{start_date}/{end_date}"
            params = {
                'apiKey': Config.POLYGON_API_KEY,
                'limit': 365
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('resultsCount', 0) == 0:
                return None
                
            df = pd.DataFrame(data['results'])
            df['Date'] = pd.to_datetime(df['t'], unit='ms')
            
            # Rename columns to match our format
            df = df.rename(columns={
                'o': 'Open',
                'h': 'High',
                'l': 'Low',
                'c': 'Close',
                'v': 'Volume'
            })
            
            # Set index and ensure timezone-naive
            df.set_index('Date', inplace=True)
            df.index = df.index.tz_localize(None)
            
            # Sort by date
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            st.error(f"Polygon.io error: {str(e)}")
            return None

    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL)
    def get_stock_data(symbol: str) -> Optional[pd.DataFrame]:
        """Fetch stock data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1y", interval="1d")
            
            if not data.empty:
                data.index = pd.to_datetime(data.index).tz_localize(None)
                return data
                
            raise ValueError(f"No data available for {symbol}")
            
        except Exception as e:
            st.error(f"Error fetching stock data: {str(e)}")
            return None

    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL)
    def get_crypto_data(symbol: str) -> Optional[pd.DataFrame]:
        """Fetch cryptocurrency data with fallback support"""
        try:
            # Try CoinGecko first
            st.info(f"Fetching {symbol} data from CoinGecko...")
            data = AssetDataFetcher._get_coingecko_data(symbol)
            if data is not None:
                return data
            
            # Fallback to Polygon.io if CoinGecko fails
            st.info(f"Falling back to Polygon.io for {symbol} data...")
            data = AssetDataFetcher._get_polygon_crypto_data(symbol)
            if data is not None:
                return data
            
            raise ValueError(f"Could not fetch data for {symbol} from any source")
            
        except Exception as e:
            st.error(f"Error fetching {symbol} data: {str(e)}")
            return None