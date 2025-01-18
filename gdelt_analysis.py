# gdelt_analysis.py (updated fetch_sentiment_data method)

class GDELTAnalyzer:
    def __init__(self):
        # Use the correct GDELT API endpoint
        self.base_url = "https://api.gdeltproject.org/api/v2/events/events"
        self.theme_filters = [
            'ECON_BANKRUPTCY', 'ECON_COST', 'ECON_DEBT', 'ECON_REFORM',
            'BUS_MARKET_CLOSE', 'BUS_MARKET_CRASH', 'BUS_MARKET_DOWN',
            'BUS_MARKET_UP', 'BUS_STOCK_MARKET', 'POLITICS'
        ]

    def fetch_sentiment_data(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch sentiment data from GDELT 2.0"""
        try:
            # Format dates for GDELT API
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            # Construct the theme filter string
            theme_filter = ' OR '.join([f'theme:{theme}' for theme in self.theme_filters])
            
            # Construct the API URL with proper parameters
            url = f"{self.base_url}?query={theme_filter}&mode=timelinevol&format=json&TIMESPAN=1&starttime={start_date}&endtime={end_date}&maxrecords=1000"
            
            logger.info(f"Fetching GDELT data with URL: {url}")
            
            # Make the API request
            response = requests.get(url, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"GDELT API request failed with status code {response.status_code}")
                raise ValueError(f"GDELT API request failed with status code {response.status_code}")
            
            # Parse the JSON response
            data = response.json()
            
            if not data:
                logger.warning("No data received from GDELT API")
                raise ValueError("No data received from GDELT API")
            
            # Convert the timeline data to DataFrame
            timeline_data = []
            for date_str, values in data.items():
                try:
                    date = pd.to_datetime(date_str)
                    timeline_data.append({
                        'ds': date,
                        'article_count': values['count'],
                        'tone_avg': values.get('avg_tone', 0),
                        'tone_std': values.get('tone_std', 0),
                        'mention_count': values.get('mentions', 0),
                        'source_count': values.get('sources', 0)
                    })
                except Exception as e:
                    logger.warning(f"Error processing date {date_str}: {str(e)}")
                    continue
            
            if not timeline_data:
                raise ValueError("No valid data points found in GDELT response")
            
            # Create DataFrame
            df = pd.DataFrame(timeline_data)
            
            # Calculate sentiment score (normalized between 0 and 1)
            df['sentiment_score'] = (df['tone_avg'] + 100) / 200
            
            # Sort by date
            df = df.sort_values('ds')
            
            logger.info(f"Successfully fetched {len(df)} data points from GDELT")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching GDELT data: {str(e)}")
            raise ValueError(f"Network error fetching GDELT data: {str(e)}")
        except Exception as e:
            logger.error(f"Error fetching GDELT data: {str(e)}")
            raise ValueError(f"Failed to fetch sentiment data: {str(e)}")

    def process_gdelt_response(self, response_text: str) -> pd.DataFrame:
        """Process GDELT API response into a DataFrame"""
        try:
            # Split the response into lines
            lines = response_text.strip().split('\n')
            if not lines:
                raise ValueError("Empty response from GDELT")

            # Parse the CSV data
            data = []
            for line in lines[1:]:  # Skip header
                try:
                    fields = line.split('\t')
                    if len(fields) >= 15:  # Ensure minimum required fields
                        data.append({
                            'date': fields[1],
                            'tone': float(fields[7]),
                            'mentions': int(fields[8]),
                            'sources': int(fields[9]),
                            'articles': int(fields[10])
                        })
                except (ValueError, IndexError) as e:
                    logger.warning(f"Error parsing line: {str(e)}")
                    continue

            if not data:
                raise ValueError("No valid data points found in GDELT response")

            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['ds'] = pd.to_datetime(df['date'], format='%Y%m%d%H%M%S')
            
            # Group by date and calculate metrics
            daily_data = df.groupby(df['ds'].dt.date).agg({
                'tone': ['mean', 'std', 'count'],
                'mentions': 'sum',
                'sources': 'mean',
                'articles': 'sum'
            }).reset_index()

            # Flatten column names
            daily_data.columns = ['ds', 'tone_avg', 'tone_std', 'article_count', 
                                'mention_count', 'source_count', 'total_articles']
            
            # Convert date column
            daily_data['ds'] = pd.to_datetime(daily_data['ds'])
            
            # Calculate sentiment score
            daily_data['sentiment_score'] = (daily_data['tone_avg'] + 100) / 200
            
            return daily_data.sort_values('ds')
            
        except Exception as e:
            logger.error(f"Error processing GDELT response: {str(e)}")
            raise ValueError(f"Failed to process GDELT response: {str(e)}")
            
def fetch_sentiment_data_backup(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Backup method to fetch sentiment data using alternative GDELT endpoint"""
        try:
            # Use the GKG (Global Knowledge Graph) endpoint
            base_url = "https://api.gdeltproject.org/api/v2/gkg/gkg"
            
            # Format dates
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            all_data = []
            current_dt = start_dt
            
            while current_dt <= end_dt:
                date_str = current_dt.strftime("%Y%m%d")
                
                # Construct URL for each day
                url = f"{base_url}?date={date_str}&format=json&maxrows=1000"
                url += "&QUERY=" + "+OR+".join([f"theme%3A{theme}" for theme in self.theme_filters])
                
                try:
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        data = response.json()
                        if data and 'articles' in data:
                            for article in data['articles']:
                                try:
                                    if 'tone' in article:
                                        all_data.append({
                                            'ds': current_dt,
                                            'tone': article['tone'],
                                            'mentions': article.get('mentions', 1),
                                            'sources': article.get('sources', 1)
                                        })
                                except Exception as e:
                                    logger.warning(f"Error processing article: {str(e)}")
                                    continue
                    
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Error fetching data for {date_str}: {str(e)}")
                
                current_dt += timedelta(days=1)
            
            if not all_data:
                raise ValueError("No data available from backup GDELT endpoint")
            
            # Convert to DataFrame and process
            df = pd.DataFrame(all_data)
            
            # Group by date and calculate metrics
            daily_data = df.groupby('ds').agg({
                'tone': ['mean', 'std', 'count'],
                'mentions': 'sum',
                'sources': 'mean'
            }).reset_index()
            
            # Flatten columns
            daily_data.columns = ['ds', 'tone_avg', 'tone_std', 'article_count', 
                                'mention_count', 'source_count']
            
            # Calculate sentiment score
            daily_data['sentiment_score'] = (daily_data['tone_avg'] + 100) / 200
            
            return daily_data.sort_values('ds')
            
        except Exception as e:
            logger.error(f"Error in backup GDELT fetch: {str(e)}")
            raise ValueError(f"Backup GDELT fetch failed: {str(e)}")

    def try_all_fetch_methods(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Try all available methods to fetch sentiment data"""
        errors = []
        
        # Try primary method
        try:
            return self.fetch_sentiment_data(start_date, end_date)
        except Exception as e:
            errors.append(f"Primary method failed: {str(e)}")
        
        # Try backup method
        try:
            return self.fetch_sentiment_data_backup(start_date, end_date)
        except Exception as e:
            errors.append(f"Backup method failed: {str(e)}")
        
        # If all methods fail, raise error with details
        raise ValueError(f"All GDELT fetch methods failed:\n" + "\n".join(errors))
        
def integrate_sentiment_analysis(sentiment_period: int) -> Optional[pd.DataFrame]:
    """Integrate sentiment analysis into the main application"""
    try:
        # Initialize GDELT analyzer
        analyzer = GDELTAnalyzer()
        
        # Try to get sentiment data using all available methods
        try:
            with st.spinner('Fetching sentiment data...'):
                sentiment_data = analyzer.try_all_fetch_methods(
                    (datetime.now() - timedelta(days=sentiment_period)).strftime("%Y-%m-%d"),
                    datetime.now().strftime("%Y-%m-%d")
                )
        except ValueError as e:
            st.error("Unable to fetch sentiment data")
            st.info("Please try again later or proceed with price-only forecast")
            logger.error(f"Sentiment data fetch failed: {str(e)}")
            return None
        
        if sentiment_data is None or sentiment_data.empty:
            st.error("No sentiment data available")
            return None
        
        # Display sentiment analysis
        st.markdown("### 📊 Market Sentiment Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_sentiment = sentiment_data['sentiment_score'].iloc[-1]
            sentiment_change = sentiment_data['sentiment_score'].iloc[-1] - sentiment_data['sentiment_score'].iloc[-2]
            st.metric(
                "Current Sentiment",
                f"{current_sentiment:.2f}",
                f"{sentiment_change:+.2f}"
            )
        
        with col2:
            st.metric(
                "Average Sentiment",
                f"{sentiment_data['sentiment_score'].mean():.2f}"
            )
        
        with col3:
            st.metric(
                "Sentiment Volatility",
                f"{sentiment_data['sentiment_score'].std():.2f}"
            )
        
        # Create sentiment trend visualization
        fig_sentiment = go.Figure()
        fig_sentiment.add_trace(
            go.Scatter(
                x=sentiment_data['ds'],
                y=sentiment_data['sentiment_score'],
                name='Sentiment Score',
                line=dict(color='purple')
            )
        )
        
        fig_sentiment.update_layout(
            title="Market Sentiment Trend",
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
            height=400
        )
        
        st.plotly_chart(fig_sentiment, use_container_width=True)
        
        return sentiment_data
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis integration: {str(e)}")
        st.error("Error integrating sentiment analysis")
        st.info("Proceeding with price-only forecast")
        return None
        
def integrate_sentiment_analysis(sentiment_period: int) -> Optional[pd.DataFrame]:
    """Integrate sentiment analysis into the main application"""
    try:
        # Initialize GDELT analyzer
        analyzer = GDELTAnalyzer()
        
        # Try to get sentiment data using all available methods
        try:
            with st.spinner('Fetching sentiment data...'):
                sentiment_data = analyzer.try_all_fetch_methods(
                    (datetime.now() - timedelta(days=sentiment_period)).strftime("%Y-%m-%d"),
                    datetime.now().strftime("%Y-%m-%d")
                )
        except ValueError as e:
            st.error("Unable to fetch sentiment data")
            st.info("Please try again later or proceed with price-only forecast")
            logger.error(f"Sentiment data fetch failed: {str(e)}")
            return None
        
        if sentiment_data is None or sentiment_data.empty:
            st.error("No sentiment data available")
            return None
        
        # Display sentiment analysis
        st.markdown("### 📊 Market Sentiment Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_sentiment = sentiment_data['sentiment_score'].iloc[-1]
            sentiment_change = sentiment_data['sentiment_score'].iloc[-1] - sentiment_data['sentiment_score'].iloc[-2]
            st.metric(
                "Current Sentiment",
                f"{current_sentiment:.2f}",
                f"{sentiment_change:+.2f}"
            )
        
        with col2:
            st.metric(
                "Average Sentiment",
                f"{sentiment_data['sentiment_score'].mean():.2f}"
            )
        
        with col3:
            st.metric(
                "Sentiment Volatility",
                f"{sentiment_data['sentiment_score'].std():.2f}"
            )
        
        # Create sentiment trend visualization
        fig_sentiment = go.Figure()
        fig_sentiment.add_trace(
            go.Scatter(
                x=sentiment_data['ds'],
                y=sentiment_data['sentiment_score'],
                name='Sentiment Score',
                line=dict(color='purple')
            )
        )
        
        fig_sentiment.update_layout(
            title="Market Sentiment Trend",
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
            height=400
        )
        
        st.plotly_chart(fig_sentiment, use_container_width=True)
        
        return sentiment_data
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis integration: {str(e)}")
        st.error("Error integrating sentiment analysis")
        st.info("Proceeding with price-only forecast")
        return None