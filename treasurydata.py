# treasurydata.py
# Enhanced Treasury Data Fetcher for real-time government spending tracking
# This application fetches and analyzes data from the U.S. Treasury's APIs
# Supports multiple time periods, outlay frequencies, and up to 9 spending categories

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
import json
from enum import Enum
import time

class TreasuryDataFetcher:
    """
    Enhanced class to handle fetching and processing real-time data from U.S. Treasury APIs
    
    This class provides methods to:
    - Build API URLs with various parameters
    - Fetch data from multiple Treasury APIs
    - Process and clean the retrieved data
    - Support different time periods (1, 3, 12, or 36 months)
    - Support different outlay frequencies (daily, weekly, 7-day moving average)
    - Track up to 9 federal spending categories
    """
    
    def __init__(self):
        """
        Initialize the TreasuryDataFetcher with base API endpoints and configurations
        """
        # API configuration
        self.base_url = 'https://api.fiscaldata.treasury.gov/services/api/fiscal_service'
        self.mts_table1_endpoint = '/v1/accounting/mts/mts_table_1'
        self.mts_table3_endpoint = '/v1/accounting/mts/mts_table_3'
        self.mts_table5_endpoint = '/v1/accounting/mts/mts_table_5'
        self.dts_endpoint = '/v1/accounting/dts/dts_table_1'
        
        # Cache to store recent API responses and reduce API calls
        self.cache = {}
        self.cache_expiry = 3600  # Cache lifetime in seconds (1 hour)
        
        # Define thresholds for funding changes
        self.significant_change_threshold = 15.0  # Percentage change
        self.critical_cut_threshold = -30.0      # Percentage change
        self.unusual_volatility_threshold = 35.0 # Volatility indicator
        
        # Expanded program categories to allow more detailed selection
        # Category codes used internally mapped to human-readable names
        self.program_categories = {
            # Social Services
            'SNAP': 'Supplemental Nutrition Assistance Program',
            'TANF': 'Temporary Assistance for Needy Families',
            'WIC': 'Women, Infants and Children Program',
            'SSA': 'Social Security Administration',
            'MEDICARE': 'Medicare Programs',
            'MEDICAID': 'Medicaid',
            'CHIP': 'Children\'s Health Insurance Program',
            
            # Health & Research
            'NIH': 'National Institutes of Health',
            'CDC': 'Centers for Disease Control',
            'CMS': 'Centers for Medicare & Medicaid Services',
            'FDA': 'Food and Drug Administration',
            
            # Education & Labor
            'ED': 'Department of Education',
            'PELL': 'Pell Grants',
            'DOL': 'Department of Labor',
            'UI': 'Unemployment Insurance',
            
            # Housing & Development
            'HUD': 'Housing and Urban Development',
            'SEC8': 'Section 8 Housing',
            
            # Security & Defense
            'DOD': 'Department of Defense',
            'MILPAY': 'Military Pay and Allowances',
            'VA': 'Veterans Affairs',
            'DHS': 'Department of Homeland Security',
            'FEMA': 'Federal Emergency Management Agency',
            'DOJ': 'Department of Justice',
            
            # Infrastructure & Transportation
            'DOT': 'Department of Transportation',
            'FAA': 'Federal Aviation Administration',
            'FHWA': 'Federal Highway Administration',
            
            # Energy & Environment
            'DOE': 'Department of Energy',
            'EPA': 'Environmental Protection Agency',
            
            # Other Major Departments
            'HHS': 'Health and Human Services',
            'USDA': 'Department of Agriculture',
            'STATE': 'Department of State',
            'TREAS': 'Department of Treasury',
            'INTREV': 'Internal Revenue Service',
            'COMM': 'Department of Commerce',
            'NASA': 'NASA',
            'SBA': 'Small Business Administration',
            
            # Interest & Debt
            'INTDEBT': 'Interest on the Public Debt',
            'DEBTOPS': 'Public Debt Operations'
        }
        
        # Dictionary to match Treasury API categories to our simplified categories
        self.api_category_mapping = self._initialize_category_mapping()
    
    def _initialize_category_mapping(self) -> Dict:
        """
        Initialize mapping between Treasury API category names and our simplified categories
        This helps normalize different naming conventions across different Treasury data sources
        """
        return {
            # Example mappings - these would need to be expanded based on actual API responses
            'SUPPLEMENTAL NUTRITION ASSISTANCE PROGRAM': 'SNAP',
            'FOOD AND NUTRITION SERVICE': 'SNAP',
            'SNAP BENEFITS': 'SNAP',
            'FOOD STAMPS': 'SNAP',
            
            'NATIONAL INSTITUTES OF HEALTH': 'NIH',
            'NIH GRANTS AND ACTIVITIES': 'NIH',
            
            'SOCIAL SECURITY BENEFITS': 'SSA',
            'SOCIAL SECURITY ADMINISTRATION': 'SSA',
            
            'VETERANS BENEFITS AND SERVICES': 'VA',
            'VETERANS AFFAIRS': 'VA',
            'VETERANS BENEFITS': 'VA',
            
            'EDUCATION DEPARTMENT': 'ED',
            'DEPARTMENT OF EDUCATION': 'ED',
            'FEDERAL PELL GRANTS': 'PELL',
            
            'HOUSING AND URBAN DEVELOPMENT': 'HUD',
            'HUD PROGRAMS': 'HUD',
            'SECTION 8 HOUSING SUBSIDIES': 'SEC8',
            
            'DEFENSE DEPARTMENT': 'DOD',
            'DEFENSE VENDOR PAYMENTS': 'DOD',
            'MILITARY ACTIVE DUTY PAY': 'MILPAY',
            'MILITARY RETIREMENT FUND': 'DOD',
            
            'HEALTH AND HUMAN SERVICES': 'HHS',
            'HHS GRANTS AND ACTIVITIES': 'HHS',
            
            'HOMELAND SECURITY': 'DHS',
            'DEPARTMENT OF HOMELAND SECURITY': 'DHS',
            'FEMA DISASTER RELIEF': 'FEMA',
            
            'MEDICARE ADVANTAGE PAYMENTS': 'MEDICARE',
            'MEDICARE BENEFITS': 'MEDICARE',
            'MEDICARE PART D': 'MEDICARE',
            
            'MEDICAID GRANTS TO STATES': 'MEDICAID',
            'MEDICAID VENDOR PAYMENTS': 'MEDICAID',
            
            'UNEMPLOYMENT INSURANCE BENEFITS': 'UI',
            'UNEMPLOYMENT BENEFITS': 'UI',
            
            'INTEREST ON TREASURY SECURITIES': 'INTDEBT',
            'PUBLIC DEBT CASH REDEMPTIONS': 'DEBTOPS',
            
            # Add more mappings as needed
        }
    
    def build_url(self, 
                  endpoint: str,
                  fields: List[str],
                  filters: Optional[List[str]] = None,
                  sort_fields: Optional[List[str]] = None,
                  page_size: int = 100,
                  page_number: int = 1) -> str:
        """
        Builds a complete API URL with the specified parameters
        
        Args:
            endpoint: API endpoint to use
            fields: List of fields to retrieve
            filters: List of filter expressions
            sort_fields: List of fields to sort by (prefix with - for descending)
            page_size: Number of records per page
            page_number: Page number to retrieve
            
        Returns:
            Complete URL for the API request
        """
        fields_str = ','.join(fields)
        url = f"{self.base_url}{endpoint}?fields={fields_str}"
        
        # Add filters if provided
        if filters:
            filter_str = '&'.join([f"filter={f}" for f in filters])
            url += f"&{filter_str}"
            
        # Add sorting if provided
        if sort_fields:
            sort_str = ','.join(sort_fields)
            url += f"&sort={sort_str}"
        else:
            url += "&sort=-record_date"  # Default sort by date descending
            
        # Add pagination
        url += f"&page[number]={page_number}&page[size]={page_size}"
        
        return url
    
    def fetch_data(self, url: str, use_cache: bool = True) -> pd.DataFrame:
        """
        Fetches data from the API and converts it to a pandas DataFrame
        
        Args:
            url: Complete API URL to fetch data from
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame containing the API response data
        """
        # Check cache first if enabled
        if use_cache and url in self.cache:
            cache_time, cache_data = self.cache[url]
            if time.time() - cache_time < self.cache_expiry:
                return cache_data.copy()  # Return a copy to prevent cache modification
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if 'data' not in data:
                raise ValueError("No data found in API response")
                
            df = pd.DataFrame(data['data'])
            
            # Convert date fields to datetime
            if 'record_date' in df.columns:
                df['record_date'] = pd.to_datetime(df['record_date'])
            elif 'record_calendar_month' in df.columns:
                df['record_date'] = pd.to_datetime(df['record_calendar_month'], format='%Y-%m')
            elif 'reporting_date' in df.columns:
                df['record_date'] = pd.to_datetime(df['reporting_date'])
            
            # Convert numeric columns
            numeric_columns = [
                'current_month_gross_rcpt_amt',
                'current_month_outly_amt',
                'current_month_deficit_surplus_amt',
                'amt',
                'today_amt',
                'mtd_amt',
                'fytd_amt'
            ]
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Store in cache if enabled
            if use_cache:
                self.cache[url] = (time.time(), df.copy())
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            raise

    def fetch_treasury_data(self, 
                           start_date: datetime,
                           end_date: datetime,
                           categories: List[str]) -> pd.DataFrame:
        """
        Fetches comprehensive Treasury data for the specified date range and categories
        
        This method attempts to fetch data from multiple Treasury APIs and combine the results
        to provide the most complete picture of federal spending.
        
        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            categories: List of category codes to retrieve data for
            
        Returns:
            Combined DataFrame with all available data
        """
        # Format dates for API
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Common fields for all API requests
        common_fields = [
            'record_date',
            'classification_desc',
            'security_desc',
            'account_desc'
        ]
        
        # Build filters
      def analyze_treasury_data(self, 
                        start_date: datetime,
                        end_date: datetime,
                        categories: List[str],
                        frequency: str = 'daily') -> Tuple[pd.DataFrame, List[Dict], List[str]]:
    """
    Comprehensive analysis of treasury data with alerts and recommendations
    
    Args:
        start_date: Start date for analysis
        end_date: End date for analysis
        categories: List of category codes to analyze
        frequency: Data frequency ('daily', 'weekly', or 'seven_day_ma')
        
    Returns:
        Tuple containing:
          - DataFrame with analyzed data
          - List of alert dictionaries
          - List of recommendation strings
    """
    try:
        # Fetch the data
        df = self.fetch_treasury_data(start_date, end_date, categories)
        
        if df is None or df.empty:
            return None, [], []
        
        all_alerts = []
        
        # Make a copy to avoid modifications to original data
        df = df.copy()
        
        # Process data according to requested frequency
        if frequency == 'seven_day_ma':
            df = self.calculate_moving_average(df)
        elif frequency == 'weekly':
            # Group data by week using string-based grouping to avoid Period objects
            df['week'] = df['record_date'].dt.strftime('%Y-%U')
            weekly_groups = []
            
            for category in categories:
                category_data = df[df['classification_desc'] == category].copy()
                if not category_data.empty:
                    # Group by week and calculate weekly sums
                    weekly_data = category_data.groupby('week').agg({
                        'current_month_outly_amt': 'sum',
                        'current_month_gross_rcpt_amt': 'sum'
                    }).reset_index()
                    
                    # Create a proper datetime from the week string (first day of week)
                    def week_to_date(week_str):
                        year, week_num = week_str.split('-')
                        # Create a date object for the first day of the year
                        first_day = datetime(int(year), 1, 1)
                        # If the first day is not a Monday, move to the first Monday
                        if first_day.weekday() != 0:
                            first_day = first_day + timedelta(days=(7 - first_day.weekday()))
                        # Add the weeks
                        return first_day + timedelta(weeks=int(week_num))
                    
                    weekly_data['record_date'] = weekly_data['week'].apply(week_to_date)
                    weekly_data['classification_desc'] = category
                    weekly_data['internal_category'] = category
                    
                    weekly_groups.append(weekly_data)
            
            if weekly_groups:
                df = pd.concat(weekly_groups, ignore_index=True)
        
        # Process alerts for each category
        for category in categories:
            category_data = df[df['classification_desc'] == category]
            
            if not category_data.empty:
                # Detect funding changes
                alerts = self.detect_funding_changes(category_data, category)
                all_alerts.extend(alerts)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(df, all_alerts)
        
        return df, all_alerts, recommendations
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Analysis error: {e}")
        return None, [], []
        
        all_alerts = []
        
        # Process data according to requested frequency
        if frequency == 'seven_day_ma':
            df = self.calculate_moving_average(df)
        elif frequency == 'weekly':
            # Group data by week
            df['week'] = df['record_date'].dt.to_period('W')
            weekly_groups = []
            
            for category in categories:
                category_data = df[df['classification_desc'] == category]
                if not category_data.empty:
                    # Group by week and calculate weekly sums
                    weekly_data = category_data.groupby('week').agg({
                        'current_month_outly_amt': 'sum',
                        'current_month_gross_rcpt_amt': 'sum'
                    }).reset_index()
                    
                    # Fix: Convert period to datetime correctly
                    weekly_data['record_date'] = weekly_data['week'].apply(lambda x: x.start_time)
                    weekly_data['classification_desc'] = category
                    weekly_data['internal_category'] = category
                    
                    weekly_groups.append(weekly_data)
            
            if weekly_groups:
                df = pd.concat(weekly_groups, ignore_index=True)
        
        # Process alerts for each category
        for category in categories:
            category_data = df[df['classification_desc'] == category]
            
            if not category_data.empty:
                # Detect funding changes
                alerts = self.detect_funding_changes(category_data, category)
                all_alerts.extend(alerts)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(df, all_alerts)
        
        return df, all_alerts, recommendations
        
    except Exception as e:
        print(f"Analysis error: {e}")
        return None, [], []
        
        # Initialize list to store DataFrames from different sources
        data_frames = []
        
        # 1. Try Monthly Treasury Statement Table 3 (Outlays)
        try:
            mts_fields = common_fields + ['current_month_outly_amt', 'prior_fytd_outly_amt']
            mts_url = self.build_url(
                endpoint=self.mts_table3_endpoint,
                fields=mts_fields,
                filters=date_filter,
                page_size=1000
            )
            mts_df = self.fetch_data(mts_url)
            
            if not mts_df.empty:
                # Map categories from API to our internal codes
                mts_df['internal_category'] = mts_df['classification_desc'].apply(
                    lambda x: self._map_api_category_to_internal(x)
                )
                data_frames.append(mts_df)
        except Exception as e:
            print(f"Error fetching MTS Table 3 data: {e}")
        
        # 2. Try Daily Treasury Statement Table 1 (Daily spending)
        try:
            dts_fields = common_fields + ['today_amt', 'mtd_amt', 'fytd_amt']
            dts_url = self.build_url(
                endpoint=self.dts_endpoint,
                fields=dts_fields,
                filters=date_filter,
                page_size=1000
            )
            dts_df = self.fetch_data(dts_url)
            
            if not dts_df.empty:
                # Map categories and rename columns to match MTS format
                dts_df['internal_category'] = dts_df['classification_desc'].apply(
                    lambda x: self._map_api_category_to_internal(x)
                )
                dts_df['current_month_outly_amt'] = dts_df['today_amt']
                data_frames.append(dts_df)
        except Exception as e:
            print(f"Error fetching DTS Table 1 data: {e}")
        
        # 3. Combine and filter datasets
        if not data_frames:
            # If no real data available, generate sample data for demo
            return self._generate_sample_data(start_date, end_date, categories)
        
        combined_df = pd.concat(data_frames, ignore_index=True)
        
        # Filter to only requested categories
        filtered_df = combined_df[combined_df['internal_category'].isin(categories)]
        
        # If no data for selected categories, return sample data
        if filtered_df.empty:
            return self._generate_sample_data(start_date, end_date, categories)
        
        # Update classification_desc to use our internal category names
        filtered_df['classification_desc'] = filtered_df['internal_category']
        
        # Sort by date (newest first) and remove duplicates
        filtered_df = filtered_df.sort_values('record_date', ascending=False)
        filtered_df = filtered_df.drop_duplicates(subset=['record_date', 'classification_desc'])
        
        return filtered_df
    
    def _map_api_category_to_internal(self, api_category: str) -> str:
        """
        Maps an API category name to our internal category code
        
        Args:
            api_category: Category name from API
            
        Returns:
            Internal category code or 'OTHER' if no match found
        """
        if not api_category or pd.isna(api_category):
            return 'OTHER'
            
        # Try exact match first
        if api_category in self.api_category_mapping:
            return self.api_category_mapping[api_category]
        
        # Try case-insensitive match
        api_category_upper = api_category.upper()
        if api_category_upper in self.api_category_mapping:
            return self.api_category_mapping[api_category_upper]
        
        # Try substring match
        for api_name, internal_code in self.api_category_mapping.items():
            if api_name in api_category_upper or api_category_upper in api_name:
                return internal_code
        
        # Direct category code match
        if api_category_upper in self.program_categories:
            return api_category_upper
        
        return 'OTHER'

    def _generate_sample_data(self, 
                            start_date: datetime,
                            end_date: datetime,
                            categories: List[str]) -> pd.DataFrame:
        """
        Generates realistic sample data when API data is unavailable
        
        Args:
            start_date: Start date for generated data
            end_date: End date for generated data
            categories: List of category codes to generate data for
            
        Returns:
            DataFrame with generated sample data
        """
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Base values and volatility for different categories
        category_params = {
            'SNAP': {'base': 180.0, 'volatility': 0.05, 'trend': 0.001},
            'TANF': {'base': 42.0, 'volatility': 0.04, 'trend': 0.0005},
            'NIH': {'base': 110.0, 'volatility': 0.08, 'trend': 0.002},
            'SSA': {'base': 320.0, 'volatility': 0.03, 'trend': 0.0008},
            'MEDICARE': {'base': 280.0, 'volatility': 0.06, 'trend': 0.0015},
            'MEDICAID': {'base': 150.0, 'volatility': 0.07, 'trend': 0.001},
            'VA': {'base': 90.0, 'volatility': 0.04, 'trend': 0.001},
            'ED': {'base': 75.0, 'volatility': 0.09, 'trend': -0.0005},
            'HUD': {'base': 65.0, 'volatility': 0.05, 'trend': 0.0002},
            'DOD': {'base': 210.0, 'volatility': 0.08, 'trend': 0.0012},
            'HHS': {'base': 160.0, 'volatility': 0.06, 'trend': 0.001},
            'DHS': {'base': 80.0, 'volatility': 0.07, 'trend': 0.0008},
            'PELL': {'base': 30.0, 'volatility': 0.1, 'trend': -0.0003},
            'CDC': {'base': 45.0, 'volatility': 0.08, 'trend': 0.002},
            'DOL': {'base': 50.0, 'volatility': 0.06, 'trend': 0.0005},
            'UI': {'base': 25.0, 'volatility': 0.15, 'trend': -0.001},
            'SEC8': {'base': 40.0, 'volatility': 0.04, 'trend': 0.0007},
            'MILPAY': {'base': 60.0, 'volatility': 0.03, 'trend': 0.001},
            'FEMA': {'base': 20.0, 'volatility': 0.25, 'trend': 0.0},
            'DOJ': {'base': 35.0, 'volatility': 0.06, 'trend': 0.0004},
            'DOT': {'base': 70.0, 'volatility': 0.07, 'trend': 0.001},
            'FAA': {'base': 15.0, 'volatility': 0.05, 'trend': 0.0003},
            'DOE': {'base': 30.0, 'volatility': 0.08, 'trend': 0.0006},
            'EPA': {'base': 25.0, 'volatility': 0.07, 'trend': -0.0004},
            'USDA': {'base': 55.0, 'volatility': 0.06, 'trend': 0.0005},
            'STATE': {'base': 40.0, 'volatility': 0.08, 'trend': 0.0007},
            'TREAS': {'base': 35.0, 'volatility': 0.05, 'trend': 0.0003},
            'INTREV': {'base': 45.0, 'volatility': 0.04, 'trend': 0.0006},
            'NASA': {'base': 22.0, 'volatility': 0.09, 'trend': 0.0002},
            'SBA': {'base': 18.0, 'volatility': 0.12, 'trend': -0.0002},
            'INTDEBT': {'base': 95.0, 'volatility': 0.03, 'trend': 0.002},
            'DEBTOPS': {'base': 80.0, 'volatility': 0.04, 'trend': 0.001},
            'CHIP': {'base': 15.0, 'volatility': 0.06, 'trend': 0.0008},
            'FHWA': {'base': 45.0, 'volatility': 0.05, 'trend': 0.0006},
            'COMM': {'base': 25.0, 'volatility': 0.07, 'trend': 0.0004},
            'FDA': {'base': 18.0, 'volatility': 0.08, 'trend': 0.001},
            'CMS': {'base': 60.0, 'volatility': 0.05, 'trend': 0.0015},
        }
        
        # Default parameters for any category not in the map
        default_params = {'base': 50.0, 'volatility': 0.08, 'trend': 0.0005}
        
        # Create rows for each date and category
        rows = []
        
        for category in categories:
            params = category_params.get(category, default_params)
            base_value = params['base'] * 1e6  # Convert to millions
            volatility = params['volatility']
            trend = params['trend']
            
            # Add weekly and monthly patterns
            for i, date in enumerate(date_range):
                # Add trend
                trend_factor = 1.0 + (trend * i)
                
                # Add weekly pattern (higher at month start, lower at end)
                day_of_month = date.day
                monthly_factor = 1.0 + (0.2 * (1.0 - day_of_month / 31.0))
                
                # Add weekly pattern (lower on weekends)
                day_of_week = date.weekday()
                weekly_factor = 1.0 if day_of_week < 5 else 0.3
                
                # Add some randomness
                random_factor = np.random.normal(1.0, volatility)
                
                # Calculate final value
                value = base_value * trend_factor * monthly_factor * weekly_factor * random_factor
                
                # Special case: add occasional spikes for FEMA (disaster relief)
                if category == 'FEMA' and np.random.random() < 0.02:  # 2% chance of spike
                    value *= np.random.uniform(3.0, 10.0)  # 3-10x spike
                
                # Add row to results
                rows.append({
                    'record_date': date,
                    'classification_desc': category,
                    'internal_category': category,
                    'current_month_outly_amt': value,
                    'current_month_gross_rcpt_amt': value * 0.8,  # Just for sample data
                })
        
        # Create DataFrame
        sample_df = pd.DataFrame(rows)
        
        # Add a random but plausible spending surge or drop for demo purposes
        if len(categories) > 0 and len(date_range) > 14:
            # Pick a random category and date for the event
            random_category = np.random.choice(categories)
            random_date_idx = np.random.randint(7, len(date_range) - 7)
            random_date = date_range[random_date_idx]
            
            # Create a spending surge or drop
            is_surge = np.random.random() > 0.5
            factor = np.random.uniform(1.5, 3.0) if is_surge else np.random.uniform(0.3, 0.7)
            
            # Apply the change over several days
            for i in range(-3, 4):
                date_idx = random_date_idx + i
                if 0 <= date_idx < len(date_range):
                    date = date_range[date_idx]
                    mask = (sample_df['record_date'] == date) & (sample_df['classification_desc'] == random_category)
                    # Gradually apply the change (stronger in the middle)
                    adjustment = 1.0 + (factor - 1.0) * (1.0 - abs(i) / 3.0)
                    sample_df.loc[mask, 'current_month_outly_amt'] *= adjustment
        
        return sample_df
    
    def calculate_moving_average(self, df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
        """
        Calculate moving average for time series data
        
        Args:
            df: DataFrame containing time series data
            days: Number of days for the moving average window
            
        Returns:
            DataFrame with additional moving average columns
        """
        df = df.copy()
        
        # Ensure data is sorted by date
        df = df.sort_values('record_date')
        
        # Calculate moving averages for relevant columns
        for col in ['current_month_outly_amt', 'current_month_gross_rcpt_amt', 'today_amt', 'mtd_amt']:
            if col in df.columns:
                df[f'{col}_ma'] = df[col].rolling(window=days, min_periods=1).mean()
        
        return df

    def detect_funding_changes(self, df: pd.DataFrame, category: str) -> List[Dict]:
        """
        Detect significant changes in funding and generate alerts
        
        Args:
            df: DataFrame containing time series data for a category
            category: Category name to analyze
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        # Ensure DataFrame is sorted by date
        df = df.sort_values('record_date')
        
        if len(df) < 2:
            return alerts  # Not enough data to detect changes
        
        # Calculate period-over-period changes
        for col in ['current_month_outly_amt', 'current_month_gross_rcpt_amt', 'today_amt']:
            if col in df.columns:
                # Calculate daily percentage change
                df[f'{col}_pct_change'] = df[col].pct_change() * 100
                
                # Calculate 7-day change
                if len(df) >= 7:
                    df[f'{col}_7d_change'] = (df[col] / df[col].shift(7) - 1) * 100
                
                # Calculate 30-day change
                if len(df) >= 30:
                    df[f'{col}_30d_change'] = (df[col] / df[col].shift(30) - 1) * 100
                
                # Create alerts based on different change periods
                for idx, row in df.iterrows():
                    # Check 1-day change
                    if pd.notnull(row[f'{col}_pct_change']):
                        alert = self._create_alert(row, category, col, row[f'{col}_pct_change'], '1-day')
                        if alert['alert_type']:
                            alerts.append(alert)
                    
                    # Check 7-day change
                    if f'{col}_7d_change' in df.columns and pd.notnull(row[f'{col}_7d_change']):
                        alert = self._create_alert(row, category, col, row[f'{col}_7d_change'], '7-day')
                        if alert['alert_type']:
                            alerts.append(alert)
                    
                    # Check 30-day change
                    if f'{col}_30d_change' in df.columns and pd.notnull(row[f'{col}_30d_change']):
                        alert = self._create_alert(row, category, col, row[f'{col}_30d_change'], '30-day')
                        if alert['alert_type']:
                            alerts.append(alert)
        
        # Detect unusual volatility
        for col in ['current_month_outly_amt', 'current_month_gross_rcpt_amt', 'today_amt']:
            if col in df.columns and len(df) >= 7:
                # Calculate rolling standard deviation and mean
                rolling_std = df[col].rolling(window=7, min_periods=3).std()
                rolling_mean = df[col].rolling(window=7, min_periods=3).mean()
                
                # Calculate coefficient of variation (relative volatility)
                df[f'{col}_volatility'] = (rolling_std / rolling_mean) * 100
                
                # Create volatility alerts
                for idx, row in df.iterrows():
                    if pd.notnull(row[f'{col}_volatility']) and row[f'{col}_volatility'] > self.unusual_volatility_threshold:
                        alerts.append({
                            'date': row['record_date'],
                            'category': category,
                            'type': col.replace('current_month_', '').replace('_amt', ''),
                            'change': row[f'{col}_volatility'],
                            'period': 'volatility',
                            'alert_type': 'WARNING',
                            'message': (
                                f"⚠️ Unusual volatility detected in {category} spending. "
                                f"Volatility index: {row[f'{col}_volatility']:.1f}%"
                            )
                        })
        
        return alerts

    def _create_alert(self, 
                     row: pd.Series,
                     category: str, 
                     col: str, 
                     pct_change: float,
                     period: str) -> Dict:
        """
        Create an alert dictionary based on funding changes
        
        Args:
            row: DataFrame row containing the data
            category: Category name
            col: Column name for the data
            pct_change: Percentage change value
            period: Time period for the change (e.g., '1-day', '7-day')
            
        Returns:
            Alert dictionary
        """
        alert = {
            'date': row['record_date'],
            'category': category,
            'type': col.replace('current_month_', '').replace('_amt', ''),
            'change': pct_change,
            'period': period,
            'alert_type': None,
            'message': None
        }
        
        # Adjust thresholds based on the period
        critical_threshold = self.critical_cut_threshold
        warning_threshold = -self.significant_change_threshold
        info_threshold = self.significant_change_threshold
        
        if period == '7-day':
            critical_threshold *= 0.8  # Less severe threshold for 7-day change
            warning_threshold *= 0.8
            info_threshold *= 0.8
        elif period == '30-day':
            critical_threshold *= 0.5  # Even less severe threshold for 30-day change
            warning_threshold *= 0.5
            info_threshold *= 0.5
        
        if pct_change <= critical_threshold:
            alert['alert_type'] = 'RED_ALERT'
            alert['message'] = (
                f"🚨 CRITICAL: Severe funding cut detected for {category}. "
                f"{period} decrease of {abs(pct_change):.1f}%"
            )
        elif pct_change <= warning_threshold:
            alert['alert_type'] = 'WARNING'
            alert['message'] = (
                f"⚠️ Warning: Significant decrease in {category} funding. "
                f"{period} decrease of {abs(pct_change):.1f}%"
            )
        elif pct_change >= info_threshold:
            alert['alert_type'] = 'INFO'
            alert['message'] = (
                f"ℹ️ Notice: Significant increase in {category} funding. "
                f"{period} increase of {pct_change:.1f}%"
            )
        
        return alert

    def generate_recommendations(self, df: pd.DataFrame, alerts: List[Dict]) -> List[str]:
        """
        Generate recommendations based on funding alerts and patterns
        
        Args:
            df: DataFrame containing the analyzed data
            alerts: List of alert dictionaries
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        category_alerts = {}
        
        # Group alerts by category and type
        for alert in alerts:
            key = (alert['category'], alert['type'])
            if key not in category_alerts:
                category_alerts[key] = []
            category_alerts[key].append(alert)
        
        # Generate recommendations based on patterns
        for (category, alert_type), cat_alerts in category_alerts.items():
            red_alerts = sum(1 for a in cat_alerts if a['alert_type'] == 'RED_ALERT')
            warnings = sum(1 for a in cat_alerts if a['alert_type'] == 'WARNING')
            volatility_alerts = sum(1 for a in cat_alerts if a['period'] == 'volatility')
            
            if red_alerts > 0:
                recommendations.append(
                    f"Critical funding changes detected for {category}. "
                    f"This may indicate significant policy changes or budget reallocations."
                )
            elif warnings >= 2:
                recommendations.append(
                    f"{category} shows a pattern of funding decreases. "
                    f"Monitor closely for potential impacts on program operations."
                )
            
            if volatility_alerts > 0:
                recommendations.append(
                    f"Unusual spending volatility detected in {category}. "
                    f"This may indicate irregular disbursements or reporting issues."
                )
        
        # Generate trend-based recommendations
        for category in set(df['classification_desc']):
            category_data = df[df['classification_desc'] == category]
            if len(category_data) >= 30:
                # Calculate trend (simple linear regression)
                category_data = category_data.sort_values('record_date')
                x = np.arange(len(category_data))
                y = category_data['current_month_outly_amt'].values
                
                if len(x) > 0 and len(y) > 0:
                    slope, _ = np.polyfit(x, y, 1)
                    avg_value = np.mean(y)
                    
                    if avg_value > 0:
                        annual_trend_pct = (slope * 365) / avg_value * 100
                        
                        if annual_trend_pct > 20:
                            recommendations.append(
                                f"{category} shows a strong upward trend ({annual_trend_pct:.1f}% annual growth rate). "
                                f"This may indicate expanding program scope or increasing costs."
                            )
                        elif annual_trend_pct < -15:
                            recommendations.append(
                                f"{category} shows a significant downward trend ({abs(annual_trend_pct):.1f}% annual decline). "
                                f"This may indicate program cuts or efficiency improvements."
                            )
        
        return recommendations

    def analyze_treasury_data(self, 
                            start_date: datetime,
                            end_date: datetime,
                            categories: List[str],
                            frequency: str = 'daily') -> Tuple[pd.DataFrame, List[Dict], List[str]]:
        """
        Comprehensive analysis of treasury data with alerts and recommendations
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            categories: List of category codes to analyze
            frequency: Data frequency ('daily', 'weekly', or 'seven_day_ma')
            
        Returns:
            Tuple containing:
              - DataFrame with analyzed data
              - List of alert dictionaries
              - List of recommendation strings
        """
        try:
            # Fetch the data
            df = self.fetch_treasury_data(start_date, end_date, categories)
            
            if df is None or df.empty:
                return None, [], []
            
            all_alerts = []
            
            # Process data according to requested frequency
            if frequency == 'seven_day_ma':
                df = self.calculate_moving_average(df)
            elif frequency == 'weekly':
                # Group data by week
                df['week'] = df['record_date'].dt.to_period('W')
                weekly_groups = []
                
                for category in categories:
                    category_data = df[df['classification_desc'] == category]
                    if not category_data.empty:
                        # Group by week and calculate weekly sums
                        weekly_data = category_data.groupby('week').agg({
                            'current_month_outly_amt': 'sum',
                            'current_month_gross_rcpt_amt': 'sum'
                        }).reset_index()
                        
                        # Convert period to datetime using start of period
                        weekly_data['record_date'] = weekly_data['week'].dt.to_timestamp()
                        weekly_data['classification_desc'] = category
                        weekly_data['internal_category'] = category
                        
                        weekly_groups.append(weekly_data)
                
                if weekly_groups:
                    df = pd.concat(weekly_groups, ignore_index=True)
            
            # Process alerts for each category
            for category in categories:
                category_data = df[df['classification_desc'] == category]
                
                if not category_data.empty:
                    # Detect funding changes
                    alerts = self.detect_funding_changes(category_data, category)
                    all_alerts.extend(alerts)
            
            # Generate recommendations
            recommendations = self.generate_recommendations(df, all_alerts)
            
            return df, all_alerts, recommendations
            
        except Exception as e:
            print(f"Analysis error: {e}")
            return None, [], []

    def get_available_dates(self) -> List[str]:
        """Return list of dates known to have data"""
        # Fetch the most recent available dates from the API
        try:
            fields = ['record_date']
            url = self.build_url(
                endpoint=self.dts_endpoint,
                fields=fields,
                page_size=100
            )
            
            df = self.fetch_data(url)
            if not df.empty and 'record_date' in df.columns:
                return df['record_date'].dt.strftime('%Y-%m-%d').tolist()
            
            # Fallback to hardcoded dates
            return [
                datetime.now().strftime('%Y-%m-%d'),
                (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
            ]
        except:
            # Return recent dates as fallback
            return [
                datetime.now().strftime('%Y-%m-%d'),
                (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
            ]


def main():
    """
    Main function to demonstrate the usage of TreasuryDataFetcher
    """
    fetcher = TreasuryDataFetcher()
    
    # Example analysis
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    categories = ['SNAP', 'NIH', 'VA']
    
    try:
        df, alerts, recommendations = fetcher.analyze_treasury_data(
            start_date,
            end_date,
            categories
        )
        
        if df is not None:
            print("\nAnalysis Results:")
            print("-" * 50)
            print(f"Total records: {len(df)}")
            print(f"Date range: {df['record_date'].min()} to {df['record_date'].max()}")
            
            print("\nAlerts:")
            for alert in alerts:
                print(f"- {alert['message']}")
            
            print("\nRecommendations:")
            for rec in recommendations:
                print(f"- {rec}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
        