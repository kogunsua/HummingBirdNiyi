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
            'account_desc',
            'current_month_outly_amt',
            'current_month_gross_rcpt_amt'
        ]
        
        # Build filters for date range
        date_filter = f"record_date:gte:{start_date_str},record_date:lte:{end_date_str}"
        
        # Initialize an empty list to store DataFrames
        dataframes = []
        
        # Try to fetch data from multiple Treasury endpoints
        for endpoint in [self.mts_table1_endpoint, self.mts_table3_endpoint, self.dts_endpoint]:
            try:
                url = self.build_url(
                    endpoint=endpoint,
                    fields=common_fields,
                    filters=[date_filter],
                    page_size=1000  # Fetch larger batches
                )
                
                df = self.fetch_data(url)
                
                # If we got data, process it
                if not df.empty:
                    # Map the classification descriptions to our category codes
                    df['internal_category'] = df['classification_desc'].apply(
                        lambda x: next((code for key, code in self.api_category_mapping.items() 
                                        if key in str(x).upper()), None)
                    )
                    
                    # Filter to only include categories we're interested in
                    filtered_df = df[df['internal_category'].isin(categories)]
                    
                    if not filtered_df.empty:
                        dataframes.append(filtered_df)
            
            except Exception as e:
                print(f"Error fetching data from {endpoint}: {e}")
                continue
        
        # Combine all dataframes
        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            
            # Sort by date descending
            combined_df = combined_df.sort_values('record_date', ascending=False)
            
            # Remove duplicates if any
            combined_df = combined_df.drop_duplicates(['record_date', 'classification_desc'])
            
            return combined_df
        else:
            # Return empty DataFrame if no data was found
            return pd.DataFrame()
    
    def get_sample_data(self, categories: List[str]) -> pd.DataFrame:
        """
        Generate sample data for demonstration purposes when API data is unavailable
        
        Args:
            categories: List of category codes to generate sample data for
            
        Returns:
            DataFrame with sample data
        """
        # Generate sample data for the past 90 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create a base for each category
        all_data = []
        
        for category in categories:
            # Only include categories we have in our mapping
            if category in self.program_categories:
                # Start with a base value that varies by category
                base_value = np.random.uniform(1e6, 1e9)  # Between 1M and 1B dollars
                
                # Generate daily values with some randomness
                daily_data = []
                for date in date_range:
                    # Add seasonal patterns
                    seasonal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * date.day_of_year / 365)
                    # Add weekly patterns
                    weekly_factor = 1.0 + 0.05 * np.sin(2 * np.pi * date.dayofweek / 7)
                    # Add some random noise
                    noise = np.random.normal(0, 0.03)
                    
                    # Calculate the daily value
                    value = base_value * seasonal_factor * weekly_factor * (1 + noise)
                    
                    daily_data.append({
                        'record_date': date,
                        'classification_desc': category,
                        'internal_category': category,
                        'current_month_outly_amt': value,
                        'current_month_gross_rcpt_amt': value * np.random.uniform(0.8, 1.2),
                        'security_desc': f"Sample data for {category}",
                        'account_desc': self.program_categories.get(category, "Unknown category")
                    })
                
                all_data.extend(daily_data)
        
        # Convert to DataFrame
        if all_data:
            return pd.DataFrame(all_data)
        else:
            return pd.DataFrame()

    def calculate_moving_average(self, df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
        """
        Calculate moving average for numeric columns in the dataframe
        
        Args:
            df: DataFrame containing the data
            window: Window size for the moving average (default: 7 days)
            
        Returns:
            DataFrame with additional moving average columns
        """
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Identify numeric columns (potential candidates for moving average)
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Columns to calculate moving average for
        ma_columns = [
            'current_month_outly_amt',
            'current_month_gross_rcpt_amt',
            'current_month_deficit_surplus_amt',
            'amt',
            'today_amt',
            'mtd_amt',
            'fytd_amt'
        ]
        
        # Filter out columns that don't exist in the dataframe
        ma_columns = [col for col in ma_columns if col in numeric_columns]
        
        # Calculate moving average for each category separately
        for category in df['classification_desc'].unique():
            category_mask = df['classification_desc'] == category
            category_data = df[category_mask].copy()
            
            # Sort by date
            category_data = category_data.sort_values('record_date')
            
            # Calculate moving average for each numeric column
            for col in ma_columns:
                if col in category_data.columns:
                    # Calculate moving average
                    ma_values = category_data[col].rolling(window=window, min_periods=1).mean()
                    
                    # Create new column name
                    ma_col = f"{col}_ma"
                    
                    # Store the MA values in the original dataframe
                    result_df.loc[category_mask, ma_col] = ma_values.values
        
        return result_df
    
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

            # If no data returned, try using sample data
            if df is None or df.empty:
                print("No data returned from API, using sample data...")
                df = self.get_sample_data(categories)
                
                # If still no data, return empty results
                if df is None or df.empty:
                    return None, [], []

            all_alerts = []

            # Process data according to requested frequency
            if frequency == 'seven_day_ma':
                df = self.calculate_moving_average(df)
            elif frequency == 'weekly':
                # Group data by week using string-based grouping to avoid Period objects
                df['week'] = df['record_date'].dt.strftime('%Y-%U')
                weekly_groups = []

                for category in categories:
                    category_data = df[df['classification_desc'] == category]
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

    def detect_funding_changes(self, df: pd.DataFrame, category: str) -> List[Dict]:
        """
        Detect significant changes in funding amounts
        
        Args:
            df: DataFrame containing data for a specific category
            category: Category name
            
        Returns:
            List of alert dictionaries with details about detected changes
        """
        if df.empty or len(df) < 2:
            return []
        
        # Sort by date
        df = df.sort_values('record_date')
        
        # Column to analyze
        amount_col = 'current_month_outly_amt'
        
        # Make sure the column exists
        if amount_col not in df.columns:
            return []
        
        alerts = []
        
        # Calculate rolling statistics to detect changes
        df['rolling_avg'] = df[amount_col].rolling(window=7, min_periods=1).mean()
        df['rolling_std'] = df[amount_col].rolling(window=7, min_periods=1).std()
        df['z_score'] = (df[amount_col] - df['rolling_avg']) / df['rolling_std'].replace(0, 1)
        df['pct_change'] = df[amount_col].pct_change() * 100
        
        # Detect large changes
        for idx, row in df.iterrows():
            if idx == 0:  # Skip first row as it has no previous
                continue
                
            # Get human-readable category name
            category_name = self.program_categories.get(category, category)
            
            # 1. Check for large percentage changes
            if abs(row['pct_change']) > self.significant_change_threshold:
                change_type = "increase" if row['pct_change'] > 0 else "decrease"
                alerts.append({
                    'alert_type': 'WARNING',
                    'date': row['record_date'],
                    'message': f"Significant {change_type} of {abs(row['pct_change']):.1f}% in {category_name} spending detected on {row['record_date'].strftime('%Y-%m-%d')}.",
                    'category': category,
                    'value': row[amount_col],
                    'change': row['pct_change']
                })
                
            # 2. Check for critical funding cuts
            if row['pct_change'] < self.critical_cut_threshold:
                alerts.append({
                    'alert_type': 'RED_ALERT',
                    'date': row['record_date'],
                    'message': f"Critical funding cut of {abs(row['pct_change']):.1f}% in {category_name} detected on {row['record_date'].strftime('%Y-%m-%d')}.",
                    'category': category,
                    'value': row[amount_col],
                    'change': row['pct_change']
                })
                
            # 3. Check for outliers using Z-score
            if abs(row['z_score']) > 3.0:  # 3 standard deviations
                change_type = "higher than" if row['z_score'] > 0 else "lower than"
                alerts.append({
                    'alert_type': 'INFO',
                    'date': row['record_date'],
                    'message': f"Unusual {category_name} spending detected on {row['record_date'].strftime('%Y-%m-%d')} - {abs(row['z_score']):.1f} standard deviations {change_type} normal.",
                    'category': category,
                    'value': row[amount_col],
                    'change': row['z_score']
                })
        
        # 4. Check overall volatility
        if not df['pct_change'].dropna().empty:
            volatility = df['pct_change'].dropna().std()
            if volatility > self.unusual_volatility_threshold:
                alerts.append({
                    'alert_type': 'WARNING',
                    'date': df['record_date'].iloc[-1],
                    'message': f"High volatility detected in {category_name} spending (volatility index: {volatility:.1f}%).",
                    'category': category,
                    'value': None,
                    'change': volatility
                })
        
        return alerts
    
    def generate_recommendations(self, df: pd.DataFrame, alerts: List[Dict]) -> List[str]:
        """
        Generate recommendations based on the data and alerts
        
        Args:
            df: DataFrame containing the analyzed data
            alerts: List of alert dictionaries
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # 1. Check if we have enough data for meaningful recommendations
        if df.empty or len(df) < 7:
            recommendations.append(
                "Insufficient data for detailed analysis. Consider expanding the time range or selecting different categories."
            )
            return recommendations
        
        # 2. Process alerts for recommendations
        critical_alerts = [a for a in alerts if a['alert_type'] == 'RED_ALERT']
        warnings = [a for a in alerts if a['alert_type'] == 'WARNING']
        
        if critical_alerts:
            # Group critical alerts by category
            categories = set([a['category'] for a in critical_alerts])
            for category in categories:
                category_name = self.program_categories.get(category, category)
                recommendations.append(
                    f"Monitor {category_name} closely due to critical funding changes. This may indicate policy shifts or budget reallocations."
                )
        
        if warnings:
            # Count warnings by category
            category_counts = {}
            for alert in warnings:
                category = alert['category']
                if category in category_counts:
                    category_counts[category] += 1
                else:
                    category_counts[category] = 1
            
            # Recommend categories with multiple warnings
            for category, count in category_counts.items():
                if count >= 2:
                    category_name = self.program_categories.get(category, category)
                    recommendations.append(
                        f"Review {category_name} spending patterns due to multiple anomalies detected. This may indicate irregular disbursement patterns."
                    )
        
        # 3. Check for long-term trends
        categories = df['classification_desc'].unique()
        for category in categories:
            category_data = df[df['classification_desc'] == category].copy()
            
            # Need at least 14 days for trend analysis
            if len(category_data) >= 14:
                category_data = category_data.sort_values('record_date')
                
                # Simple linear regression to detect trend
                try:
                    import numpy as np
                    from scipy import stats
                    
                    x = range(len(category_data))
                    y = category_data['current_month_outly_amt'].values
                    
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    
                    # Calculate percent change over the period
                    total_change = (slope * len(x)) / intercept * 100 if intercept != 0 else 0
                    
                    # Check if trend is statistically significant
                    if p_value < 0.05 and abs(total_change) > 5:
                        category_name = self.program_categories.get(category, category)
                        direction = "increasing" if slope > 0 else "decreasing"
                        recommendations.append(
                            f"{category_name} shows a statistically significant {direction} trend of {abs(total_change):.1f}% over the period. This indicates a sustained policy direction."
                        )
                except:
                    # If regression fails, skip this category
                    pass
        
        # 4. Add general recommendations if we have few specific ones
        if len(recommendations) < 2:
            recommendations.append(
                "Consider analyzing a longer time period to identify meaningful patterns in federal spending."
            )
            recommendations.append(
                "Compare multiple related categories to gain better context for spending patterns."
            )
        
        return recommendations

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
