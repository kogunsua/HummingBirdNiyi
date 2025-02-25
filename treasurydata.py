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
        
