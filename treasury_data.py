#treasurydata.py
# Treasury Data Fetcher Application
# This application fetches and analyzes data from the U.S. Treasury's Monthly Treasury Statement (MTS) Table 1 API
# The API provides detailed information about government receipts, outlays, and deficit/surplus

# Import required libraries
import requests          # For making HTTP requests to the API
import pandas as pd      # For data manipulation and analysis
from datetime import datetime  # For handling dates
import matplotlib.pyplot as plt  # For creating visualizations
from typing import Dict, List, Optional  # For type hints
import json  # For handling JSON data

class TreasuryDataFetcher:
    """
    A class to handle fetching and processing data from the U.S. Treasury's MTS API
    
    This class provides methods to:
    - Build API URLs with various parameters
    - Fetch data from the API
    - Process and clean the retrieved data
    - Create visualizations of the data
    """
    
    def __init__(self):
        """
        Initialize the TreasuryDataFetcher with base API endpoints
        The base URL and endpoint are set as instance variables for flexibility
        """
        self.base_url = 'https://api.fiscaldata.treasury.gov/services/api/fiscal_service'
        self.endpoint = '/v1/accounting/mts/mts_table_1'
        
    def build_url(self, 
                  fields: List[str],
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None,
                  page_size: int = 100,
                  page_number: int = 1) -> str:
        """
        Builds a complete API URL with the specified parameters
        
        Parameters:
        -----------
        fields : List[str]
            List of field names to retrieve from the API
        start_date : Optional[str]
            Start date for filtering data (format: YYYY-MM-DD)
        end_date : Optional[str]
            End date for filtering data (format: YYYY-MM-DD)
        page_size : int
            Number of records to return per page (default: 100)
        page_number : int
            Page number to retrieve (default: 1)
            
        Returns:
        --------
        str
            Complete API URL with all parameters
        """
        # Join the requested fields with commas
        fields_str = ','.join(fields)
        url = f"{self.base_url}{self.endpoint}?fields={fields_str}"
        
        # Add date range filters if provided
        if start_date:
            url += f"&filter=record_date:gte:{start_date}"
        if end_date:
            url += f"&filter=record_date:lte:{end_date}"
            
        # Add sorting parameter (newest dates first)
        url += "&sort=-record_date"
        
        # Add pagination parameters
        url += f"&page[number]={page_number}&page[size]={page_size}"
        
        return url
    
    def fetch_data(self, url: str) -> pd.DataFrame:
        """
        Fetches data from the API and converts it to a pandas DataFrame
        
        Parameters:
        -----------
        url : str
            Complete API URL to fetch data from
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing the processed API response data
        
        Raises:
        -------
        requests.exceptions.RequestException
            If the API request fails
        ValueError
            If the API response doesn't contain data
        """
        try:
            # Make the API request
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for bad status codes
            data = response.json()
            
            # Check if the response contains data
            if 'data' not in data:
                raise ValueError("No data found in API response")
                
            # Convert the JSON data to a DataFrame
            df = pd.DataFrame(data['data'])
            
            # Convert date strings to datetime objects
            df['record_date'] = pd.to_datetime(df['record_date'])
            
            # Convert numeric columns to float type
            numeric_columns = ['current_month_gross_rcpt_amt']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            raise
            
    def plot_receipts_by_classification(self, df: pd.DataFrame):
        """
        Creates a bar plot showing gross receipts by classification
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the MTS data to visualize
            
        Returns:
        --------
        matplotlib.pyplot
            Plot object that can be displayed or saved
        """
        # Create a new figure with specified size
        plt.figure(figsize=(12, 6))
        
        # Group the data by classification and sum the receipts
        plot_data = df.groupby('classification_desc')['current_month_gross_rcpt_amt'].sum()
        
        # Create the bar plot
        plot_data.plot(kind='bar')
        
        # Customize the plot
        plt.title('Gross Receipts by Classification')
        plt.xlabel('Classification')
        plt.ylabel('Gross Receipts (USD)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return plt

def main():
    """
    Main function to demonstrate the usage of TreasuryDataFetcher
    
    This function:
    1. Initializes the TreasuryDataFetcher
    2. Fetches recent Treasury data
    3. Displays summary statistics
    4. Creates and saves a visualization
    """
    # Create an instance of the TreasuryDataFetcher
    fetcher = TreasuryDataFetcher()
    
    # Define the fields we want to retrieve from the API
    fields = [
        'record_date',
        'classification_desc',
        'current_month_gross_rcpt_amt'
    ]
    
    # Get the current date for the end_date parameter
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        # Build the API URL and fetch the data
        url = fetcher.build_url(
            fields=fields,
            end_date=end_date,
            page_size=100
        )
        
        # Fetch and process the data
        df = fetcher.fetch_data(url)
        
        # Display summary statistics
        print("\nData Summary:")
        print("-" * 50)
        print(f"Total records: {len(df)}")
        print(f"Date range: {df['record_date'].min()} to {df['record_date'].max()}")
        print("\nTotal receipts by classification:")
        print(df.groupby('classification_desc')['current_month_gross_rcpt_amt'].sum())
        
        # Create and save the visualization
        plt = fetcher.plot_receipts_by_classification(df)
        plt.savefig('treasury_receipts.png')
        print("\nPlot saved as 'treasury_receipts.png'")
        
    except Exception as e:
        print(f"An error occurred: {e}")

# Standard boilerplate to only run the main function if this file is executed directly
if __name__ == "__main__":
    main()