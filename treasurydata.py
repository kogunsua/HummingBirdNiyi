#treasurydata.py
# treasurydata.py
# Treasury Data Fetcher Application
# This application fetches and analyzes data from the U.S. Treasury's Monthly Treasury Statement (MTS) Table 1 API
# The API provides detailed information about government receipts, outlays, and deficit/surplus

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import json
from enum import Enum

class TreasuryDataFetcher:
    """
    A class to handle fetching and processing data from the U.S. Treasury's MTS API
    
    This class provides methods to:
    - Build API URLs with various parameters
    - Fetch data from the API
    - Process and clean the retrieved data
    - Create visualizations of the data
    - Monitor funding changes and generate alerts
    """
    
    def __init__(self):
        """
        Initialize the TreasuryDataFetcher with base API endpoints and configurations
        """
        self.base_url = 'https://api.fiscaldata.treasury.gov/services/api/fiscal_service'
        self.endpoint = '/v1/accounting/mts/mts_table_1'
        self.outlays_endpoint = '/v1/accounting/mts/mts_table_3'
        
        # Define thresholds for funding changes
        self.significant_change_threshold = 15.0
        self.critical_cut_threshold = -30.0
        
        # Define program categories
        self.program_categories = {
            'SNAP': 'Supplemental Nutrition Assistance Program',
            'NIH': 'National Institutes of Health',
            'SSA': 'Social Security Administration',
            'VA': 'Veterans Affairs',
            'ED': 'Department of Education',
            'HUD': 'Housing and Urban Development',
            'DOD': 'Department of Defense',
            'HHS': 'Health and Human Services',
            'DHS': 'Department of Homeland Security'
        }
        
    def build_url(self, 
                  fields: List[str],
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None,
                  page_size: int = 100,
                  page_number: int = 1,
                  outlays: bool = False) -> str:
        """
        Builds a complete API URL with the specified parameters
        """
        fields_str = ','.join(fields)
        endpoint = self.outlays_endpoint if outlays else self.endpoint
        url = f"{self.base_url}{endpoint}?fields={fields_str}"
        
        if start_date:
            url += f"&filter=record_date:gte:{start_date}"
        if end_date:
            url += f"&filter=record_date:lte:{end_date}"
            
        url += "&sort=-record_date"
        url += f"&page[number]={page_number}&page[size]={page_size}"
        
        return url
    
    def fetch_data(self, url: str) -> pd.DataFrame:
        """
        Fetches data from the API and converts it to a pandas DataFrame
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if 'data' not in data:
                raise ValueError("No data found in API response")
                
            df = pd.DataFrame(data['data'])
            df['record_date'] = pd.to_datetime(df['record_date'])
            
            numeric_columns = [
                'current_month_gross_rcpt_amt',
                'current_month_outly_amt'
            ]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            raise

    def calculate_moving_average(self, df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
        """
        Calculate moving average for time series data
        """
        df = df.copy()
        for col in ['current_month_outly_amt', 'current_month_gross_rcpt_amt']:
            if col in df.columns:
                df[f'{col}_ma'] = df[col].rolling(window=days).mean()
        return df

    def detect_funding_changes(self, df: pd.DataFrame, category: str) -> List[Dict]:
        """
        Detect significant changes in funding and generate alerts
        """
        alerts = []
        df = df.sort_values('record_date')
        
        for col in ['current_month_outly_amt', 'current_month_gross_rcpt_amt']:
            if col in df.columns:
                df[f'{col}_pct_change'] = df[col].pct_change() * 100
                
                for idx, row in df.iterrows():
                    pct_change = row[f'{col}_pct_change']
                    if pd.notnull(pct_change):
                        alert = self._create_alert(row, category, col, pct_change)
                        if alert['alert_type']:
                            alerts.append(alert)
        
        return alerts

    def _create_alert(self, row: pd.Series, category: str, col: str, pct_change: float) -> Dict:
        """
        Create an alert dictionary based on funding changes
        """
        alert = {
            'date': row['record_date'],
            'category': category,
            'type': col.replace('current_month_', '').replace('_amt', ''),
            'change': pct_change,
            'alert_type': None,
            'message': None
        }
        
        if pct_change <= self.critical_cut_threshold:
            alert['alert_type'] = 'RED_ALERT'
            alert['message'] = (
                f"🚨 CRITICAL: Severe funding cut detected for {category}. "
                f"Decrease of {abs(pct_change):.1f}%"
            )
        elif pct_change <= -self.significant_change_threshold:
            alert['alert_type'] = 'WARNING'
            alert['message'] = (
                f"⚠️ Warning: Significant decrease in {category} funding. "
                f"Decrease of {abs(pct_change):.1f}%"
            )
        elif pct_change >= self.significant_change_threshold:
            alert['alert_type'] = 'INFO'
            alert['message'] = (
                f"ℹ️ Notice: Significant increase in {category} funding. "
                f"Increase of {pct_change:.1f}%"
            )
        
        return alert

    def generate_recommendations(self, alerts: List[Dict]) -> List[str]:
        """
        Generate recommendations based on funding alerts
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
            
            if red_alerts > 0:
                recommendations.append(
                    f"🚨 URGENT: {category} has experienced critical {alert_type} cuts. "
                    "Immediate review and contingency planning recommended."
                )
            elif warnings >= 2:
                recommendations.append(
                    f"⚠️ ATTENTION: {category} shows a pattern of {alert_type} decreases. "
                    "Consider strategic planning to address potential impacts."
                )
        
        return recommendations

    def plot_treasury_visualization(self, 
                                  df: pd.DataFrame,
                                  frequency: str = 'daily',
                                  categories: Optional[List[str]] = None,
                                  plot_type: str = 'outlays') -> plt.Figure:
        """
        Create an enhanced visualization of treasury data with alerts
        """
        fig, ax = plt.subplots(figsize=(15, 8))
        
        amount_column = (
            'current_month_outly_amt' 
            if plot_type == 'outlays' 
            else 'current_month_gross_rcpt_amt'
        )
        
        categories = categories or df['classification_desc'].unique()
        
        for category in categories:
            category_data = df[df['classification_desc'] == category].copy()
            
            if frequency == 'seven_day_ma':
                category_data = self.calculate_moving_average(category_data)
                plot_data = category_data.set_index('record_date')[f'{amount_column}_ma']
                label = f"{category} (7-day MA)"
            elif frequency == 'weekly':
                category_data['week'] = category_data['record_date'].dt.to_period('W')
                plot_data = category_data.groupby('week')[amount_column].sum()
                label = f"{category} (Weekly)"
            else:  # daily
                plot_data = category_data.set_index('record_date')[amount_column]
                label = f"{category} (Daily)"
            
            # Convert to billions
            plot_data = plot_data / 1e9
            
            # Plot the data
            ax.plot(plot_data.index, plot_data.values, label=label, marker='o', markersize=4)
            
            # Add alert indicators
            alerts = self.detect_funding_changes(category_data, category)
            for alert in alerts:
                if alert['alert_type'] == 'RED_ALERT':
                    ax.axvline(x=alert['date'], color='red', linestyle='--', alpha=0.3)
        
        title_type = "Outlays" if plot_type == 'outlays' else "Receipts"
        ax.set_title(f'Treasury {title_type} by Category')
        ax.set_xlabel('Date')
        ax.set_ylabel(f'{title_type} (Billions USD)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def analyze_treasury_data(self, 
                            start_date: datetime,
                            end_date: datetime,
                            categories: Optional[List[str]] = None,
                            frequency: str = 'daily') -> Tuple[pd.DataFrame, List[Dict], List[str]]:
        """
        Comprehensive analysis of treasury data with alerts and recommendations
        """
        try:
            fields = [
                'record_date',
                'classification_desc',
                'current_month_gross_rcpt_amt',
                'current_month_outly_amt'
            ]
            
            url = self.build_url(
                fields=fields,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                outlays=True
            )
            
            df = self.fetch_data(url)
            
            if df is not None:
                categories = categories or list(self.program_categories.keys())[:9]
                all_alerts = []
                
                for category in categories:
                    category_data = df[df['classification_desc'] == category]
                    alerts = self.detect_funding_changes(category_data, category)
                    all_alerts.extend(alerts)
                
                recommendations = self.generate_recommendations(all_alerts)
                
                return df, all_alerts, recommendations
            
            return None, [], []
            
        except Exception as e:
            print(f"Analysis error: {e}")
            return None, [], []

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
            
            # Create and save visualizations
            fig = fetcher.plot_treasury_visualization(df, categories=categories)
            fig.savefig('treasury_analysis.png', bbox_inches='tight')
            print("\nAnalysis plot saved as 'treasury_analysis.png'")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
