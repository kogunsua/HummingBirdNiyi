
# treasurydata.py
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import json
from enum import Enum
import streamlit as st

class TimeRange(Enum):
    ONE_MONTH = 1
    THREE_MONTHS = 3
    TWELVE_MONTHS = 12
    THIRTY_SIX_MONTHS = 36

class FrequencyType(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    SEVEN_DAY_MA = "7day_ma"

class TreasuryDataFetcher:
    """Enhanced TreasuryDataFetcher with integration for Streamlit and advanced analysis features"""
    
    def __init__(self):
        self.base_url = 'https://api.fiscaldata.treasury.gov/services/api/fiscal_service'
        self.endpoint = '/v1/accounting/mts/mts_table_1'
        self.outlays_endpoint = '/v1/accounting/mts/mts_table_3'
        
        # Define thresholds for funding changes
        self.significant_change_threshold = 15.0
        self.critical_cut_threshold = -30.0
        
        # Define known program categories
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
        """Build API URL with parameters"""
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
        """Fetch and process data from the API"""
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            if 'data' not in data:
                raise ValueError("No data found in API response")
            
            df = pd.DataFrame(data['data'])
            df['record_date'] = pd.to_datetime(df['record_date'])
            
            numeric_columns = ['current_month_gross_rcpt_amt', 'current_month_outly_amt']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching data: {e}")
            raise

    def calculate_moving_average(self, df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
        """Calculate moving average for the data"""
        df = df.copy()
        if 'current_month_outly_amt' in df.columns:
            df['moving_average'] = df['current_month_outly_amt'].rolling(window=days).mean()
        if 'current_month_gross_rcpt_amt' in df.columns:
            df['receipts_ma'] = df['current_month_gross_rcpt_amt'].rolling(window=days).mean()
        return df

    def detect_funding_changes(self, df: pd.DataFrame, category: str) -> List[Dict]:
        """Detect significant changes in funding and generate alerts"""
        alerts = []
        df = df.sort_values('record_date')
        
        for col in ['current_month_outly_amt', 'current_month_gross_rcpt_amt']:
            if col in df.columns:
                df[f'{col}_pct_change'] = df[col].pct_change() * 100
                
                for idx, row in df.iterrows():
                    pct_change = row[f'{col}_pct_change']
                    if pd.notnull(pct_change):
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
                            alert['message'] = f"CRITICAL: Severe funding cut detected for {category}"
                        elif pct_change <= -self.significant_change_threshold:
                            alert['alert_type'] = 'WARNING'
                            alert['message'] = f"Significant decrease in {category} funding"
                        elif pct_change >= self.significant_change_threshold:
                            alert['alert_type'] = 'INFO'
                            alert['message'] = f"Significant increase in {category} funding"
                        
                        if alert['alert_type']:
                            alerts.append(alert)
        
        return alerts

    def generate_recommendations(self, alerts: List[Dict]) -> List[str]:
        """Generate recommendations based on funding alerts"""
        recommendations = []
        category_alerts = {}
        
        for alert in alerts:
            key = (alert['category'], alert['type'])
            if key not in category_alerts:
                category_alerts[key] = []
            category_alerts[key].append(alert)
        
        for (category, alert_type), cat_alerts in category_alerts.items():
            red_alerts = sum(1 for a in cat_alerts if a['alert_type'] == 'RED_ALERT')
            warnings = sum(1 for a in cat_alerts if a['alert_type'] == 'WARNING')
            
            if red_alerts > 0:
                recommendations.append(
                    f"🚨 URGENT: {category} has experienced critical {alert_type} changes. "
                    "Immediate review recommended."
                )
            elif warnings >= 2:
                recommendations.append(
                    f"⚠️ ATTENTION: {category} shows a pattern of {alert_type} decreases. "
                    "Consider monitoring closely."
                )
        
        return recommendations

    def plot_treasury_data(self, 
                          df: pd.DataFrame, 
                          frequency: FrequencyType,
                          categories: List[str],
                          time_range: TimeRange,
                          plot_type: str = 'outlays'):
        """Create an enhanced visualization of treasury data"""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        amount_column = 'current_month_outly_amt' if plot_type == 'outlays' else 'current_month_gross_rcpt_amt'
        
        for category in categories:
            category_data = df[df['classification_desc'] == category].copy()
            
            if frequency == FrequencyType.SEVEN_DAY_MA:
                category_data = self.calculate_moving_average(category_data)
                plot_data = category_data.set_index('record_date')[
                    'moving_average' if plot_type == 'outlays' else 'receipts_ma'
                ]
                label = f"{category} (7-day MA)"
            elif frequency == FrequencyType.WEEKLY:
                category_data['week'] = category_data['record_date'].dt.to_period('W')
                plot_data = category_data.groupby('week')[amount_column].sum()
                label = f"{category} (Weekly)"
            else:
                plot_data = category_data.set_index('record_date')[amount_column]
                label = f"{category} (Daily)"
            
            # Convert to billions
            plot_data = plot_data / 1e9
            
            ax.plot(plot_data.index, plot_data.values, label=label, marker='o', markersize=4)
            
            # Add alert indicators
            alerts = self.detect_funding_changes(category_data, category)
            for alert in alerts:
                if alert['alert_type'] == 'RED_ALERT':
                    ax.axvline(x=alert['date'], color='red', linestyle='--', alpha=0.3)
        
        title_type = "Outlays" if plot_type == 'outlays' else "Receipts"
        ax.set_title(f'Treasury {title_type} by Category ({time_range.value} months)')
        ax.set_xlabel('Date')
        ax.set_ylabel(f'{title_type} (Billions USD)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def streamlit_analysis(self, start_date: datetime, end_date: datetime, 
                          frequency: FrequencyType, categories: List[str]):
        """Perform analysis for Streamlit interface"""
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
                # Calculate time range
                days_diff = (end_date - start_date).days
                time_range = TimeRange.ONE_MONTH
                if days_diff > 90:
                    time_range = TimeRange.THREE_MONTHS
                elif days_diff > 365:
                    time_range = TimeRange.TWELVE_MONTHS
                
                # Generate alerts and recommendations
                all_alerts = []
                for category in categories:
                    category_data = df[df['classification_desc'] == category]
                    alerts = self.detect_funding_changes(category_data, category)
                    all_alerts.extend(alerts)
                
                recommendations = self.generate_recommendations(all_alerts)
                
                return df, all_alerts, recommendations, time_range
            
            return None, [], [], TimeRange.ONE_MONTH
            
        except Exception as e:
            st.error(f"Analysis error: {e}")
            return None, [], [], TimeRange.ONE_MONTH

def display_treasury_section():
    """Display the treasury analysis section in Streamlit"""
    st.title("Treasury Statement Analysis")
    
    # Create input columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        end_date = st.date_input(
            "End Date",
            datetime.now().date(),
            max_value=datetime.now().date()
        )
    
    with col2:
        days_back = st.slider(
            "Analysis Period (days)",
            30, 365, 90,
            help="Select the number of days to analyze"
        )
    
    with col3:
        frequency = st.selectbox(
            "Data Frequency",
            [freq.value for freq in FrequencyType],
            format_func=lambda x: x.title()
        )
    
    start_date = end_date - timedelta(days=days_back)
    
    # Category selection
    categories = st.multiselect(
        "Select Categories to Analyze (max 9)",
        list(TreasuryDataFetcher().program_categories.keys()),
        default=['SNAP', 'NIH', 'VA'],
        max_selections=9
    )
    
    return start_date, end_date, FrequencyType(frequency), categories

def main():
    """Main function for standalone testing"""
    fetcher = TreasuryDataFetcher()
    
    # Example usage
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    frequency = FrequencyType.DAILY
    categories = ['SNAP', 'NIH', 'VA']
    
    df, alerts, recommendations, time_range = fetcher.streamlit_analysis(
        start_date, 
        end_date, 
        frequency, 
        categories
    )
    
    if df is not None:
        print("\nAnalysis Results:")
        print("-" * 50)
        print(f"Total records: {len(df)}")
        print("\nAlerts:")
        for alert in alerts:
            print(f"- {alert['message']}")
        print("\nRecommendations:")
        for rec in recommendations:
            print(f"- {rec}")

if __name__ == "__main__":
    main()