# treasury_interface.py

import streamlit as st
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Optional
from treasurydata import TreasuryDataFetcher

def display_treasury_dashboard():
    """Main function to display the Treasury dashboard"""
    st.title("🏦 U.S. Treasury Monthly Statement Analysis")
    
    # Initialize TreasuryDataFetcher
    fetcher = TreasuryDataFetcher()
    
    # Create sidebar controls
    with st.sidebar:
        st.header("Analysis Configuration")
        
        # Specific date selection instead of date range
        st.subheader("Select Treasury Report Date")
        available_dates = fetcher.get_available_dates()
        
        if 'selected_date' not in st.session_state:
            st.session_state.selected_date = available_dates[0]
        
        selected_date = st.selectbox(
            "Report Date",
            options=available_dates,
            index=available_dates.index(st.session_state.selected_date),
            help="Select a specific Treasury report date"
        )
        
        st.session_state.selected_date = selected_date
        
        # Show info about date selection approach
        st.info(f"Selected report date: {selected_date}")
        st.write("The Treasury API requires specific report dates. Date ranges are less reliable.")
        
        # Category selection
        st.subheader("Select Categories")
        default_categories = ['DOD', 'HHS', 'VA']
        
        if 'categories' not in st.session_state:
            st.session_state.categories = default_categories
            
        categories = st.multiselect(
            "Program Categories",
            options=list(fetcher.program_categories.keys()),
            default=st.session_state.categories,
            help="Choose departments/programs to analyze"
        )
        
        st.session_state.categories = categories
        
        # Analysis type
        st.subheader("Analysis Options")
        analysis_type = st.radio(
            "Data Type",
            options=['outlays', 'receipts'],
            help="Choose between outlays or receipts analysis"
        )
        
        frequency = st.selectbox(
            "Data Frequency",
            options=['daily'],  # Removed weekly and seven_day_ma as they require multiple dates
            help="Choose data aggregation frequency"
        )

    # Main content area
    if st.button("🔄 Run Analysis", key="run_analysis"):
        if not categories:
            st.warning("Please select at least one category to analyze")
            return
            
        with st.spinner(f"Fetching Treasury data for {selected_date}..."):
            try:
                # Get treasury data using specific date
                df, alerts, recommendations = fetcher.analyze_treasury_data(
                    specific_date=selected_date,
                    categories=categories,
                    frequency=frequency
                )
                
                if not df.empty:
                    # Create tabs for different views
                    overview_tab, alerts_tab, details_tab = st.tabs([
                        "📊 Overview",
                        "🚨 Alerts",
                        "📋 Detailed Analysis"
                    ])
                    
                    with overview_tab:
                        display_overview(df, fetcher, categories, frequency, analysis_type)
                    
                    with alerts_tab:
                        display_alerts(alerts, recommendations)
                    
                    with details_tab:
                        display_detailed_analysis(df, categories, analysis_type)
                else:
                    st.error("No data available for the selected parameters.")
                    st.info("""
                    This could be due to:
                    - The selected date might not have data available
                    - The selected categories might not have records for this period
                    - The Treasury API might be experiencing issues
                    
                    Try selecting a different date or different categories.
                    """)
            
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.info("""
                Troubleshooting tips:
                - Check your internet connection
                - Try a different date
                - Try different categories
                - The Treasury API might be temporarily unavailable
                """)
    else:
        # Display instructions when first loading
        st.info("""
        ## How to use this dashboard
        
        1. Select a specific Treasury report date from the sidebar
        2. Choose one or more program categories to analyze
        3. Select whether to view outlays or receipts data
        4. Click "Run Analysis" to generate the visualization
        
        This dashboard uses the U.S. Treasury's Monthly Treasury Statement (MTS) API.
        """)

def display_overview(df: pd.DataFrame, fetcher: TreasuryDataFetcher, 
                    categories: List[str], frequency: str, analysis_type: str):
    """Display overview of Treasury data"""
    st.subheader("Treasury Data Overview")
    
    # Create visualization
    fig = create_treasury_plot(df, fetcher, categories, analysis_type)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display summary metrics
    st.subheader("Summary Metrics")
    
    # Calculate metrics for each category
    metrics = calculate_summary_metrics(df, categories, analysis_type)
    
    # Display metrics in columns
    if metrics:
        cols = st.columns(len(metrics))
        for col, metric in zip(cols, metrics):
            with col:
                st.metric(
                    label=metric['category'],
                    value=f"${metric['current_value']:.2f}B"
                )
    else:
        st.info("No metrics available for the selected data")

def create_treasury_plot(df: pd.DataFrame, fetcher: TreasuryDataFetcher,
                        categories: List[str], plot_type: str) -> go.Figure:
    """Create interactive Plotly visualization of Treasury data"""
    fig = go.Figure()
    
    amount_column = (
        'current_month_outly_amt' 
        if plot_type == 'outlays' 
        else 'current_month_gross_rcpt_amt'
    )
    
    # Filter for selected categories
    display_df = df[df['classification_desc'].isin(categories)].copy()
    
    if display_df.empty:
        return fig
        
    # Convert to billions for display
    if amount_column in display_df.columns:
        display_df[amount_column] = display_df[amount_column] / 1e9
        
        # Create bar chart - better for single date display
        fig = go.Figure(data=[
            go.Bar(
                x=display_df['classification_desc'],
                y=display_df[amount_column],
                text=display_df[amount_column].apply(lambda x: f'${x:.2f}B'),
                textposition='auto',
                hoverinfo='text',
                name=plot_type.capitalize()
            )
        ])
    
    title_type = "Outlays" if plot_type == 'outlays' else "Receipts"
    fig.update_layout(
        title=f'Treasury {title_type} by Category',
        xaxis_title='Category',
        yaxis_title=f'{title_type} (Billions USD)',
        height=500,
        showlegend=True,
        hovermode='closest'
    )
    
    return fig

def calculate_summary_metrics(df: pd.DataFrame, categories: List[str], 
                            analysis_type: str) -> List[dict]:
    """Calculate summary metrics for each category"""
    metrics = []
    column = ('current_month_outly_amt' if analysis_type == 'outlays' 
              else 'current_month_gross_rcpt_amt')
    
    for category in categories:
        category_data = df[df['classification_desc'] == category]
        if not category_data.empty and column in category_data.columns:
            current_value = category_data[column].iloc[0] / 1e9  # Convert to billions
            
            metrics.append({
                'category': category,
                'current_value': current_value
            })
    
    return metrics

def display_alerts(alerts: List[dict], recommendations: List[str]):
    """Display alerts and recommendations"""
    st.subheader("🚨 Active Alerts")
    
    if not alerts:
        st.info("No alerts detected for the selected parameters.")
    else:
        # Group alerts by type
        for alert_type in ['RED_ALERT', 'WARNING', 'INFO']:
            type_alerts = [a for a in alerts if a['alert_type'] == alert_type]
            if type_alerts:
                if alert_type == 'RED_ALERT':
                    st.error("Critical Alerts")
                elif alert_type == 'WARNING':
                    st.warning("Warning Alerts")
                else:
                    st.info("Information Alerts")
                
                for alert in type_alerts:
                    st.markdown(f"- {alert['message']}")
    
    st.subheader("📋 Recommendations")
    if not recommendations:
        st.info("No recommendations generated for the current analysis.")
    else:
        for rec in recommendations:
            st.markdown(f"- {rec}")

def display_detailed_analysis(df: pd.DataFrame, categories: List[str], analysis_type: str):
    """Display detailed analysis of Treasury data"""
    st.subheader("Detailed Analysis")
    
    # Show raw data table
    st.markdown("### Treasury Data Table")
    
    # Prepare data for display
    amount_col = ('current_month_outly_amt' if analysis_type == 'outlays' 
                 else 'current_month_gross_rcpt_amt')
    
    if amount_col in df.columns:
        display_cols = ['classification_desc', 'record_date', amount_col]
        display_df = df[df['classification_desc'].isin(categories)][display_cols].copy()
        
        if not display_df.empty:
            # Format amounts to billions
            display_df[amount_col] = display_df[amount_col] / 1e9
            
            # Rename columns for display
            display_df = display_df.rename(columns={
                'classification_desc': 'Category',
                'record_date': 'Date',
                amount_col: 'Amount (Billions $)'
            })
            
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No data available for the selected categories.")
    
    # Create individual category analysis
    st.markdown("### Category Details")
    
    for category in categories:
        category_data = df[df['classification_desc'] == category]
        if not category_data.empty and amount_col in category_data.columns:
            st.markdown(f"#### {category}")
            
            # Calculate total amount in billions
            amount = category_data[amount_col].iloc[0] / 1e9
            
            # Display metric
            st.metric(
                "Amount", 
                f"${amount:.2f}B"
            )
            
            # Show category raw data
            with st.expander(f"View raw data for {category}"):
                st.dataframe(category_data)
            
            st.markdown("---")
        else:
            st.info(f"No data available for {category}")

def display_treasury_diagnostics():
    """Display diagnostic information about the Treasury API"""
    st.subheader("🔍 Treasury API Diagnostics")
    
    if st.button("Run API Test"):
        with st.spinner("Testing Treasury API..."):
            try:
                # Basic test of API connectivity
                import requests
                base_url = 'https://api.fiscaldata.treasury.gov/services/api/fiscal_service'
                endpoint = '/v1/accounting/mts/mts_table_1'
                test_url = f"{base_url}{endpoint}?page[number]=1&page[size]=1"
                
                response = requests.get(test_url)
                
                if response.status_code == 200:
                    st.success(f"✅ API connection successful (Status code: {response.status_code})")
                    
                    # Try a specific known date
                    test_date = "2023-05-31"
                    test_url = f"{base_url}{endpoint}?fields=record_date&filter=record_date:eq:{test_date}&page[size]=1"
                    date_response = requests.get(test_url)
                    
                    if date_response.status_code == 200 and date_response.json().get('data'):
                        st.success(f"✅ Successfully retrieved data for {test_date}")
                    else:
                        st.warning(f"⚠️ No data available for test date {test_date}")
                    
                    # Try a sample of specific dates to find available data
                    st.subheader("Testing Sample Dates")
                    test_dates = [
                        "2023-05-31", "2023-04-30", "2023-03-31", 
                        "2023-02-28", "2023-01-31", "2022-12-31"
                    ]
                    
                    available_dates = []
                    for date in test_dates:
                        test_url = f"{base_url}{endpoint}?fields=record_date&filter=record_date:eq:{date}&page[size]=1"
                        date_response = requests.get(test_url)
                        
                        if date_response.status_code == 200 and date_response.json().get('data'):
                            st.success(f"✅ {date}: Data available")
                            available_dates.append(date)
                        else:
                            st.warning(f"⚠️ {date}: No data available")
                    
                    if available_dates:
                        st.info(f"Use these dates in your analysis: {', '.join(available_dates)}")
                else:
                    st.error(f"❌ API connection failed (Status code: {response.status_code})")
            
            except Exception as e:
                st.error(f"❌ Error testing API: {str(e)}")

def display_treasury_tab():
    """Display combined Treasury tab with diagnostics and dashboard"""
    st.title("🏦 U.S. Treasury Analysis")
    
    # Create tabs for dashboard and diagnostics
    dashboard_tab, diagnostics_tab = st.tabs(["Dashboard", "Diagnostics"])
    
    with dashboard_tab:
        display_treasury_dashboard()
    
    with diagnostics_tab
