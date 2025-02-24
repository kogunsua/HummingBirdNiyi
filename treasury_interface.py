# treasury_interface.py

import streamlit as st
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Optional
from treasurydata import TreasuryDataFetcher

def display_treasury_dashboard_internal():
    """Main function to display the Treasury dashboard"""
    st.title("🏦 U.S. Treasury Daily Statement Analysis")
    
    # Initialize TreasuryDataFetcher
    fetcher = TreasuryDataFetcher()
    
    # Create sidebar controls
    with st.sidebar:
        st.header("Analysis Configuration")
        
        # Use specific date selection instead of date range
        st.subheader("Select Treasury Report Date")
        available_dates = fetcher.get_available_dates()
        
        selected_date = st.selectbox(
            "Report Date",
            options=available_dates,
            index=0,
            help="Select a specific Treasury report date"
        )
        
        # Show information about date selection approach
        st.info(f"Selected report date: {selected_date}")
        st.write("The Treasury API works best with specific report dates.")
        
        # Category selection
        st.subheader("Select Categories")
        default_categories = ['SNAP', 'NIH', 'VA']
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
            options=['daily', 'weekly', 'seven_day_ma'],
            help="Choose data aggregation frequency"
        )

    # Main content area
    if st.button("🔄 Run Analysis", key="run_analysis"):
        if not categories:
            st.warning("Please select at least one category to analyze")
            return
            
        with st.spinner("Fetching Treasury data..."):
            try:
                # We'll continue to use the existing function signature but modify how it works internally
                # to use specific date filtering instead of date ranges
                start_date = datetime.now() - timedelta(days=1)  # Just a placeholder
                end_date = datetime.now()  # Just a placeholder
                
                df, alerts, recommendations = fetcher.analyze_treasury_data(
                    start_date=start_date,
                    end_date=end_date,
                    categories=categories,
                    frequency=frequency
                )
                
                if df is not None and not df.empty:
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
                    - The selected categories might not have records for this period
                    - The Treasury API might be experiencing issues
                    
                    Try selecting different categories.
                    """)
            
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.info("""
                Troubleshooting tips:
                - Check your internet connection
                - Try different categories
                - The Treasury API might be temporarily unavailable
                """)
    else:
        # Display instructions when first loading
        st.info("""
        ## How to use this dashboard
        
        1. Select a Treasury report date from the sidebar
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
    fig = create_treasury_plot(df, fetcher, categories, frequency, analysis_type)
    st.plotly_chart(fig, use_container_width=True)
    
    # Display summary metrics
    st.subheader("Summary Metrics")
    
    # Calculate metrics for each category
    metrics = calculate_summary_metrics(df, categories, analysis_type)
    
    # Display metrics in columns
    cols = st.columns(len(metrics)) if metrics else st.columns(1)
    if metrics:
        for col, metric in zip(cols, metrics):
            with col:
                st.metric(
                    label=metric['category'],
                    value=f"${metric['current_value']:.2f}B",
                    delta=f"{metric['change']:.1f}%" if 'change' in metric else None
                )
    else:
        st.info("No metrics available for the selected data")

def create_treasury_plot(df: pd.DataFrame, fetcher: TreasuryDataFetcher,
                        categories: List[str], frequency: str, plot_type: str) -> go.Figure:
    """Create interactive Plotly visualization of Treasury data"""
    fig = go.Figure()
    
    amount_column = (
        'current_month_outly_amt' 
        if plot_type == 'outlays' 
        else 'current_month_gross_rcpt_amt'
    )
    
    for category in categories:
        category_data = df[df['classification_desc'] == category].copy()
        
        if category_data.empty:
            continue
            
        if frequency == 'seven_day_ma':
            category_data = fetcher.calculate_moving_average(category_data)
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
        
        # Add trace
        fig.add_trace(
            go.Bar(
                x=[str(x).split()[0] for x in plot_data.index],  # Convert datetime to string for bar chart
                y=plot_data.values,
                name=label
            )
        )
        
        # Add alert indicators
        alerts = fetcher.detect_funding_changes(category_data, category)
        for alert in alerts:
            if alert['alert_type'] == 'RED_ALERT':
                fig.add_vline(
                    x=str(alert['date']).split()[0],
                    line_dash="dash",
                    line_color="red",
                    opacity=0.3
                )
    
    title_type = "Outlays" if plot_type == 'outlays' else "Receipts"
    fig.update_layout(
        title=f'Treasury {title_type} by Category',
        xaxis_title='Date',
        yaxis_title=f'{title_type} (Billions USD)',
        height=600,
        showlegend=True,
        hovermode='x unified'
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
        if not category_data.empty and len(category_data) > 0:
            current_value = category_data[column].iloc[0] / 1e9  # Convert to billions
            
            metric = {
                'category': category,
                'current_value': current_value
            }
            
            # Add change metric if we have more than one data point
            if len(category_data) > 1:
                prev_value = category_data[column].iloc[1] / 1e9
                if prev_value != 0:
                    pct_change = ((current_value / prev_value) - 1) * 100
                    metric['change'] = pct_change
            
            metrics.append(metric)
    
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
    
    # Create trend analysis
    st.markdown("### Data Table")
    
    # Show a filtered dataframe with just the relevant categories
    filtered_df = df[df['classification_desc'].isin(categories)].copy()
    
    if not filtered_df.empty:
        # Select columns to display
        display_cols = ['classification_desc', 'record_date']
        amount_col = ('current_month_outly_amt' if analysis_type == 'outlays' 
                     else 'current_month_gross_rcpt_amt')
        
        if amount_col in filtered_df.columns:
            display_cols.append(amount_col)
            filtered_df[amount_col] = filtered_df[amount_col] / 1e9  # Convert to billions
            
            # Rename columns for display
            renamed_df = filtered_df[display_cols].rename(columns={
                'classification_desc': 'Category',
                'record_date': 'Date',
                amount_col: f'Amount (Billions $)'
            })
            
            st.dataframe(renamed_df)
        else:
            st.info(f"No {amount_col} data available")
    else:
        st.info("No data available for the selected categories")
    
    # Show individual category details
    for category in categories:
        category_data = df[df['classification_desc'] == category]
        if not category_data.empty:
            st.markdown(f"#### {category}")
            
            # Calculate trends
            column = ('current_month_outly_amt' if analysis_type == 'outlays' 
                     else 'current_month_gross_rcpt_amt')
            
            if column in category_data.columns:
                current = category_data[column].iloc[0] / 1e9
                
                # Display metrics
                st.metric("Current Value", f"${current:.2f}B")
                
                # Show raw data
                with st.expander(f"View raw data for {category}"):
                    st.dataframe(category_data)
                
                # Add a separator
                st.markdown("---")
            else:
                st.info(f"No {column} data available for {category}")
        else:
            st.info(f"No data available for {category}")

def display_treasury_tab():
    """Display Treasury tab with improved diagnostics and dashboard"""
    # Create tabs for dashboard and diagnostics
    dashboard_tab, diagnostics_tab = st.tabs(["Dashboard", "Diagnostics"])
    
    with dashboard_tab:
        display_treasury_dashboard_internal()
    
    with diagnostics_tab:
        st.title("🔍 Treasury API Diagnostics")
        
        if st.button("Run API Test"):
            with st.spinner("Testing Treasury API..."):
                try:
                    # Basic test of API connectivity
                    import requests
                    base_url = 'https://api.fiscaldata.treasury.gov/services/api/fiscal_service'
                    endpoint = '/v1/accounting/mts/mts_table_1'
                    
                    # Test basic connectivity
                    test_url = f"{base_url}{endpoint}?page[number]=1&page[size]=1"
                    response = requests.get(test_url)
                    
                    if response.status_code == 200:
                        st.success(f"✅ API connection successful (Status code: {response.status_code})")
                        
                        # Try example API request that works
                        test_date = "2023-05-31"
                        example_url = f"{base_url}{endpoint}?fields=record_date,classification_desc,current_month_gross_rcpt_amt&filter=record_date:eq:{test_date}&page[size]=5"
                        st.code(example_url, language="bash")
                        
                        sample_response = requests.get(example_url)
                        if sample_response.status_code == 200:
                            sample_data = sample_response.json()
                            if 'data' in sample_data and sample_data['data']:
                                st.success(f"✅ Successfully retrieved data with exact date filter")
                                st.json(sample_data)
                            else:
                                st.warning(f"⚠️ No data returned for sample request")
                        else:
                            st.error(f"❌ Sample request failed (Status code: {sample_response.status_code})")
                    else:
                        st.error(f"❌ API connection failed (Status code: {response.status_code})")
                
                except Exception as e:
                    st.error(f"❌ Error testing API: {str(e)}")

# This function is the public entry point
def display_treasury_dashboard():
    """Public entry point to display the Treasury dashboard"""
    # Call display_treasury_tab which includes both the dashboard and diagnostics
    display_treasury_tab()

if __name__ == "__main__":
    st.set_page_config(
        page_title="Treasury Dashboard",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    display_treasury_dashboard()
