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
    st.title("🏦 U.S. Treasury Daily Statement Analysis")
    
    # Initialize TreasuryDataFetcher
    fetcher = TreasuryDataFetcher()
    
    # Create sidebar controls
    with st.sidebar:
        st.header("Analysis Configuration")
        
        # Date range selection
        st.subheader("Select Date Range")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        start_date = st.date_input(
            "Start Date",
            value=start_date,
            max_value=end_date
        )
        
        end_date = st.date_input(
            "End Date",
            value=end_date,
            max_value=datetime.now()
        )
        
        # Category selection
        st.subheader("Select Categories")
        categories = st.multiselect(
            "Program Categories",
            options=list(fetcher.program_categories.keys()),
            default=['SNAP', 'NIH', 'VA'],
            help="Choose departments/programs to analyze"
        )
        
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
    if st.button("🔄 Run Analysis"):
        with st.spinner("Fetching Treasury data..."):
            try:
                # Get treasury data
                df, alerts, recommendations = fetcher.analyze_treasury_data(
                    start_date=datetime.combine(start_date, datetime.min.time()),
                    end_date=datetime.combine(end_date, datetime.min.time()),
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
            
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")

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
    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics):
        with col:
            st.metric(
                label=metric['category'],
                value=f"${metric['current_value']:.2f}B",
                delta=f"{metric['change']:.1f}%"
            )

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
            go.Scatter(
                x=plot_data.index,
                y=plot_data.values,
                name=label,
                mode='lines+markers',
                marker=dict(size=6)
            )
        )
        
        # Add alert indicators
        alerts = fetcher.detect_funding_changes(category_data, category)
        for alert in alerts:
            if alert['alert_type'] == 'RED_ALERT':
                fig.add_vline(
                    x=alert['date'],
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
        if not category_data.empty:
            current_value = category_data[column].iloc[0] / 1e9  # Convert to billions
            prev_value = category_data[column].iloc[1] / 1e9 if len(category_data) > 1 else current_value
            pct_change = ((current_value / prev_value) - 1) * 100 if prev_value != 0 else 0
            
            metrics.append({
                'category': category,
                'current_value': current_value,
                'change': pct_change
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
    
    # Create trend analysis
    st.markdown("### Trend Analysis")
    
    for category in categories:
        category_data = df[df['classification_desc'] == category]
        if not category_data.empty:
            st.markdown(f"#### {category}")
            
            # Calculate trends
            column = ('current_month_outly_amt' if analysis_type == 'outlays' 
                     else 'current_month_gross_rcpt_amt')
            
            current = category_data[column].iloc[0] / 1e9
            avg_30d = category_data[column].head(30).mean() / 1e9
            avg_90d = category_data[column].mean() / 1e9
            
            # Display metrics
            cols = st.columns(3)
            with cols[0]:
                st.metric("Current Value", f"${current:.2f}B")
            with cols[1]:
                st.metric("30-Day Average", f"${avg_30d:.2f}B")
            with cols[2]:
                st.metric("90-Day Average", f"${avg_90d:.2f}B")
            
            # Add a separator
            st.markdown("---")