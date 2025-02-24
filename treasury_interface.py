# treasury_interface.py

import streamlit as st
from datetime import datetime, timedelta
import plotly.graph_objects as go
import pandas as pd
from typing import List, Optional, Dict
from treasurydata import TreasuryDataFetcher

def display_treasury_dashboard():
    """Main function to display the Federal Spending Dashboard"""
    st.title("🏦 Federal Government Spending Tracker")
    st.subheader("Real-time spending visualization across government agencies")
    
    # Initialize TreasuryDataFetcher
    fetcher = TreasuryDataFetcher()
    
    # Create sidebar controls
    with st.sidebar:
        st.header("Visualization Controls")
        
        # Time period selection
        st.subheader("Select Time Period")
        time_period = st.radio(
            "Time Period",
            options=["1 month", "3 months", "12 months", "36 months"],
            index=1,  # Default to 3 months
            help="Select the time period for analysis"
        )
        
        # Calculate dates based on selection
        end_date = datetime.now().date()
        
        if time_period == "1 month":
            start_date = end_date - timedelta(days=30)
        elif time_period == "3 months":
            start_date = end_date - timedelta(days=90)
        elif time_period == "12 months":
            start_date = end_date - timedelta(days=365)
        else:  # 36 months
            start_date = end_date - timedelta(days=365*3)
        
        # Show selected date range
        st.info(f"Period: {start_date.strftime('%b %d, %Y')} to {end_date.strftime('%b %d, %Y')}")
        
        # Outlay frequency selection
        st.subheader("Outlay Frequency")
        frequency = st.radio(
            "Display outlays as:",
            options=[
                "Daily outlays",
                "Weekly outlays",
                "7-day moving average"
            ],
            help="Choose how to display the outlay data"
        )
        
        # Map UI selection to data format
        frequency_map = {
            "Daily outlays": "daily",
            "Weekly outlays": "weekly",
            "7-day moving average": "seven_day_ma"
        }
        
        # Category selection - allow up to 9 categories
        st.subheader("Federal Outlay Categories")
        
        # Get all available categories
        all_categories = list(fetcher.program_categories.keys())
        
        # Option to search or select
        category_select_method = st.radio(
            "Category selection method:",
            options=["Select from list", "Search for categories"],
            help="Choose how you want to select categories"
        )
        
        if category_select_method == "Select from list":
            # Display human-readable names but store the internal code
            category_map = {v: k for k, v in fetcher.program_categories.items()}
            selected_categories = st.multiselect(
                "Select up to 9 categories:",
                options=list(category_map.keys()),
                default=[fetcher.program_categories["SNAP"], fetcher.program_categories["VA"]],
                max_selections=9,
                help="Choose up to 9 departments/programs to analyze"
            )
            # Convert back to internal codes
            categories = [category_map[cat] for cat in selected_categories]
        else:
            # Search functionality
            search_term = st.text_input(
                "Search for categories",
                help="Type to search for departments/programs"
            ).lower()
            
            filtered_categories = []
            if search_term:
                for code, name in fetcher.program_categories.items():
                    if search_term in code.lower() or search_term in name.lower():
                        filtered_categories.append((code, name))
            else:
                filtered_categories = list(fetcher.program_categories.items())
            
            # Display search results with checkboxes
            st.write("Select from search results:")
            selected_categories = []
            
            # Limit to first 15 matches for UI clarity
            for code, name in filtered_categories[:15]:
                if st.checkbox(f"{name} ({code})", key=f"search_{code}"):
                    selected_categories.append(code)
            
            # Enforce max of 9 categories
            if len(selected_categories) > 9:
                st.warning("Only the first 9 selected categories will be displayed.")
                categories = selected_categories[:9]
            else:
                categories = selected_categories

        # Analytics options
        st.subheader("Additional Options")
        show_trend_lines = st.checkbox(
            "Show trend lines",
            value=True,
            help="Display linear trend lines for each category"
        )
        
        normalize_data = st.checkbox(
            "Normalize data (percentage of total)",
            value=False,
            help="Show data as percentage of total spending instead of absolute values"
        )

    # Main content area
    if not categories:
        st.warning("Please select at least one category to analyze")
    else:
        with st.spinner("Fetching latest Treasury data..."):
            try:
                # Convert UI selections to API parameters
                freq_param = frequency_map[frequency]
                
                # Get treasury data
                df, alerts, recommendations = fetcher.analyze_treasury_data(
                    start_date=datetime.combine(start_date, datetime.min.time()),
                    end_date=datetime.combine(end_date, datetime.min.time()),
                    categories=categories,
                    frequency=freq_param
                )
                
                if df is not None and not df.empty:
                    # Create tabs for different views
                    visualization_tab, alerts_tab, details_tab = st.tabs([
                        "📊 Spending Visualization",
                        "🚨 Notable Changes",
                        "📋 Detailed Analysis"
                    ])
                    
                    with visualization_tab:
                        display_spending_visualization(
                            df, 
                            fetcher, 
                            categories, 
                            freq_param, 
                            "outlays",
                            show_trend_lines,
                            normalize_data
                        )
                    
                    with alerts_tab:
                        display_spending_alerts(alerts, recommendations)
                    
                    with details_tab:
                        display_detailed_analysis(df, categories, "outlays")
                else:
                    st.error("No data available for the selected parameters.")
                    st.info("""
                    This could be due to:
                    - The selected time period might not have data available yet
                    - The selected categories might not have records for this period
                    - There may be a delay in updating the Treasury data
                    
                    Try selecting a different time period or different categories.
                    """)
            
            except Exception as e:
                st.error(f"Error retrieving spending data: {str(e)}")
                st.info("""
                Troubleshooting tips:
                - Check your internet connection
                - Try a smaller time period
                - Try different categories
                - The Treasury API might be temporarily unavailable
                """)

def display_spending_visualization(
    df: pd.DataFrame, 
    fetcher: TreasuryDataFetcher, 
    categories: List[str], 
    frequency: str, 
    analysis_type: str,
    show_trend_lines: bool = True,
    normalize_data: bool = False
):
    """Display interactive visualization of federal spending data"""
    st.subheader("Federal Spending Visualization")
    
    # Add description
    st.markdown("""
    This visualization shows federal government outlays (spending) across selected categories.
    Data is sourced directly from the U.S. Treasury's Daily Treasury Statements.
    """)
    
    # Create visualization
    fig = create_spending_plot(
        df, 
        fetcher, 
        categories, 
        frequency, 
        analysis_type,
        show_trend_lines,
        normalize_data
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display summary metrics - use columns to create a dashboard feel
    st.subheader("Spending Summary (In Billions of Dollars)")
    
    # Calculate metrics for each category
    metrics = calculate_summary_metrics(df, categories, analysis_type)
    
    # Create a grid of metrics
    if metrics:
        # Determine how many columns we need
        cols_per_row = 3
        num_rows = (len(metrics) + cols_per_row - 1) // cols_per_row
        
        for row in range(num_rows):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                metric_idx = row * cols_per_row + col_idx
                if metric_idx < len(metrics):
                    metric = metrics[metric_idx]
                    with cols[col_idx]:
                        st.metric(
                            label=f"{metric['category']}",
                            value=f"${metric['current_value']:.2f}B",
                            delta=f"{metric['change']:.1f}%",
                            delta_color="inverse" if analysis_type == "outlays" else "normal"
                        )
    else:
        st.info("No spending metrics available for the selected data")
    
    # Add data download option
    if not df.empty:
        st.download_button(
            label="Download data as CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name=f"federal_spending_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )

def create_spending_plot(
    df: pd.DataFrame, 
    fetcher: TreasuryDataFetcher,
    categories: List[str], 
    frequency: str, 
    plot_type: str,
    show_trend_lines: bool = True,
    normalize_data: bool = False
) -> go.Figure:
    """Create enhanced interactive Plotly visualization of federal spending data"""
    fig = go.Figure()
    
    # Determine which column to use for the data
    amount_column = 'current_month_outly_amt' if plot_type == 'outlays' else 'current_month_gross_rcpt_amt'
    
    # Calculate total spending per date if normalization is requested
    date_totals = {}
    if normalize_data:
        for date in df['record_date'].unique():
            date_df = df[df['record_date'] == date]
            date_totals[date] = date_df[amount_column].sum()
    
    # Define a color palette for categories
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    # Process each selected category
    for i, category in enumerate(categories):
        # Safety check to prevent index errors
        color_idx = i % len(colors)
        
        category_data = df[df['classification_desc'] == category].copy()
        
        if category_data.empty:
            continue
            
        # Create the appropriate data series based on frequency
        if frequency == 'seven_day_ma':
            category_data = fetcher.calculate_moving_average(category_data)
            x_values = category_data['record_date']
            y_values = category_data[f'{amount_column}_ma'] / 1e9  # Convert to billions
            label = f"{category} (7-day MA)"
        elif frequency == 'weekly':
            category_data['week'] = category_data['record_date'].dt.to_period('W')
            weekly_data = category_data.groupby('week')[amount_column].sum() / 1e9
            x_values = [pd.Period(x).to_timestamp() for x in weekly_data.index]
            y_values = weekly_data.values
            label = f"{category} (Weekly)"
        else:  # daily
            x_values = category_data['record_date']
            y_values = category_data[amount_column] / 1e9  # Convert to billions
            label = f"{category} (Daily)"
        
        # Normalize data if requested
        if normalize_data:
            normalized_values = []
            for i, date in enumerate(x_values):
                if date in date_totals and date_totals[date] > 0:
                    normalized_values.append((y_values[i] * 1e9 / date_totals[date]) * 100)
                else:
                    normalized_values.append(0)
            y_values = normalized_values
        
        # Add trace for this category
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                name=label,
                mode='lines+markers',
                marker=dict(size=6, color=colors[color_idx]),
                line=dict(width=2, color=colors[color_idx]),
                hovertemplate='%{x}<br>%{y:.2f}' + (' %' if normalize_data else 'B $')
            )
        )
        
        # Add trend line if requested
        if show_trend_lines and len(x_values) > 1:
            # Convert x values to numeric for trendline calculation
            x_numeric = range(len(x_values))
            
            # Calculate trend line using numpy's polyfit
            import numpy as np
            z = np.polyfit(x_numeric, y_values, 1)
            p = np.poly1d(z)
            trend_y = [p(x) for x in x_numeric]
            
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=trend_y,
                    name=f"{category} trend",
                    mode='lines',
                    line=dict(color=colors[color_idx], width=1, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
        
        # Add alert indicators for significant changes
        alerts = fetcher.detect_funding_changes(category_data, category)
        for alert in alerts:
            if alert['alert_type'] == 'RED_ALERT':
                fig.add_vline(
                    x=alert['date'],
                    line_dash="dash",
                    line_color="red",
                    opacity=0.3,
                    annotation_text="Significant Change",
                    annotation_position="top right"
                )
    
    # Customize the figure layout
    title_text = "Federal Government Outlays by Category" if plot_type == 'outlays' else "Federal Government Receipts by Category"
    y_axis_title = "Percentage of Total Spending" if normalize_data else "Billions of Dollars"
    
    fig.update_layout(
        title={
            'text': title_text,
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Date',
        yaxis_title=y_axis_title,
        legend_title="Categories",
        height=600,
        showlegend=True,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=60, r=30, t=80, b=100)
    )
    
    # Add range selector for time navigation
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    
    return fig

def calculate_summary_metrics(df: pd.DataFrame, categories: List[str], 
                             analysis_type: str) -> List[Dict]:
    """Calculate enhanced summary metrics for each category"""
    metrics = []
    column = 'current_month_outly_amt' if analysis_type == 'outlays' else 'current_month_gross_rcpt_amt'
    
    for category in categories:
        category_data = df[df['classification_desc'] == category]
        if not category_data.empty and len(category_data) > 0:
            # Get the most recent value (in billions)
            current_value = category_data[column].iloc[0] / 1e9
            
            # Calculate percentage change
            if len(category_data) > 1:
                prev_value = category_data[column].iloc[1] / 1e9
                pct_change = ((current_value / prev_value) - 1) * 100 if prev_value != 0 else 0
            else:
                pct_change = 0
            
            # Calculate additional metrics
            total_value = category_data[column].sum() / 1e9 if len(category_data) > 0 else 0
            avg_value = category_data[column].mean() / 1e9 if len(category_data) > 0 else 0
            
            metrics.append({
                'category': category,
                'current_value': current_value,
                'change': pct_change,
                'total': total_value,
                'average': avg_value
            })
    
    return metrics

def display_spending_alerts(alerts: List[Dict], recommendations: List[str]):
    """Display enhanced alerts and recommendations for spending changes"""
    st.subheader("Notable Spending Changes")
    
    # Create tabs for different alert levels
    if alerts:
        alert_tabs = st.tabs(["🔴 Critical Changes", "⚠️ Significant Changes", "ℹ️ Notable Trends"])
        
        with alert_tabs[0]:
            critical_alerts = [a for a in alerts if a['alert_type'] == 'RED_ALERT']
            if critical_alerts:
                for alert in critical_alerts:
                    st.error(alert['message'])
            else:
                st.info("No critical spending changes detected in the selected period.")
        
        with alert_tabs[1]:
            warning_alerts = [a for a in alerts if a['alert_type'] == 'WARNING']
            if warning_alerts:
                for alert in warning_alerts:
                    st.warning(alert['message'])
            else:
                st.info("No significant spending changes detected in the selected period.")
        
        with alert_tabs[2]:
            info_alerts = [a for a in alerts if a['alert_type'] == 'INFO']
            if info_alerts:
                for alert in info_alerts:
                    st.info(alert['message'])
            else:
                st.info("No notable spending trends detected in the selected period.")
    else:
        st.info("No significant spending changes detected for the selected parameters.")
    
    # Display recommendations with more structure
    st.subheader("Analysis & Recommendations")
    if recommendations:
        for i, rec in enumerate(recommendations):
            st.markdown(f"**{i+1}.** {rec}")
    else:
        st.info("No specific recommendations for the current spending analysis.")
    
    # Add historical context
    st.subheader("Historical Context")
    st.markdown("""
    Spending patterns should be interpreted in the context of:
    - Budget cycles (fiscal year begins October 1)
    - Supplemental appropriations
    - Continuing resolutions
    - Seasonal program spending patterns
    
    Some spending fluctuations are normal and expected based on the government's fiscal calendar.
    """)

def display_detailed_analysis(df: pd.DataFrame, categories: List[str], analysis_type: str):
    """Display detailed spending analysis with enhanced metrics"""
    st.subheader("Detailed Spending Analysis")
    
    # Option to select specific category for detailed view
    selected_category = st.selectbox(
        "Select category for detailed analysis",
        options=categories
    )
    
    # Get data for selected category
    category_data = df[df['classification_desc'] == selected_category]
    
    if not category_data.empty:
        # Create two columns for metrics and chart
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader(f"{selected_category} Metrics")
            
            # Calculate key metrics
            column = 'current_month_outly_amt' if analysis_type == 'outlays' else 'current_month_gross_rcpt_amt'
            current = category_data[column].iloc[0] / 1e9
            avg_30d = category_data[column].head(30).mean() / 1e9 if len(category_data) >= 30 else None
            avg_90d = category_data[column].mean() / 1e9
            median = category_data[column].median() / 1e9
            total = category_data[column].sum() / 1e9
            
            # Calculate volatility
            std = category_data[column].std() / 1e9
            volatility = (std / avg_90d) * 100 if avg_90d > 0 else 0
            
            # Display metrics
            st.metric("Latest Value", f"${current:.2f}B")
            if avg_30d is not None:
                st.metric("30-Day Average", f"${avg_30d:.2f}B")
            st.metric("Average", f"${avg_90d:.2f}B")
            st.metric("Median", f"${median:.2f}B")
            st.metric("Total", f"${total:.2f}B")
            st.metric("Volatility", f"{volatility:.1f}%")
            
            # Display information about the category
            st.subheader("About This Category")
            st.markdown(f"""
            This visualization shows spending for **{selected_category}** based on 
            the U.S. Treasury's Daily Treasury Statements.
            
            *Note: Labels have been edited for clarity and consistency over time 
            and may not exactly match those found on the Daily Treasury Statements.*
            """)
        
        with col2:
            st.subheader("Detailed Analysis")
            
            # Create a detailed chart for this category
            detailed_fig = go.Figure()
            
            # Plot the main data
            detailed_fig.add_trace(
                go.Scatter(
                    x=category_data['record_date'],
                    y=category_data[column] / 1e9,
                    name=f"{selected_category}",
                    mode='lines+markers',
                    marker=dict(size=8),
                    line=dict(width=2),
                )
            )
            
            # Add 7-day moving average
            ma_data = fetcher.calculate_moving_average(category_data)
            detailed_fig.add_trace(
                go.Scatter(
                    x=ma_data['record_date'],
                    y=ma_data[f'{column}_ma'] / 1e9,
                    name="7-day MA",
                    mode='lines',
                    line=dict(width=2, dash='dot'),
                )
            )
            
            # Update layout
            detailed_fig.update_layout(
                title=f"{selected_category} Spending Analysis",
                xaxis_title="Date",
                yaxis_title="Billions of Dollars",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(detailed_fig, use_container_width=True)
            
            # Add a historical comparison
            st.subheader("Historical Statistics")
            
            # Calculate year-over-year change if enough data
            if len(category_data) > 365:
                # This is simplified - would need actual YoY calculation
                st.metric(
                    "Year-over-Year Change",
                    f"{(current/avg_90d - 1) * 100:.1f}%"
                )
            
            # Add data table with key stats
            st.subheader("Daily Spending Data")
            st.dataframe(
                category_data[['record_date', column]].rename(
                    columns={column: 'Amount (USD)'}
                ).sort_values('record_date', ascending=False).head(10)
            )
    else:
        st.info(f"No data available for {selected_category} in the selected time period.")

if __name__ == "__main__":
    st.set_page_config(
        page_title="Federal Spending Tracker",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    display_treasury_dashboard()