import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from utils.data_generator import generate_customer_data, generate_revenue_data
from utils.models import CustomerSegmentation, PriceOptimizer
from utils.metrics import calculate_clv, calculate_roi

st.set_page_config(page_title="Revenue Analysis", page_icon="💹", layout="wide")

st.title("💹 Revenue Impact Analysis")
st.markdown("### ROI Calculator and Business Outcome Modeling")

# Sidebar controls
st.sidebar.header("Analysis Parameters")

analysis_period = st.sidebar.selectbox(
    "Analysis Period",
    ["1 Year", "2 Years", "3 Years", "5 Years"]
)

market_scenario = st.sidebar.selectbox(
    "Market Scenario",
    ["Conservative", "Base Case", "Optimistic", "Stress Test"]
)

implementation_timeline = st.sidebar.selectbox(
    "Implementation Timeline",
    ["Immediate", "3 Months", "6 Months", "12 Months"]
)

# Convert period to months
period_months = {
    "1 Year": 12,
    "2 Years": 24,
    "3 Years": 36,
    "5 Years": 60
}[analysis_period]

# Market scenario multipliers
scenario_multipliers = {
    "Conservative": 0.8,
    "Base Case": 1.0,
    "Optimistic": 1.3,
    "Stress Test": 0.6
}

multiplier = scenario_multipliers[market_scenario]

st.header(f"Revenue Analysis - {market_scenario} Scenario ({analysis_period})")

# Generate revenue analysis data
@st.cache_data
def load_revenue_analysis_data():
    customers = generate_customer_data(3000, 42)
    revenue_data = generate_revenue_data(customers)
    return customers, revenue_data

customers_df, revenue_df = load_revenue_analysis_data()

# Merge datasets
full_df = customers_df.merge(revenue_df, on='customer_id', how='inner')

# Customer segmentation for revenue analysis
segmentation_model = CustomerSegmentation(n_clusters=5, random_state=42)
segments = segmentation_model.fit_predict(customers_df[['income', 'credit_score', 'debt_to_income', 'age', 'savings_rate']])
full_df['segment'] = segments

# Current vs Optimized Revenue Analysis
st.header("Revenue Optimization Impact")

col1, col2, col3, col4 = st.columns(4)

# Calculate current metrics
current_total_revenue = full_df['current_revenue'].sum()
optimized_total_revenue = full_df['optimized_revenue'].sum() * multiplier
revenue_increase = optimized_total_revenue - current_total_revenue
percentage_increase = (revenue_increase / current_total_revenue) * 100

with col1:
    st.metric(
        label="Current Annual Revenue",
        value=f"${current_total_revenue:,.0f}",
        help="Revenue with current pricing strategy"
    )

with col2:
    st.metric(
        label="Optimized Revenue Potential",
        value=f"${optimized_total_revenue:,.0f}",
        delta=f"+{percentage_increase:.1f}%",
        help=f"Potential revenue with optimization ({market_scenario} scenario)"
    )

with col3:
    st.metric(
        label="Additional Annual Revenue",
        value=f"${revenue_increase:,.0f}",
        help="Incremental revenue from optimization"
    )

with col4:
    avg_customer_lift = revenue_increase / len(full_df)
    st.metric(
        label="Revenue Lift per Customer",
        value=f"${avg_customer_lift:,.0f}",
        help="Average additional revenue per customer"
    )

# Revenue by Segment Analysis
st.header("Revenue Analysis by Customer Segment")

segment_revenue_analysis = []
segment_labels = ['Conservative', 'Balanced', 'Growth-Oriented', 'High-Risk', 'Premium']

for segment in range(5):
    segment_data = full_df[full_df['segment'] == segment]
    
    if len(segment_data) > 0:
        current_rev = segment_data['current_revenue'].sum()
        optimized_rev = segment_data['optimized_revenue'].sum() * multiplier
        segment_lift = optimized_rev - current_rev
        segment_lift_pct = (segment_lift / current_rev) * 100 if current_rev > 0 else 0
        
        segment_revenue_analysis.append({
            'Segment': segment_labels[segment],
            'Customers': len(segment_data),
            'Current Revenue': current_rev,
            'Optimized Revenue': optimized_rev,
            'Additional Revenue': segment_lift,
            'Revenue Lift %': segment_lift_pct,
            'Avg Current Revenue per Customer': current_rev / len(segment_data),
            'Avg Optimized Revenue per Customer': optimized_rev / len(segment_data)
        })

segment_df = pd.DataFrame(segment_revenue_analysis)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Revenue by Segment")
    fig_segment_revenue = px.bar(
        segment_df,
        x='Segment',
        y=['Current Revenue', 'Optimized Revenue'],
        title="Current vs Optimized Revenue by Segment",
        labels={'value': 'Revenue ($)', 'variable': 'Revenue Type'}
    )
    fig_segment_revenue.update_layout(yaxis_tickformat='$,.0f')
    st.plotly_chart(fig_segment_revenue, use_container_width=True)

with col2:
    st.subheader("Revenue Lift by Segment")
    fig_segment_lift = px.bar(
        segment_df,
        x='Segment',
        y='Revenue Lift %',
        title="Revenue Lift Percentage by Segment",
        labels={'Revenue Lift %': 'Revenue Lift (%)'}
    )
    fig_segment_lift.update_layout(yaxis_tickformat='.1f')
    st.plotly_chart(fig_segment_lift, use_container_width=True)

# Detailed segment table
st.subheader("Detailed Segment Analysis")

display_df = segment_df.copy()
for col in ['Current Revenue', 'Optimized Revenue', 'Additional Revenue']:
    display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}")
    
for col in ['Avg Current Revenue per Customer', 'Avg Optimized Revenue per Customer']:
    display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}")
    
display_df['Revenue Lift %'] = display_df['Revenue Lift %'].apply(lambda x: f"{x:.1f}%")

st.dataframe(display_df, use_container_width=True, hide_index=True)

# Time-based Revenue Projection
st.header("Multi-Year Revenue Projection")

# Generate monthly projections
months = range(1, period_months + 1)
implementation_delay = {
    "Immediate": 0,
    "3 Months": 3,
    "6 Months": 6,
    "12 Months": 12
}[implementation_timeline]

monthly_projections = []

for month in months:
    # Current revenue (with growth)
    base_growth_rate = 0.02  # 2% monthly growth
    current_monthly = current_total_revenue * (1 + base_growth_rate) ** month / 12
    
    # Optimized revenue (with ramp-up)
    if month <= implementation_delay:
        optimized_monthly = current_monthly
    else:
        ramp_up_factor = min(1.0, (month - implementation_delay) / 6)  # 6-month ramp-up
        optimization_benefit = (optimized_total_revenue - current_total_revenue) / 12
        optimized_monthly = current_monthly + (optimization_benefit * ramp_up_factor)
    
    monthly_projections.append({
        'Month': month,
        'Date': (datetime.now() + timedelta(days=month*30)).strftime('%Y-%m'),
        'Current Revenue': current_monthly,
        'Optimized Revenue': optimized_monthly,
        'Incremental Revenue': optimized_monthly - current_monthly,
        'Cumulative Benefit': sum([p['Incremental Revenue'] for p in monthly_projections]) + (optimized_monthly - current_monthly)
    })

projections_df = pd.DataFrame(monthly_projections)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Monthly Revenue Projection")
    fig_monthly = px.line(
        projections_df,
        x='Month',
        y=['Current Revenue', 'Optimized Revenue'],
        title=f"Revenue Projection - {analysis_period}",
        labels={'value': 'Monthly Revenue ($)', 'variable': 'Revenue Type'}
    )
    fig_monthly.update_layout(yaxis_tickformat='$,.0f')
    st.plotly_chart(fig_monthly, use_container_width=True)

with col2:
    st.subheader("Cumulative Revenue Benefit")
    fig_cumulative = px.area(
        projections_df,
        x='Month',
        y='Cumulative Benefit',
        title=f"Cumulative Revenue Benefit - {analysis_period}",
        labels={'Cumulative Benefit': 'Cumulative Additional Revenue ($)'}
    )
    fig_cumulative.update_layout(yaxis_tickformat='$,.0f')
    st.plotly_chart(fig_cumulative, use_container_width=True)

# ROI Analysis
st.header("Return on Investment (ROI) Analysis")

# Implementation costs
col1, col2 = st.columns(2)

with col1:
    st.subheader("Implementation Costs")
    
    # Cost assumptions
    technology_cost = st.number_input("Technology Implementation ($)", 0, 1000000, 250000, 25000)
    personnel_cost = st.number_input("Personnel/Training Cost ($)", 0, 500000, 150000, 25000)
    marketing_cost = st.number_input("Marketing/Communication ($)", 0, 200000, 75000, 10000)
    ongoing_monthly_cost = st.number_input("Monthly Ongoing Costs ($)", 0, 50000, 10000, 1000)
    
    total_upfront_cost = technology_cost + personnel_cost + marketing_cost
    total_ongoing_cost = ongoing_monthly_cost * period_months
    total_cost = total_upfront_cost + total_ongoing_cost
    
    st.write(f"**Total Upfront Cost**: ${total_upfront_cost:,.0f}")
    st.write(f"**Total Ongoing Cost ({analysis_period})**: ${total_ongoing_cost:,.0f}")
    st.write(f"**Total Investment**: ${total_cost:,.0f}")

with col2:
    st.subheader("ROI Metrics")
    
    total_benefit = projections_df['Cumulative Benefit'].iloc[-1]
    net_benefit = total_benefit - total_cost
    roi_percentage = (net_benefit / total_cost) * 100 if total_cost > 0 else 0
    payback_period = None
    
    # Calculate payback period
    cumulative_net_benefit = 0
    for i, row in projections_df.iterrows():
        cumulative_net_benefit += row['Incremental Revenue']
        if cumulative_net_benefit >= total_cost:
            payback_period = row['Month']
            break
    
    st.metric("Total Benefit", f"${total_benefit:,.0f}")
    st.metric("Net Benefit", f"${net_benefit:,.0f}")
    st.metric("ROI", f"{roi_percentage:.1f}%")
    
    if payback_period:
        st.metric("Payback Period", f"{payback_period} months")
    else:
        st.metric("Payback Period", "Beyond analysis period")

# Risk-Adjusted Analysis
st.header("Risk-Adjusted Financial Analysis")

col1, col2, col3 = st.columns(3)

# Sensitivity analysis
scenarios = {
    "Best Case": 1.5,
    "Base Case": 1.0,
    "Worst Case": 0.5
}

scenario_analysis = []

for scenario_name, scenario_mult in scenarios.items():
    scenario_benefit = total_benefit * scenario_mult
    scenario_net_benefit = scenario_benefit - total_cost
    scenario_roi = (scenario_net_benefit / total_cost) * 100 if total_cost > 0 else 0
    
    scenario_analysis.append({
        'Scenario': scenario_name,
        'Total Benefit': scenario_benefit,
        'Net Benefit': scenario_net_benefit,
        'ROI': scenario_roi
    })

scenario_df = pd.DataFrame(scenario_analysis)

with col1:
    st.subheader("Scenario Analysis")
    display_scenario_df = scenario_df.copy()
    display_scenario_df['Total Benefit'] = display_scenario_df['Total Benefit'].apply(lambda x: f"${x:,.0f}")
    display_scenario_df['Net Benefit'] = display_scenario_df['Net Benefit'].apply(lambda x: f"${x:,.0f}")
    display_scenario_df['ROI'] = display_scenario_df['ROI'].apply(lambda x: f"{x:.1f}%")
    st.dataframe(display_scenario_df, use_container_width=True, hide_index=True)

with col2:
    st.subheader("Risk Factors")
    
    risk_factors = []
    
    if roi_percentage < 20:
        risk_factors.append("Low ROI (<20%)")
    
    if payback_period and payback_period > 24:
        risk_factors.append("Long payback period (>24 months)")
    
    if total_cost > current_total_revenue * 0.1:
        risk_factors.append("High implementation cost (>10% of revenue)")
    
    market_risk_scenarios = ["Economic Uncertainty", "Stress Test"]
    if market_scenario in market_risk_scenarios:
        risk_factors.append(f"Market risk ({market_scenario})")
    
    if risk_factors:
        for risk in risk_factors:
            st.warning(f"⚠️ {risk}")
    else:
        st.success("✅ Low risk profile")

with col3:
    st.subheader("Financial Health Indicators")
    
    # Financial ratios
    revenue_growth_rate = (percentage_increase / 100) / (period_months / 12)  # Annualized
    investment_ratio = total_cost / current_total_revenue
    benefit_cost_ratio = total_benefit / total_cost if total_cost > 0 else 0
    
    st.metric("Annual Revenue Growth", f"{revenue_growth_rate:.1%}")
    st.metric("Investment-to-Revenue Ratio", f"{investment_ratio:.1%}")
    st.metric("Benefit-Cost Ratio", f"{benefit_cost_ratio:.2f}")

# Customer Lifetime Value Impact
st.header("Customer Lifetime Value (CLV) Analysis")

# Calculate CLV for each segment before and after optimization
clv_analysis = []

for segment in range(5):
    segment_data = full_df[full_df['segment'] == segment]
    
    if len(segment_data) > 0:
        # Current CLV
        avg_monthly_revenue_current = segment_data['current_revenue'].mean() / 12
        current_clv = calculate_clv(avg_monthly_revenue_current, 0.85, 24)
        
        # Optimized CLV
        avg_monthly_revenue_optimized = segment_data['optimized_revenue'].mean() * multiplier / 12
        optimized_clv = calculate_clv(avg_monthly_revenue_optimized, 0.90, 24)  # Assume better retention with optimization
        
        clv_lift = optimized_clv - current_clv
        clv_lift_pct = (clv_lift / current_clv) * 100 if current_clv > 0 else 0
        
        clv_analysis.append({
            'Segment': segment_labels[segment],
            'Customers': len(segment_data),
            'Current CLV': current_clv,
            'Optimized CLV': optimized_clv,
            'CLV Lift': clv_lift,
            'CLV Lift %': clv_lift_pct,
            'Total CLV Impact': clv_lift * len(segment_data)
        })

clv_df = pd.DataFrame(clv_analysis)

col1, col2 = st.columns(2)

with col1:
    st.subheader("CLV by Segment")
    fig_clv = px.bar(
        clv_df,
        x='Segment',
        y=['Current CLV', 'Optimized CLV'],
        title="Customer Lifetime Value by Segment",
        labels={'value': 'CLV ($)', 'variable': 'CLV Type'}
    )
    fig_clv.update_layout(yaxis_tickformat='$,.0f')
    st.plotly_chart(fig_clv, use_container_width=True)

with col2:
    st.subheader("Total CLV Impact")
    fig_clv_impact = px.bar(
        clv_df,
        x='Segment',
        y='Total CLV Impact',
        title="Total CLV Impact by Segment",
        labels={'Total CLV Impact': 'Total CLV Impact ($)'}
    )
    fig_clv_impact.update_layout(yaxis_tickformat='$,.0f')
    st.plotly_chart(fig_clv_impact, use_container_width=True)

# Key Performance Indicators Dashboard
st.header("📊 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

total_clv_impact = clv_df['Total CLV Impact'].sum()
avg_clv_lift = clv_df['CLV Lift'].mean()
customer_acquisition_cost = total_cost / len(full_df)
ltv_cac_ratio = avg_clv_lift / customer_acquisition_cost if customer_acquisition_cost > 0 else 0

with col1:
    st.metric(
        label="Total CLV Impact",
        value=f"${total_clv_impact:,.0f}",
        help="Total increase in customer lifetime value across all segments"
    )

with col2:
    st.metric(
        label="Average CLV Lift",
        value=f"${avg_clv_lift:,.0f}",
        help="Average increase in CLV per customer"
    )

with col3:
    st.metric(
        label="Customer Acquisition Cost",
        value=f"${customer_acquisition_cost:,.0f}",
        help="Implementation cost per customer"
    )

with col4:
    st.metric(
        label="LTV:CAC Ratio",
        value=f"{ltv_cac_ratio:.2f}",
        help="Lifetime value lift to customer acquisition cost ratio"
    )

# Executive Summary
st.header("📈 Executive Summary")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Financial Impact")
    st.write(f"**Investment Required**: ${total_cost:,.0f}")
    st.write(f"**Total Revenue Benefit ({analysis_period})**: ${total_benefit:,.0f}")
    st.write(f"**Net Benefit**: ${net_benefit:,.0f}")
    st.write(f"**Return on Investment**: {roi_percentage:.1f}%")
    
    if payback_period:
        st.write(f"**Payback Period**: {payback_period} months")
    else:
        st.write("**Payback Period**: Beyond analysis period")

with col2:
    st.subheader("Strategic Recommendations")
    
    if roi_percentage > 50 and payback_period and payback_period <= 18:
        st.success("**STRONG RECOMMENDATION: PROCEED**")
        st.write("High ROI with reasonable payback period. Strong business case for implementation.")
    elif roi_percentage > 25 and payback_period and payback_period <= 24:
        st.info("**RECOMMENDATION: PROCEED WITH MONITORING**")
        st.write("Positive ROI with acceptable payback. Monitor implementation closely.")
    elif roi_percentage > 0:
        st.warning("**CONDITIONAL RECOMMENDATION**")
        st.write("Positive ROI but consider risks and alternative investments.")
    else:
        st.error("**NOT RECOMMENDED**")
        st.write("Negative ROI. Reconsider strategy or implementation approach.")

# Export functionality
st.header("Export Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Export Revenue Analysis"):
        csv = segment_df.to_csv(index=False)
        st.download_button(
            label="Download Revenue Analysis",
            data=csv,
            file_name=f"revenue_analysis_{market_scenario.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("📈 Export Projections"):
        csv = projections_df.to_csv(index=False)
        st.download_button(
            label="Download Projections",
            data=csv,
            file_name=f"revenue_projections_{analysis_period.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )

with col3:
    if st.button("💰 Export ROI Analysis"):
        roi_summary = {
            'Metric': ['Total Investment', 'Total Benefit', 'Net Benefit', 'ROI %', 'Payback Period (months)'],
            'Value': [total_cost, total_benefit, net_benefit, roi_percentage, payback_period or 'N/A']
        }
        roi_df = pd.DataFrame(roi_summary)
        csv = roi_df.to_csv(index=False)
        st.download_button(
            label="Download ROI Analysis",
            data=csv,
            file_name=f"roi_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

st.markdown("---")
st.markdown("**Disclaimer**: Financial projections are based on historical data and statistical models. Actual results may vary due to market conditions, competitive factors, and implementation effectiveness. Always consult with financial advisors before making significant business decisions.")
