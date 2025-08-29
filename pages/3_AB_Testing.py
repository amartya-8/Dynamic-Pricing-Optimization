import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from utils.data_generator import generate_ab_test_data
from utils.metrics import calculate_statistical_power, calculate_conversion_rate

st.set_page_config(page_title="A/B Testing", page_icon="🧪", layout="wide")

st.title("🧪 A/B Testing Framework")
st.markdown("### Statistical Testing for Pricing Strategy Validation")

# Sidebar controls
st.sidebar.header("A/B Test Configuration")

test_duration = st.sidebar.slider("Test Duration (days)", 7, 90, 30, 7)
control_price = st.sidebar.number_input("Control Price ($)", 1000, 100000, 15000, 1000)
treatment_price = st.sidebar.number_input("Treatment Price ($)", 1000, 100000, 17000, 1000)
sample_size_per_group = st.sidebar.slider("Sample Size per Group", 100, 5000, 1000, 100)
significance_level = st.sidebar.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01)
product_type = st.sidebar.selectbox("Product Type", ["Personal Loan", "Credit Card", "Mortgage"])

# Calculate price difference
price_difference = ((treatment_price - control_price) / control_price) * 100

st.header(f"A/B Test: {product_type} Pricing Strategy")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Control Price",
        value=f"${control_price:,}",
        help="Current/baseline pricing"
    )

with col2:
    st.metric(
        label="Treatment Price",
        value=f"${treatment_price:,}",
        delta=f"{price_difference:+.1f}%",
        help="New pricing strategy being tested"
    )

with col3:
    st.metric(
        label="Sample Size",
        value=f"{sample_size_per_group * 2:,}",
        help="Total customers in the experiment"
    )

with col4:
    st.metric(
        label="Test Duration",
        value=f"{test_duration} days",
        help="Duration of the A/B test"
    )

# Generate A/B test data
@st.cache_data
def load_ab_test_data(sample_size, control_price, treatment_price, test_duration):
    return generate_ab_test_data(sample_size, control_price, treatment_price, test_duration)

ab_test_data = load_ab_test_data(sample_size_per_group, control_price, treatment_price, test_duration)

# Split data by group
control_data = ab_test_data[ab_test_data['group'] == 'control']
treatment_data = ab_test_data[ab_test_data['group'] == 'treatment']

# Calculate key metrics
control_conversion = control_data['converted'].mean()
treatment_conversion = treatment_data['converted'].mean()
control_revenue = control_data['revenue'].mean()
treatment_revenue = treatment_data['revenue'].mean()

# Statistical significance tests
st.header("Statistical Analysis")

# Conversion rate test (Chi-square test)
control_conversions = control_data['converted'].sum()
control_total = len(control_data)
treatment_conversions = treatment_data['converted'].sum()
treatment_total = len(treatment_data)

# Contingency table for chi-square test
contingency_table = np.array([
    [control_conversions, control_total - control_conversions],
    [treatment_conversions, treatment_total - treatment_conversions]
])

chi2, p_value_chi2, dof, expected = chi2_contingency(contingency_table)

# Revenue test (t-test)
t_stat, p_value_ttest = ttest_ind(treatment_data['revenue'], control_data['revenue'])

# Effect size (Cohen's d) for revenue
pooled_std = np.sqrt(((len(control_data) - 1) * control_data['revenue'].var() + 
                     (len(treatment_data) - 1) * treatment_data['revenue'].var()) / 
                     (len(control_data) + len(treatment_data) - 2))
cohens_d = (treatment_revenue - control_revenue) / pooled_std

col1, col2 = st.columns(2)

with col1:
    st.subheader("Conversion Rate Analysis")
    
    # Conversion rate metrics
    conversion_lift = ((treatment_conversion - control_conversion) / control_conversion) * 100
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.metric(
            label="Control Conversion",
            value=f"{control_conversion:.2%}",
            help=f"{control_conversions} out of {control_total} customers"
        )
    
    with col_b:
        st.metric(
            label="Treatment Conversion", 
            value=f"{treatment_conversion:.2%}",
            delta=f"{conversion_lift:+.1f}%",
            help=f"{treatment_conversions} out of {treatment_total} customers"
        )
    
    # Statistical significance for conversion
    is_conversion_significant = p_value_chi2 < significance_level
    
    st.write(f"**Chi-square test p-value**: {p_value_chi2:.4f}")
    
    if is_conversion_significant:
        st.success(f"✅ **Statistically Significant** (p < {significance_level})")
        st.write("The difference in conversion rates is statistically significant.")
    else:
        st.warning(f"⚠️ **Not Statistically Significant** (p ≥ {significance_level})")
        st.write("The difference in conversion rates is not statistically significant.")

with col2:
    st.subheader("Revenue Analysis")
    
    # Revenue metrics
    revenue_lift = ((treatment_revenue - control_revenue) / control_revenue) * 100
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.metric(
            label="Control Revenue",
            value=f"${control_revenue:,.0f}",
            help="Average revenue per customer"
        )
    
    with col_b:
        st.metric(
            label="Treatment Revenue",
            value=f"${treatment_revenue:,.0f}",
            delta=f"{revenue_lift:+.1f}%",
            help="Average revenue per customer"
        )
    
    # Statistical significance for revenue
    is_revenue_significant = p_value_ttest < significance_level
    
    st.write(f"**T-test p-value**: {p_value_ttest:.4f}")
    st.write(f"**Effect Size (Cohen's d)**: {cohens_d:.3f}")
    
    if is_revenue_significant:
        st.success(f"✅ **Statistically Significant** (p < {significance_level})")
        st.write("The difference in revenue is statistically significant.")
    else:
        st.warning(f"⚠️ **Not Statistically Significant** (p ≥ {significance_level})")
        st.write("The difference in revenue is not statistically significant.")

# Confidence intervals
st.header("Confidence Intervals")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Conversion Rate Confidence Intervals")
    
    # 95% CI for conversion rates
    z_score = stats.norm.ppf(1 - significance_level/2)
    
    control_ci_lower = control_conversion - z_score * np.sqrt(control_conversion * (1 - control_conversion) / control_total)
    control_ci_upper = control_conversion + z_score * np.sqrt(control_conversion * (1 - control_conversion) / control_total)
    
    treatment_ci_lower = treatment_conversion - z_score * np.sqrt(treatment_conversion * (1 - treatment_conversion) / treatment_total)
    treatment_ci_upper = treatment_conversion + z_score * np.sqrt(treatment_conversion * (1 - treatment_conversion) / treatment_total)
    
    fig_ci_conv = go.Figure()
    
    fig_ci_conv.add_trace(go.Scatter(
        x=['Control', 'Treatment'],
        y=[control_conversion, treatment_conversion],
        mode='markers',
        marker=dict(size=10, color=['blue', 'red']),
        name='Conversion Rate'
    ))
    
    fig_ci_conv.add_trace(go.Scatter(
        x=['Control', 'Control', 'Treatment', 'Treatment'],
        y=[control_ci_lower, control_ci_upper, treatment_ci_lower, treatment_ci_upper],
        mode='markers',
        marker=dict(size=5, symbol='line-ns', color=['blue', 'blue', 'red', 'red']),
        name='95% CI',
        showlegend=False
    ))
    
    for i, (group, lower, upper, rate) in enumerate([('Control', control_ci_lower, control_ci_upper, control_conversion),
                                                    ('Treatment', treatment_ci_lower, treatment_ci_upper, treatment_conversion)]):
        fig_ci_conv.add_shape(
            type="line",
            x0=i, y0=lower,
            x1=i, y1=upper,
            line=dict(color="blue" if group == 'Control' else "red", width=3)
        )
    
    fig_ci_conv.update_layout(
        title="Conversion Rate with 95% Confidence Intervals",
        yaxis_title="Conversion Rate",
        yaxis=dict(tickformat='.2%')
    )
    
    st.plotly_chart(fig_ci_conv, use_container_width=True)
    
    st.write(f"Control 95% CI: [{control_ci_lower:.3f}, {control_ci_upper:.3f}]")
    st.write(f"Treatment 95% CI: [{treatment_ci_lower:.3f}, {treatment_ci_upper:.3f}]")

with col2:
    st.subheader("Revenue Confidence Intervals")
    
    # 95% CI for revenue
    control_revenue_std = control_data['revenue'].std()
    treatment_revenue_std = treatment_data['revenue'].std()
    
    control_revenue_ci_lower = control_revenue - z_score * (control_revenue_std / np.sqrt(control_total))
    control_revenue_ci_upper = control_revenue + z_score * (control_revenue_std / np.sqrt(control_total))
    
    treatment_revenue_ci_lower = treatment_revenue - z_score * (treatment_revenue_std / np.sqrt(treatment_total))
    treatment_revenue_ci_upper = treatment_revenue + z_score * (treatment_revenue_std / np.sqrt(treatment_total))
    
    fig_ci_rev = go.Figure()
    
    fig_ci_rev.add_trace(go.Scatter(
        x=['Control', 'Treatment'],
        y=[control_revenue, treatment_revenue],
        mode='markers',
        marker=dict(size=10, color=['blue', 'red']),
        name='Average Revenue'
    ))
    
    for i, (group, lower, upper, revenue) in enumerate([('Control', control_revenue_ci_lower, control_revenue_ci_upper, control_revenue),
                                                       ('Treatment', treatment_revenue_ci_lower, treatment_revenue_ci_upper, treatment_revenue)]):
        fig_ci_rev.add_shape(
            type="line",
            x0=i, y0=lower,
            x1=i, y1=upper,
            line=dict(color="blue" if group == 'Control' else "red", width=3)
        )
    
    fig_ci_rev.update_layout(
        title="Revenue with 95% Confidence Intervals",
        yaxis_title="Revenue per Customer ($)",
        yaxis=dict(tickformat='$,.0f')
    )
    
    st.plotly_chart(fig_ci_rev, use_container_width=True)
    
    st.write(f"Control 95% CI: [${control_revenue_ci_lower:,.0f}, ${control_revenue_ci_upper:,.0f}]")
    st.write(f"Treatment 95% CI: [${treatment_revenue_ci_lower:,.0f}, ${treatment_revenue_ci_upper:,.0f}]")

# Time series analysis
st.header("Performance Over Time")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Daily Conversion Rates")
    
    # Aggregate by day
    daily_stats = ab_test_data.groupby(['date', 'group']).agg({
        'converted': ['sum', 'count']
    }).reset_index()
    
    daily_stats.columns = ['date', 'group', 'conversions', 'total']
    daily_stats['conversion_rate'] = daily_stats['conversions'] / daily_stats['total']
    
    fig_daily_conv = px.line(
        daily_stats,
        x='date',
        y='conversion_rate',
        color='group',
        title="Daily Conversion Rates",
        labels={'conversion_rate': 'Conversion Rate', 'date': 'Date'}
    )
    fig_daily_conv.update_yaxis(tickformat='.2%')
    st.plotly_chart(fig_daily_conv, use_container_width=True)

with col2:
    st.subheader("Daily Revenue per Customer")
    
    daily_revenue = ab_test_data.groupby(['date', 'group'])['revenue'].mean().reset_index()
    
    fig_daily_rev = px.line(
        daily_revenue,
        x='date',
        y='revenue',
        color='group',
        title="Daily Average Revenue per Customer",
        labels={'revenue': 'Revenue per Customer ($)', 'date': 'Date'}
    )
    fig_daily_rev.update_yaxis(tickformat='$,.0f')
    st.plotly_chart(fig_daily_rev, use_container_width=True)

# Power analysis and sample size calculations
st.header("Statistical Power Analysis")

col1, col2, col3 = st.columns(3)

# Calculate achieved power
observed_effect_size = abs(treatment_conversion - control_conversion)
achieved_power = calculate_statistical_power(
    effect_size=observed_effect_size,
    sample_size=sample_size_per_group,
    alpha=significance_level
)

with col1:
    st.metric(
        label="Statistical Power",
        value=f"{achieved_power:.2%}",
        help="Probability of detecting the effect if it truly exists"
    )

with col2:
    st.metric(
        label="Effect Size",
        value=f"{observed_effect_size:.4f}",
        help="Magnitude of the difference between groups"
    )

with col3:
    # Calculate minimum detectable effect
    min_effect = 2 * z_score * np.sqrt(2 * control_conversion * (1 - control_conversion) / sample_size_per_group)
    st.metric(
        label="Min Detectable Effect",
        value=f"{min_effect:.4f}",
        help="Smallest effect size detectable with 80% power"
    )

# Sample size recommendation
st.subheader("Sample Size Recommendations")

target_effects = [0.01, 0.02, 0.03, 0.05]  # 1%, 2%, 3%, 5% effect sizes
sample_sizes = []

for effect in target_effects:
    # Calculate required sample size for 80% power
    z_alpha = stats.norm.ppf(1 - significance_level/2)
    z_beta = stats.norm.ppf(0.8)  # 80% power
    
    p1 = control_conversion
    p2 = p1 + effect
    p_pooled = (p1 + p2) / 2
    
    n = (z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled)) + 
         z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))**2 / (p2 - p1)**2
    
    sample_sizes.append(int(np.ceil(n)))

sample_size_df = pd.DataFrame({
    'Target Effect Size': [f"{e:.1%}" for e in target_effects],
    'Required Sample Size per Group': sample_sizes,
    'Total Sample Size': [s * 2 for s in sample_sizes],
    'Test Duration (approx days)': [max(7, s // 50) for s in sample_sizes]  # Assuming ~50 customers per day
})

st.dataframe(sample_size_df, use_container_width=True, hide_index=True)

# Business Impact Analysis
st.header("💰 Business Impact Analysis")

# Calculate total business impact
total_customers_treated = treatment_total
revenue_per_customer_lift = treatment_revenue - control_revenue
total_additional_revenue = total_customers_treated * revenue_per_customer_lift

# Projected annual impact
annual_customers = total_customers_treated * (365 / test_duration)
annual_additional_revenue = annual_customers * revenue_per_customer_lift

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Revenue Lift per Customer",
        value=f"${revenue_per_customer_lift:,.0f}",
        delta=f"{revenue_lift:+.1f}%"
    )

with col2:
    st.metric(
        label="Test Period Additional Revenue",
        value=f"${total_additional_revenue:,.0f}",
        help=f"Additional revenue from {total_customers_treated} customers"
    )

with col3:
    st.metric(
        label="Projected Annual Impact",
        value=f"${annual_additional_revenue:,.0f}",
        help="Projected additional annual revenue if treatment is implemented"
    )

# Risk assessment
st.subheader("Risk Assessment")

risk_factors = []

if not is_conversion_significant:
    risk_factors.append("⚠️ **Conversion difference not statistically significant** - Risk of false positive")

if not is_revenue_significant:
    risk_factors.append("⚠️ **Revenue difference not statistically significant** - Risk of false positive")

if achieved_power < 0.8:
    risk_factors.append(f"⚠️ **Low statistical power ({achieved_power:.1%})** - High risk of missing true effects")

if abs(price_difference) > 20:
    risk_factors.append("⚠️ **Large price change (>20%)** - May impact customer satisfaction")

if cohens_d < 0.2:
    risk_factors.append("⚠️ **Small effect size** - Practical significance may be limited")

if risk_factors:
    st.warning("**Risk Factors Identified:**")
    for risk in risk_factors:
        st.write(risk)
else:
    st.success("✅ **Low Risk** - Test shows strong statistical evidence with acceptable risk levels")

# Decision framework
st.header("🎯 Decision Framework")

decision_criteria = {
    'Statistical Significance (Conversion)': '✅ Yes' if is_conversion_significant else '❌ No',
    'Statistical Significance (Revenue)': '✅ Yes' if is_revenue_significant else '❌ No',
    'Adequate Statistical Power': '✅ Yes' if achieved_power >= 0.8 else '❌ No',
    'Meaningful Effect Size': '✅ Yes' if abs(cohens_d) >= 0.2 else '❌ No',
    'Business Impact': '✅ Positive' if revenue_per_customer_lift > 0 else '❌ Negative'
}

decision_df = pd.DataFrame(list(decision_criteria.items()), columns=['Criteria', 'Status'])
st.dataframe(decision_df, use_container_width=True, hide_index=True)

# Final recommendation
positive_criteria = sum(1 for status in decision_criteria.values() if '✅' in status)
total_criteria = len(decision_criteria)

if positive_criteria >= 4:
    st.success(f"**RECOMMENDATION: IMPLEMENT** ({positive_criteria}/{total_criteria} criteria met)")
    st.write("The treatment pricing strategy shows strong evidence of positive impact.")
elif positive_criteria >= 3:
    st.warning(f"**RECOMMENDATION: CONSIDER WITH CAUTION** ({positive_criteria}/{total_criteria} criteria met)")
    st.write("Mixed results - consider additional testing or gradual rollout.")
else:
    st.error(f"**RECOMMENDATION: DO NOT IMPLEMENT** ({positive_criteria}/{total_criteria} criteria met)")
    st.write("Insufficient evidence to support the new pricing strategy.")

# Export functionality
st.header("Export Test Results")

col1, col2 = st.columns(2)

with col1:
    if st.button("📊 Export Test Summary"):
        summary_data = {
            'Metric': ['Control Conversion', 'Treatment Conversion', 'Conversion Lift', 
                      'Control Revenue', 'Treatment Revenue', 'Revenue Lift',
                      'P-value (Conversion)', 'P-value (Revenue)', 'Statistical Power'],
            'Value': [f"{control_conversion:.2%}", f"{treatment_conversion:.2%}", f"{conversion_lift:+.1f}%",
                     f"${control_revenue:,.0f}", f"${treatment_revenue:,.0f}", f"{revenue_lift:+.1f}%",
                     f"{p_value_chi2:.4f}", f"{p_value_ttest:.4f}", f"{achieved_power:.2%}"]
        }
        summary_df = pd.DataFrame(summary_data)
        csv = summary_df.to_csv(index=False)
        st.download_button(
            label="Download Test Summary",
            data=csv,
            file_name=f"ab_test_summary_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("📈 Export Raw Data"):
        csv = ab_test_data.to_csv(index=False)
        st.download_button(
            label="Download Raw Test Data",
            data=csv,
            file_name=f"ab_test_raw_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

st.markdown("---")
st.markdown("**Note**: A/B testing results should be interpreted in the context of business objectives and external factors. Always validate results through multiple tests and consider practical significance alongside statistical significance.")
