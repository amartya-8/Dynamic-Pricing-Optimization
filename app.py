import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.data_generator import generate_customer_data, generate_market_data
from utils.models import CustomerSegmentation, PriceOptimizer
from utils.metrics import calculate_clv, calculate_conversion_rate

# Configure page
st.set_page_config(
    page_title="Dynamic Pricing Optimization Platform",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("💰 Dynamic Pricing Optimization Platform")
st.markdown("### Machine Learning-Powered Financial Product Pricing")

# Sidebar navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("Use the pages in the sidebar to explore different aspects of the pricing optimization platform.")

# Overview section
st.header("Platform Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Customer Segments",
        value="5",
        help="Number of distinct customer segments identified"
    )

with col2:
    st.metric(
        label="Product Categories",
        value="3",
        help="Financial products: Loans, Credit Cards, Mortgages"
    )

with col3:
    st.metric(
        label="ML Models",
        value="4",
        help="Segmentation, Pricing, Conversion, CLV models"
    )

with col4:
    st.metric(
        label="A/B Tests",
        value="Active",
        help="Real-time A/B testing framework"
    )

# Generate sample data for overview
@st.cache_data
def load_overview_data():
    np.random.seed(42)
    customers = generate_customer_data(1000)
    market_data = generate_market_data()
    return customers, market_data

customers_df, market_df = load_overview_data()

# Key Insights Section
st.header("Key Business Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Distribution by Segment")
    
    # Simple segmentation for overview
    segmentation_model = CustomerSegmentation()
    numeric_features = ['income', 'credit_score', 'debt_to_income', 'age', 'savings_rate']
    segments = segmentation_model.fit_predict(customers_df[numeric_features])
    customers_df['segment'] = segments
    
    segment_counts = pd.Series(segments).value_counts().sort_index()
    segment_labels = ['Conservative', 'Balanced', 'Growth-Oriented', 'High-Risk', 'Premium']
    
    fig_pie = px.pie(
        values=segment_counts.values,
        names=[segment_labels[i] for i in segment_counts.index],
        title="Customer Segmentation Distribution"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("Revenue Potential by Product")
    
    # Calculate revenue potential
    product_revenue = {
        'Personal Loans': np.random.normal(15000, 3000, 100).clip(5000, 25000).mean(),
        'Credit Cards': np.random.normal(8000, 2000, 100).clip(2000, 15000).mean(),
        'Mortgages': np.random.normal(250000, 50000, 100).clip(150000, 400000).mean()
    }
    
    fig_bar = px.bar(
        x=list(product_revenue.keys()),
        y=list(product_revenue.values()),
        title="Average Revenue Potential by Product",
        labels={'x': 'Product Type', 'y': 'Average Revenue ($)'}
    )
    fig_bar.update_layout(yaxis_tickformat='$,.0f')
    st.plotly_chart(fig_bar, use_container_width=True)

# Market Trends Section
st.header("Market Intelligence")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Interest Rate Trends")
    fig_line = px.line(
        market_df,
        x='date',
        y='interest_rate',
        title="Federal Interest Rate Trend",
        labels={'interest_rate': 'Interest Rate (%)', 'date': 'Date'}
    )
    st.plotly_chart(fig_line, use_container_width=True)

with col2:
    st.subheader("Market Volatility Index")
    fig_volatility = px.line(
        market_df,
        x='date',
        y='volatility',
        title="Market Volatility Index",
        labels={'volatility': 'Volatility Index', 'date': 'Date'}
    )
    st.plotly_chart(fig_volatility, use_container_width=True)

# Technology Stack
st.header("Technology Stack")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Machine Learning")
    st.markdown("""
    - **Scikit-learn**: Customer segmentation (K-Means)
    - **XGBoost**: Price optimization models
    - **Statistical Testing**: A/B test validation
    """)

with col2:
    st.subheader("Data Processing")
    st.markdown("""
    - **Pandas**: Data manipulation and analysis
    - **NumPy**: Numerical computations
    - **SciPy**: Statistical analysis
    """)

with col3:
    st.subheader("Visualization")
    st.markdown("""
    - **Plotly**: Interactive charts and dashboards
    - **Streamlit**: Web application framework
    - **Matplotlib/Seaborn**: Statistical plots
    """)

# Feature Highlights
st.header("Platform Features")

features = [
    {
        "title": "🎯 Customer Segmentation",
        "description": "Advanced K-Means clustering to identify distinct customer segments based on risk profiles, demographics, and financial behavior."
    },
    {
        "title": "📈 Price Optimization", 
        "description": "Gradient Boosting models to determine optimal pricing strategies for different financial products and customer segments."
    },
    {
        "title": "🧪 A/B Testing Framework",
        "description": "Statistical testing framework to validate pricing strategies with confidence intervals and significance testing."
    },
    {
        "title": "💹 Revenue Impact Analysis",
        "description": "Comprehensive ROI calculator showing potential business outcomes and revenue optimization opportunities."
    }
]

for i, feature in enumerate(features):
    with st.container():
        st.subheader(feature["title"])
        st.write(feature["description"])
        if i < len(features) - 1:
            st.divider()

# Footer
st.markdown("---")
st.markdown(
    """
    **Dynamic Pricing Optimization Platform** - Built with Streamlit, scikit-learn, and XGBoost  
    Navigate to the different pages using the sidebar to explore each component in detail.
    """
)
