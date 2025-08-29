import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

from utils.data_generator import generate_customer_data
from utils.models import CustomerSegmentation
from utils.metrics import calculate_clv

st.set_page_config(page_title="Customer Segmentation", page_icon="🎯", layout="wide")

st.title("🎯 Customer Segmentation Analysis")
st.markdown("### K-Means Clustering for Financial Product Customers")

# Sidebar controls
st.sidebar.header("Segmentation Parameters")
n_customers = st.sidebar.slider("Number of Customers", 500, 5000, 2000, 500)
n_clusters = st.sidebar.slider("Number of Segments", 3, 8, 5, 1)
random_seed = st.sidebar.number_input("Random Seed", 0, 999, 42)

# Generate customer data
@st.cache_data
def load_customer_data(n_customers, random_seed):
    return generate_customer_data(n_customers, random_seed)

customers_df = load_customer_data(n_customers, random_seed)

# Feature selection for clustering
st.header("Feature Engineering")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Available Features")
    features_for_clustering = st.multiselect(
        "Select features for clustering:",
        options=['age', 'income', 'credit_score', 'debt_to_income', 'employment_years', 
                'existing_products', 'transaction_frequency', 'savings_rate'],
        default=['income', 'credit_score', 'debt_to_income', 'age', 'savings_rate']
    )

with col2:
    st.subheader("Data Preview")
    st.dataframe(customers_df[features_for_clustering].head(), use_container_width=True)

if len(features_for_clustering) >= 2:
    # Perform segmentation
    segmentation_model = CustomerSegmentation(n_clusters=n_clusters, random_state=random_seed)
    
    # Fit and predict
    segments = segmentation_model.fit_predict(customers_df[features_for_clustering])
    customers_df['segment'] = segments
    
    # Calculate segment statistics
    segment_stats = customers_df.groupby('segment').agg({
        'income': ['mean', 'std'],
        'credit_score': ['mean', 'std'],
        'debt_to_income': ['mean', 'std'],
        'age': ['mean', 'std'],
        'savings_rate': ['mean', 'std']
    }).round(2)
    
    # Display results
    st.header("Segmentation Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Segment Distribution")
        segment_counts = pd.Series(segments).value_counts().sort_index()
        
        fig_pie = px.pie(
            values=segment_counts.values,
            names=[f'Segment {i}' for i in segment_counts.index],
            title=f"Distribution of {n_clusters} Customer Segments"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("Segment Characteristics")
        
        # Create segment profiles
        segment_profiles = []
        for i in range(n_clusters):
            segment_data = customers_df[customers_df['segment'] == i]
            avg_income = segment_data['income'].mean()
            avg_credit = segment_data['credit_score'].mean()
            avg_age = segment_data['age'].mean()
            count = len(segment_data)
            
            # Determine segment personality
            if avg_income > 80000 and avg_credit > 750:
                profile = "Premium"
            elif avg_income > 60000 and avg_credit > 700:
                profile = "Growth-Oriented"
            elif avg_credit > 650:
                profile = "Balanced"
            elif avg_income < 40000:
                profile = "Conservative"
            else:
                profile = "High-Risk"
            
            segment_profiles.append({
                'Segment': i,
                'Profile': profile,
                'Count': count,
                'Avg Income': f"${avg_income:,.0f}",
                'Avg Credit Score': f"{avg_credit:.0f}",
                'Avg Age': f"{avg_age:.0f}"
            })
        
        profiles_df = pd.DataFrame(segment_profiles)
        st.dataframe(profiles_df, use_container_width=True, hide_index=True)
    
    # Detailed Analysis
    st.header("Detailed Segment Analysis")
    
    # 2D Scatter plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Income vs Credit Score")
        fig_scatter1 = px.scatter(
            customers_df,
            x='income',
            y='credit_score',
            color='segment',
            title="Customer Segments: Income vs Credit Score",
            labels={'income': 'Annual Income ($)', 'credit_score': 'Credit Score'},
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        st.plotly_chart(fig_scatter1, use_container_width=True)
    
    with col2:
        st.subheader("Age vs Debt-to-Income")
        fig_scatter2 = px.scatter(
            customers_df,
            x='age',
            y='debt_to_income',
            color='segment',
            title="Customer Segments: Age vs Debt-to-Income Ratio",
            labels={'age': 'Age (years)', 'debt_to_income': 'Debt-to-Income Ratio'},
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        st.plotly_chart(fig_scatter2, use_container_width=True)
    
    # Radar chart for segment comparison
    st.subheader("Segment Comparison Radar Chart")
    
    # Prepare data for radar chart
    radar_metrics = ['income', 'credit_score', 'age', 'debt_to_income', 'savings_rate']
    
    # Normalize metrics for radar chart (0-1 scale)
    scaler = StandardScaler()
    normalized_data = customers_df[radar_metrics].copy()
    
    # Handle debt_to_income (lower is better)
    normalized_data['debt_to_income'] = 1 - (normalized_data['debt_to_income'] / normalized_data['debt_to_income'].max())
    
    # Normalize other metrics
    for metric in ['income', 'credit_score', 'age', 'savings_rate']:
        normalized_data[metric] = (normalized_data[metric] - normalized_data[metric].min()) / (normalized_data[metric].max() - normalized_data[metric].min())
    
    normalized_data['segment'] = customers_df['segment']
    
    # Create radar chart
    fig_radar = go.Figure()
    
    for segment in range(n_clusters):
        segment_data = normalized_data[normalized_data['segment'] == segment]
        avg_values = segment_data[radar_metrics].mean()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=avg_values.values.tolist() + [avg_values.values[0]],  # Close the polygon
            theta=radar_metrics + [radar_metrics[0]],
            fill='toself',
            name=f'Segment {segment}',
            line_color=px.colors.qualitative.Set1[segment % len(px.colors.qualitative.Set1)]
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="Normalized Segment Profiles"
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Customer Lifetime Value Analysis
    st.header("Customer Lifetime Value by Segment")
    
    # Calculate CLV for each customer
    customers_df['clv'] = customers_df.apply(lambda row: calculate_clv(
        monthly_revenue=row['income'] * 0.02,  # Assume 2% of income as monthly revenue
        retention_rate=0.85 if row['credit_score'] > 700 else 0.75,
        months=24
    ), axis=1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CLV by segment
        clv_by_segment = customers_df.groupby('segment')['clv'].agg(['mean', 'median', 'std']).round(2)
        clv_by_segment.columns = ['Mean CLV', 'Median CLV', 'CLV Std Dev']
        clv_by_segment['Segment Profile'] = [profiles_df.iloc[i]['Profile'] for i in clv_by_segment.index]
        
        st.subheader("CLV Statistics by Segment")
        st.dataframe(clv_by_segment, use_container_width=True)
    
    with col2:
        # CLV distribution plot
        fig_clv = px.box(
            customers_df,
            x='segment',
            y='clv',
            title="Customer Lifetime Value Distribution by Segment",
            labels={'segment': 'Segment', 'clv': 'Customer Lifetime Value ($)'}
        )
        st.plotly_chart(fig_clv, use_container_width=True)
    
    # Actionable Insights
    st.header("🎯 Actionable Business Insights")
    
    # Calculate insights
    highest_clv_segment = clv_by_segment['Mean CLV'].idxmax()
    lowest_clv_segment = clv_by_segment['Mean CLV'].idxmin()
    total_customers = len(customers_df)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Highest Value Segment",
            value=f"Segment {highest_clv_segment}",
            delta=f"${clv_by_segment.loc[highest_clv_segment, 'Mean CLV']:,.0f} avg CLV"
        )
    
    with col2:
        st.metric(
            label="Total Customers Analyzed",
            value=f"{total_customers:,}",
            delta=f"{n_clusters} segments identified"
        )
    
    with col3:
        premium_segment_size = len(customers_df[customers_df['segment'] == highest_clv_segment])
        premium_percentage = (premium_segment_size / total_customers) * 100
        st.metric(
            label="Premium Segment Size",
            value=f"{premium_percentage:.1f}%",
            delta=f"{premium_segment_size} customers"
        )
    
    # Strategic recommendations
    st.subheader("Strategic Recommendations")
    
    recommendations = []
    
    for i, row in clv_by_segment.iterrows():
        profile = row['Segment Profile']
        mean_clv = row['Mean CLV']
        segment_size = len(customers_df[customers_df['segment'] == i])
        
        if profile == "Premium":
            rec = f"**Segment {i} ({profile})**: Focus on retention and upselling. High CLV (${mean_clv:,.0f}) customers."
        elif profile == "Growth-Oriented":
            rec = f"**Segment {i} ({profile})**: Target for product expansion and premium offerings. Strong growth potential."
        elif profile == "Conservative":
            rec = f"**Segment {i} ({profile})**: Offer low-risk, stable products. Focus on building trust and gradual engagement."
        elif profile == "High-Risk":
            rec = f"**Segment {i} ({profile})**: Implement risk mitigation strategies. Require additional verification."
        else:
            rec = f"**Segment {i} ({profile})**: Standard marketing approach. Monitor for segment migration opportunities."
        
        recommendations.append(rec + f" ({segment_size} customers)")
    
    for rec in recommendations:
        st.write(f"• {rec}")
    
    # Export functionality
    st.header("Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export Segment Data"):
            csv = customers_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"customer_segments_{n_clusters}_clusters.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📈 Export CLV Analysis"):
            clv_csv = clv_by_segment.to_csv()
            st.download_button(
                label="Download CLV Analysis",
                data=clv_csv,
                file_name=f"clv_analysis_{n_clusters}_segments.csv",
                mime="text/csv"
            )

else:
    st.warning("Please select at least 2 features for clustering analysis.")

# Model Performance Metrics
if len(features_for_clustering) >= 2:
    st.header("Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Silhouette score
        silhouette_score = segmentation_model.silhouette_score_
        st.metric(
            label="Silhouette Score",
            value=f"{silhouette_score:.3f}",
            help="Measures how similar objects are to their own cluster vs other clusters. Range: -1 to 1 (higher is better)"
        )
    
    with col2:
        # Inertia (within-cluster sum of squares)
        inertia = segmentation_model.inertia_
        st.metric(
            label="Inertia",
            value=f"{inertia:,.0f}",
            help="Sum of squared distances of samples to their closest cluster centers (lower is better)"
        )
    
    with col3:
        # Calinski-Harabasz score
        ch_score = segmentation_model.calinski_harabasz_score_
        st.metric(
            label="Calinski-Harabasz Score",
            value=f"{ch_score:.0f}",
            help="Ratio of sum of between-clusters dispersion and within-cluster dispersion (higher is better)"
        )
