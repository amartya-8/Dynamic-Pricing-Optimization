import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.data_generator import generate_customer_data, generate_pricing_data
from utils.models import PriceOptimizer, CustomerSegmentation
from utils.metrics import calculate_conversion_rate

st.set_page_config(page_title="Price Optimization", page_icon="📈", layout="wide")

st.title("📈 Price Optimization Engine")
st.markdown("### Gradient Boosting for Optimal Financial Product Pricing")

# Sidebar controls
st.sidebar.header("Optimization Parameters")
product_type = st.sidebar.selectbox(
    "Product Type",
    ["Personal Loan", "Credit Card", "Mortgage"]
)

market_conditions = st.sidebar.selectbox(
    "Market Conditions",
    ["Stable", "Rising Interest Rates", "Economic Uncertainty", "Competitive Market"]
)

optimization_goal = st.sidebar.selectbox(
    "Optimization Goal",
    ["Maximize Revenue", "Maximize Conversion", "Maximize Profit", "Balance Risk-Return"]
)

# Generate data
@st.cache_data
def load_pricing_data():
    customers = generate_customer_data(2000, 42)
    pricing_data = generate_pricing_data(customers, product_type)
    return customers, pricing_data

customers_df, pricing_df = load_pricing_data()

# Merge customer and pricing data
full_df = customers_df.merge(pricing_df, on='customer_id', how='inner')

# Customer segmentation for pricing
segmentation_model = CustomerSegmentation(n_clusters=5, random_state=42)
segments = segmentation_model.fit_predict(customers_df[['income', 'credit_score', 'debt_to_income', 'age', 'savings_rate']])
full_df['segment'] = segments

st.header(f"Price Optimization for {product_type}")

# Display current pricing overview
col1, col2, col3, col4 = st.columns(4)

with col1:
    avg_price = full_df['current_price'].mean()
    st.metric(
        label="Average Current Price",
        value=f"${avg_price:,.0f}",
        help="Current average pricing across all customer segments"
    )

with col2:
    avg_conversion = full_df['conversion_rate'].mean() * 100
    st.metric(
        label="Average Conversion Rate",
        value=f"{avg_conversion:.1f}%",
        help="Current conversion rate across all customers"
    )

with col3:
    total_revenue = (full_df['current_price'] * full_df['conversion_rate']).sum()
    st.metric(
        label="Total Revenue",
        value=f"${total_revenue:,.0f}",
        help="Current total revenue from pricing strategy"
    )

with col4:
    customer_count = len(full_df)
    st.metric(
        label="Total Customers",
        value=f"{customer_count:,}",
        help="Number of customers in pricing analysis"
    )

# Price optimization model
st.header("Price Optimization Model")

# Feature engineering for price optimization
feature_columns = ['income', 'credit_score', 'debt_to_income', 'age', 'employment_years', 
                  'existing_products', 'transaction_frequency', 'savings_rate', 'segment']

# Encode categorical variables
le = LabelEncoder()
full_df_encoded = full_df.copy()
for col in ['risk_category', 'region']:
    if col in full_df_encoded.columns:
        full_df_encoded[col] = le.fit_transform(full_df_encoded[col])
        feature_columns.append(col)

# Market condition encoding
market_encoding = {
    "Stable": 0,
    "Rising Interest Rates": 1, 
    "Economic Uncertainty": 2,
    "Competitive Market": 3
}
full_df_encoded['market_condition'] = market_encoding[market_conditions]
feature_columns.append('market_condition')

# Train price optimization model
X = full_df_encoded[feature_columns]
y_price = full_df_encoded['optimal_price']
y_conversion = full_df_encoded['conversion_rate']

# Split data
X_train, X_test, y_price_train, y_price_test, y_conversion_train, y_conversion_test = train_test_split(
    X, y_price, y_conversion, test_size=0.2, random_state=42
)

# Train models
@st.cache_resource
def train_pricing_models(X_train, y_price_train, y_conversion_train):
    # Price prediction model
    price_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    price_model.fit(X_train, y_price_train)
    
    # Conversion rate prediction model
    conversion_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    conversion_model.fit(X_train, y_conversion_train)
    
    return price_model, conversion_model

price_model, conversion_model = train_pricing_models(X_train, y_price_train, y_conversion_train)

# Model performance
col1, col2 = st.columns(2)

with col1:
    st.subheader("Price Prediction Model Performance")
    
    price_pred = price_model.predict(X_test)
    price_mae = mean_absolute_error(y_price_test, price_pred)
    price_r2 = r2_score(y_price_test, price_pred)
    
    st.metric("Mean Absolute Error", f"${price_mae:,.0f}")
    st.metric("R² Score", f"{price_r2:.3f}")
    
    # Actual vs Predicted plot
    fig_price_pred = px.scatter(
        x=y_price_test, 
        y=price_pred,
        title="Actual vs Predicted Prices",
        labels={'x': 'Actual Price ($)', 'y': 'Predicted Price ($)'}
    )
    fig_price_pred.add_trace(
        go.Scatter(x=[y_price_test.min(), y_price_test.max()], 
                  y=[y_price_test.min(), y_price_test.max()],
                  mode='lines', name='Perfect Prediction', line=dict(dash='dash'))
    )
    st.plotly_chart(fig_price_pred, use_container_width=True)

with col2:
    st.subheader("Conversion Rate Model Performance")
    
    conversion_pred = conversion_model.predict(X_test)
    conversion_mae = mean_absolute_error(y_conversion_test, conversion_pred)
    conversion_r2 = r2_score(y_conversion_test, conversion_pred)
    
    st.metric("Mean Absolute Error", f"{conversion_mae:.4f}")
    st.metric("R² Score", f"{conversion_r2:.3f}")
    
    # Actual vs Predicted plot
    fig_conversion_pred = px.scatter(
        x=y_conversion_test, 
        y=conversion_pred,
        title="Actual vs Predicted Conversion Rates",
        labels={'x': 'Actual Conversion Rate', 'y': 'Predicted Conversion Rate'}
    )
    fig_conversion_pred.add_trace(
        go.Scatter(x=[y_conversion_test.min(), y_conversion_test.max()], 
                  y=[y_conversion_test.min(), y_conversion_test.max()],
                  mode='lines', name='Perfect Prediction', line=dict(dash='dash'))
    )
    st.plotly_chart(fig_conversion_pred, use_container_width=True)

# Feature importance
st.header("Feature Importance Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Price Optimization Features")
    price_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': price_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig_price_importance = px.bar(
        price_importance.head(10),
        x='Importance',
        y='Feature',
        orientation='h',
        title="Top 10 Features for Price Optimization"
    )
    st.plotly_chart(fig_price_importance, use_container_width=True)

with col2:
    st.subheader("Conversion Rate Features")
    conversion_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': conversion_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig_conversion_importance = px.bar(
        conversion_importance.head(10),
        x='Importance',
        y='Feature',
        orientation='h',
        title="Top 10 Features for Conversion Prediction"
    )
    st.plotly_chart(fig_conversion_importance, use_container_width=True)

# Pricing recommendations by segment
st.header("Segment-Based Pricing Recommendations")

# Generate recommendations for each segment
segment_recommendations = []

for segment in range(5):
    segment_data = full_df[full_df['segment'] == segment]
    
    if len(segment_data) > 0:
        avg_income = segment_data['income'].mean()
        avg_credit = segment_data['credit_score'].mean()
        avg_current_price = segment_data['current_price'].mean()
        avg_conversion = segment_data['conversion_rate'].mean()
        
        # Predict optimal pricing
        segment_features = segment_data[feature_columns].mean().values.reshape(1, -1)
        optimal_price = price_model.predict(segment_features)[0]
        predicted_conversion = conversion_model.predict(segment_features)[0]
        
        # Calculate potential impact
        current_revenue = avg_current_price * avg_conversion
        optimal_revenue = optimal_price * predicted_conversion
        revenue_lift = ((optimal_revenue - current_revenue) / current_revenue) * 100
        
        segment_recommendations.append({
            'Segment': segment,
            'Customers': len(segment_data),
            'Avg Income': f"${avg_income:,.0f}",
            'Avg Credit Score': f"{avg_credit:.0f}",
            'Current Price': f"${avg_current_price:,.0f}",
            'Optimal Price': f"${optimal_price:,.0f}",
            'Current Conversion': f"{avg_conversion:.1%}",
            'Predicted Conversion': f"{predicted_conversion:.1%}",
            'Revenue Lift': f"{revenue_lift:+.1f}%"
        })

recommendations_df = pd.DataFrame(segment_recommendations)
st.dataframe(recommendations_df, use_container_width=True, hide_index=True)

# Price sensitivity analysis
st.header("Price Sensitivity Analysis")

col1, col2 = st.columns(2)

with col1:
    # Select a segment for analysis
    selected_segment = st.selectbox("Select Segment for Price Sensitivity", range(5))
    
    segment_data = full_df[full_df['segment'] == selected_segment]
    if len(segment_data) > 0:
        base_features = segment_data[feature_columns].mean().values.reshape(1, -1)
        
        # Generate price range
        base_price = segment_data['current_price'].mean()
        price_range = np.linspace(base_price * 0.7, base_price * 1.3, 50)
        
        conversions = []
        revenues = []
        
        for price in price_range:
            # Update price in features (if price is a direct feature)
            test_features = base_features.copy()
            predicted_conversion = conversion_model.predict(test_features)[0]
            
            # Apply price elasticity (simplified model)
            price_elasticity = -0.5  # Typical financial product elasticity
            relative_price = price / base_price
            adjusted_conversion = predicted_conversion * (relative_price ** price_elasticity)
            adjusted_conversion = max(0.01, min(0.99, adjusted_conversion))  # Keep realistic bounds
            
            conversions.append(adjusted_conversion)
            revenues.append(price * adjusted_conversion)
        
        # Plot sensitivity
        fig_sensitivity = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Conversion Rate vs Price', 'Revenue vs Price'],
            vertical_spacing=0.12
        )
        
        fig_sensitivity.add_trace(
            go.Scatter(x=price_range, y=conversions, mode='lines', name='Conversion Rate'),
            row=1, col=1
        )
        
        fig_sensitivity.add_trace(
            go.Scatter(x=price_range, y=revenues, mode='lines', name='Revenue', line_color='red'),
            row=2, col=1
        )
        
        # Mark optimal point
        optimal_idx = np.argmax(revenues)
        fig_sensitivity.add_vline(x=price_range[optimal_idx], line_dash="dash", line_color="green",
                                annotation_text="Optimal Price")
        
        fig_sensitivity.update_xaxes(title_text="Price ($)", row=2, col=1)
        fig_sensitivity.update_yaxes(title_text="Conversion Rate", row=1, col=1)
        fig_sensitivity.update_yaxes(title_text="Revenue per Customer ($)", row=2, col=1)
        
        st.plotly_chart(fig_sensitivity, use_container_width=True)

with col2:
    st.subheader("Pricing Strategy Simulator")
    
    # Interactive pricing simulator
    test_income = st.slider("Customer Income ($)", 20000, 200000, 60000, 5000)
    test_credit = st.slider("Credit Score", 300, 850, 700, 10)
    test_age = st.slider("Age", 18, 80, 35, 1)
    test_dti = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3, 0.05)
    
    # Create test customer profile
    test_customer = pd.DataFrame({
        'income': [test_income],
        'credit_score': [test_credit],
        'debt_to_income': [test_dti],
        'age': [test_age],
        'employment_years': [5],
        'existing_products': [2],
        'transaction_frequency': [10],
        'savings_rate': [0.15],
        'segment': [2],  # Default segment
        'market_condition': [market_encoding[market_conditions]]
    })
    
    if len(feature_columns) <= len(test_customer.columns):
        # Predict pricing
        predicted_price = price_model.predict(test_customer[feature_columns[:len(test_customer.columns)]])[0]
        predicted_conversion = conversion_model.predict(test_customer[feature_columns[:len(test_customer.columns)]])[0]
        predicted_revenue = predicted_price * predicted_conversion
        
        st.markdown("### Predictions for Test Customer:")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric("Optimal Price", f"${predicted_price:,.0f}")
        
        with col_b:
            st.metric("Predicted Conversion", f"{predicted_conversion:.1%}")
        
        with col_c:
            st.metric("Expected Revenue", f"${predicted_revenue:,.0f}")

# Business Impact Summary
st.header("💰 Business Impact Analysis")

col1, col2, col3 = st.columns(3)

# Calculate total impact
total_current_revenue = (full_df['current_price'] * full_df['conversion_rate']).sum()

# Predict optimal revenues for all customers
optimal_prices = price_model.predict(X)
optimal_conversions = conversion_model.predict(X)
optimal_revenues = optimal_prices * optimal_conversions
total_optimal_revenue = optimal_revenues.sum()

revenue_increase = total_optimal_revenue - total_current_revenue
percentage_increase = (revenue_increase / total_current_revenue) * 100

with col1:
    st.metric(
        label="Current Total Revenue",
        value=f"${total_current_revenue:,.0f}",
        help="Revenue with current pricing strategy"
    )

with col2:
    st.metric(
        label="Optimized Revenue Potential",
        value=f"${total_optimal_revenue:,.0f}",
        delta=f"+{percentage_increase:.1f}%",
        help="Potential revenue with ML-optimized pricing"
    )

with col3:
    st.metric(
        label="Additional Revenue",
        value=f"${revenue_increase:,.0f}",
        delta=f"${revenue_increase/len(full_df):,.0f} per customer",
        help="Total additional revenue from optimization"
    )

# Export functionality
st.header("Export Results")

col1, col2 = st.columns(2)

with col1:
    if st.button("📊 Export Pricing Recommendations"):
        csv = recommendations_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"pricing_recommendations_{product_type.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("📈 Export Detailed Analysis"):
        # Create detailed analysis
        detailed_df = full_df.copy()
        detailed_df['predicted_optimal_price'] = price_model.predict(X)
        detailed_df['predicted_conversion'] = conversion_model.predict(X)
        detailed_df['predicted_revenue'] = detailed_df['predicted_optimal_price'] * detailed_df['predicted_conversion']
        detailed_df['current_revenue'] = detailed_df['current_price'] * detailed_df['conversion_rate']
        detailed_df['revenue_lift'] = detailed_df['predicted_revenue'] - detailed_df['current_revenue']
        
        csv = detailed_df.to_csv(index=False)
        st.download_button(
            label="Download Detailed Analysis",
            data=csv,
            file_name=f"detailed_pricing_analysis_{product_type.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )

st.markdown("---")
st.markdown("**Note**: This pricing optimization model uses gradient boosting to predict optimal prices based on customer characteristics, market conditions, and business objectives. Results should be validated through A/B testing before full implementation.")
