import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_customer_data(n_customers=2000, random_seed=42):
    """Generate synthetic customer data for financial products"""
    np.random.seed(random_seed)
    
    customers = []
    
    for i in range(n_customers):
        # Basic demographics
        age = np.random.normal(45, 15)
        age = max(18, min(80, age))  # Constrain age between 18-80
        
        # Income correlated with age (peak earning years)
        age_factor = 1.0 if age < 30 else (1.5 if age < 50 else 1.2 if age < 65 else 0.8)
        base_income = np.random.lognormal(10.5, 0.5) * age_factor
        income = max(20000, min(500000, base_income))
        
        # Credit score correlated with age and income
        income_factor = min(2.0, income / 50000)
        age_factor = 1.2 if age > 30 else 1.0
        credit_base = 650 + (income_factor * 50) + (age_factor * 25) + np.random.normal(0, 40)
        credit_score = max(300, min(850, credit_base))
        
        # Employment years
        employment_years = min(age - 18, np.random.exponential(8))
        employment_years = max(0, employment_years)
        
        # Debt-to-income ratio (lower for higher income/credit)
        dti_base = np.random.beta(2, 5)  # Skewed towards lower DTI
        dti_adjustment = max(0.1, min(1.2, 1 - (credit_score - 300) / 550))
        debt_to_income = dti_base * dti_adjustment
        
        # Other financial characteristics
        existing_products = np.random.poisson(2)
        transaction_frequency = np.random.gamma(3, 5)  # Monthly transactions
        
        # Savings rate (higher for higher income)
        savings_rate = min(0.4, max(0.02, np.random.beta(2, 8) * (income / 100000)))
        
        # Risk category based on credit score and DTI
        if credit_score >= 750 and debt_to_income <= 0.3:
            risk_category = 'Low'
        elif credit_score >= 650 and debt_to_income <= 0.5:
            risk_category = 'Medium'
        else:
            risk_category = 'High'
        
        # Geographic region
        regions = ['Northeast', 'Southeast', 'Midwest', 'Southwest', 'West']
        region = np.random.choice(regions)
        
        customers.append({
            'customer_id': f'CUST_{i+1:06d}',
            'age': round(age, 0),
            'income': round(income, 0),
            'credit_score': round(credit_score, 0),
            'employment_years': round(employment_years, 1),
            'debt_to_income': round(debt_to_income, 3),
            'existing_products': existing_products,
            'transaction_frequency': round(transaction_frequency, 1),
            'savings_rate': round(savings_rate, 3),
            'risk_category': risk_category,
            'region': region
        })
    
    return pd.DataFrame(customers)

def generate_market_data(days_back=365):
    """Generate synthetic market data"""
    np.random.seed(42)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    dates = pd.date_range(start_date, end_date, freq='D')
    
    # Generate correlated market indicators
    base_interest_rate = 4.5  # Starting federal rate
    market_data = []
    
    for i, date in enumerate(dates):
        # Interest rate trend with noise
        trend = 0.001 * i  # Gradual increase
        noise = np.random.normal(0, 0.05)
        interest_rate = max(0.5, base_interest_rate + trend + noise)
        
        # Market volatility
        volatility = abs(np.random.normal(0.2, 0.1))
        
        # Economic indicators
        unemployment_rate = max(2.0, min(15.0, 4.5 + np.random.normal(0, 0.5)))
        gdp_growth = np.random.normal(2.5, 1.0)
        inflation_rate = max(0, np.random.normal(3.0, 0.8))
        
        market_data.append({
            'date': date,
            'interest_rate': round(interest_rate, 3),
            'volatility': round(volatility, 3),
            'unemployment_rate': round(unemployment_rate, 2),
            'gdp_growth': round(gdp_growth, 2),
            'inflation_rate': round(inflation_rate, 2)
        })
    
    return pd.DataFrame(market_data)

def generate_pricing_data(customers_df, product_type):
    """Generate pricing data for different financial products"""
    np.random.seed(42)
    
    pricing_data = []
    
    for _, customer in customers_df.iterrows():
        # Base pricing logic by product type
        if product_type == "Personal Loan":
            base_price = 15000  # Base loan amount
            risk_multiplier = 1.2 if customer['risk_category'] == 'High' else (1.1 if customer['risk_category'] == 'Medium' else 1.0)
            income_factor = min(2.0, customer['income'] / 50000)
            
        elif product_type == "Credit Card":
            base_price = 5000  # Base credit limit
            risk_multiplier = 0.8 if customer['risk_category'] == 'High' else (1.0 if customer['risk_category'] == 'Medium' else 1.3)
            income_factor = min(3.0, customer['income'] / 30000)
            
        else:  # Mortgage
            base_price = 250000  # Base mortgage amount
            risk_multiplier = 0.9 if customer['risk_category'] == 'High' else (1.0 if customer['risk_category'] == 'Medium' else 1.2)
            income_factor = min(5.0, customer['income'] / 40000)
        
        # Current price (suboptimal)
        current_price = base_price * risk_multiplier * income_factor * np.random.uniform(0.9, 1.1)
        
        # Optimal price (ML-optimized)
        optimization_factor = 1.1 + (customer['credit_score'] - 650) / 1000  # Better prices for better credit
        optimal_price = current_price * optimization_factor
        
        # Conversion rate based on price and customer characteristics
        price_sensitivity = 0.5 if customer['income'] > 80000 else 1.0
        base_conversion = 0.15  # 15% base conversion
        
        # Current conversion rate
        price_factor_current = max(0.1, 1 - (current_price - base_price) / base_price * price_sensitivity)
        credit_factor = (customer['credit_score'] - 300) / 550
        conversion_rate = base_conversion * price_factor_current * (0.5 + credit_factor)
        conversion_rate = max(0.01, min(0.8, conversion_rate))
        
        pricing_data.append({
            'customer_id': customer['customer_id'],
            'product_type': product_type,
            'current_price': round(current_price, 0),
            'optimal_price': round(optimal_price, 0),
            'conversion_rate': round(conversion_rate, 4)
        })
    
    return pd.DataFrame(pricing_data)

def generate_ab_test_data(sample_size_per_group, control_price, treatment_price, test_duration):
    """Generate A/B test data"""
    np.random.seed(42)
    
    ab_test_data = []
    
    # Generate test dates
    start_date = datetime.now() - timedelta(days=test_duration)
    
    for group in ['control', 'treatment']:
        price = control_price if group == 'control' else treatment_price
        
        for i in range(sample_size_per_group):
            # Random customer characteristics
            age = np.random.normal(45, 15)
            age = max(18, min(80, age))
            
            income = max(20000, min(200000, np.random.lognormal(10.8, 0.6)))
            credit_score = max(300, min(850, np.random.normal(700, 80)))
            
            # Conversion based on price sensitivity and customer characteristics
            base_conversion_rate = 0.12  # 12% base conversion
            
            # Price sensitivity (higher income = less sensitive)
            price_sensitivity = max(0.1, 1 - (income - 30000) / 100000)
            
            # Price effect on conversion
            price_effect = 1 - (price - 10000) / 50000 * price_sensitivity
            price_effect = max(0.1, price_effect)
            
            # Credit score effect
            credit_effect = 0.5 + (credit_score - 300) / 550
            
            # Final conversion probability
            conversion_prob = base_conversion_rate * price_effect * credit_effect
            conversion_prob = max(0.01, min(0.8, conversion_prob))
            
            # Determine if customer converted
            converted = np.random.binomial(1, conversion_prob)
            
            # Revenue (only if converted)
            revenue = price if converted else 0
            
            # Random test date
            test_date = start_date + timedelta(days=np.random.randint(0, test_duration))
            
            ab_test_data.append({
                'customer_id': f'{group.upper()}_{i+1:06d}',
                'group': group,
                'date': test_date.date(),
                'age': round(age),
                'income': round(income),
                'credit_score': round(credit_score),
                'price': price,
                'converted': converted,
                'revenue': revenue,
                'conversion_probability': round(conversion_prob, 4)
            })
    
    return pd.DataFrame(ab_test_data)

def generate_revenue_data(customers_df):
    """Generate current and optimized revenue data for customers"""
    np.random.seed(42)
    
    revenue_data = []
    
    for _, customer in customers_df.iterrows():
        # Base revenue calculation
        income_factor = customer['income'] / 50000
        credit_factor = (customer['credit_score'] - 300) / 550
        products_factor = 1 + customer['existing_products'] * 0.2
        
        # Current revenue (suboptimal)
        base_revenue = 2000 * income_factor * credit_factor * products_factor
        current_revenue = base_revenue * np.random.uniform(0.8, 1.2)
        
        # Optimized revenue (15-30% improvement potential)
        optimization_multiplier = 1.15 + (credit_factor * 0.15)  # Better customers have more upside
        optimized_revenue = current_revenue * optimization_multiplier
        
        revenue_data.append({
            'customer_id': customer['customer_id'],
            'current_revenue': round(current_revenue, 2),
            'optimized_revenue': round(optimized_revenue, 2),
            'revenue_lift': round(optimized_revenue - current_revenue, 2),
            'revenue_lift_pct': round((optimized_revenue - current_revenue) / current_revenue * 100, 2)
        })
    
    return pd.DataFrame(revenue_data)
