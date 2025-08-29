import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

def calculate_clv(monthly_revenue, retention_rate, months=24):
    """
    Calculate Customer Lifetime Value (CLV)
    
    Args:
        monthly_revenue (float): Average monthly revenue per customer
        retention_rate (float): Monthly customer retention rate (0-1)
        months (int): Number of months to project
    
    Returns:
        float: Customer Lifetime Value
    """
    if retention_rate <= 0 or retention_rate > 1:
        raise ValueError("Retention rate must be between 0 and 1")
    
    if monthly_revenue < 0:
        raise ValueError("Monthly revenue must be non-negative")
    
    # CLV = Monthly Revenue × (Retention Rate / (1 + Discount Rate - Retention Rate))
    # Simplified version without discount rate for financial products
    if retention_rate == 1:
        clv = monthly_revenue * months
    else:
        # Geometric series sum for declining customer base
        clv = monthly_revenue * (1 - retention_rate ** months) / (1 - retention_rate)
    
    return max(0, clv)

def calculate_conversion_rate(conversions, total_exposures):
    """
    Calculate conversion rate with confidence intervals
    
    Args:
        conversions (int): Number of conversions
        total_exposures (int): Total number of exposures/opportunities
    
    Returns:
        dict: Conversion rate, confidence intervals, and statistics
    """
    if total_exposures <= 0:
        raise ValueError("Total exposures must be positive")
    
    if conversions < 0 or conversions > total_exposures:
        raise ValueError("Conversions must be between 0 and total exposures")
    
    conversion_rate = conversions / total_exposures
    
    # Wilson score interval (more accurate for small samples)
    z = 1.96  # 95% confidence
    n = total_exposures
    p = conversion_rate
    
    denominator = 1 + (z**2 / n)
    center = (p + (z**2 / (2*n))) / denominator
    margin = z * math.sqrt((p*(1-p) + z**2/(4*n)) / n) / denominator
    
    ci_lower = max(0, center - margin)
    ci_upper = min(1, center + margin)
    
    # Standard error
    standard_error = math.sqrt(p * (1 - p) / n) if n > 0 else 0
    
    return {
        'conversion_rate': conversion_rate,
        'confidence_interval_lower': ci_lower,
        'confidence_interval_upper': ci_upper,
        'standard_error': standard_error,
        'conversions': conversions,
        'total_exposures': total_exposures
    }

def calculate_statistical_power(effect_size, sample_size, alpha=0.05):
    """
    Calculate statistical power for A/B test
    
    Args:
        effect_size (float): Effect size (difference between groups)
        sample_size (int): Sample size per group
        alpha (float): Significance level (default 0.05)
    
    Returns:
        float: Statistical power (0-1)
    """
    if sample_size <= 0:
        raise ValueError("Sample size must be positive")
    
    if alpha <= 0 or alpha >= 1:
        raise ValueError("Alpha must be between 0 and 1")
    
    # Z-scores for two-tailed test
    z_alpha = stats.norm.ppf(1 - alpha/2)
    
    # Standard error for difference in proportions
    # Simplified calculation assuming equal sample sizes and balanced design
    pooled_variance = 0.25  # Maximum variance for binomial (p=0.5)
    standard_error = math.sqrt(2 * pooled_variance / sample_size)
    
    if standard_error == 0:
        return 1.0 if effect_size > 0 else 0.0
    
    # Non-centrality parameter
    z_beta = (abs(effect_size) - z_alpha * standard_error) / standard_error
    
    # Power is probability of rejecting null when alternative is true
    power = 1 - stats.norm.cdf(z_beta)
    
    return max(0, min(1, power))

def calculate_sample_size_needed(effect_size, power=0.8, alpha=0.05):
    """
    Calculate required sample size for A/B test
    
    Args:
        effect_size (float): Minimum detectable effect size
        power (float): Desired statistical power (default 0.8)
        alpha (float): Significance level (default 0.05)
    
    Returns:
        int: Required sample size per group
    """
    if effect_size <= 0:
        raise ValueError("Effect size must be positive")
    
    if power <= 0 or power >= 1:
        raise ValueError("Power must be between 0 and 1")
    
    # Z-scores
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Sample size calculation for difference in proportions
    # Using conservative estimate (p=0.5 for maximum variance)
    p = 0.5
    variance = 2 * p * (1 - p)
    
    sample_size = (z_alpha + z_beta)**2 * variance / (effect_size**2)
    
    return max(1, int(np.ceil(sample_size)))

def calculate_roi(investment, returns, time_period_months=12):
    """
    Calculate Return on Investment (ROI)
    
    Args:
        investment (float): Total investment amount
        returns (float): Total returns/benefits
        time_period_months (int): Time period in months
    
    Returns:
        dict: ROI metrics
    """
    if investment <= 0:
        raise ValueError("Investment must be positive")
    
    # Basic ROI
    roi = (returns - investment) / investment
    
    # Annualized ROI
    if time_period_months > 0:
        annualized_roi = ((returns / investment) ** (12 / time_period_months)) - 1
    else:
        annualized_roi = roi
    
    # Net present value (simplified, no discount rate)
    npv = returns - investment
    
    # Return on investment ratio
    roi_ratio = returns / investment if investment > 0 else 0
    
    # Payback period (months)
    monthly_return = returns / time_period_months if time_period_months > 0 else returns
    payback_period = investment / monthly_return if monthly_return > 0 else float('inf')
    
    return {
        'roi_percentage': roi * 100,
        'annualized_roi_percentage': annualized_roi * 100,
        'npv': npv,
        'roi_ratio': roi_ratio,
        'payback_period_months': payback_period,
        'investment': investment,
        'returns': returns,
        'net_benefit': npv
    }

def calculate_price_elasticity(price_changes, demand_changes):
    """
    Calculate price elasticity of demand
    
    Args:
        price_changes (array-like): Percentage changes in price
        demand_changes (array-like): Percentage changes in demand/conversion
    
    Returns:
        float: Price elasticity coefficient
    """
    price_changes = np.array(price_changes)
    demand_changes = np.array(demand_changes)
    
    if len(price_changes) != len(demand_changes):
        raise ValueError("Price and demand changes must have same length")
    
    if len(price_changes) < 2:
        raise ValueError("Need at least 2 data points to calculate elasticity")
    
    # Remove zero price changes to avoid division by zero
    non_zero_mask = price_changes != 0
    if not np.any(non_zero_mask):
        return 0.0
    
    price_changes = price_changes[non_zero_mask]
    demand_changes = demand_changes[non_zero_mask]
    
    # Elasticity = % change in demand / % change in price
    elasticities = demand_changes / price_changes
    
    # Return median elasticity to reduce impact of outliers
    return np.median(elasticities)

def calculate_confidence_interval(data, confidence=0.95):
    """
    Calculate confidence interval for a dataset
    
    Args:
        data (array-like): Dataset
        confidence (float): Confidence level (default 0.95)
    
    Returns:
        dict: Mean, confidence intervals, and statistics
    """
    data = np.array(data)
    
    if len(data) == 0:
        raise ValueError("Data cannot be empty")
    
    n = len(data)
    mean = np.mean(data)
    std_error = stats.sem(data)  # Standard error of mean
    
    # T-distribution for small samples, normal for large samples
    if n < 30:
        t_value = stats.t.ppf((1 + confidence) / 2, df=n-1)
        margin_error = t_value * std_error
    else:
        z_value = stats.norm.ppf((1 + confidence) / 2)
        margin_error = z_value * std_error
    
    ci_lower = mean - margin_error
    ci_upper = mean + margin_error
    
    return {
        'mean': mean,
        'confidence_interval_lower': ci_lower,
        'confidence_interval_upper': ci_upper,
        'standard_error': std_error,
        'margin_of_error': margin_error,
        'sample_size': n,
        'confidence_level': confidence
    }

def calculate_lift(control_metric, treatment_metric):
    """
    Calculate lift between control and treatment groups
    
    Args:
        control_metric (float): Control group metric value
        treatment_metric (float): Treatment group metric value
    
    Returns:
        dict: Lift metrics
    """
    if control_metric == 0:
        if treatment_metric == 0:
            return {
                'absolute_lift': 0,
                'relative_lift_percentage': 0,
                'control_value': control_metric,
                'treatment_value': treatment_metric
            }
        else:
            # Cannot calculate percentage lift when control is zero
            return {
                'absolute_lift': treatment_metric - control_metric,
                'relative_lift_percentage': float('inf'),
                'control_value': control_metric,
                'treatment_value': treatment_metric
            }
    
    absolute_lift = treatment_metric - control_metric
    relative_lift = (treatment_metric - control_metric) / control_metric * 100
    
    return {
        'absolute_lift': absolute_lift,
        'relative_lift_percentage': relative_lift,
        'control_value': control_metric,
        'treatment_value': treatment_metric,
        'improvement_factor': treatment_metric / control_metric if control_metric != 0 else 1
    }

def calculate_statistical_significance(control_data, treatment_data, test_type='ttest'):
    """
    Calculate statistical significance between two groups
    
    Args:
        control_data (array-like): Control group data
        treatment_data (array-like): Treatment group data
        test_type (str): Type of test ('ttest', 'chi2', 'mannwhitney')
    
    Returns:
        dict: Statistical test results
    """
    control_data = np.array(control_data)
    treatment_data = np.array(treatment_data)
    
    if len(control_data) == 0 or len(treatment_data) == 0:
        raise ValueError("Both groups must have data")
    
    if test_type == 'ttest':
        # Independent t-test
        statistic, p_value = stats.ttest_ind(treatment_data, control_data)
        test_name = "Independent T-Test"
        
    elif test_type == 'chi2':
        # Chi-square test for categorical data
        # Assuming binary outcomes (0/1)
        control_success = np.sum(control_data)
        control_total = len(control_data)
        treatment_success = np.sum(treatment_data)
        treatment_total = len(treatment_data)
        
        contingency_table = np.array([
            [control_success, control_total - control_success],
            [treatment_success, treatment_total - treatment_success]
        ])
        
        statistic, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        test_name = "Chi-Square Test"
        
    elif test_type == 'mannwhitney':
        # Mann-Whitney U test (non-parametric)
        statistic, p_value = stats.mannwhitneyu(treatment_data, control_data, alternative='two-sided')
        test_name = "Mann-Whitney U Test"
        
    else:
        raise ValueError("Invalid test_type. Choose 'ttest', 'chi2', or 'mannwhitney'")
    
    # Effect size (Cohen's d for t-test)
    if test_type == 'ttest':
        pooled_std = np.sqrt(((len(control_data) - 1) * np.var(control_data, ddof=1) +
                             (len(treatment_data) - 1) * np.var(treatment_data, ddof=1)) /
                            (len(control_data) + len(treatment_data) - 2))
        cohens_d = (np.mean(treatment_data) - np.mean(control_data)) / pooled_std
    else:
        cohens_d = None
    
    return {
        'test_name': test_name,
        'test_statistic': statistic,
        'p_value': p_value,
        'is_significant_005': p_value < 0.05,
        'is_significant_001': p_value < 0.01,
        'cohens_d': cohens_d,
        'control_mean': np.mean(control_data),
        'treatment_mean': np.mean(treatment_data),
        'control_size': len(control_data),
        'treatment_size': len(treatment_data)
    }

def calculate_revenue_metrics(prices, quantities, costs=None):
    """
    Calculate comprehensive revenue metrics
    
    Args:
        prices (array-like): Prices per unit
        quantities (array-like): Quantities sold
        costs (array-like, optional): Costs per unit
    
    Returns:
        dict: Revenue metrics
    """
    prices = np.array(prices)
    quantities = np.array(quantities)
    
    if len(prices) != len(quantities):
        raise ValueError("Prices and quantities must have same length")
    
    # Revenue calculations
    revenues = prices * quantities
    total_revenue = np.sum(revenues)
    average_revenue = np.mean(revenues)
    
    # Price and quantity metrics
    average_price = np.mean(prices)
    total_quantity = np.sum(quantities)
    average_quantity = np.mean(quantities)
    
    # Revenue metrics
    metrics = {
        'total_revenue': total_revenue,
        'average_revenue': average_revenue,
        'average_price': average_price,
        'total_quantity': total_quantity,
        'average_quantity': average_quantity,
        'revenue_variance': np.var(revenues),
        'revenue_std': np.std(revenues)
    }
    
    # Add profit metrics if costs provided
    if costs is not None:
        costs = np.array(costs)
        if len(costs) != len(prices):
            raise ValueError("Costs must have same length as prices")
        
        profits = (prices - costs) * quantities
        total_profit = np.sum(profits)
        profit_margin = total_profit / total_revenue if total_revenue > 0 else 0
        
        metrics.update({
            'total_profit': total_profit,
            'average_profit': np.mean(profits),
            'profit_margin_percentage': profit_margin * 100,
            'total_costs': np.sum(costs * quantities)
        })
    
    return metrics

def calculate_churn_probability(customer_features, model_type='logistic'):
    """
    Simple churn probability calculation based on customer features
    
    Args:
        customer_features (dict): Customer feature dictionary
        model_type (str): Type of model to simulate
    
    Returns:
        float: Churn probability (0-1)
    """
    # Simplified churn model based on common financial services factors
    base_churn = 0.15  # 15% base annual churn rate
    
    # Adjust based on customer characteristics
    churn_prob = base_churn
    
    # Credit score impact (higher credit = lower churn)
    if 'credit_score' in customer_features:
        credit_score = customer_features['credit_score']
        credit_factor = max(0.5, min(1.5, (850 - credit_score) / 550))
        churn_prob *= credit_factor
    
    # Income impact (higher income = lower churn)
    if 'income' in customer_features:
        income = customer_features['income']
        income_factor = max(0.7, min(1.3, 80000 / max(income, 20000)))
        churn_prob *= income_factor
    
    # Age impact (middle-aged customers more stable)
    if 'age' in customer_features:
        age = customer_features['age']
        if 35 <= age <= 55:
            churn_prob *= 0.8  # Lower churn for stable age group
        elif age < 25:
            churn_prob *= 1.3  # Higher churn for young customers
    
    # Product usage impact
    if 'existing_products' in customer_features:
        products = customer_features['existing_products']
        if products >= 3:
            churn_prob *= 0.7  # Lower churn for multi-product customers
    
    return max(0.01, min(0.99, churn_prob))

def calculate_customer_value_score(customer_data):
    """
    Calculate a composite customer value score
    
    Args:
        customer_data (dict): Customer data dictionary
    
    Returns:
        dict: Customer value metrics
    """
    score = 0
    max_score = 100
    
    # Income component (30% of score)
    if 'income' in customer_data:
        income = customer_data['income']
        income_score = min(30, (income / 100000) * 30)
        score += income_score
    
    # Credit score component (25% of score)
    if 'credit_score' in customer_data:
        credit_score = customer_data['credit_score']
        credit_score_normalized = (credit_score - 300) / 550  # Normalize to 0-1
        score += credit_score_normalized * 25
    
    # Product engagement (20% of score)
    if 'existing_products' in customer_data:
        products = customer_data['existing_products']
        product_score = min(20, products * 5)
        score += product_score
    
    # Financial stability (25% of score)
    stability_score = 0
    if 'debt_to_income' in customer_data:
        dti = customer_data['debt_to_income']
        stability_score += max(0, (0.5 - dti) / 0.5 * 15)  # Lower DTI is better
    
    if 'savings_rate' in customer_data:
        savings_rate = customer_data['savings_rate']
        stability_score += min(10, savings_rate * 50)  # Higher savings rate is better
    
    score += stability_score
    
    # Normalize to percentage
    value_score_percentage = (score / max_score) * 100
    
    # Categorize customer
    if value_score_percentage >= 80:
        category = 'Premium'
    elif value_score_percentage >= 60:
        category = 'High Value'
    elif value_score_percentage >= 40:
        category = 'Medium Value'
    elif value_score_percentage >= 20:
        category = 'Low Value'
    else:
        category = 'At Risk'
    
    return {
        'value_score': round(value_score_percentage, 2),
        'value_category': category,
        'score_components': {
            'income_component': income_score if 'income' in customer_data else 0,
            'credit_component': credit_score_normalized * 25 if 'credit_score' in customer_data else 0,
            'engagement_component': min(20, customer_data.get('existing_products', 0) * 5),
            'stability_component': stability_score
        }
    }

