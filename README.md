# Dynamic Pricing Optimization Platform

A comprehensive Machine Learning-powered financial product pricing optimization platform built with Streamlit. This application helps financial institutions optimize pricing strategies for loans, credit cards, and mortgages through customer segmentation, dynamic pricing models, A/B testing frameworks, and revenue impact analysis.

## 🚀 Features

- **Customer Segmentation**: Advanced K-Means clustering to identify distinct customer segments
- **Price Optimization**: XGBoost models for optimal pricing strategies
- **A/B Testing Framework**: Statistical testing for pricing strategy validation
- **Revenue Analysis**: ROI calculator and business outcome modeling
- **Interactive Dashboards**: Real-time visualizations with Plotly

## 🛠 Technology Stack

- **Frontend**: Streamlit
- **Machine Learning**: scikit-learn, XGBoost
- **Data Processing**: pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Statistical Analysis**: SciPy

## 📊 ML Models

- **K-Means Clustering** for customer segmentation
- **XGBoost Regressor** for price optimization
- **Gradient Boosting** for conversion rate prediction
- **Statistical Power Analysis** for A/B testing

## 🎯 Business Impact

- Customer segmentation with 5 distinct profiles
- Dynamic pricing optimization with 15-30% revenue uplift potential
- A/B testing framework with statistical significance validation
- ROI analysis with payback period calculations

## 🚀 Getting Started

### Prerequisites

```bash
pip install streamlit pandas numpy plotly scikit-learn xgboost scipy seaborn matplotlib
```

### Running the Application

```bash
streamlit run app.py --server.port 5000
```

Visit `http://localhost:5000` to view the application.

## 📁 Project Structure

```
├── app.py                          # Main Streamlit application
├── pages/
│   ├── 1_Customer_Segmentation.py  # Customer segmentation analysis
│   ├── 2_Price_Optimization.py     # Price optimization engine
│   ├── 3_AB_Testing.py            # A/B testing framework
│   └── 4_Revenue_Analysis.py       # Revenue impact analysis
├── utils/
│   ├── data_generator.py           # Synthetic data generation
│   ├── models.py                   # ML models and algorithms
│   └── metrics.py                  # Business metrics calculations
└── .streamlit/
    └── config.toml                 # Streamlit configuration
```

## 🔧 Configuration

The application uses `.streamlit/config.toml` for server configuration:

```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000
```

## 💰 Revenue Optimization Features

### Customer Segmentation
- 5 customer segments: Conservative, Balanced, Growth-Oriented, High-Risk, Premium
- Feature importance analysis
- CLV (Customer Lifetime Value) calculations

### Price Optimization
- ML-powered optimal pricing recommendations
- Price sensitivity analysis
- Revenue maximization strategies

### A/B Testing
- Statistical significance testing
- Confidence intervals
- Power analysis and sample size recommendations

### Revenue Analysis
- Multi-year revenue projections
- ROI calculations with different scenarios
- Business impact assessment

## 📈 Performance Metrics

- **Silhouette Score**: Customer segmentation quality
- **R² Score**: Model prediction accuracy  
- **Statistical Power**: A/B test reliability
- **Revenue Lift**: Business impact measurement

## 🎨 Key Visualizations

- Interactive customer segment distribution charts
- Price sensitivity curves
- A/B test performance dashboards
- Revenue projection timelines
- ROI scenario analysis

## 🔍 Use Cases

Perfect for financial institutions looking to:
- Optimize loan pricing strategies
- Improve credit card conversion rates
- Enhance mortgage pricing models
- Validate pricing changes through A/B testing
- Maximize revenue per customer

## 📊 Data Features

The platform analyzes customer data including:
- Demographics (age, income, employment)
- Financial metrics (credit score, debt-to-income ratio)
- Behavioral data (transaction frequency, product usage)
- Risk assessments and market conditions

## 🚀 Deployment

This application is designed for easy deployment on platforms like:
- Streamlit