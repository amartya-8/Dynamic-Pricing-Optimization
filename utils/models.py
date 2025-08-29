import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

class CustomerSegmentation:
    """K-Means clustering for customer segmentation"""
    
    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.is_fitted = False
        
    def fit_predict(self, X):
        """Fit the model and predict cluster labels"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit K-means
        labels = self.kmeans.fit_predict(X_scaled)
        
        # Calculate performance metrics
        self.silhouette_score_ = silhouette_score(X_scaled, labels)
        self.calinski_harabasz_score_ = calinski_harabasz_score(X_scaled, labels)
        self.inertia_ = self.kmeans.inertia_
        
        self.is_fitted = True
        return labels
    
    def predict(self, X):
        """Predict cluster labels for new data"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.kmeans.predict(X_scaled)
    
    def get_cluster_centers(self):
        """Get cluster centers in original feature space"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Transform centers back to original scale
        centers_scaled = self.kmeans.cluster_centers_
        centers_original = self.scaler.inverse_transform(centers_scaled)
        return centers_original

class PriceOptimizer:
    """Gradient Boosting model for price optimization"""
    
    def __init__(self, objective='maximize_revenue', random_state=42):
        self.objective = objective
        self.random_state = random_state
        
        # Initialize models
        self.price_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=random_state
        )
        
        self.conversion_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=random_state
        )
        
        self.is_fitted = False
        self.feature_names = None
    
    def fit(self, X, y_price, y_conversion):
        """Fit both price and conversion models"""
        self.feature_names = list(X.columns) if hasattr(X, 'columns') else None
        
        # Fit price model
        self.price_model.fit(X, y_price)
        
        # Fit conversion model
        self.conversion_model.fit(X, y_conversion)
        
        # Calculate performance metrics
        self.price_score_ = self.price_model.score(X, y_price)
        self.conversion_score_ = self.conversion_model.score(X, y_conversion)
        
        # Cross-validation scores
        self.price_cv_scores_ = cross_val_score(self.price_model, X, y_price, cv=5)
        self.conversion_cv_scores_ = cross_val_score(self.conversion_model, X, y_conversion, cv=5)
        
        self.is_fitted = True
        return self
    
    def predict_price(self, X):
        """Predict optimal prices"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.price_model.predict(X)
    
    def predict_conversion(self, X):
        """Predict conversion rates"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.conversion_model.predict(X)
    
    def predict_revenue(self, X):
        """Predict expected revenue (price * conversion)"""
        prices = self.predict_price(X)
        conversions = self.predict_conversion(X)
        return prices * conversions
    
    def optimize_price(self, X, price_bounds=None):
        """Find optimal price for maximum revenue"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before optimization")
        
        # If no bounds provided, use ±50% of predicted price
        base_prices = self.predict_price(X)
        
        if price_bounds is None:
            lower_bounds = base_prices * 0.5
            upper_bounds = base_prices * 1.5
        else:
            lower_bounds = price_bounds[0]
            upper_bounds = price_bounds[1]
        
        optimal_prices = []
        
        for i in range(len(X)):
            customer_features = X.iloc[i:i+1] if hasattr(X, 'iloc') else X[i:i+1]
            
            # Grid search for optimal price
            price_range = np.linspace(lower_bounds[i], upper_bounds[i], 50)
            max_revenue = 0
            optimal_price = base_prices[i]
            
            for price in price_range:
                # Create modified features with new price (if price is a feature)
                test_features = customer_features.copy()
                
                # Predict conversion for this price
                predicted_conversion = self.predict_conversion(test_features)[0]
                
                # Apply price elasticity (simplified model)
                relative_price = price / base_prices[i]
                price_elasticity = -0.5  # Typical elasticity for financial products
                adjusted_conversion = predicted_conversion * (relative_price ** price_elasticity)
                adjusted_conversion = max(0.01, min(0.99, adjusted_conversion))
                
                # Calculate revenue
                revenue = price * adjusted_conversion
                
                if revenue > max_revenue:
                    max_revenue = revenue
                    optimal_price = price
            
            optimal_prices.append(optimal_price)
        
        return np.array(optimal_prices)
    
    def get_feature_importance(self):
        """Get feature importance from both models"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        importance_dict = {
            'price_importance': self.price_model.feature_importances_,
            'conversion_importance': self.conversion_model.feature_importances_
        }
        
        if self.feature_names:
            importance_dict['feature_names'] = self.feature_names
        
        return importance_dict

class ConversionRateModel:
    """Specialized model for conversion rate prediction"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=random_state,
            objective='reg:squarederror'
        )
        self.is_fitted = False
    
    def fit(self, X, y):
        """Fit the conversion rate model"""
        # Ensure conversion rates are bounded [0, 1]
        y_bounded = np.clip(y, 0.001, 0.999)
        
        self.model.fit(X, y_bounded)
        self.score_ = self.model.score(X, y_bounded)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Predict conversion rates"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = self.model.predict(X)
        # Ensure predictions are in valid range
        return np.clip(predictions, 0.001, 0.999)
    
    def predict_proba_range(self, X, confidence_interval=0.95):
        """Predict conversion rates with confidence intervals"""
        # This is a simplified implementation
        # In practice, you might use quantile regression or ensemble methods
        base_predictions = self.predict(X)
        
        # Estimate uncertainty based on residuals (simplified)
        std_error = 0.05  # Placeholder - would calculate from validation set
        
        z_score = 1.96 if confidence_interval == 0.95 else 2.58  # 95% or 99% CI
        
        lower_bound = np.clip(base_predictions - z_score * std_error, 0.001, 0.999)
        upper_bound = np.clip(base_predictions + z_score * std_error, 0.001, 0.999)
        
        return base_predictions, lower_bound, upper_bound

class RiskAssessmentModel:
    """Model for risk assessment in pricing"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=random_state
        )
        self.is_fitted = False
        self.risk_thresholds = {'low': 0.3, 'medium': 0.6, 'high': 1.0}
    
    def fit(self, X, y_risk_score):
        """Fit the risk assessment model"""
        self.model.fit(X, y_risk_score)
        self.score_ = self.model.score(X, y_risk_score)
        self.is_fitted = True
        return self
    
    def predict_risk_score(self, X):
        """Predict continuous risk scores"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        return self.model.predict(X)
    
    def predict_risk_category(self, X):
        """Predict risk categories (low/medium/high)"""
        risk_scores = self.predict_risk_score(X)
        
        categories = []
        for score in risk_scores:
            if score <= self.risk_thresholds['low']:
                categories.append('low')
            elif score <= self.risk_thresholds['medium']:
                categories.append('medium')
            else:
                categories.append('high')
        
        return np.array(categories)
    
    def get_risk_adjustment_factors(self, X):
        """Get pricing adjustment factors based on risk"""
        risk_scores = self.predict_risk_score(X)
        
        # Convert risk scores to pricing multipliers
        # Higher risk = higher prices (for loans) or lower limits (for credit)
        adjustment_factors = 1 + (risk_scores - 0.5) * 0.5  # Scale to reasonable range
        
        return np.clip(adjustment_factors, 0.5, 2.0)  # Reasonable bounds

class MarketTrendAnalyzer:
    """Analyze market trends affecting pricing"""
    
    def __init__(self):
        self.trend_model = None
        self.seasonality_factors = None
        self.is_fitted = False
    
    def fit(self, market_data):
        """Fit market trend analysis"""
        # Simple linear trend for interest rates
        dates_numeric = pd.to_datetime(market_data['date']).astype(int)
        
        # Fit trend line for interest rates
        coeffs = np.polyfit(dates_numeric, market_data['interest_rate'], 1)
        self.trend_model = coeffs
        
        # Calculate seasonality (monthly patterns)
        market_data['month'] = pd.to_datetime(market_data['date']).dt.month
        monthly_avg = market_data.groupby('month')['interest_rate'].mean()
        overall_avg = market_data['interest_rate'].mean()
        self.seasonality_factors = (monthly_avg / overall_avg).to_dict()
        
        self.is_fitted = True
        return self
    
    def predict_trend(self, future_dates):
        """Predict future market trends"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        future_dates_numeric = pd.to_datetime(future_dates).astype(int)
        trend_predictions = np.polyval(self.trend_model, future_dates_numeric)
        
        return trend_predictions
    
    def get_seasonality_adjustment(self, dates):
        """Get seasonality adjustments for given dates"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        months = pd.to_datetime(dates).month
        adjustments = [self.seasonality_factors.get(month, 1.0) for month in months]
        
        return np.array(adjustments)
