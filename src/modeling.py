"""
Regression modeling utilities for used car price prediction.

This module provides functions for training linear regression models,
evaluating performance, and generating diagnostic plots.
"""

from typing import Dict, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


def prepare_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and test sets.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Proportion of data to use for testing (0.0 to 1.0)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
        
    Example:
        >>> X_train, X_test, y_train, y_test = prepare_train_test_split(X, y)
        >>> print(f"Training: {len(X_train)}, Test: {len(X_test)}")
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    
    print(f"Training set: {len(X_train)} samples ({100*(1-test_size):.0f}%)")
    print(f"Test set: {len(X_test)} samples ({100*test_size:.0f}%)")
    
    return X_train, X_test, y_train, y_test


def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_statsmodels: bool = True
) -> Any:
    """
    Train linear regression model using sklearn or statsmodels.
    
    Args:
        X_train: Training features
        y_train: Training target
        use_statsmodels: If True, uses statsmodels OLS (provides detailed stats)
                        If False, uses sklearn LinearRegression (faster)
        
    Returns:
        Fitted model object (statsmodels RegressionResultsWrapper or sklearn LinearRegression)
        
    Example:
        >>> model = train_linear_regression(X_train, y_train, use_statsmodels=True)
        >>> print(model.rsquared)  # R-squared value
    """
    print("="*60)
    print("TRAINING LINEAR REGRESSION MODEL")
    print("="*60)
    
    if use_statsmodels:
        # Add constant for intercept (statsmodels requires explicit constant)
        X_train_const = sm.add_constant(X_train)
        
        # Fit OLS model
        model = sm.OLS(y_train, X_train_const).fit()
        
        print(f"\nModel trained with statsmodels OLS")
        print(f"R-squared: {model.rsquared:.4f}")
        print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
        print(f"F-statistic: {model.fvalue:.2f} (p-value: {model.f_pvalue:.2e})")
        
        return model
    else:
        # Sklearn LinearRegression
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Calculate R-squared manually
        y_pred_train = model.predict(X_train)
        r2 = r2_score(y_train, y_pred_train)
        
        print(f"\nModel trained with sklearn LinearRegression")
        print(f"R-squared (train): {r2:.4f}")
        print(f"Number of features: {X_train.shape[1]}")
        
        return model


def train_robust_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cov_type: str = 'HC3'
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Train OLS regression with robust standard errors (heteroscedasticity-consistent).
    
    Robust errors (HC3) provide valid inference even when homoscedasticity
    assumption is violated (common in price data).
    
    Args:
        X_train: Training features
        y_train: Training target
        cov_type: Covariance estimator type ('HC0', 'HC1', 'HC2', 'HC3')
                  HC3 is most conservative, recommended for finite samples
        
    Returns:
        Fitted OLS model with robust standard errors
        
    Example:
        >>> model = train_robust_regression(X_train, y_train, cov_type='HC3')
        >>> print(model.summary())
    """
    print("="*60)
    print("TRAINING ROBUST LINEAR REGRESSION (HC3)")
    print("="*60)
    
    # Add constant for intercept
    X_train_const = sm.add_constant(X_train)
    
    # Fit OLS model
    model_ols = sm.OLS(y_train, X_train_const).fit()
    
    # Get robust covariance estimator
    model_robust = model_ols.get_robustcov_results(cov_type=cov_type)
    
    print(f"\nRobust regression trained (cov_type={cov_type})")
    print(f"R-squared: {model_robust.rsquared:.4f}")
    print(f"Adjusted R-squared: {model_robust.rsquared_adj:.4f}")
    print("Note: Coefficients same as OLS, but standard errors adjusted for heteroscedasticity")
    
    return model_robust


def evaluate_regression(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    log_target: bool = False
) -> Dict[str, float]:
    """
    Evaluate regression model on test set.
    
    Calculates RMSE, MAE, R², and MAPE. If target was log-transformed,
    applies exp() to convert predictions back to original scale.
    
    Args:
        model: Trained regression model
        X_test: Test features
        y_test: Test target (original scale)
        model_name: Name for display purposes
        log_target: Whether predictions are in log scale (need exp transform)
        
    Returns:
        Dictionary with evaluation metrics
        
    Example:
        >>> metrics = evaluate_regression(model, X_test, y_test, "OLS Baseline")
        >>> print(f"Test RMSE: {metrics['rmse']:.2f}")
    """
    print(f"\n{'='*60}")
    print(f"EVALUATION: {model_name}")
    print(f"{'='*60}")
    
    # Handle statsmodels vs sklearn prediction
    if hasattr(model, 'predict') and 'statsmodels' in str(type(model)):
        # Statsmodels - add constant to test data
        X_test_const = sm.add_constant(X_test)
        y_pred = model.predict(X_test_const)
    else:
        # Sklearn
        y_pred = model.predict(X_test)
    
    # If predictions are in log scale, convert back
    if log_target:
        y_pred = np.exp(y_pred)
        print("Note: Predictions transformed from log scale to original scale")
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"\nTest Set Performance:")
    print(f"  RMSE: {rmse:.2f} lakhs")
    print(f"  MAE:  {mae:.2f} lakhs")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'predictions': y_pred
    }


def compare_models(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create comparison table of model performances.
    
    Args:
        results: Dictionary mapping model names to evaluation metrics
        
    Returns:
        DataFrame with model comparison
        
    Example:
        >>> comparison = compare_models({
        ...     'OLS Baseline': metrics1,
        ...     'Robust OLS': metrics2,
        ...     'Log-Price Model': metrics3
        ... })
        >>> print(comparison)
    """
    comparison_data = []
    
    for model_name, metrics in results.items():
        comparison_data.append({
            'Model': model_name,
            'RMSE (lakhs)': f"{metrics['rmse']:.2f}",
            'MAE (lakhs)': f"{metrics['mae']:.2f}",
            'R²': f"{metrics['r2']:.4f}",
            'MAPE (%)': f"{metrics['mape']:.2f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by RMSE (lower is better)
    comparison_df['_rmse_sort'] = comparison_df['RMSE (lakhs)'].astype(float)
    comparison_df = comparison_df.sort_values('_rmse_sort')
    comparison_df = comparison_df.drop('_rmse_sort', axis=1)
    
    return comparison_df


def get_feature_importance(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Extract top N most important features by absolute coefficient value.
    
    For OLS/robust regression, feature importance is approximated by
    standardized coefficient magnitude (assumes features are scaled).
    
    Args:
        model: Trained statsmodels regression model
        top_n: Number of top features to return
        
    Returns:
        DataFrame with features ranked by absolute coefficient
        
    Example:
        >>> importance = get_feature_importance(model, top_n=10)
        >>> print(importance)
    """
    if not hasattr(model, 'params'):
        print("Error: Model must be statsmodels OLS/RegressionResults")
        return pd.DataFrame()
    
    # Get coefficients (exclude constant)
    coeffs = model.params.drop('const', errors='ignore')
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'Feature': coeffs.index,
        'Coefficient': coeffs.values,
        'Abs_Coefficient': np.abs(coeffs.values)
    })
    
    # Sort by absolute value
    importance_df = importance_df.sort_values('Abs_Coefficient', ascending=False)
    importance_df = importance_df.head(top_n)
    
    print(f"\nTop {top_n} Features by Coefficient Magnitude:")
    print(importance_df[['Feature', 'Coefficient']].to_string(index=False))
    
    return importance_df


def calculate_vif(X: pd.DataFrame, threshold: float = 10.0) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for multicollinearity detection.
    
    VIF > 10 indicates high multicollinearity (consider removing feature).
    VIF > 5 is moderate concern.
    
    Args:
        X: Feature DataFrame (should be numeric)
        threshold: VIF threshold for flagging high multicollinearity
        
    Returns:
        DataFrame with VIF values for each feature
        
    Example:
        >>> vif_df = calculate_vif(X_train_numeric)
        >>> high_vif = vif_df[vif_df['VIF'] > 10]
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    print("Calculating VIF (Variance Inflation Factor)...")
    
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [
        variance_inflation_factor(X.values, i) 
        for i in range(X.shape[1])
    ]
    
    vif_data = vif_data.sort_values('VIF', ascending=False)
    
    # Flag high VIF
    high_vif = vif_data[vif_data['VIF'] > threshold]
    
    if len(high_vif) > 0:
        print(f"\n⚠️ {len(high_vif)} features with VIF > {threshold} (multicollinearity concern):")
        print(high_vif.to_string(index=False))
    else:
        print(f"\n✓ All features have VIF ≤ {threshold}")
    
    return vif_data
