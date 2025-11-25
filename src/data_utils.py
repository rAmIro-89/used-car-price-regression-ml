"""
Data utilities for used car price prediction.

This module provides functions for loading, cleaning, and preprocessing
used car data extracted from exploratory notebooks.
"""

from typing import Tuple, List
import pandas as pd
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    """
    Load used car dataset from CSV file.
    
    Automatically handles:
    - Renaming 'price(in lakhs)' to 'price'
    - Dropping 'Unnamed: 0' index column
    
    Args:
        path: Path to CSV file containing used car data
        
    Returns:
        DataFrame with standardized column names
        
    Example:
        >>> df = load_data('../data/raw/Used Car Dataset.csv')
        >>> print(df.shape)
        (6019, 13)
    """
    df = pd.read_csv(path)
    
    # Standardize target column name
    if 'price(in lakhs)' in df.columns:
        df = df.rename(columns={'price(in lakhs)': 'price'})
    
    # Drop index artifact if present
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')
    
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic data cleaning operations.
    
    Handles:
    - Removes low-utility columns (car_name, registration_year, insurance_validity)
    - Removes duplicate rows
    - Reports missing values (user decides on imputation strategy)
    
    Args:
        df: Raw DataFrame with potential data quality issues
        
    Returns:
        Cleaned DataFrame ready for feature engineering
        
    Example:
        >>> df_clean = basic_cleaning(df)
        >>> print(df_clean.duplicated().sum())
        0
    """
    df = df.copy()
    
    # Remove low-utility columns
    cols_to_drop = [
        'car_name',           # High cardinality, not generalizable
        'registration_year',  # Redundant with model_year/age
        'insurance_validity'  # Weak predictor for intrinsic value
    ]
    
    existing_drops = [col for col in cols_to_drop if col in df.columns]
    if existing_drops:
        df = df.drop(columns=existing_drops)
        print(f"Removed columns: {existing_drops}")
    
    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate rows")
    
    # Report missing values (don't auto-fill for regression data)
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\nMissing values detected:")
        print(missing[missing > 0])
        print("Consider imputation strategy based on business logic")
    else:
        print("No missing values detected")
    
    print(f"Cleaning complete. Final shape: {df.shape}")
    return df


def create_vehicle_age(df: pd.DataFrame, current_year: int = 2024) -> pd.DataFrame:
    """
    Create vehicle age feature from model_year.
    
    Args:
        df: DataFrame with 'model_year' column
        current_year: Reference year for age calculation
        
    Returns:
        DataFrame with new 'vehicle_age' column
        
    Example:
        >>> df_with_age = create_vehicle_age(df, current_year=2024)
        >>> print(df_with_age[['model_year', 'vehicle_age']].head())
    """
    if 'model_year' not in df.columns:
        print("Warning: 'model_year' column not found")
        return df
    
    df = df.copy()
    df['vehicle_age'] = current_year - df['model_year']
    
    # Ensure no negative ages
    if (df['vehicle_age'] < 0).any():
        print(f"Warning: {(df['vehicle_age'] < 0).sum()} vehicles have negative age")
        df = df[df['vehicle_age'] >= 0]
    
    print(f"Created 'vehicle_age' feature. Range: {df['vehicle_age'].min()}-{df['vehicle_age'].max()} years")
    return df


def get_feature_types(df: pd.DataFrame, target_col: str = 'price') -> dict:
    """
    Identify numeric and categorical features in DataFrame.
    
    Args:
        df: DataFrame to analyze
        target_col: Name of target variable to exclude from features
        
    Returns:
        Dictionary with 'numeric' and 'categorical' keys containing column lists
        
    Example:
        >>> feature_types = get_feature_types(df)
        >>> print(f"Numeric: {len(feature_types['numeric'])}")
        >>> print(f"Categorical: {len(feature_types['categorical'])}")
    """
    # Exclude target from features
    features_df = df.drop(columns=[target_col], errors='ignore')
    
    numeric_features = features_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = features_df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    
    return {
        'numeric': numeric_features,
        'categorical': categorical_features
    }


def split_features_target(
    df: pd.DataFrame, 
    target_col: str = 'price'
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features (X) and target (y).
    
    Args:
        df: Complete DataFrame including target column
        target_col: Name of the target variable column
        
    Returns:
        Tuple of (X, y) where X is features and y is target
        
    Example:
        >>> X, y = split_features_target(df)
        >>> print(f"Features: {X.shape}, Target: {y.shape}")
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nTarget statistics:")
    print(f"  Mean: {y.mean():.2f} lakhs")
    print(f"  Median: {y.median():.2f} lakhs")
    print(f"  Std: {y.std():.2f} lakhs")
    print(f"  Range: {y.min():.2f} - {y.max():.2f} lakhs")
    
    return X, y


def detect_outliers_iqr(
    series: pd.Series, 
    multiplier: float = 1.5
) -> pd.Series:
    """
    Detect outliers using Interquartile Range (IQR) method.
    
    Args:
        series: Numeric pandas Series
        multiplier: IQR multiplier (1.5 = standard, 3.0 = extreme only)
        
    Returns:
        Boolean Series where True indicates outlier
        
    Example:
        >>> outliers = detect_outliers_iqr(df['price'])
        >>> print(f"Found {outliers.sum()} outliers")
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = (series < lower_bound) | (series > upper_bound)
    
    print(f"Outlier detection for '{series.name}':")
    print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"  Outliers: {outliers.sum()} ({100*outliers.sum()/len(series):.2f}%)")
    
    return outliers


def apply_log_transformation(
    series: pd.Series, 
    offset: float = 1.0
) -> pd.Series:
    """
    Apply log transformation to handle skewed distributions.
    
    Useful for right-skewed price data to stabilize variance
    and improve linear regression assumptions.
    
    Args:
        series: Numeric pandas Series (must be positive)
        offset: Small value to add before log (handles zeros)
        
    Returns:
        Log-transformed Series
        
    Example:
        >>> log_price = apply_log_transformation(df['price'])
        >>> print(f"Original skew: {df['price'].skew():.2f}")
        >>> print(f"Log skew: {log_price.skew():.2f}")
    """
    if (series <= 0).any():
        print(f"Warning: {(series <= 0).sum()} non-positive values found. Using offset={offset}")
    
    log_series = np.log(series + offset)
    
    print(f"Log transformation applied to '{series.name}':")
    print(f"  Original: mean={series.mean():.2f}, std={series.std():.2f}, skew={series.skew():.2f}")
    print(f"  Log: mean={log_series.mean():.2f}, std={log_series.std():.2f}, skew={log_series.skew():.2f}")
    
    return log_series
