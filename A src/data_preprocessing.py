import pandas as pd
import numpy as np

def load_and_preprocess_data(file_path):
    """
    Load and preprocess stock data
    
    Args:
        file_path: Excel file path
    
    Returns:
        Preprocessed DataFrame
    """
    # Load data
    df = pd.read_excel(file_path)
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Ensure date column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Remove unnecessary columns
    df = df.drop(['Name'], axis=1) if 'Name' in df.columns else df
    
    # Create target variable (predict next day's close price)
    df['target'] = df['close'].shift(-1)
    
    # Remove last row (no next day data)
    df = df.dropna().reset_index(drop=True)
    
    return df

def create_features(df):
    """
    Create additional features
    
    Args:
        df: Original DataFrame
    
    Returns:
        DataFrame with new features
    """
    # Date-related features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    
    # Technical indicator features already exist, can add more here
    # For example: moving average differences, volume changes, etc.
    
    return df

def split_data_by_date(df, start_date, end_date):
    """
    Split data by date range
    
    Args:
        df: DataFrame
        start_date: Start date
        end_date: End date
    
    Returns:
        Split data
    """
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    return df[mask]
