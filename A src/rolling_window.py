import pandas as pd
from data_preprocessing import split_data_by_date

def generate_rolling_windows(start_year=2013, end_year=2017):
    """
    Generate rolling window date ranges
    
    Args:
        start_year: Start year
        end_year: End year (test set start year)
    
    Returns:
        List of rolling windows, each containing train and validation date ranges
    """
    windows = []
    
    for year in range(start_year, end_year):
        train_start = pd.Timestamp(f'{year}-03-11')
        train_end = pd.Timestamp(f'{year+1}-03-11')
        val_start = pd.Timestamp(f'{year+1}-03-11')
        val_end = pd.Timestamp(f'{year+2}-03-11')
        
        windows.append({
            'train_start': train_start,
            'train_end': train_end,
            'val_start': val_start,
            'val_end': val_end
        })
    
    # Test set
    test_start = pd.Timestamp(f'{end_year}-03-11')
    
    return windows, test_start

def prepare_rolling_window_data(df, window):
    """
    Prepare training and validation data for a specific window
    
    Args:
        df: Complete DataFrame
        window: Window configuration
    
    Returns:
        X_train, y_train, X_val, y_val
    """
    # Split training and validation sets
    train_data = split_data_by_date(df, window['train_start'], window['train_end'])
    val_data = split_data_by_date(df, window['val_start'], window['val_end'])
    
    # Feature columns (exclude date and target)
    feature_columns = [col for col in df.columns if col not in ['date', 'target']]
    
    # Prepare features and target variables
    X_train = train_data[feature_columns]
    y_train = train_data['target']
    X_val = val_data[feature_columns]
    y_val = val_data['target']
    
    return X_train, y_train, X_val, y_val

def prepare_test_data(df, test_start):
    """
    Prepare test set data
    
    Args:
        df: Complete DataFrame
        test_start: Test set start date
    
    Returns:
        X_test, y_test, test_dates
    """
    test_data = df[df['date'] >= test_start]
    
    # Feature columns (exclude date and target)
    feature_columns = [col for col in df.columns if col not in ['date', 'target']]
    
    X_test = test_data[feature_columns]
    y_test = test_data['target']
    test_dates = test_data['date']
    
    return X_test, y_test, test_dates
