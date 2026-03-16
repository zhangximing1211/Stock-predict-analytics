import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def train_xgboost(X_train, y_train, X_val, y_val, params=None):
    """
    Train XGBoost model
    
    Args:
        X_train: Training set features
        y_train: Training set target
        X_val: Validation set features
        y_val: Validation set target
        params: XGBoost parameters
    
    Returns:
        Trained model
    """
    # Default parameters
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42
        }
    
    # Create model
    model = xgb.XGBRegressor(**params)
    
    # Train model
    model.fit(
        X_train, y_train
    )
    
    return model

def predict_xgboost(model, X):
    """
    Predict using XGBoost model
    
    Args:
        model: Trained model
        X: Feature data
    
    Returns:
        Prediction results
    """
    return model.predict(X)

def evaluate_model(y_true, y_pred):
    """
    Evaluate model performance
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Evaluation metrics dictionary
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }
