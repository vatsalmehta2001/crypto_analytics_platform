import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import xgboost as xgb
import lightgbm as lgb
import os
import joblib
from datetime import datetime, timedelta
import json
from tqdm import tqdm

class CryptoPricePredictor:
    """
    Class for predicting cryptocurrency prices using various machine learning models.
    Integrates price data with sentiment features for enhanced predictions.
    """
    
    def __init__(self, model_dir='models/saved'):
        """
        Initialize the price predictor with a directory to save models.
        
        Args:
            model_dir (str): Directory to save trained models
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Scalers for data normalization
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Configure GPU memory growth to avoid OOM errors
        self._configure_gpu()
    
    def _configure_gpu(self):
        """Configure TensorFlow to use GPU efficiently."""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU is available: {len(gpus)} device(s)")
            else:
                print("No GPU found, using CPU")
        except Exception as e:
            print(f"Error configuring GPU: {e}")
    
    def prepare_time_series_data(self, df, target_col='close', window_size=10, forecast_horizon=1, 
                                 feature_cols=None, include_sentiment=True):
        """
        Prepare time series data for sequence prediction.
        
        Args:
            df (pd.DataFrame): DataFrame with price and sentiment data
            target_col (str): Column to predict (e.g., 'close')
            window_size (int): Number of past days to use for prediction
            forecast_horizon (int): Number of days ahead to predict
            feature_cols (list): List of feature columns to include
            include_sentiment (bool): Whether to include sentiment features
            
        Returns:
            tuple: X (features), y (targets), and feature names
        """
        if feature_cols is None:
            # Default technical indicators
            feature_cols = ['close', 'high', 'low', 'open', 
                          'sma_7', 'sma_20', 'rsi_14', 
                          'macd', 'bb_upper', 'bb_lower', 'volatility_7']
        
        # Add sentiment features if requested
        sentiment_cols = []
        if include_sentiment and any(col.startswith('sentiment') for col in df.columns):
            sentiment_cols = [col for col in df.columns if col.startswith('sentiment') or 
                              col.endswith('ratio') or col == 'sentiment_score']
            feature_cols = feature_cols + sentiment_cols
        
        # Ensure all required columns exist
        available_cols = [col for col in feature_cols if col in df.columns]
        if len(available_cols) < len(feature_cols):
            missing = set(feature_cols) - set(available_cols)
            print(f"Warning: Missing columns: {missing}")
            feature_cols = available_cols
        
        # Make sure target column is in the data
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        # Create a copy of the dataframe with selected columns
        data = df[feature_cols + [target_col]].copy()
        data = data.dropna()
        
        # Scale the price data
        price_data = data[[target_col]].values
        scaled_price = self.price_scaler.fit_transform(price_data)
        
        # Scale the feature data
        feature_data = data[feature_cols].values
        scaled_features = self.feature_scaler.fit_transform(feature_data)
        
        # Create sequences
        X, y = [], []
        for i in range(len(data) - window_size - forecast_horizon + 1):
            # Get window_size days of all features as input sequence
            X.append(scaled_features[i:i+window_size])
            # Get target price 'forecast_horizon' days in the future
            y.append(scaled_price[i+window_size+forecast_horizon-1])
        
        return np.array(X), np.array(y), feature_cols
    
    def split_data(self, X, y, test_size=0.2, validation_size=0.1):
        """
        Split data into train, validation, and test sets.
        
        Args:
            X (np.array): Feature sequences
            y (np.array): Target values
            test_size (float): Proportion of data for testing
            validation_size (float): Proportion of training data for validation
            
        Returns:
            tuple: Train, validation, and test datasets
        """
        # First split into train and test
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # Then split training data into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=validation_size, shuffle=False
        )
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def build_lstm_model(self, input_shape, output_shape=1, lstm_units=50, dropout_rate=0.2):
        """
        Build an LSTM model for time series prediction.
        
        Args:
            input_shape (tuple): Shape of input sequences (window_size, n_features)
            output_shape (int): Number of output units
            lstm_units (int): Number of LSTM units
            dropout_rate (float): Dropout rate for regularization
            
        Returns:
            tf.keras.Model: Compiled LSTM model
        """
        model = Sequential()
        
        # First LSTM layer with return sequences for stacking
        model.add(LSTM(units=lstm_units, return_sequences=True, 
                       input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        
        # Second LSTM layer
        model.add(LSTM(units=lstm_units, return_sequences=False))
        model.add(Dropout(dropout_rate))
        
        # Dense layers
        model.add(Dense(25, activation='relu'))
        model.add(Dense(output_shape))
        
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        return model
    
    def build_bidirectional_lstm_model(self, input_shape, output_shape=1, lstm_units=50, dropout_rate=0.2):
        """
        Build a bidirectional LSTM model for time series prediction.
        
        Args:
            input_shape (tuple): Shape of input sequences (window_size, n_features)
            output_shape (int): Number of output units
            lstm_units (int): Number of LSTM units
            dropout_rate (float): Dropout rate for regularization
            
        Returns:
            tf.keras.Model: Compiled bidirectional LSTM model
        """
        model = Sequential()
        
        # Bidirectional LSTM layers
        model.add(Bidirectional(LSTM(units=lstm_units, return_sequences=True), 
                                input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        
        model.add(Bidirectional(LSTM(units=lstm_units, return_sequences=False)))
        model.add(Dropout(dropout_rate))
        
        # Dense layers
        model.add(Dense(25, activation='relu'))
        model.add(Dense(output_shape))
        
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        return model
    
    def train_lstm_model(self, train_data, val_data, model_type='lstm', epochs=100, batch_size=32, 
                         patience=10, coin_name='generic'):
        """
        Train an LSTM model with the given data.
        
        Args:
            train_data (tuple): X_train and y_train
            val_data (tuple): X_val and y_val
            model_type (str): Type of model ('lstm' or 'bidirectional')
            epochs (int): Maximum number of training epochs
            batch_size (int): Batch size for training
            patience (int): Early stopping patience
            coin_name (str): Name of the coin for model saving
            
        Returns:
            tf.keras.Model: Trained model
        """
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Get input shape from data
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # Build the appropriate model
        if model_type == 'lstm':
            model = self.build_lstm_model(input_shape)
            model_name = f"{coin_name}_lstm_model"
        elif model_type == 'bidirectional':
            model = self.build_bidirectional_lstm_model(input_shape)
            model_name = f"{coin_name}_bidirectional_lstm_model"
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ModelCheckpoint(
                filepath=os.path.join(self.model_dir, f"{model_name}.h5"),
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save the scalers
        joblib.dump(self.price_scaler, os.path.join(self.model_dir, f"{coin_name}_price_scaler.pkl"))
        joblib.dump(self.feature_scaler, os.path.join(self.model_dir, f"{coin_name}_feature_scaler.pkl"))
        
        return model, history
    
    def train_xgboost_model(self, train_data, val_data, coin_name='generic', **params):
        """
        Train an XGBoost model for price prediction.
        
        Args:
            train_data (tuple): X_train and y_train
            val_data (tuple): X_val and y_val
            coin_name (str): Name of the coin for model saving
            **params: XGBoost parameters
            
        Returns:
            xgb.Booster: Trained XGBoost model
        """
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        # Reshape sequential data for XGBoost
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        X_val_reshaped = X_val.reshape(X_val.shape[0], -1)
        
        # Default XGBoost parameters
        xgb_params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 1000,
            'early_stopping_rounds': 50
        }
        
        # Update with any provided parameters
        xgb_params.update(params)
        
        # Create DMatrix objects
        dtrain = xgb.DMatrix(X_train_reshaped, label=y_train)
        dval = xgb.DMatrix(X_val_reshaped, label=y_val)
        
        # Train the model
        model = xgb.train(
            xgb_params,
            dtrain,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=xgb_params['early_stopping_rounds'],
            verbose_eval=50
        )
        
        # Save the model
        model_path = os.path.join(self.model_dir, f"{coin_name}_xgboost_model.json")
        model.save_model(model_path)
        
        return model
    
    def evaluate_model(self, model, test_data, coin_name='generic', model_type='lstm', denormalize=True):
        """
        Evaluate a trained model on test data.
        
        Args:
            model: Trained model (LSTM or XGBoost)
            test_data (tuple): X_test and y_test
            coin_name (str): Name of the coin
            model_type (str): Type of model ('lstm', 'bidirectional', 'xgboost')
            denormalize (bool): Whether to denormalize predictions
            
        Returns:
            dict: Evaluation metrics
        """
        X_test, y_test = test_data
        
        # Make predictions based on model type
        if model_type in ['lstm', 'bidirectional']:
            y_pred = model.predict(X_test)
        elif model_type == 'xgboost':
            # Reshape for XGBoost
            X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
            dtest = xgb.DMatrix(X_test_reshaped)
            y_pred = model.predict(dtest).reshape(-1, 1)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Denormalize if requested
        if denormalize:
            y_test = self.price_scaler.inverse_transform(y_test)
            y_pred = self.price_scaler.inverse_transform(y_pred)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Print evaluation results
        print(f"\nEvaluation for {coin_name} using {model_type}:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"R²: {r2:.4f}")
        
        # Return metrics as dictionary
        metrics = {
            'coin': coin_name,
            'model_type': model_type,
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'r2': float(r2)
        }
        
        # Save metrics to file
        metrics_path = os.path.join(self.model_dir, f"{coin_name}_{model_type}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        return metrics
    
    def make_future_predictions(self, model, last_sequence, days_ahead=7, model_type='lstm'):
        """
        Make predictions for future days based on the last known sequence.
        
        Args:
            model: Trained model
            last_sequence (np.array): Last known sequence of data
            days_ahead (int): Number of days to predict ahead
            model_type (str): Type of model ('lstm', 'bidirectional', 'xgboost')
            
        Returns:
            np.array: Predicted prices for future days
        """
        # Make a copy of the last sequence
        curr_sequence = last_sequence.copy()
        predictions = []
        
        for _ in range(days_ahead):
            # Reshape for prediction
            if model_type in ['lstm', 'bidirectional']:
                # For LSTM, keep the 3D shape (samples, timesteps, features)
                pred_sequence = curr_sequence.reshape(1, curr_sequence.shape[0], curr_sequence.shape[1])
                prediction = model.predict(pred_sequence)
            elif model_type == 'xgboost':
                # For XGBoost, flatten to 2D
                pred_sequence = curr_sequence.reshape(1, -1)
                dtest = xgb.DMatrix(pred_sequence)
                prediction = model.predict(dtest).reshape(-1, 1)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Store prediction
            predictions.append(prediction[0][0])
            
            # Update sequence for next prediction (slide window forward)
            # Assuming close price is the first feature
            curr_sequence = np.roll(curr_sequence, -1, axis=0)
            curr_sequence[-1, 0] = prediction[0][0]
        
        # Convert scaled predictions back to original scale
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.price_scaler.inverse_transform(predictions)
        
        return predictions.flatten()
    
    def plot_predictions(self, y_true, y_pred, coin_name='', model_type=''):
        """
        Plot actual vs predicted prices.
        
        Args:
            y_true (np.array): Actual prices
            y_pred (np.array): Predicted prices
            coin_name (str): Name of the coin
            model_type (str): Type of model
            
        Returns:
            matplotlib.figure.Figure: The plot figure
        """
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='Actual', color='blue')
        plt.plot(y_pred, label='Predicted', color='red', linestyle='--')
        plt.title(f'{coin_name} Price Prediction using {model_type}')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        os.makedirs('reports/figures', exist_ok=True)
        plt.savefig(f'reports/figures/{coin_name}_{model_type}_prediction.png')
        
        return plt.gcf()
    
    def plot_future_predictions(self, historical_prices, future_predictions, coin_name='', model_type=''):
        """
        Plot historical prices with future predictions.
        
        Args:
            historical_prices (np.array): Historical prices
            future_predictions (np.array): Predicted future prices
            coin_name (str): Name of the coin
            model_type (str): Type of model
            
        Returns:
            matplotlib.figure.Figure: The plot figure
        """
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        plt.plot(range(len(historical_prices)), historical_prices, label='Historical', color='blue')
        
        # Plot future predictions
        plt.plot(
            range(len(historical_prices) - 1, len(historical_prices) + len(future_predictions) - 1),
            future_predictions,
            label='Predicted',
            color='red',
            linestyle='--'
        )
        
        # Add vertical line to separate historical from predicted
        plt.axvline(x=len(historical_prices) - 1, color='green', linestyle='-', alpha=0.5)
        plt.text(len(historical_prices) - 1, min(historical_prices), 'Now', rotation=90, alpha=0.5)
        
        plt.title(f'{coin_name} Future Price Prediction using {model_type}')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        os.makedirs('reports/figures', exist_ok=True)
        plt.savefig(f'reports/figures/{coin_name}_{model_type}_future_prediction.png')
        
        return plt.gcf()
    
    def predict_with_sentiment(self, df, model, window_size=10, coin_name='generic', model_type='lstm'):
        """
        Make predictions with a model that incorporates sentiment data.
        
        Args:
            df (pd.DataFrame): DataFrame with price and sentiment data
            model: Trained model
            window_size (int): Window size used during training
            coin_name (str): Name of the coin
            model_type (str): Type of model
            
        Returns:
            pd.DataFrame: DataFrame with actual and predicted prices
        """
        # Prepare sequences for prediction
        X, y, feature_cols = self.prepare_time_series_data(
            df, window_size=window_size, include_sentiment=True
        )
        
        # Make predictions
        if model_type in ['lstm', 'bidirectional']:
            predictions = model.predict(X)
        elif model_type == 'xgboost':
            X_reshaped = X.reshape(X.shape[0], -1)
            dtest = xgb.DMatrix(X_reshaped)
            predictions = model.predict(dtest).reshape(-1, 1)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Denormalize predictions
        predictions = self.price_scaler.inverse_transform(predictions)
        actual_prices = self.price_scaler.inverse_transform(y)
        
        # Create result DataFrame
        results = pd.DataFrame({
            'date': df['date'].iloc[window_size:window_size+len(predictions)].values,
            'actual': actual_prices.flatten(),
            'predicted': predictions.flatten()
        })
        
        return results

# Example usage
if __name__ == "__main__":
    # Example code for using the predictor
    from src.data.data_processor import CryptoDataProcessor
    
    # Initialize data processor
    data_processor = CryptoDataProcessor(
        raw_data_path="data/raw",
        processed_data_path="data/processed"
    )
    
    # Load example data
    data_processor.load_price_data(coin_list=["BTC"])
    data_processor.load_news_data()
    
    # Process and align data
    aligned_data = data_processor.prepare_all_coins_data(coin_list=["BTC"])
    btc_data = aligned_data.get("BTC")
    
    if btc_data is not None:
        # Initialize price predictor
        predictor = CryptoPricePredictor(model_dir='models/saved')
        
        # Prepare data for LSTM
        X, y, feature_cols = predictor.prepare_time_series_data(
            btc_data, target_col='close', window_size=10, include_sentiment=True
        )
        
        # Split data
        train_data, val_data, test_data = predictor.split_data(X, y)
        
        # Train LSTM model
        lstm_model, history = predictor.train_lstm_model(
            train_data, val_data, model_type='bidirectional', 
            epochs=50, coin_name='BTC'
        )
        
        # Evaluate model
        metrics = predictor.evaluate_model(
            lstm_model, test_data, coin_name='BTC', 
            model_type='bidirectional'
        )
        
        # Make future predictions
        last_sequence = X[-1:][0]
        future_prices = predictor.make_future_predictions(
            lstm_model, last_sequence, days_ahead=7, model_type='bidirectional'
        )
        
        print(f"\nPredicted prices for next 7 days: {future_prices}")