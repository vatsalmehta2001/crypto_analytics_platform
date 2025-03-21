import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'visualization'))
import matplotlib_config

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.api import VAR
import os
from datetime import datetime, timedelta
import json
from tqdm import tqdm

class CryptoCorrelationAnalyzer:
    """
    Class for analyzing correlations between sentiment data and cryptocurrency price movements.
    Provides methods for time-lagged correlation analysis, Granger causality tests, and more.
    """
    
    def __init__(self, output_dir='reports'):
        """
        Initialize the correlation analyzer.
        
        Args:
            output_dir (str): Directory to save analysis results
        """
        self.output_dir = output_dir
        self.max_columns = 15  # Maximum number of columns for heatmap to avoid memory issues
        self.max_fig_size = (10, 8)  # Maximum figure size to prevent memory errors
        
        plt.rcParams['figure.max_open_warning'] = 0
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['figure.figsize'] = (8, 6)
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'correlation'), exist_ok=True)
    
    def pearson_correlation(self, df, price_col='close', sentiment_cols=None):
        """
        Compute Pearson correlation between price and sentiment features.
        
        Args:
            df (pd.DataFrame): DataFrame with price and sentiment data
            price_col (str): Price column to use
            sentiment_cols (list): List of sentiment columns to analyze
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        try:
            # Filter to include only numeric columns
            numeric_df = df.select_dtypes(include=['number']).copy()
            
            if sentiment_cols is None:
                # Auto-detect sentiment columns from numeric columns only
                sentiment_cols = [col for col in numeric_df.columns if 
                               'sentiment' in col or 
                               'polarity' in col or 
                               'ratio' in col or 
                               'score' in col]
                
                # Limit the number of columns to avoid oversized plots
                if len(sentiment_cols) > self.max_columns:
                    print(f"Limiting correlation analysis to {self.max_columns} sentiment columns to avoid oversized plots")
                    sentiment_cols = sentiment_cols[:self.max_columns]
            else:
                # Filter to keep only numeric sentiment columns
                sentiment_cols = [col for col in sentiment_cols if col in numeric_df.columns]
            
            # Ensure price column exists
            if price_col not in numeric_df.columns:
                raise ValueError(f"Price column '{price_col}' not found in data")
            
            # Select relevant columns
            columns_to_analyze = [price_col] + sentiment_cols
            corr_data = numeric_df[columns_to_analyze].copy()
            
            # Drop any NaN values
            corr_data = corr_data.dropna()
            
            # Check for constant columns that would cause correlation to be undefined
            for col in corr_data.columns:
                if corr_data[col].std() == 0:
                    print(f"Warning: Column '{col}' is constant, removing from correlation analysis")
                    corr_data = corr_data.drop(columns=[col])
            
            # Compute correlation matrix
            correlation_matrix = corr_data.corr(method='pearson')
            
            return correlation_matrix
        
        except Exception as e:
            print(f"Error in pearson_correlation: {e}")
            # Return a minimal correlation matrix with just the price column
            return pd.DataFrame([[1.0]], index=[price_col], columns=[price_col])
    
    def spearman_correlation(self, df, price_col='close', sentiment_cols=None):
        """
        Compute Spearman rank correlation between price and sentiment features.
        
        Args:
            df (pd.DataFrame): DataFrame with price and sentiment data
            price_col (str): Price column to use
            sentiment_cols (list): List of sentiment columns to analyze
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        try:
            # Filter to include only numeric columns
            numeric_df = df.select_dtypes(include=['number']).copy()
            
            if sentiment_cols is None:
                # Auto-detect sentiment columns from numeric columns only
                sentiment_cols = [col for col in numeric_df.columns if 
                               'sentiment' in col or 
                               'polarity' in col or 
                               'ratio' in col or 
                               'score' in col]
                
                # Limit the number of columns to avoid oversized plots
                if len(sentiment_cols) > self.max_columns:
                    print(f"Limiting correlation analysis to {self.max_columns} sentiment columns to avoid oversized plots")
                    sentiment_cols = sentiment_cols[:self.max_columns]
            else:
                # Filter to keep only numeric sentiment columns
                sentiment_cols = [col for col in sentiment_cols if col in numeric_df.columns]
            
            # Ensure price column exists
            if price_col not in numeric_df.columns:
                raise ValueError(f"Price column '{price_col}' not found in data")
            
            # Select relevant columns
            columns_to_analyze = [price_col] + sentiment_cols
            corr_data = numeric_df[columns_to_analyze].copy()
            
            # Drop any NaN values
            corr_data = corr_data.dropna()
            
            # Check for constant columns that would cause correlation to be undefined
            for col in corr_data.columns:
                if corr_data[col].std() == 0:
                    print(f"Warning: Column '{col}' is constant, removing from correlation analysis")
                    corr_data = corr_data.drop(columns=[col])
            
            # Compute correlation matrix
            correlation_matrix = corr_data.corr(method='spearman')
            
            return correlation_matrix
        
        except Exception as e:
            print(f"Error in spearman_correlation: {e}")
            # Return a minimal correlation matrix with just the price column
            return pd.DataFrame([[1.0]], index=[price_col], columns=[price_col])
    
    def plot_correlation_heatmap(self, correlation_matrix, title='Correlation Heatmap', figsize=(10, 8), save_path=None):
        """
        Plot a heatmap of the correlation matrix.
        
        Args:
            correlation_matrix (pd.DataFrame): Correlation matrix
            title (str): Plot title
            figsize (tuple): Figure size
            save_path (str): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The heatmap figure
        """
        try:
            # Limit matrix size if too large
            if correlation_matrix.shape[0] > self.max_columns:
                print(f"Correlation matrix too large ({correlation_matrix.shape}), limiting to {self.max_columns} columns")
                correlation_matrix = correlation_matrix.iloc[:self.max_columns, :self.max_columns]
            
            # Limit figure size to prevent memory errors
            figsize = (min(figsize[0], self.max_fig_size[0]), min(figsize[1], self.max_fig_size[1]))
            
            plt.figure(figsize=figsize)
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(
                correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                vmin=-1, 
                vmax=1, 
                center=0, 
                fmt='.2f',
                mask=mask
            )
            plt.title(title)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=200, bbox_inches='tight')
            
            return plt.gcf()
        
        except Exception as e:
            print(f"Error plotting correlation heatmap: {e}")
            plt.figure(figsize=(6, 4))
            plt.text(0.5, 0.5, f"Error creating heatmap: {str(e)}", 
                    ha='center', va='center', fontsize=10, color='red')
            return plt.gcf()
    
    def time_lagged_correlation(self, df, price_col='close', sentiment_col='sentiment_score', 
                              max_lag=10, method='pearson'):
        """
        Compute time-lagged correlation between price and sentiment.
        
        Args:
            df (pd.DataFrame): DataFrame with price and sentiment data
            price_col (str): Price column to use
            sentiment_col (str): Sentiment column to analyze
            max_lag (int): Maximum number of days to lag
            method (str): Correlation method ('pearson' or 'spearman')
            
        Returns:
            pd.DataFrame: Lagged correlations
        """
        try:
            # Filter to include only numeric columns
            numeric_df = df.select_dtypes(include=['number']).copy()
            
            # Ensure required columns exist
            if price_col not in numeric_df.columns:
                raise ValueError(f"Price column '{price_col}' not found in data")
            
            if sentiment_col not in numeric_df.columns:
                raise ValueError(f"Sentiment column '{sentiment_col}' not found in data")
            
            # Calculate price changes (returns)
            numeric_df['price_change'] = numeric_df[price_col].pct_change()
            
            # Drop missing values to ensure clean data
            valid_data = numeric_df.dropna(subset=['price_change', sentiment_col])
            
            # Initialize results dictionary
            correlations = {}
            p_values = {}
            
            # Loop through each lag value from -max_lag to +max_lag
            for lag in range(-max_lag, max_lag + 1):
                try:
                    if lag == 0:
                        # Direct correlation, no lag
                        if method == 'pearson':
                            corr, p_value = stats.pearsonr(valid_data[sentiment_col], valid_data['price_change'])
                        else:
                            corr, p_value = stats.spearmanr(valid_data[sentiment_col], valid_data['price_change'])
                    elif lag > 0:
                        # Sentiment leads price (positive lag)
                        if len(valid_data) <= lag:
                            continue
                            
                        x = valid_data[sentiment_col].iloc[:-lag].values
                        y = valid_data['price_change'].iloc[lag:].values
                        
                        # Ensure arrays have same length
                        min_len = min(len(x), len(y))
                        x = x[:min_len]
                        y = y[:min_len]
                        
                        if len(x) < 2:
                            continue
                            
                        if method == 'pearson':
                            corr, p_value = stats.pearsonr(x, y)
                        else:
                            corr, p_value = stats.spearmanr(x, y)
                    else:
                        # Price leads sentiment (negative lag)
                        lag_abs = abs(lag)
                        if len(valid_data) <= lag_abs:
                            continue
                            
                        x = valid_data['price_change'].iloc[:-lag_abs].values
                        y = valid_data[sentiment_col].iloc[lag_abs:].values
                        
                        # Ensure arrays have same length
                        min_len = min(len(x), len(y))
                        x = x[:min_len]
                        y = y[:min_len]
                        
                        if len(x) < 2:
                            continue
                            
                        if method == 'pearson':
                            corr, p_value = stats.pearsonr(x, y)
                        else:
                            corr, p_value = stats.spearmanr(x, y)
                    
                    # Store results
                    correlations[lag] = corr
                    p_values[lag] = p_value
                    
                except Exception as e:
                    print(f"Error calculating correlation for lag {lag}: {e}")
                    # Skip this lag
                    continue
            
            # Create results DataFrame
            results = pd.DataFrame({
                'lag': list(correlations.keys()),
                'correlation': list(correlations.values()),
                'p_value': list(p_values.values()),
                'significant': [p < 0.05 for p in p_values.values()]
            })
            
            # Sort by lag
            results = results.sort_values('lag').reset_index(drop=True)
            
            return results
            
        except Exception as e:
            print(f"Error in time_lagged_correlation: {e}")
            # Return empty DataFrame with expected structure
            return pd.DataFrame(columns=['lag', 'correlation', 'p_value', 'significant'])
    
    def plot_lagged_correlation(self, lagged_corr_df, title='Time-Lagged Correlation', figsize=(10, 6), save_path=None):
        """
        Plot time-lagged correlation between price and sentiment.
        
        Args:
            lagged_corr_df (pd.DataFrame): DataFrame with lagged correlations
            title (str): Plot title
            figsize (tuple): Figure size
            save_path (str): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The figure
        """
        try:
            # Check if DataFrame is empty
            if lagged_corr_df.empty:
                plt.figure(figsize=(6, 4))
                plt.text(0.5, 0.5, "No lagged correlation data available", 
                        ha='center', va='center', fontsize=12)
                plt.title(title)
                return plt.gcf()
            
            # Limit figure size to prevent memory errors
            figsize = (min(figsize[0], self.max_fig_size[0]), min(figsize[1], self.max_fig_size[1]))
            plt.figure(figsize=figsize)
            
            # Plot correlations
            plt.bar(
                lagged_corr_df['lag'], 
                lagged_corr_df['correlation'], 
                alpha=0.7,
                color=lagged_corr_df['significant'].map({True: 'blue', False: 'gray'})
            )
            
            # Add significance markers
            for idx, row in lagged_corr_df.iterrows():
                if row['significant']:
                    plt.text(
                        row['lag'], 
                        row['correlation'] + 0.02 * np.sign(row['correlation']),
                        '*', 
                        ha='center', 
                        fontsize=14
                    )
            
            # Add reference line at zero
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add reference line for significance threshold
            plt.axhline(y=0.2, color='red', linestyle='--', alpha=0.3)
            plt.axhline(y=-0.2, color='red', linestyle='--', alpha=0.3)
            
            # Add labels and title
            plt.xlabel('Lag (days)')
            plt.ylabel('Correlation')
            plt.title(title)
            plt.grid(True, alpha=0.3)
            
            # Annotate the direction of causality
            max_lag = lagged_corr_df['lag'].abs().max() if not lagged_corr_df.empty else 5
            plt.text(-max_lag * 0.8, 0.9, 'Price leads Sentiment', ha='center')
            plt.text(max_lag * 0.8, 0.9, 'Sentiment leads Price', ha='center')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=200, bbox_inches='tight')
            
            return plt.gcf()
            
        except Exception as e:
            print(f"Error plotting lagged correlation: {e}")
            plt.figure(figsize=(6, 4))
            plt.text(0.5, 0.5, f"Error creating lagged correlation plot: {str(e)}", 
                    ha='center', va='center', fontsize=10, color='red')
            return plt.gcf()
    
    def granger_causality_test(self, df, x_col, y_col, max_lag=10):
        """
        Perform Granger causality test to determine if x causes y.
        
        Args:
            df (pd.DataFrame): DataFrame with time series data
            x_col (str): Column name for potential causal variable
            y_col (str): Column name for potential effect variable
            max_lag (int): Maximum number of lags to test
            
        Returns:
            dict: Dictionary of test results
        """
        try:
            # Filter to include only numeric columns
            numeric_df = df.select_dtypes(include=['number']).copy()
            
            # Ensure columns exist
            if x_col not in numeric_df.columns:
                raise ValueError(f"Column '{x_col}' not found in data")
            if y_col not in numeric_df.columns:
                raise ValueError(f"Column '{y_col}' not found in data")
            
            # Create a dataframe with only the relevant columns
            data = pd.DataFrame({
                x_col: numeric_df[x_col],
                y_col: numeric_df[y_col]
            }).dropna()
            
            # Make sure we have enough data
            if len(data) <= max_lag + 1:
                return {
                    'x_causes_y': {'error': 'Not enough data points for testing'},
                    'y_causes_x': {'error': 'Not enough data points for testing'}
                }
            
            # Perform ADF test to check stationarity
            try:
                x_adf = adfuller(data[x_col])
                y_adf = adfuller(data[y_col])
                
                # Take first difference if non-stationary
                if x_adf[1] > 0.05:
                    data[f'{x_col}_diff'] = data[x_col].diff().dropna()
                    x_col = f'{x_col}_diff'
                
                if y_adf[1] > 0.05:
                    data[f'{y_col}_diff'] = data[y_col].diff().dropna()
                    y_col = f'{y_col}_diff'
                
                # Drop NAs introduced by differencing
                data = data.dropna()
            except:
                # If ADF test fails, continue with original data
                pass
            
            # Check if we still have enough data after preprocessing
            if len(data) <= max_lag + 1:
                return {
                    'x_causes_y': {'error': 'Not enough data points after preprocessing'},
                    'y_causes_x': {'error': 'Not enough data points after preprocessing'}
                }
            
            # Adjust max_lag if needed
            max_lag = min(max_lag, len(data) // 5)  # Rule of thumb: at least 5 observations per parameter
            
            # Run Granger causality test
            results = {}
            
            # Test if x Granger-causes y
            try:
                xy_test = grangercausalitytests(data[[y_col, x_col]], maxlag=max_lag, verbose=False)
                results['x_causes_y'] = {
                    'lags': {},
                    'min_p_value': min([xy_test[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1)]),
                    'optimal_lag': min(
                        range(1, max_lag+1), 
                        key=lambda lag: xy_test[lag][0]['ssr_ftest'][1]
                    )
                }
                for lag in range(1, max_lag+1):
                    results['x_causes_y']['lags'][lag] = {
                        'f_stat': xy_test[lag][0]['ssr_ftest'][0],
                        'p_value': xy_test[lag][0]['ssr_ftest'][1],
                        'significant': xy_test[lag][0]['ssr_ftest'][1] < 0.05
                    }
            except Exception as e:
                results['x_causes_y'] = {'error': str(e)}
            
            # Test if y Granger-causes x
            try:
                yx_test = grangercausalitytests(data[[x_col, y_col]], maxlag=max_lag, verbose=False)
                results['y_causes_x'] = {
                    'lags': {},
                    'min_p_value': min([yx_test[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1)]),
                    'optimal_lag': min(
                        range(1, max_lag+1), 
                        key=lambda lag: yx_test[lag][0]['ssr_ftest'][1]
                    )
                }
                for lag in range(1, max_lag+1):
                    results['y_causes_x']['lags'][lag] = {
                        'f_stat': yx_test[lag][0]['ssr_ftest'][0],
                        'p_value': yx_test[lag][0]['ssr_ftest'][1],
                        'significant': yx_test[lag][0]['ssr_ftest'][1] < 0.05
                    }
            except Exception as e:
                results['y_causes_x'] = {'error': str(e)}
            
            return results
            
        except Exception as e:
            print(f"Error in granger_causality_test: {e}")
            return {
                'x_causes_y': {'error': str(e)},
                'y_causes_x': {'error': str(e)}
            }
    
    def analyze_correlation_over_time(self, df, price_col='close', sentiment_col='sentiment_score',
                                     window_size=30, step=7, method='pearson'):
        """
        Analyze how correlation between price and sentiment changes over time.
        
        Args:
            df (pd.DataFrame): DataFrame with price and sentiment data
            price_col (str): Price column to use
            sentiment_col (str): Sentiment column to analyze
            window_size (int): Size of rolling window in days
            step (int): Number of days to step forward for each window
            method (str): Correlation method ('pearson' or 'spearman')
            
        Returns:
            pd.DataFrame: Rolling correlations over time
        """
        try:
            # Filter to include only numeric columns
            numeric_df = df.select_dtypes(include=['number']).copy()
            
            # Ensure columns exist
            if price_col not in numeric_df.columns:
                raise ValueError(f"Price column '{price_col}' not found in data")
            if sentiment_col not in numeric_df.columns:
                raise ValueError(f"Sentiment column '{sentiment_col}' not found in data")
            
            # Calculate price changes
            numeric_df['price_change'] = numeric_df[price_col].pct_change()
            
            # Drop NAs to ensure clean data
            valid_data = numeric_df.dropna(subset=['price_change', sentiment_col])
            
            # Initialize results
            results = []
            
            # Ensure dataframe is sorted by date
            if 'date' in df.columns:
                date_col = df['date']
                valid_data = pd.concat([valid_data, date_col], axis=1)
                valid_data = valid_data.sort_values('date').reset_index(drop=True)
            
            # Check if we have enough data
            if len(valid_data) < window_size:
                print(f"Warning: Not enough data for window size {window_size}. Have {len(valid_data)} rows.")
                if len(valid_data) < 5:  # Too little data for any meaningful window
                    return pd.DataFrame(columns=['start_date', 'end_date', 'correlation', 'p_value', 'significant'])
                
                # Reduce window size to accommodate smaller dataset
                window_size = max(5, len(valid_data) // 2)
                print(f"Reduced window size to {window_size}")
            
            # Sliding window analysis
            for start_idx in range(0, len(valid_data) - window_size, step):
                end_idx = start_idx + window_size
                window_df = valid_data.iloc[start_idx:end_idx].copy()
                
                # Calculate correlation for this window
                try:
                    if method == 'pearson':
                        corr, p_value = stats.pearsonr(
                            window_df[sentiment_col], 
                            window_df['price_change']
                        )
                    elif method == 'spearman':
                        corr, p_value = stats.spearmanr(
                            window_df[sentiment_col], 
                            window_df['price_change']
                        )
                    else:
                        raise ValueError(f"Unknown correlation method: {method}")
                    
                    # Get start and end dates for this window
                    if 'date' in window_df.columns:
                        start_date = window_df['date'].iloc[0]
                        end_date = window_df['date'].iloc[-1]
                    else:
                        start_date = start_idx
                        end_date = end_idx
                    
                    # Store results
                    results.append({
                        'start_date': start_date,
                        'end_date': end_date,
                        'correlation': corr,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    })
                except Exception as e:
                    print(f"Skipping window {start_idx}-{end_idx} due to error: {e}")
            
            # Create DataFrame from results
            results_df = pd.DataFrame(results) if results else pd.DataFrame(
                columns=['start_date', 'end_date', 'correlation', 'p_value', 'significant']
            )
            
            return results_df
            
        except Exception as e:
            print(f"Error in analyze_correlation_over_time: {e}")
            return pd.DataFrame(columns=['start_date', 'end_date', 'correlation', 'p_value', 'significant'])
    
    def plot_rolling_correlation(self, rolling_corr_df, title='Rolling Correlation', figsize=(10, 6), save_path=None):
        """
        Plot rolling correlation between price and sentiment.
        
        Args:
            rolling_corr_df (pd.DataFrame): DataFrame with rolling correlations
            title (str): Plot title
            figsize (tuple): Figure size
            save_path (str): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The figure
        """
        try:
            # Check if DataFrame is empty
            if rolling_corr_df.empty:
                plt.figure(figsize=(6, 4))
                plt.text(0.5, 0.5, "No rolling correlation data available", 
                        ha='center', va='center', fontsize=12)
                plt.title(title)
                return plt.gcf()
            
            # Limit figure size to prevent memory errors
            figsize = (min(figsize[0], self.max_fig_size[0]), min(figsize[1], self.max_fig_size[1]))
            plt.figure(figsize=figsize)
            
            # Create x-axis based on dates if available
            if isinstance(rolling_corr_df['end_date'].iloc[0], (datetime, pd.Timestamp)):
                x = rolling_corr_df['end_date']
            else:
                x = range(len(rolling_corr_df))
            
            # Plot correlations
            plt.plot(x, rolling_corr_df['correlation'], marker='o', linestyle='-', markersize=4)
            
            # Color significant points differently
            significant_mask = rolling_corr_df['significant']
            if any(significant_mask):
                plt.scatter(
                    x[significant_mask], 
                    rolling_corr_df.loc[significant_mask, 'correlation'],
                    color='red',
                    s=50,
                    label='Significant (p<0.05)',
                    zorder=5
                )
            
            # Add reference line at zero
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add trend line if enough points
            if len(rolling_corr_df) > 2:
                try:
                    z = np.polyfit(range(len(rolling_corr_df)), rolling_corr_df['correlation'], 1)
                    p = np.poly1d(z)
                    plt.plot(x, p(range(len(rolling_corr_df))), "r--", alpha=0.5, label='Trend')
                except Exception as e:
                    print(f"Could not plot trend line: {e}")
            
            # Add labels and title
            plt.xlabel('Time')
            plt.ylabel('Correlation')
            plt.title(title)
            plt.grid(True, alpha=0.3)
            
            # Add legend if we have significant points or a trend line
            if any(significant_mask) or len(rolling_corr_df) > 2:
                plt.legend()
            
            # Format x-axis if using dates
            if isinstance(rolling_corr_df['end_date'].iloc[0], (datetime, pd.Timestamp)):
                plt.gcf().autofmt_xdate()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=200, bbox_inches='tight')
            
            return plt.gcf()
            
        except Exception as e:
            print(f"Error plotting rolling correlation: {e}")
            plt.figure(figsize=(6, 4))
            plt.text(0.5, 0.5, f"Error creating rolling correlation plot: {str(e)}", 
                    ha='center', va='center', fontsize=10, color='red')
            return plt.gcf()
    
    def analyze_event_impact(self, df, event_date, event_name, window_before=10, window_after=10,
                           price_col='close', sentiment_col='sentiment_score'):
        """
        Analyze the impact of a specific event on price and sentiment.
        
        Args:
            df (pd.DataFrame): DataFrame with price and sentiment data
            event_date (str or datetime): Date of the event
            event_name (str): Name of the event
            window_before (int): Number of days to analyze before the event
            window_after (int): Number of days to analyze after the event
            price_col (str): Price column to use
            sentiment_col (str): Sentiment column to analyze
            
        Returns:
            dict: Event impact analysis results
        """
        try:
            # Filter to include only numeric columns
            numeric_df = df.select_dtypes(include=['number']).copy()
            
            # We need the date column
            if 'date' not in df.columns:
                raise ValueError("DataFrame must have a 'date' column")
            
            # Add date column to numeric_df
            numeric_df['date'] = df['date']
            
            # Convert event_date to datetime if it's a string
            if isinstance(event_date, str):
                event_date = pd.to_datetime(event_date)
            
            # Ensure price and sentiment columns exist
            if price_col not in numeric_df.columns:
                raise ValueError(f"Price column '{price_col}' not found in data")
            if sentiment_col not in numeric_df.columns:
                raise ValueError(f"Sentiment column '{sentiment_col}' not found in data")
            
            # Convert dataframe date to datetime if needed
            if not pd.api.types.is_datetime64_dtype(numeric_df['date']):
                numeric_df['date'] = pd.to_datetime(numeric_df['date'])
            
            # Find the event date in the dataframe
            event_idx = numeric_df[numeric_df['date'] == event_date].index
            if len(event_idx) == 0:
                # If exact date not found, find the closest date
                closest_date_idx = (numeric_df['date'] - event_date).abs().idxmin()
                event_idx = [closest_date_idx]
                event_date = numeric_df.loc[closest_date_idx, 'date']
            
            event_idx = event_idx[0]
            
            # Get data before and after the event
            before_start = max(0, event_idx - window_before)
            after_end = min(len(numeric_df), event_idx + window_after + 1)
            
            before_event = numeric_df.iloc[before_start:event_idx].copy()
            after_event = numeric_df.iloc[event_idx+1:after_end].copy()
            event_day = numeric_df.iloc[event_idx:event_idx+1].copy()
            
            # Calculate price changes
            df_window = numeric_df.iloc[before_start:after_end].copy()
            df_window['price_change'] = df_window[price_col].pct_change()
            
            # Calculate basic statistics
            avg_sentiment_before = before_event[sentiment_col].mean() if not before_event.empty else None
            avg_sentiment_after = after_event[sentiment_col].mean() if not after_event.empty else None
            sentiment_change = avg_sentiment_after - avg_sentiment_before if avg_sentiment_before is not None and avg_sentiment_after is not None else None
            
            price_before = before_event[price_col].iloc[-1] if len(before_event) > 0 else None
            price_event = event_day[price_col].iloc[0] if len(event_day) > 0 else None
            price_after = after_event[price_col].iloc[-1] if len(after_event) > 0 else None
            
            try:
                price_change_at_event = ((price_event / price_before) - 1) * 100 if price_before and price_event and price_before > 0 else None
            except:
                price_change_at_event = None
                
            try:
                price_change_after = ((price_after / price_event) - 1) * 100 if price_event and price_after and price_event > 0 else None
            except:
                price_change_after = None
            
            # Calculate correlation before and after
            try:
                corr_before = before_event[[price_col, sentiment_col]].corr().iloc[0, 1] if len(before_event) > 1 else None
            except:
                corr_before = None
                
            try:
                corr_after = after_event[[price_col, sentiment_col]].corr().iloc[0, 1] if len(after_event) > 1 else None
            except:
                corr_after = None
            
            # Collect results
            results = {
                'event_name': event_name,
                'event_date': event_date,
                'sentiment': {
                    'before': float(avg_sentiment_before) if avg_sentiment_before is not None else None,
                    'event_day': float(event_day[sentiment_col].iloc[0]) if len(event_day) > 0 else None,
                    'after': float(avg_sentiment_after) if avg_sentiment_after is not None else None,
                    'change': float(sentiment_change) if sentiment_change is not None else None,
                    'change_pct': float((sentiment_change / avg_sentiment_before) * 100) 
                                if sentiment_change is not None and avg_sentiment_before and avg_sentiment_before != 0 else None
                },
                'price': {
                    'before': float(price_before) if price_before is not None else None,
                    'event_day': float(price_event) if price_event is not None else None,
                    'after': float(price_after) if price_after is not None else None,
                    'change_at_event_pct': float(price_change_at_event) if price_change_at_event is not None else None,
                    'change_after_event_pct': float(price_change_after) if price_change_after is not None else None
                },
                'correlation': {
                    'before': float(corr_before) if corr_before is not None else None,
                    'after': float(corr_after) if corr_after is not None else None,
                    'change': float(corr_after - corr_before) if corr_before is not None and corr_after is not None else None
                },
                'window': {
                    'before_days': window_before,
                    'after_days': window_after
                }
            }
            
            return results
            
        except Exception as e:
            print(f"Error in analyze_event_impact: {e}")
            return {
                'event_name': event_name,
                'event_date': event_date,
                'error': str(e),
                'window': {
                    'before_days': window_before,
                    'after_days': window_after
                }
            }
    
    def plot_event_impact(self, df, event_date, event_name, window_before=10, window_after=10,
                        price_col='close', sentiment_col='sentiment_score', save_path=None):
        """
        Plot the impact of a specific event on price and sentiment.
        
        Args:
            df (pd.DataFrame): DataFrame with price and sentiment data
            event_date (str or datetime): Date of the event
            event_name (str): Name of the event
            window_before (int): Number of days to analyze before the event
            window_after (int): Number of days to analyze after the event
            price_col (str): Price column to use
            sentiment_col (str): Sentiment column to analyze
            save_path (str): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The figure
        """
        try:
            # Filter to include only numeric columns but keep the date column
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            df_numeric = df[numeric_cols + ['date']].copy()
            
            # Convert event_date to datetime if it's a string
            if isinstance(event_date, str):
                event_date = pd.to_datetime(event_date)
            
            # Ensure dataframe date is datetime
            if not pd.api.types.is_datetime64_dtype(df_numeric['date']):
                df_numeric['date'] = pd.to_datetime(df_numeric['date'])
            
            # Ensure required columns exist
            if price_col not in df_numeric.columns:
                raise ValueError(f"Price column '{price_col}' not found in data")
            if sentiment_col not in df_numeric.columns:
                raise ValueError(f"Sentiment column '{sentiment_col}' not found in data")
            
            # Find the event date index
            event_idx = df_numeric[df_numeric['date'] == event_date].index
            if len(event_idx) == 0:
                closest_date_idx = (df_numeric['date'] - event_date).abs().idxmin()
                event_idx = [closest_date_idx]
                event_date = df_numeric.loc[closest_date_idx, 'date']
            
            event_idx = event_idx[0]
            
            # Get data window
            before_start = max(0, event_idx - window_before)
            after_end = min(len(df_numeric), event_idx + window_after + 1)
            df_window = df_numeric.iloc[before_start:after_end].copy()
            
            # Check if we have enough data
            if len(df_window) < 3:
                fig = plt.figure(figsize=(8, 5))
                plt.text(0.5, 0.5, "Not enough data for event impact analysis", 
                        ha='center', va='center', fontsize=14)
                plt.title(f'Impact of {event_name} - Insufficient Data')
                plt.tight_layout()
                return fig
            
            # Limit figure size to prevent memory errors
            figsize = (min(12, self.max_fig_size[0]), min(6, self.max_fig_size[1]))
            fig, ax1 = plt.subplots(figsize=figsize)
            
            # Plot price data
            ax1.set_xlabel('Date')
            ax1.set_ylabel(f'{price_col.capitalize()} Price', color='tab:blue')
            ax1.plot(df_window['date'], df_window[price_col], 'b-', label=price_col.capitalize(), linewidth=2)
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            
            # Add sentiment data on secondary axis
            ax2 = ax1.twinx()
            ax2.set_ylabel('Sentiment', color='tab:red')
            ax2.plot(df_window['date'], df_window[sentiment_col], 'r-', label='Sentiment', linewidth=2)
            ax2.tick_params(axis='y', labelcolor='tab:red')
            
            # Mark the event date
            event_date_val = df_window[df_window['date'] == event_date]['date'].iloc[0]
            ax1.axvline(x=event_date_val, color='g', linestyle='--', alpha=0.7, linewidth=2)
            
            # Add event label positioned above the chart
            y_pos = max(df_window[price_col]) * 1.05
            ax1.text(event_date_val, y_pos, event_name, 
                    rotation=90, verticalalignment='bottom', 
                    horizontalalignment='center', fontweight='bold')
            
            # Add legends and title
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            plt.title(f'Impact of {event_name} on {price_col.capitalize()} and Sentiment')
            fig.autofmt_xdate()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=200, bbox_inches='tight')
            
            return fig
            
        except Exception as e:
            print(f"Error plotting event impact: {e}")
            plt.figure(figsize=(8, 5))
            plt.text(0.5, 0.5, f"Error creating event impact plot: {str(e)}", 
                    ha='center', va='center', fontsize=10, color='red')
            return plt.gcf()
    
    def compare_coins_sentiment_correlation(self, coin_data_dict, sentiment_col='sentiment_score', 
                                         price_col='close', method='pearson'):
        """
        Compare sentiment-price correlation across multiple cryptocurrencies.
        
        Args:
            coin_data_dict (dict): Dictionary of coin tickers to DataFrames
            sentiment_col (str): Sentiment column to analyze
            price_col (str): Price column to use
            method (str): Correlation method ('pearson' or 'spearman')
            
        Returns:
            pd.DataFrame: Correlation comparison results
        """
        results = []
        
        for coin, df in coin_data_dict.items():
            # Filter to numeric columns
            numeric_df = df.select_dtypes(include=['number']).copy()
            
            # Ensure required columns exist
            if price_col not in numeric_df.columns:
                print(f"Warning: Price column '{price_col}' not found for {coin}, skipping")
                continue
            if sentiment_col not in numeric_df.columns:
                print(f"Warning: Sentiment column '{sentiment_col}' not found for {coin}, skipping")
                continue
            
            # Calculate price returns
            numeric_df['price_change'] = numeric_df[price_col].pct_change()
            
            # Drop NAs for clean calculation
            valid_data = numeric_df.dropna(subset=['price_change', sentiment_col])
            
            # Calculate correlation with sentiment
            try:
                if method == 'pearson':
                    corr, p_value = stats.pearsonr(valid_data[sentiment_col], valid_data['price_change'])
                elif method == 'spearman':
                    corr, p_value = stats.spearmanr(valid_data[sentiment_col], valid_data['price_change'])
                else:
                    raise ValueError(f"Unknown correlation method: {method}")
                
                # Calculate volatility
                volatility = valid_data['price_change'].std() * np.sqrt(365)  # Annualized
                
                # Calculate average sentiment
                avg_sentiment = valid_data[sentiment_col].mean()
                sentiment_volatility = valid_data[sentiment_col].std()
                
                # Store results
                results.append({
                    'coin': coin,
                    'correlation': corr,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'price_volatility': volatility,
                    'avg_sentiment': avg_sentiment,
                    'sentiment_volatility': sentiment_volatility,
                    'data_points': len(valid_data)
                })
            except Exception as e:
                print(f"Error calculating correlation for {coin}: {e}")
        
        # Convert to DataFrame and sort by correlation strength
        if results:
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('correlation', key=abs, ascending=False).reset_index(drop=True)
            return results_df
        else:
            return pd.DataFrame(columns=['coin', 'correlation', 'p_value', 'significant', 
                                       'price_volatility', 'avg_sentiment', 
                                       'sentiment_volatility', 'data_points'])
    
    def plot_coin_comparison(self, comparison_df, metric='correlation', figsize=(10, 6), save_path=None):
        """
        Plot comparison of sentiment metrics across multiple cryptocurrencies.
        
        Args:
            comparison_df (pd.DataFrame): DataFrame with coin comparison data
            metric (str): Metric to plot (correlation, price_volatility, etc.)
            figsize (tuple): Figure size
            save_path (str): Path to save the figure
            
        Returns:
            matplotlib.figure.Figure: The figure
        """
        try:
            # Check if DataFrame is empty or missing required columns
            if comparison_df.empty or metric not in comparison_df.columns or 'coin' not in comparison_df.columns:
                plt.figure(figsize=(6, 4))
                plt.text(0.5, 0.5, "Insufficient data for coin comparison", 
                        ha='center', va='center', fontsize=12)
                return plt.gcf()
            
            # Limit figure size to prevent memory errors
            figsize = (min(figsize[0], self.max_fig_size[0]), min(figsize[1], self.max_fig_size[1]))
            plt.figure(figsize=figsize)
            
            # Sort by absolute correlation value if plotting correlation
            if metric == 'correlation':
                plot_df = comparison_df.sort_values('correlation', key=abs, ascending=False)
                
                # Use blue for positive correlations, red for negative
                colors = ['red' if val < 0 else 'blue' for val in plot_df[metric]]
                
                # Add significance markers
                alpha = [1.0 if sig else 0.5 for sig in plot_df['significant']]
            else:
                plot_df = comparison_df.sort_values(metric, ascending=False)
                colors = 'blue'  # Single color for other metrics
                alpha = 0.7
            
            # Create the bar chart
            bars = plt.bar(
                plot_df['coin'], 
                plot_df[metric], 
                color=colors,
                alpha=alpha
            )
            
            # Add reference line at zero for correlation
            if metric == 'correlation':
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom')
            
            # Format y-axis based on metric
            if metric == 'correlation':
                plt.ylabel('Correlation Coefficient')
            elif 'volatility' in metric:
                plt.ylabel(f'{metric.replace("_", " ").title()}')
            else:
                plt.ylabel(metric.replace('_', ' ').title())
            
            plt.xlabel('Cryptocurrency')
            plt.title(f'Cryptocurrency {metric.replace("_", " ").title()} Comparison')
            plt.grid(axis='y', alpha=0.3)
            
            # Add legend for correlation
            if metric == 'correlation':
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='blue', alpha=1.0, label='Positive (significant)'),
                    Patch(facecolor='blue', alpha=0.5, label='Positive (not significant)'),
                    Patch(facecolor='red', alpha=1.0, label='Negative (significant)'),
                    Patch(facecolor='red', alpha=0.5, label='Negative (not significant)')
                ]
                plt.legend(handles=legend_elements, loc='best')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=200, bbox_inches='tight')
            
            return plt.gcf()
            
        except Exception as e:
            print(f"Error plotting coin comparison: {e}")
            plt.figure(figsize=(6, 4))
            plt.text(0.5, 0.5, f"Error creating coin comparison plot: {str(e)}", 
                    ha='center', va='center', fontsize=10, color='red')
            return plt.gcf()

# Example usage
if __name__ == "__main__":
    # Example code for using the analyzer
    from src.data.data_processor import CryptoDataProcessor
    
    # Initialize data processor
    data_processor = CryptoDataProcessor(
        raw_data_path="data/raw",
        processed_data_path="data/processed"
    )
    
    # Load example data
    data_processor.load_price_data(coin_list=["BTC", "ETH"])
    data_processor.load_news_data()
    
    # Process and align data
    aligned_data = data_processor.prepare_all_coins_data(coin_list=["BTC", "ETH"])
    btc_data = aligned_data.get("BTC")
    
    if btc_data is not None:
        # Initialize correlation analyzer
        analyzer = CryptoCorrelationAnalyzer(output_dir='reports')
        
        # Calculate correlations
        correlation_matrix = analyzer.pearson_correlation(btc_data)
        
        # Plot correlation heatmap
        heatmap_fig = analyzer.plot_correlation_heatmap(
            correlation_matrix,
            title='BTC Price-Sentiment Correlation',
            save_path='reports/figures/btc_correlation_heatmap.png'
        )
        
        # Calculate time-lagged correlations
        lagged_corr = analyzer.time_lagged_correlation(
            btc_data,
            price_col='close',
            sentiment_col='sentiment_score',
            max_lag=10
        )
        
        # Plot lagged correlations
        lagged_fig = analyzer.plot_lagged_correlation(
            lagged_corr,
            title='BTC Time-Lagged Correlation: Price vs Sentiment',
            save_path='reports/figures/btc_lagged_correlation.png'
        )
        
        # Perform Granger causality test
        granger_results = analyzer.granger_causality_test(
            btc_data,
            x_col='sentiment_score',
            y_col='close',
            max_lag=5
        )
        
        # Print Granger causality results
        print("\nGranger Causality Test Results:")
        print(f"Does sentiment Granger-cause price? p-value: {granger_results['x_causes_y'].get('min_p_value', 'Error')}")
        print(f"Does price Granger-cause sentiment? p-value: {granger_results['y_causes_x'].get('min_p_value', 'Error')}")
        
        # Calculate rolling correlations
        rolling_corr = analyzer.analyze_correlation_over_time(
            btc_data,
            price_col='close',
            sentiment_col='sentiment_score',
            window_size=30
        )
        
        # Plot rolling correlations
        rolling_fig = analyzer.plot_rolling_correlation(
            rolling_corr,
            title='BTC Rolling Correlation: Price vs Sentiment',
            save_path='reports/figures/btc_rolling_correlation.png'
        )
        
        # Compare coins if both BTC and ETH data are available
        if "ETH" in aligned_data:
            eth_data = aligned_data.get("ETH")
            
            # Compare correlations across coins
            comparison = analyzer.compare_coins_sentiment_correlation(
                {"BTC": btc_data, "ETH": eth_data},
                sentiment_col='sentiment_score',
                price_col='close'
            )
            
            print("\nCoin Comparison Results:")
            print(comparison)
            
            # Plot comparison
            comparison_fig = analyzer.plot_coin_comparison(
                comparison,
                metric='correlation',
                save_path='reports/figures/coin_correlation_comparison.png'
            )
            
            # Save comparison results
            comparison.to_csv('reports/correlation/coin_sentiment_correlation_comparison.csv', index=False)
            
        print("\nCorrelation analysis completed. Results saved to 'reports' directory.")