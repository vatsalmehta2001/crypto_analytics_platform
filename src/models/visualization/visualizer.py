import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
from wordcloud import WordCloud
import json
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

class CryptoVisualizer:
    """
    Class for creating visualizations of cryptocurrency data and sentiment analysis.
    Provides methods for creating interactive charts and dashboard components.
    """
    
    def __init__(self, output_dir='reports/figures'):
        """
        Initialize the visualizer with an output directory.
        
        Args:
            output_dir (str): Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Suppress deprecation warnings
        warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    def plot_price_sentiment_time_series(self, df, price_col='close', sentiment_col='sentiment_score',
                                       title=None, coin_name='', save_path=None):
        """
        Create an interactive time series plot of price and sentiment.
        
        Args:
            df (pd.DataFrame): DataFrame with price and sentiment data
            price_col (str): Price column to plot
            sentiment_col (str): Sentiment column to plot
            title (str): Plot title
            coin_name (str): Name of the cryptocurrency
            save_path (str): Path to save the HTML file
            
        Returns:
            plotly.graph_objects.Figure: The figure
        """
        if title is None:
            title = f'{coin_name} Price and Sentiment Over Time'
        
        # Ensure date column is datetime
        if 'date' in df.columns and not pd.api.types.is_datetime64_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Create subplot with two y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add price line
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df[price_col],
                name=f'{price_col.capitalize()} Price',
                line=dict(color='blue')
            ),
            secondary_y=False
        )
        
        # Add sentiment line
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df[sentiment_col],
                name='Sentiment Score',
                line=dict(color='red')
            ),
            secondary_y=True
        )
        
        # Add layout details
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
            hovermode='x unified',
            height=600,
            template='plotly_white'
        )
        
        fig.update_yaxes(title_text=f"{price_col.capitalize()} Price", secondary_y=False)
        fig.update_yaxes(title_text="Sentiment Score", secondary_y=True)
        
        # Save the figure if a path is provided
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_candlestick_with_sentiment(self, df, sentiment_col='sentiment_score', coin_name='',
                                      title=None, save_path=None):
        """
        Create a candlestick chart with sentiment overlay.
        
        Args:
            df (pd.DataFrame): DataFrame with OHLC and sentiment data
            sentiment_col (str): Sentiment column to plot
            coin_name (str): Name of the cryptocurrency
            title (str): Plot title
            save_path (str): Path to save the HTML file
            
        Returns:
            plotly.graph_objects.Figure: The figure
        """
        if title is None:
            title = f'{coin_name} Price Candlestick with Sentiment'
        
        # Ensure date column is datetime
        if 'date' in df.columns and not pd.api.types.is_datetime64_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
            
        # Create subplot with two y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ),
            secondary_y=False
        )
        
        # Add sentiment line
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df[sentiment_col],
                name='Sentiment Score',
                line=dict(color='purple', width=2)
            ),
            secondary_y=True
        )
        
        # Add SMA lines if available
        for col in ['sma_7', 'sma_20', 'sma_50']:
            if col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['date'],
                        y=df[col],
                        name=col.upper(),
                        line=dict(width=1)
                    ),
                    secondary_y=False
                )
        
        # Add layout details
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
            hovermode='x unified',
            height=700,
            template='plotly_white',
            xaxis_rangeslider_visible=False
        )
        
        fig.update_yaxes(title_text="Price", secondary_y=False)
        fig.update_yaxes(title_text="Sentiment Score", secondary_y=True)
        
        # Save the figure if a path is provided
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_correlation_heatmap_plotly(self, correlation_matrix, title='Correlation Heatmap',
                                      save_path=None):
        """
        Create an interactive correlation heatmap.
        
        Args:
            correlation_matrix (pd.DataFrame): Correlation matrix
            title (str): Plot title
            save_path (str): Path to save the HTML file
            
        Returns:
            plotly.graph_objects.Figure: The figure
        """
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix), k=1)
        mask = mask.astype(bool)
        correlation_matrix_masked = correlation_matrix.copy()
        correlation_matrix_masked.values[mask] = np.nan
        
        # Create heatmap
        fig = px.imshow(
            correlation_matrix_masked,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            text_auto='.2f'
        )
        
        fig.update_layout(
            title=title,
            height=800,
            width=900,
            xaxis_title='',
            yaxis_title='',
            xaxis={'side': 'top'},
            template='plotly_white'
        )
        
        # Save the figure if a path is provided
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_technical_indicators(self, df, coin_name='', title=None, save_path=None):
        """
        Plot price with technical indicators.
        
        Args:
            df (pd.DataFrame): DataFrame with price and technical indicators
            coin_name (str): Name of the cryptocurrency
            title (str): Plot title
            save_path (str): Path to save the HTML file
            
        Returns:
            plotly.graph_objects.Figure: The figure
        """
        if title is None:
            title = f'{coin_name} Technical Indicators'
        
        # Ensure date column is datetime
        if 'date' in df.columns and not pd.api.types.is_datetime64_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
            
        # Create subplots: 3 rows, 1 column
        fig = make_subplots(
            rows=3, 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=('Price and Moving Averages', 'RSI', 'MACD')
        )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add Moving Averages
        for ma_col, color in [('sma_7', 'blue'), ('sma_20', 'orange'), ('sma_50', 'green')]:
            if ma_col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['date'],
                        y=df[ma_col],
                        name=ma_col.upper(),
                        line=dict(color=color, width=1.5)
                    ),
                    row=1, col=1
                )
        
        # Add Bollinger Bands
        for bb_col, color, dash in [('bb_upper', 'red', 'dash'), ('bb_middle', 'purple', 'solid'), ('bb_lower', 'red', 'dash')]:
            if bb_col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['date'],
                        y=df[bb_col],
                        name=bb_col.upper(),
                        line=dict(color=color, width=1, dash=dash)
                    ),
                    row=1, col=1
                )
        
        # Add RSI
        if 'rsi_14' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['rsi_14'],
                    name='RSI (14)',
                    line=dict(color='blue', width=1.5)
                ),
                row=2, col=1
            )
            
            # Add RSI reference lines
            fig.add_shape(
                type='line',
                x0=df['date'].iloc[0],
                y0=70,
                x1=df['date'].iloc[-1],
                y1=70,
                line=dict(color='red', width=1, dash='dash'),
                row=2, col=1
            )
            
            fig.add_shape(
                type='line',
                x0=df['date'].iloc[0],
                y0=30,
                x1=df['date'].iloc[-1],
                y1=30,
                line=dict(color='green', width=1, dash='dash'),
                row=2, col=1
            )
        
        # Add MACD
        if all(col in df.columns for col in ['macd', 'macd_signal']):
            # MACD Line
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['macd'],
                    name='MACD',
                    line=dict(color='blue', width=1.5)
                ),
                row=3, col=1
            )
            
            # Signal Line
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['macd_signal'],
                    name='Signal',
                    line=dict(color='red', width=1.5)
                ),
                row=3, col=1
            )
            
            # Histogram
            if 'macd_hist' in df.columns:
                colors = ['green' if val >= 0 else 'red' for val in df['macd_hist']]
                fig.add_trace(
                    go.Bar(
                        x=df['date'],
                        y=df['macd_hist'],
                        name='Histogram',
                        marker_color=colors
                    ),
                    row=3, col=1
                )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=900,
            template='plotly_white',
            xaxis_rangeslider_visible=False
        )
        
        # Update y-axis titles
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=3, col=1)
        
        # Save the figure if a path is provided
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_future_prediction(self, historical_df, future_predictions, prediction_dates=None,
                             coin_name='', model_type='', price_col='close',
                             title=None, save_path=None):
        """
        Plot historical prices with future predictions.
        
        Args:
            historical_df (pd.DataFrame): DataFrame with historical price data
            future_predictions (np.array): Array of predicted future prices
            prediction_dates (list): List of dates for predictions (if None, will be generated)
            coin_name (str): Name of the cryptocurrency
            model_type (str): Type of model used
            price_col (str): Column with price data
            title (str): Plot title
            save_path (str): Path to save the HTML file
            
        Returns:
            plotly.graph_objects.Figure: The figure
        """
        if title is None:
            title = f'{coin_name} Future Price Prediction ({model_type})'
        
        # Ensure data is properly formatted
        if not isinstance(historical_df, pd.DataFrame) or historical_df.empty:
            fig = go.Figure()
            fig.update_layout(
                title="Error: No historical data available",
                annotations=[dict(text="No data to display", showarrow=False, xref="paper", yref="paper")]
            )
            return fig
            
        # Ensure date column is datetime
        if 'date' in historical_df.columns and not pd.api.types.is_datetime64_dtype(historical_df['date']):
            historical_df['date'] = pd.to_datetime(historical_df['date'])
            
        # Sort historical data by date
        historical_df = historical_df.sort_values('date')
        
        # Check if we have prediction data
        if future_predictions is None or len(future_predictions) == 0:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=historical_df['date'],
                    y=historical_df[price_col],
                    name='Historical',
                    line=dict(color='blue', width=2)
                )
            )
            fig.update_layout(
                title=f"{coin_name} Price (No Predictions Available)",
                xaxis_title='Date',
                yaxis_title='Price',
                template='plotly_white'
            )
            return fig
        
        # Generate prediction dates if not provided
        if prediction_dates is None:
            last_date = historical_df['date'].iloc[-1]
            prediction_dates = [last_date + timedelta(days=i+1) for i in range(len(future_predictions))]
        
        # Convert prediction_dates to datetime if they're strings
        if isinstance(prediction_dates[0], str):
            prediction_dates = [pd.to_datetime(date) for date in prediction_dates]
        
        fig = go.Figure()
        
        # Add historical prices
        fig.add_trace(
            go.Scatter(
                x=historical_df['date'],
                y=historical_df[price_col],
                name='Historical',
                line=dict(color='blue', width=2)
            )
        )
        
        # Add vertical line at prediction start
        fig.add_vline(
            x=historical_df['date'].iloc[-1],
            line=dict(color='green', width=2, dash='dash')
        )
        
        # Add future predictions
        fig.add_trace(
            go.Scatter(
                x=prediction_dates,
                y=future_predictions,
                name='Predicted',
                line=dict(color='red', width=2, dash='dash')
            )
        )
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price',
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
            hovermode='x unified',
            template='plotly_white',
            height=600
        )
        
        # Save the figure if a path is provided
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_prediction_vs_actual(self, results_df, coin_name='', model_type='', 
                                title=None, save_path=None):
        """
        Plot predicted vs actual prices.
        
        Args:
            results_df (pd.DataFrame): DataFrame with 'date', 'actual', and 'predicted' columns
            coin_name (str): Name of the cryptocurrency
            model_type (str): Type of model used
            title (str): Plot title
            save_path (str): Path to save the HTML file
            
        Returns:
            plotly.graph_objects.Figure: The figure
        """
        if title is None:
            title = f'{coin_name} Actual vs Predicted Prices ({model_type})'
        
        # Check if results DataFrame is valid
        if not isinstance(results_df, pd.DataFrame) or results_df.empty:
            fig = go.Figure()
            fig.update_layout(
                title="Error: No prediction results available",
                annotations=[dict(text="No data to display", showarrow=False, xref="paper", yref="paper")]
            )
            return fig
            
        # Ensure required columns exist
        required_cols = ['date', 'actual', 'predicted']
        if not all(col in results_df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in results_df.columns]
            fig = go.Figure()
            fig.update_layout(
                title=f"Error: Missing columns {missing_cols}",
                annotations=[dict(text=f"Missing columns: {missing_cols}", showarrow=False, xref="paper", yref="paper")]
            )
            return fig
            
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_dtype(results_df['date']):
            results_df['date'] = pd.to_datetime(results_df['date'])
            
        fig = go.Figure()
        
        # Add actual prices
        fig.add_trace(
            go.Scatter(
                x=results_df['date'],
                y=results_df['actual'],
                name='Actual',
                line=dict(color='blue', width=2)
            )
        )
        
        # Add predicted prices
        fig.add_trace(
            go.Scatter(
                x=results_df['date'],
                y=results_df['predicted'],
                name='Predicted',
                line=dict(color='red', width=2, dash='dash')
            )
        )
        
        # Calculate error metrics
        mse = ((results_df['actual'] - results_df['predicted']) ** 2).mean()
        mape = (abs(results_df['actual'] - results_df['predicted']) / results_df['actual']).mean() * 100
        
        # Add annotations with error metrics
        fig.add_annotation(
            x=0.05,
            y=0.95,
            text=f"MSE: {mse:.2f}<br>MAPE: {mape:.2f}%",
            showarrow=False,
            xref="paper",
            yref="paper",
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price',
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
            hovermode='x unified',
            template='plotly_white',
            height=600
        )
        
        # Save the figure if a path is provided
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_multi_coin_comparison(self, coin_data_dict, metric='close', 
                                 normalize=True, title=None, save_path=None):
        """
        Plot a comparison of multiple cryptocurrencies over time.
        
        Args:
            coin_data_dict (dict): Dictionary mapping coin names to DataFrames
            metric (str): Column to plot (e.g., 'close', 'sentiment_score')
            normalize (bool): Whether to normalize values for comparison
            title (str): Plot title
            save_path (str): Path to save the HTML file
            
        Returns:
            plotly.graph_objects.Figure: The figure
        """
        if title is None:
            title = f'Multi-Coin Comparison: {metric.capitalize()}'
        
        fig = go.Figure()
        
        for coin, df in coin_data_dict.items():
            if metric in df.columns:
                # Ensure date column is datetime
                if 'date' in df.columns and not pd.api.types.is_datetime64_dtype(df['date']):
                    df['date'] = pd.to_datetime(df['date'])
                
                # Sort by date
                df = df.sort_values('date')
                
                # Normalize if requested
                if normalize:
                    if df[metric].iloc[0] != 0:  # Avoid division by zero
                        values = df[metric] / df[metric].iloc[0] * 100
                        y_title = f'Normalized {metric.capitalize()} (First Day = 100)'
                    else:
                        values = df[metric]
                        y_title = metric.capitalize()
                else:
                    values = df[metric]
                    y_title = metric.capitalize()
                
                # Add trace
                fig.add_trace(
                    go.Scatter(
                        x=df['date'],
                        y=values,
                        name=coin,
                        mode='lines'
                    )
                )
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title=y_title,
            legend_title='Coin',
            hovermode='x unified',
            template='plotly_white',
            height=600
        )
        
        # Save the figure if a path is provided
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_sentiment_dashboard_components(self, df, coin_name='', save_dir=None):
        """
        Create a set of dashboard components for sentiment analysis.
        
        Args:
            df (pd.DataFrame): DataFrame with price and sentiment data
            coin_name (str): Name of the cryptocurrency
            save_dir (str): Directory to save the HTML files
            
        Returns:
            dict: Dictionary of created figures
        """
        if save_dir is None:
            save_dir = os.path.join(self.output_dir, coin_name.lower())
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Handle empty dataframe
        if df is None or df.empty:
            print(f"Warning: Empty dataframe provided for {coin_name}")
            return {}
            
        # Ensure date column is datetime
        if 'date' in df.columns and not pd.api.types.is_datetime64_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
            
        figures = {}
        
        try:
            # Price and sentiment time series
            figures['price_sentiment'] = self.plot_price_sentiment_time_series(
                df,
                coin_name=coin_name,
                save_path=os.path.join(save_dir, 'price_sentiment.html')
            )
            
            # Candlestick with sentiment
            figures['candlestick'] = self.plot_candlestick_with_sentiment(
                df,
                coin_name=coin_name,
                save_path=os.path.join(save_dir, 'candlestick.html')
            )
            
            # Technical indicators
            figures['technical'] = self.plot_technical_indicators(
                df,
                coin_name=coin_name,
                save_path=os.path.join(save_dir, 'technical.html')
            )
            
            # Create a summary JSON with metadata
            summary = {
                'coin': coin_name,
                'date_range': [df['date'].min().strftime('%Y-%m-%d'), df['date'].max().strftime('%Y-%m-%d')],
                'data_points': len(df),
                'available_charts': list(figures.keys()),
                'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(os.path.join(save_dir, 'dashboard_summary.json'), 'w') as f:
                json.dump(summary, f, indent=4)
        
        except Exception as e:
            print(f"Error creating dashboard components for {coin_name}: {e}")
            import traceback
            traceback.print_exc()
        
        return figures
        
    def plot_sentiment_distribution(self, df, sentiment_col='sentiment_score', bins=30,
                                  title=None, coin_name='', save_path=None):
        """
        Plot the distribution of sentiment values.
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment data
            sentiment_col (str): Sentiment column to plot
            bins (int): Number of bins for histogram
            title (str): Plot title
            coin_name (str): Name of the cryptocurrency
            save_path (str): Path to save the HTML file
            
        Returns:
            plotly.graph_objects.Figure: The figure
        """
        if title is None:
            title = f'{coin_name} Sentiment Distribution'
        
        # Handle empty or invalid dataframe
        if df is None or df.empty or sentiment_col not in df.columns:
            fig = go.Figure()
            fig.update_layout(
                title="Error: No valid sentiment data",
                annotations=[dict(text="No data to display", showarrow=False, xref="paper", yref="paper")]
            )
            return fig
            
        # Simple histogram with density curve
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(
            go.Histogram(
                x=df[sentiment_col],
                nbinsx=bins,
                histnorm='probability density',
                name='Sentiment Distribution',
                marker_color='blue',
                opacity=0.6
            )
        )
        
        # Add density curve (using KDE approximation)
        # First, get histogram data
        hist_vals, bin_edges = np.histogram(
            df[sentiment_col].dropna(), 
            bins=bins, 
            density=True
        )
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        # Smooth the histogram values for a curve-like appearance
        from scipy.signal import savgol_filter
        if len(hist_vals) >= 5:  # Need at least 5 points for savgol filter
            smooth_vals = savgol_filter(hist_vals, min(5, len(hist_vals)), 3)
            
            fig.add_trace(
                go.Scatter(
                    x=bin_centers,
                    y=smooth_vals,
                    mode='lines',
                    name='Density',
                    line=dict(color='red', width=2)
                )
            )
        
        # Add mean line
        mean_val = df[sentiment_col].mean()
        fig.add_vline(
            x=mean_val,
            line=dict(color='green', width=2, dash='dash'),
            annotation_text=f"Mean: {mean_val:.3f}",
            annotation_position="top right"
        )
        
        fig.update_layout(
            title=title,
            xaxis_title='Sentiment Score',
            yaxis_title='Density',
            template='plotly_white',
            height=500,
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
        )
        
        # Save the figure if a path is provided
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_model_comparison(self, metrics_df, metric='rmse', title=None, save_path=None):
        """
        Plot a comparison of multiple model metrics.
        
        Args:
            metrics_df (pd.DataFrame): DataFrame with model metrics
            metric (str): Metric to plot ('rmse', 'mae', 'mape', or 'r2')
            title (str): Plot title
            save_path (str): Path to save the HTML file
            
        Returns:
            plotly.graph_objects.Figure: The figure
        """
        if title is None:
            title = f'Model Comparison: {metric.upper()}'
        
        # Handle empty dataframe
        if metrics_df is None or metrics_df.empty:
            fig = go.Figure()
            fig.update_layout(
                title="Error: No model metrics data",
                annotations=[dict(text="No data to display", showarrow=False, xref="paper", yref="paper")]
            )
            return fig
            
        # Ensure required columns exist
        required_cols = ['coin', 'model_type', metric]
        if not all(col in metrics_df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in metrics_df.columns]
            fig = go.Figure()
            fig.update_layout(
                title=f"Error: Missing columns {missing_cols}",
                annotations=[dict(text=f"Missing columns: {missing_cols}", showarrow=False, xref="paper", yref="paper")]
            )
            return fig
        
        # Create the plot
        fig = px.bar(
            metrics_df,
            x='coin',
            y=metric,
            color='model_type',
            barmode='group',
            labels={metric: metric.upper()},
            title=title
        )
        
        # Add text labels
        fig.update_traces(
            texttemplate='%{y:.3f}',
            textposition='outside'
        )
        
        fig.update_layout(
            xaxis_title='Cryptocurrency',
            yaxis_title=metric.upper(),
            legend_title='Model Type',
            template='plotly_white',
            height=600
        )
        
        # Save the figure if a path is provided
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_rolling_correlations_heatmap(self, df_dict, window_size=30, step=7, 
                                        sentiment_col='sentiment_score', price_col='close',
                                        title=None, save_path=None):
        """
        Create a heatmap of rolling correlations between sentiment and price for multiple coins.
        
        Args:
            df_dict (dict): Dictionary mapping coin names to DataFrames
            window_size (int): Size of rolling window in days
            step (int): Number of days to step forward for each window
            sentiment_col (str): Sentiment column to analyze
            price_col (str): Price column to analyze
            title (str): Plot title
            save_path (str): Path to save the HTML file
            
        Returns:
            plotly.graph_objects.Figure: The figure
        """
        if title is None:
            title = f'Rolling Sentiment-Price Correlations (Window: {window_size} days)'
        
        # Calculate rolling correlations for each coin
        results = {}
        
        for coin, df in df_dict.items():
            try:
                # Make sure we have required columns
                if sentiment_col not in df.columns or price_col not in df.columns:
                    print(f"Warning: Missing required columns for {coin}")
                    continue
                
                # Make sure we have enough data
                if len(df) < window_size:
                    print(f"Warning: Not enough data for {coin} - need at least {window_size} rows")
                    continue
                    
                # Ensure date column is datetime
                if 'date' in df.columns and not pd.api.types.is_datetime64_dtype(df['date']):
                    df['date'] = pd.to_datetime(df['date'])
                
                # Calculate price changes
                df['price_change'] = df[price_col].pct_change()
                
                # Initialize results
                coin_results = []
                
                # Ensure dataframe is sorted by date
                if 'date' in df.columns:
                    df = df.sort_values('date').reset_index(drop=True)
                
                # Sliding window analysis
                for start_idx in range(0, len(df) - window_size, step):
                    end_idx = start_idx + window_size
                    window_df = df.iloc[start_idx:end_idx].copy()
                    
                    # Get mid-point date for this window
                    if 'date' in window_df.columns:
                        mid_date = window_df['date'].iloc[window_size // 2]
                    else:
                        mid_date = start_idx + window_size // 2
                    
                    # Calculate correlation for this window - handling potential NaN values
                    window_df_clean = window_df[[sentiment_col, 'price_change']].dropna()
                    if len(window_df_clean) > 5:  # Require at least 5 data points for meaningful correlation
                        # Use try/except to handle potential issues with correlation calculation
                        try:
                            corr = window_df_clean.corr().iloc[0, 1]
                            # Only store if correlation is not NaN
                            if not np.isnan(corr):
                                coin_results.append({
                                    'date': mid_date,
                                    'correlation': corr
                                })
                        except Exception as e:
                            print(f"Error calculating correlation for {coin} at window {start_idx}-{end_idx}: {e}")
                
                if coin_results:
                    results[coin] = pd.DataFrame(coin_results)
            except Exception as e:
                print(f"Error processing {coin}: {e}")
        
        # If no results were calculated, return empty figure
        if not results:
            fig = go.Figure()
            fig.update_layout(
                title="Error: Not enough data to calculate rolling correlations",
                annotations=[dict(text="Insufficient data", showarrow=False, xref="paper", yref="paper")]
            )
            return fig
        
        # Create a common date range for all coins
        all_dates = pd.concat([df['date'] for df in results.values()]).drop_duplicates().sort_values()
        if len(all_dates) == 0:
            fig = go.Figure()
            fig.update_layout(
                title="Error: No valid dates found",
                annotations=[dict(text="No valid dates", showarrow=False, xref="paper", yref="paper")]
            )
            return fig
            
        date_range = pd.date_range(start=all_dates.min(), end=all_dates.max(), freq='D')
        
        # Create a pivot table of correlations
        pivot_data = []
        
        for coin, df in results.items():
            try:
                # Resample to daily and interpolate
                resampled = df.set_index('date').reindex(date_range)
                resampled['correlation'] = resampled['correlation'].interpolate(method='time')
                
                for date, row in resampled.iterrows():
                    if not np.isnan(row['correlation']):
                        pivot_data.append({
                            'date': date,
                            'coin': coin,
                            'correlation': row['correlation']
                        })
            except Exception as e:
                print(f"Error processing {coin} for heatmap: {e}")