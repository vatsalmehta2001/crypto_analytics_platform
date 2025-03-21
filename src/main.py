# Temporary flag to disable visualization to focus on data processing
DISABLE_VISUALIZATION = True
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import json
from tqdm import tqdm

# Import project modules
from src.data.data_processor import CryptoDataProcessor
from models.sentiment.sentiment_analyzer import CryptoSentimentAnalyzer
from models.correlation.correlation_analyzer import CryptoCorrelationAnalyzer
from models.price.price_predictor import CryptoPricePredictor
from models.visualization.visualizer import CryptoVisualizer

def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs = [
        'data/processed',
        'data/market',
        'models/saved',
        'reports/figures',
        'reports/correlation',
        'dashboard/assets'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Ensured directory exists: {dir_path}")

def process_data(raw_data_path, output_path, coin_list=None):
    """Process raw cryptocurrency and sentiment data."""
    print("\n=== PROCESSING DATA ===")
    
    # Initialize data processor
    processor = CryptoDataProcessor(
        raw_data_path=raw_data_path,
        processed_data_path=output_path
    )
    
    # Load price data
    processor.load_price_data(coin_list=coin_list)
    
    # Load news data
    processor.load_news_data()
    
    # Process and align data
    print("\nAligning price and sentiment data...")
    aligned_data = processor.prepare_all_coins_data(coin_list=coin_list)
    
    print(f"\nProcessed data for {len(aligned_data)} coins:")
    for coin, df in aligned_data.items():
        print(f"  - {coin}: {len(df)} records from {df['date'].min()} to {df['date'].max()}")
    
    return aligned_data

def process_data_twitter(raw_data_path, output_path, coin_list=None, bearer_token=None):
    """Process raw cryptocurrency data with Twitter sentiment."""
    print("\n=== PROCESSING DATA WITH TWITTER SENTIMENT ===")
    
    # Initialize data processor
    processor = CryptoDataProcessor(
        raw_data_path=raw_data_path,
        processed_data_path=output_path
    )
    
    # Load price data
    processor.load_price_data(coin_list=coin_list)
    
    # Process and align data with Twitter sentiment
    print("\nAligning price and Twitter sentiment data...")
    aligned_data = processor.prepare_all_coins_data_twitter(coin_list=coin_list, bearer_token=bearer_token)
    
    print(f"\nProcessed data for {len(aligned_data)} coins with Twitter sentiment:")
    for coin, df in aligned_data.items():
        print(f"  - {coin}: {len(df)} records from {df['date'].min()} to {df['date'].max()}")
    
    return aligned_data

def analyze_sentiment(aligned_data, output_dir):
    """Perform advanced sentiment analysis on processed data."""
    print("\n=== SENTIMENT ANALYSIS ===")
    
    # Initialize sentiment analyzer
    analyzer = CryptoSentimentAnalyzer(model_type='vader')  # Using VADER for simplicity
    
    # Placeholder for future sentiment analysis enhancements
    # This could involve re-running sentiment analysis on news text or
    # analyzing specific coin mentions within the news corpus
    
    print("Sentiment analysis previously performed during data processing.")
    print("Additional sentiment analysis features can be implemented here.")
    
    return True

def analyze_correlations(aligned_data, output_dir):
    """Analyze correlations between sentiment and price movements."""
    print("\n=== CORRELATION ANALYSIS ===")
    
    # Initialize correlation analyzer
    analyzer = CryptoCorrelationAnalyzer(output_dir=output_dir)
    
    results = {}
    
    for coin, df in aligned_data.items():
        print(f"\nAnalyzing correlations for {coin}...")
        
        # Calculate Pearson correlation
        correlation_matrix = analyzer.pearson_correlation(df)
        
        # Plot correlation heatmap
        heatmap_fig = analyzer.plot_correlation_heatmap(
            correlation_matrix,
            title=f'{coin} Price-Sentiment Correlation',
            save_path=os.path.join(output_dir, 'figures', f'{coin.lower()}_correlation_heatmap.png')
        )
        
        # Calculate time-lagged correlations
        lagged_corr = analyzer.time_lagged_correlation(
            df,
            price_col='close',
            sentiment_col='sentiment_score',
            max_lag=10
        )
        
        # Plot lagged correlations
        lagged_fig = analyzer.plot_lagged_correlation(
            lagged_corr,
            title=f'{coin} Time-Lagged Correlation: Price vs Sentiment',
            save_path=os.path.join(output_dir, 'figures', f'{coin.lower()}_lagged_correlation.png')
        )
        
        # Perform Granger causality test
        granger_results = analyzer.granger_causality_test(
            df,
            x_col='sentiment_score',
            y_col='close',
            max_lag=5
        )
        
        # Calculate rolling correlations
        rolling_corr = analyzer.analyze_correlation_over_time(
            df,
            price_col='close',
            sentiment_col='sentiment_score',
            window_size=30
        )
        
        # Plot rolling correlations
        rolling_fig = analyzer.plot_rolling_correlation(
            rolling_corr,
            title=f'{coin} Rolling Correlation: Price vs Sentiment',
            save_path=os.path.join(output_dir, 'figures', f'{coin.lower()}_rolling_correlation.png')
        )
        
        # Store results
        results[coin] = {
            'pearson_correlation': correlation_matrix.to_dict(),
            'lagged_correlation': lagged_corr.to_dict(),
            'granger_causality': granger_results,
            'rolling_correlation': rolling_corr.to_dict()
        }
    
    # Compare correlations across coins
    if len(aligned_data) > 1:
        print("\nComparing correlations across coins...")
        comparison = analyzer.compare_coins_sentiment_correlation(
            aligned_data,
            sentiment_col='sentiment_score',
            price_col='close'
        )
        
        # Save comparison results
        comparison.to_csv(os.path.join(output_dir, 'correlation', 'coin_sentiment_correlation_comparison.csv'), index=False)
        
        # Add to results
        results['comparison'] = comparison.to_dict()
    
    # Save all correlation results
    with open(os.path.join(output_dir, 'correlation', 'correlation_analysis_results.json'), 'w') as f:
        # Convert any non-serializable values
        json.dump(results, f, default=str, indent=4)
    
    print(f"\nCorrelation analysis completed. Results saved to '{output_dir}/correlation'")
    
    return results

def train_prediction_models(aligned_data, models_dir):
    """Train machine learning models to predict cryptocurrency prices."""
    print("\n=== TRAINING PREDICTION MODELS ===")
    
    # Initialize price predictor
    predictor = CryptoPricePredictor(model_dir=models_dir)
    
    model_metrics = []
    prediction_results = {}
    
    for coin, df in aligned_data.items():
        print(f"\nTraining models for {coin}...")
        
        # Prepare data for LSTM
        X, y, feature_cols = predictor.prepare_time_series_data(
            df, target_col='close', window_size=10, include_sentiment=True
        )
        
        # Split data
        train_data, val_data, test_data = predictor.split_data(X, y)
        
        # Train LSTM model
        print(f"Training LSTM model for {coin}...")
        lstm_model, history = predictor.train_lstm_model(
            train_data, val_data, model_type='lstm', 
            epochs=50, coin_name=coin
        )
        
        # Evaluate LSTM model
        lstm_metrics = predictor.evaluate_model(
            lstm_model, test_data, coin_name=coin, 
            model_type='lstm'
        )
        model_metrics.append(lstm_metrics)
        
        # Train XGBoost model
        print(f"Training XGBoost model for {coin}...")
        xgb_model = predictor.train_xgboost_model(
            train_data, val_data, coin_name=coin
        )
        
        # Evaluate XGBoost model
        xgb_metrics = predictor.evaluate_model(
            xgb_model, test_data, coin_name=coin, 
            model_type='xgboost'
        )
        model_metrics.append(xgb_metrics)
        
        # Make future predictions with LSTM
        last_sequence = X[-1:][0]
        future_prices_lstm = predictor.make_future_predictions(
            lstm_model, last_sequence, days_ahead=7, model_type='lstm'
        )
        
        # Make predictions on test data
        lstm_results = predictor.predict_with_sentiment(
            df, lstm_model, window_size=10, coin_name=coin, model_type='lstm'
        )
        
        # Store results
        prediction_results[coin] = {
            'lstm_future': future_prices_lstm.tolist(),
            'forecast_dates': [(df['date'].iloc[-1] + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d') 
                              for i in range(len(future_prices_lstm))],
            'test_predictions': lstm_results.to_dict()
        }
    
    # Save model metrics
    metrics_df = pd.DataFrame(model_metrics)
    metrics_df.to_csv(os.path.join(models_dir, 'model_metrics.csv'), index=False)
    
    # Save prediction results
    with open(os.path.join(models_dir, 'prediction_results.json'), 'w') as f:
        json.dump(prediction_results, f, default=str, indent=4)
    
    print(f"\nPrediction models trained and evaluated. Results saved to '{models_dir}'")
    
    return metrics_df, prediction_results

def create_visualizations(aligned_data, model_metrics, prediction_results, output_dir):
    """Create visualizations and dashboard components."""
    print("\n=== CREATING VISUALIZATIONS ===")
    
    # Initialize visualizer
    visualizer = CryptoVisualizer(output_dir=os.path.join(output_dir, 'figures'))
    
    # Create visualizations for each coin
    for coin, df in aligned_data.items():
        print(f"\nCreating visualizations for {coin}...")
        
        # Price and sentiment time series
        price_sentiment_fig = visualizer.plot_price_sentiment_time_series(
            df, 
            coin_name=coin,
            save_path=os.path.join(output_dir, 'figures', f'{coin.lower()}_price_sentiment.html')
        )
        
        # Candlestick with sentiment
        candlestick_fig = visualizer.plot_candlestick_with_sentiment(
            df,
            coin_name=coin,
            save_path=os.path.join(output_dir, 'figures', f'{coin.lower()}_candlestick.html')
        )
        
        # Technical indicators
        technical_fig = visualizer.plot_technical_indicators(
            df,
            coin_name=coin,
            save_path=os.path.join(output_dir, 'figures', f'{coin.lower()}_technical.html')
        )
        
        # Prediction vs actual (if available)
        if coin in prediction_results:
            # Get prediction results
            lstm_results = pd.DataFrame(prediction_results[coin]['test_predictions'])
            
            prediction_fig = visualizer.plot_prediction_vs_actual(
                lstm_results,
                coin_name=coin,
                model_type='LSTM',
                save_path=os.path.join(output_dir, 'figures', f'{coin.lower()}_prediction.html')
            )
            
            # Future predictions
            future_predictions = np.array(prediction_results[coin]['lstm_future'])
            forecast_dates = [pd.to_datetime(date) for date in prediction_results[coin]['forecast_dates']]
            
            future_fig = visualizer.plot_future_prediction(
                df,
                future_predictions,
                prediction_dates=forecast_dates,
                coin_name=coin,
                model_type='LSTM',
                save_path=os.path.join(output_dir, 'figures', f'{coin.lower()}_future_prediction.html')
            )
    
    # Create model comparison visualization
    if model_metrics is not None and len(model_metrics) > 1:
        for metric in ['rmse', 'mae', 'mape', 'r2']:
            model_fig = visualizer.plot_model_comparison(
                model_metrics,
                metric=metric,
                save_path=os.path.join(output_dir, 'figures', f'model_comparison_{metric}.html')
            )
    
    # Create multi-coin comparison if multiple coins are available
    if len(aligned_data) > 1:
        # Price comparison
        price_comparison_fig = visualizer.plot_multi_coin_comparison(
            aligned_data,
            metric='close',
            normalize=True,
            save_path=os.path.join(output_dir, 'figures', 'multi_coin_price_comparison.html')
        )
        
        # Sentiment comparison
        sentiment_comparison_fig = visualizer.plot_multi_coin_comparison(
            aligned_data,
            metric='sentiment_score',
            normalize=False,
            save_path=os.path.join(output_dir, 'figures', 'multi_coin_sentiment_comparison.html')
        )
        
        # Rolling correlations heatmap
        rolling_corr_fig = visualizer.plot_rolling_correlations_heatmap(
            aligned_data,
            window_size=30,
            step=7,
            save_path=os.path.join(output_dir, 'figures', 'rolling_correlations_heatmap.html')
        )
    
    # Create dashboard components for each coin
    print("\nCreating dashboard components...")
    dashboard_dir = 'dashboard/assets'
    
    for coin, df in aligned_data.items():
        components = visualizer.create_sentiment_dashboard_components(
            df,
            coin_name=coin,
            save_dir=os.path.join(dashboard_dir, coin.lower())
        )
    
    print(f"\nVisualizations created and saved to '{output_dir}/figures' and '{dashboard_dir}'")

def main(args):
    """Main function to run the complete pipeline."""
    print("\n=== CRYPTO SENTIMENT ANALYSIS PLATFORM ===")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup directories
    setup_directories()
    
    # Process data (with Twitter sentiment if specified)
    if args.use_twitter and args.twitter_token:
        print("\n=== USING TWITTER FOR SENTIMENT ANALYSIS ===")
        aligned_data = process_data_twitter(
            raw_data_path=args.raw_data_path,
            output_path=args.processed_data_path,
            coin_list=args.coins.split(',') if args.coins else None,
            bearer_token=args.twitter_token
        )
    else:
        aligned_data = process_data(
            raw_data_path=args.raw_data_path,
            output_path=args.processed_data_path,
            coin_list=args.coins.split(',') if args.coins else None
        )
    
    # Analyze sentiment (already done during data processing, placeholder for enhancements)
    analyze_sentiment(aligned_data, args.output_dir)
    
    # Analyze correlations
    correlation_results = analyze_correlations(aligned_data, args.output_dir)
    
    # Train prediction models
    if not args.skip_models:
        model_metrics, prediction_results = train_prediction_models(aligned_data, args.models_dir)
    else:
        print("\nSkipping model training as requested.")
        model_metrics, prediction_results = None, {}
    
    # Create visualizations
    create_visualizations(aligned_data, model_metrics, prediction_results, args.output_dir)
    
    print(f"\nPipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crypto Sentiment Analysis Platform')
    parser.add_argument('--raw-data-path', type=str, default='data/raw',
                        help='Path to raw data directory')
    parser.add_argument('--processed-data-path', type=str, default='data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--output-dir', type=str, default='reports',
                        help='Path to output directory for reports and figures')
    parser.add_argument('--models-dir', type=str, default='models/saved',
                        help='Path to directory for saving trained models')
    parser.add_argument('--coins', type=str, default=None,
                        help='Comma-separated list of coin tickers to analyze (e.g., BTC,ETH,SOL)')
    parser.add_argument('--skip-models', action='store_true',
                        help='Skip model training (useful for quick analysis)')
    parser.add_argument('--visualize-only', action='store_true',
                        help='Only create visualizations from existing processed data')
    parser.add_argument('--use-twitter', action='store_true',
                        help='Use Twitter data instead of news data for sentiment analysis')
    parser.add_argument('--twitter-token', type=str, default=None,
                        help='Twitter API bearer token for sentiment analysis')
    
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()