import os
import sys
import pandas as pd
from data.data_processor import CryptoDataProcessor

# Input your Twitter bearer token
BEARER_TOKEN = "YOUR_BEARER_TOKEN_HERE"  # Replace with your actual token

def test_twitter_sentiment():
    """Test Twitter sentiment analysis functionality"""
    processor = CryptoDataProcessor(
        raw_data_path="data/raw",
        processed_data_path="data/processed"
    )
    
    # Load only BTC and ICP for testing
    processor.load_price_data(coin_list=["BTC", "ICP"])
    
    # Get Twitter sentiment and combine with price data
    print("Processing Twitter sentiment data...")
    aligned_data = processor.prepare_all_coins_data_twitter(
        coin_list=["BTC", "ICP"],
        bearer_token=BEARER_TOKEN
    )
    
    # Print sample of results
    for coin, df in aligned_data.items():
        print(f"\n{coin} sample data with Twitter sentiment:")
        print(df[['date', 'close', 'sentiment_score', 'tweet_volume']].head())
        
        # Check if sentiment varies
        if 'sentiment_score' in df.columns:
            unique_values = df['sentiment_score'].nunique()
            print(f"Number of unique sentiment values: {unique_values}")
            print(f"Sentiment range: {df['sentiment_score'].min()} to {df['sentiment_score'].max()}")
    
    return aligned_data

if __name__ == "__main__":
    test_twitter_sentiment()