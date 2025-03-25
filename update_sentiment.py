import asyncio
import pandas as pd
import os
import json
from datetime import datetime
from src.models.sentiment.sentiment_analyzer import TwscrapeTwitterSentimentAnalyzer
from src.models.sentiment.sentiment_analyzer import EnhancedSentimentAnalyzer

async def update_sentiment_data():
    """Fetch latest sentiment data and update the processed files."""
    # Initialize the analyzers
    twitter_analyzer = TwscrapeTwitterSentimentAnalyzer()
    sentiment_analyzer = EnhancedSentimentAnalyzer(model_dir='models/saved')
    
    # Optional: Add Twitter account for better rate limits
    # await twitter_analyzer.add_twitter_account("username", "password")
    
    # Get list of coins from processed data directory
    data_dir = 'data/processed'
    files = [f for f in os.listdir(data_dir) if f.endswith('_processed.csv')]
    coins = [f.split('_processed.csv')[0] for f in files]
    
    # Output directory for latest sentiment
    output_dir = 'data/latest_sentiment'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each coin
    results = {}
    
    for coin in coins:
        print(f"\nUpdating sentiment for {coin}...")
        
        try:
            # Get latest Twitter sentiment (last 2 days)
            tweets_df = await twitter_analyzer.get_crypto_sentiment_async(
                coin, max_results=50, days_back=2
            )
            
            if tweets_df.empty:
                print(f"No tweets found for {coin}")
                continue
            
            # Use enhanced sentiment analyzer
            enhanced_df = sentiment_analyzer.analyze_dataframe(tweets_df)
            
            # Calculate overall sentiment score
            sentiment_score = sentiment_analyzer.calculate_sentiment_score(enhanced_df)
            sentiment_level = sentiment_analyzer.get_market_sentiment_level(sentiment_score)
            
            # Save results
            results[coin] = {
                'score': sentiment_score,
                'level': sentiment_level,
                'tweets_analyzed': len(enhanced_df),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save detailed results to CSV
            enhanced_df.to_csv(os.path.join(output_dir, f"{coin}_latest.csv"), index=False)
            
            print(f"Sentiment for {coin}: {sentiment_score:.1f}/100 - {sentiment_level}")
            
        except Exception as e:
            print(f"Error updating sentiment for {coin}: {e}")
    
    # Save summary to JSON
    with open(os.path.join(output_dir, 'sentiment_summary.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nSentiment update complete. Results saved to {output_dir}")

if __name__ == "__main__":
    asyncio.run(update_sentiment_data())