import asyncio
import pandas as pd
import os
from tqdm import tqdm
from src.models.sentiment.sentiment_analyzer import EnhancedSentimentAnalyzer

async def train_model():
    """Train the enhanced sentiment model using available crypto tweet data."""
    # Initialize the analyzer
    analyzer = EnhancedSentimentAnalyzer(model_dir='models/saved')
    
    # Collect tweet data from processed Twitter files
    data_dir = 'data/twitter_sentiment'
    
    if not os.path.exists(data_dir):
        print(f"Twitter data directory not found: {data_dir}")
        print("First collect tweet data using sentiment_analyzer.py script")
        return
    
    # Find all tweet files
    tweet_files = [f for f in os.listdir(data_dir) if f.endswith('_tweets.csv')]
    
    if not tweet_files:
        print(f"No tweet files found in {data_dir}")
        return
    
    print(f"Found {len(tweet_files)} tweet files")
    
    # Combine tweet data
    all_tweets = []
    for file in tqdm(tweet_files, desc="Loading tweet data"):
        try:
            coin = file.split('_tweets.csv')[0]
            df = pd.read_csv(os.path.join(data_dir, file))
            # Add coin label
            df['coin'] = coin
            all_tweets.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not all_tweets:
        print("No tweet data could be loaded")
        return
    
    # Combine all tweets
    combined_tweets = pd.concat(all_tweets, ignore_index=True)
    print(f"Combined {len(combined_tweets)} tweets for training")
    
    # Train the model
    training_result = analyzer.train_model(combined_tweets)
    
    if 'error' in training_result:
        print(f"Error during training: {training_result['error']}")
    else:
        print(f"Model training complete. Accuracy: {training_result.get('accuracy', 'N/A')}")
        print("Model saved to models/saved directory")

if __name__ == "__main__":
    asyncio.run(train_model())