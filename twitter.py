import tweepy
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Twitter credentials
bearer_token = "AAAAAAAAAAAAAAAAAAAAAAhH0AEAAAAAJSLfUglW3uLvPhr8LgKPhF8nyjk%3D70hLqvwhycoakcbYtsb1caseZqSm07tOjkQ11iUcOOLKcl2ZJS"  # Replace with your actual token

# Initialize Twitter client
client = tweepy.Client(bearer_token=bearer_token)

# Test function to get tweets and analyze sentiment
def get_crypto_sentiment(coin_symbol, max_results=10):
    # Create search query
    query = f"#{coin_symbol} OR {coin_symbol} -is:retweet lang:en"
    
    print(f"Searching for: {query}")
    
    # Search tweets
    tweets = client.search_recent_tweets(
        query=query,
        max_results=max_results,
        tweet_fields=['created_at']
    )
    
    if not tweets.data:
        print("No tweets found")
        return pd.DataFrame()
    
    # Process tweets
    results = []
    for tweet in tweets.data:
        # Get sentiment scores
        sentiment = analyzer.polarity_scores(tweet.text)
        
        # Add to results
        results.append({
            'date': tweet.created_at,
            'text': tweet.text,
            'sentiment_score': sentiment['compound'],
            'positive_ratio': sentiment['pos'],
            'negative_ratio': sentiment['neg'],
            'neutral_ratio': sentiment['neu'],
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    return df

# Test with Bitcoin
btc_sentiment = get_crypto_sentiment('BTC', max_results=10)
print(btc_sentiment[['sentiment_score', 'text']].head())