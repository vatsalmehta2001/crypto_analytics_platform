import asyncio
from twscrape import API, gather
from twscrape.logger import set_log_level
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datetime

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Initialize twscrape API
api = API()
set_log_level("ERROR")  # Reduce logging noise

# This is a drop-in replacement for the original twitter.py script
def get_crypto_sentiment(coin_symbol, max_results=10):
    """
    Get sentiment for a cryptocurrency from Twitter using twscrape.
    This function maintains the same interface as the original Twitter API version.
    
    Args:
        coin_symbol (str): Cryptocurrency ticker symbol (e.g., 'BTC')
        max_results (int): Maximum number of tweets to retrieve
        
    Returns:
        pd.DataFrame: DataFrame with tweets and sentiment data
    """
    # Create search query
    query = f"#{coin_symbol} OR {coin_symbol} -filter:retweets lang:en"
    
    print(f"Searching for: {query}")
    
    # Set time window (last 7 days)
    end_time = datetime.datetime.utcnow()
    start_time = end_time - datetime.timedelta(days=7)
    
    # Format dates for Twitter search
    since_date = start_time.strftime('%Y-%m-%d')
    until_date = end_time.strftime('%Y-%m-%d')
    
    # Add date range to query
    full_query = f"{query} since:{since_date} until:{until_date}"
    
    # Function to run async code
    async def fetch_tweets():
        try:
            # Collect tweets using twscrape
            tweets = await gather(api.search(full_query, limit=max_results))
            
            if not tweets:
                print("No tweets found")
                return pd.DataFrame()
            
            # Process tweets
            results = []
            for tweet in tweets:
                # Get sentiment scores
                sentiment = analyzer.polarity_scores(tweet.rawContent)
                
                # Add to results
                results.append({
                    'date': tweet.date,
                    'text': tweet.rawContent,
                    'sentiment_score': sentiment['compound'],
                    'positive_ratio': sentiment['pos'],
                    'negative_ratio': sentiment['neg'],
                    'neutral_ratio': sentiment['neu'],
                })
            
            # Create DataFrame
            return pd.DataFrame(results)
        
        except Exception as e:
            print(f"Error fetching tweets: {e}")
            return pd.DataFrame()
    
    # Run async function and return results
    return asyncio.run(fetch_tweets())

# Example usage
if __name__ == "__main__":
    # Define async test function for adding accounts (optional)
    async def add_account():
        """
        Optionally add a Twitter account to improve rate limits
        """
        try:
            # Uncomment and replace with real credentials to add account
            # await api.pool.add_account("your_username", "your_password")
            # await api.pool.login_all()
            print("To improve scraping capabilities, add a Twitter account.")
            print("Uncomment and edit the code in this function with your credentials.")
        except Exception as e:
            print(f"Error adding account: {e}")
    
    # Run the account addition (optional)
    # asyncio.run(add_account())
    
    # Test with Bitcoin
    btc_sentiment = get_crypto_sentiment('BTC', max_results=10)
    print(btc_sentiment[['sentiment_score', 'text']].head())