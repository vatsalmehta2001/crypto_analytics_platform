import asyncio
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta
from tqdm import tqdm
from twscrape import API, gather
from twscrape.logger import set_log_level
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class TwscrapeTwitterSentimentAnalyzer:
    """
    Class for analyzing cryptocurrency sentiment from Twitter data using Twscrape.
    Uses VADER for sentiment analysis and Twscrape for data retrieval.
    """
    
    def __init__(self, raw_data_path='data/raw'):
        """
        Initialize the Twitter sentiment analyzer.
        
        Args:
            raw_data_path (str): Path to raw data directory containing coin CSV files
        """
        # Disable excessive logging from twscrape
        set_log_level("ERROR")
        
        # Initialize twscrape API
        self.api = API()
        self.analyzer = SentimentIntensityAnalyzer()
        self.request_count = 0
        self.last_request_time = None
        self.raw_data_path = raw_data_path
        self.coin_map = self._build_coin_map()
    
    def _build_coin_map(self):
        """
        Build a map of coin symbols to their common names based on files in raw data directory.
        
        Returns:
            dict: Mapping of coin symbols to common names and keywords
        """
        coin_map = {}
        
        # Get all CSV files in the raw data directory
        try:
            files = [f for f in os.listdir(self.raw_data_path) if f.endswith('.csv') and f != 'cryptonews.csv']
            coin_symbols = [os.path.splitext(f)[0] for f in files]
            
            # Build default mappings
            for symbol in coin_symbols:
                coin_map[symbol] = {
                    'symbol': symbol,
                    'name': symbol,
                    'keywords': [symbol]
                }
            
            # Add known mappings for popular coins
            common_coins = {
                "BTC": {
                    'name': 'Bitcoin',
                    'keywords': ['bitcoin', 'btc', 'crypto']
                },
                "ETH": {
                    'name': 'Ethereum',
                    'keywords': ['ethereum', 'eth', 'vitalik']
                },
                "XRP": {
                    'name': 'Ripple',
                    'keywords': ['ripple', 'xrp']
                },
                "SOL": {
                    'name': 'Solana',
                    'keywords': ['solana', 'sol']
                },
                "ADA": {
                    'name': 'Cardano',
                    'keywords': ['cardano', 'ada', 'hoskinson']
                },
                "DOGE": {
                    'name': 'Dogecoin',
                    'keywords': ['dogecoin', 'doge']
                },
                "DOT": {
                    'name': 'Polkadot',
                    'keywords': ['polkadot', 'dot']
                },
                "LINK": {
                    'name': 'Chainlink',
                    'keywords': ['chainlink', 'link']
                },
                "AVAX": {
                    'name': 'Avalanche',
                    'keywords': ['avalanche', 'avax']
                },
                "MATIC": {
                    'name': 'Polygon',
                    'keywords': ['polygon', 'matic']
                }
            }
            
            # Update common coins found in the directory
            for symbol, data in common_coins.items():
                if symbol in coin_map:
                    coin_map[symbol].update(data)
            
            print(f"Found {len(coin_map)} coins in raw data directory")
            
        except Exception as e:
            print(f"Error building coin map: {e}")
        
        return coin_map
    
    async def add_twitter_account(self, username, password, email=None, email_password=None):
        """
        Add a Twitter account to the API pool for better rate limits.
        This step is optional but recommended for better performance.
        
        Args:
            username (str): Twitter username
            password (str): Twitter password
            email (str, optional): Email associated with Twitter account
            email_password (str, optional): Email password
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if email and email_password:
                # Add account with email for handling challenges
                await self.api.pool.add_account(username, password, email, email_password)
            else:
                # Add basic account
                await self.api.pool.add_account(username, password)
            
            # Try to login
            await self.api.pool.login_all()
            print(f"Successfully added Twitter account: {username}")
            return True
        except Exception as e:
            print(f"Error adding Twitter account: {e}")
            return False
    
    async def get_crypto_sentiment_async(self, coin_symbol, max_results=100, days_back=7):
        """
        Async method to get sentiment for a cryptocurrency from Twitter using Twscrape.
        
        Args:
            coin_symbol (str): Cryptocurrency ticker symbol (e.g., 'BTC')
            max_results (int): Maximum number of tweets to retrieve
            days_back (int): Number of days to look back
            
        Returns:
            pd.DataFrame: DataFrame with tweets and sentiment data
        """
        # Create search query with coin-specific terms
        query = self._build_query(coin_symbol)
        print(f"Searching for: {query}")
        
        try:
            # Set time window
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=days_back)
            
            # Format dates for Twitter search
            since_date = start_time.strftime('%Y-%m-%d')
            until_date = end_time.strftime('%Y-%m-%d')
            
            # Build full query with date range
            full_query = f"{query} since:{since_date} until:{until_date}"
            
            # Search tweets
            tweets = await gather(self.api.search(full_query, limit=max_results))
            
            if not tweets:
                print("No tweets found")
                return pd.DataFrame()
            
            # Process tweets
            results = []
            for tweet in tqdm(tweets, desc=f"Analyzing {coin_symbol} tweets"):
                # Get sentiment scores
                sentiment = self.analyzer.polarity_scores(tweet.rawContent)
                
                # Add to results
                results.append({
                    'date': tweet.date,
                    'text': tweet.rawContent[:280],  # Limit text length for storage
                    'sentiment_score': sentiment['compound'],
                    'positive_ratio': sentiment['pos'],
                    'negative_ratio': sentiment['neg'],
                    'neutral_ratio': sentiment['neu'],
                    'retweet_count': tweet.retweetCount,
                    'like_count': tweet.likeCount,
                    'reply_count': tweet.replyCount,
                    'user': tweet.user.username,
                    'tweet_id': tweet.id
                })
            
            # Create DataFrame
            df = pd.DataFrame(results)
            
            # Add engagement-weighted sentiment
            if not df.empty:
                df['engagement'] = df['retweet_count'] + df['like_count'] + df['reply_count'] + 1
                df['weighted_sentiment'] = df['sentiment_score'] * np.log1p(df['engagement'])  # Log-scale engagement for more balanced weighting
            
            return df
            
        except Exception as e:
            print(f"Error fetching tweets: {e}")
            return pd.DataFrame()
    
    def get_crypto_sentiment(self, coin_symbol, max_results=100, days_back=7):
        """
        Synchronous wrapper for get_crypto_sentiment_async.
        
        Args:
            coin_symbol (str): Cryptocurrency ticker symbol (e.g., 'BTC')
            max_results (int): Maximum number of tweets to retrieve
            days_back (int): Number of days to look back
            
        Returns:
            pd.DataFrame: DataFrame with tweets and sentiment data
        """
        return asyncio.run(self.get_crypto_sentiment_async(coin_symbol, max_results, days_back))
    
    def _build_query(self, coin_symbol):
        """
        Build a query string for searching tweets based on coin symbol.
        
        Args:
            coin_symbol (str): Cryptocurrency ticker symbol
            
        Returns:
            str: Search query string
        """
        coin_symbol = coin_symbol.upper()
        
        # Use coin map if available
        if coin_symbol in self.coin_map:
            coin_data = self.coin_map[coin_symbol]
            query_parts = []
            
            # Add hashtag and symbol
            query_parts.append(f"#{coin_symbol}")
            query_parts.append(coin_symbol)
            
            # Add coin name if different from symbol
            if coin_data['name'].lower() != coin_symbol.lower():
                query_parts.append(coin_data['name'])
                query_parts.append(f"#{coin_data['name'].replace(' ', '')}")
            
            # Add additional keywords
            for keyword in coin_data['keywords']:
                if keyword.lower() != coin_symbol.lower() and keyword.lower() != coin_data['name'].lower():
                    query_parts.append(keyword)
            
            # Join with OR
            query = " OR ".join(f"({part})" for part in query_parts)
        else:
            # Base query with hashtag and symbol mention
            query = f"#{coin_symbol} OR {coin_symbol}"
        
        # Add filters for quality
        query += " -filter:retweets lang:en"
        
        return query
    
    def aggregate_daily_sentiment(self, tweet_df):
        """
        Aggregate tweet-level sentiment data to daily level.
        
        Args:
            tweet_df (pd.DataFrame): DataFrame with tweet-level sentiment data
            
        Returns:
            pd.DataFrame: Daily aggregated sentiment data
        """
        if tweet_df.empty:
            return pd.DataFrame()
        
        # Convert date to datetime if needed
        if not pd.api.types.is_datetime64_dtype(tweet_df['date']):
            tweet_df['date'] = pd.to_datetime(tweet_df['date'])
        
        # Extract date only for grouping
        tweet_df['date_day'] = tweet_df['date'].dt.date
        
        # Aggregate by day
        daily_agg = tweet_df.groupby('date_day').agg({
            'sentiment_score': 'mean',
            'positive_ratio': 'mean',
            'negative_ratio': 'mean',
            'neutral_ratio': 'mean',
            'text': 'count',
            'engagement': 'sum',
            'weighted_sentiment': 'sum'
        }).reset_index()
        
        # Calculate engagement-weighted sentiment
        daily_agg['sentiment_score_weighted'] = daily_agg['weighted_sentiment'] / daily_agg['engagement']
        
        # Rename count column to volume
        daily_agg.rename(columns={'text': 'tweet_volume'}, inplace=True)
        
        # Convert date_day back to datetime
        daily_agg['date'] = pd.to_datetime(daily_agg['date_day'])
        daily_agg.drop('date_day', axis=1, inplace=True)
        
        return daily_agg
    
    async def process_all_coins_async(self, max_results_per_coin=100, days_back=30, output_dir='data/twitter_sentiment'):
        """
        Async method to process Twitter sentiment for all coins in the raw data directory.
        
        Args:
            max_results_per_coin (int): Maximum number of tweets to retrieve per coin
            days_back (int): Number of days to look back
            output_dir (str): Directory to save results
            
        Returns:
            dict: Dictionary of coin symbols to sentiment DataFrames
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        for coin_symbol in self.coin_map.keys():
            print(f"\nProcessing Twitter sentiment for {coin_symbol}...")
            
            # Get sentiment from Twitter
            tweets_df = await self.get_crypto_sentiment_async(
                coin_symbol, 
                max_results=max_results_per_coin,
                days_back=days_back
            )
            
            if tweets_df.empty:
                print(f"No tweets found for {coin_symbol}")
                continue
            
            # Aggregate by day
            daily_sentiment = self.aggregate_daily_sentiment(tweets_df)
            
            # Store results
            results[coin_symbol] = daily_sentiment
            
            # Save to CSV
            tweets_output_path = os.path.join(output_dir, f"{coin_symbol}_tweets.csv")
            daily_output_path = os.path.join(output_dir, f"{coin_symbol}_daily.csv")
            
            tweets_df.to_csv(tweets_output_path, index=False)
            daily_sentiment.to_csv(daily_output_path, index=False)
            
            print(f"Saved Twitter sentiment for {coin_symbol}")
            
            # Add a short delay between coins to be nice to Twitter
            await asyncio.sleep(5)
        
        return results
    
    def process_all_coins(self, max_results_per_coin=100, days_back=30, output_dir='data/twitter_sentiment'):
        """
        Synchronous wrapper for process_all_coins_async.
        
        Args:
            max_results_per_coin (int): Maximum number of tweets to retrieve per coin
            days_back (int): Number of days to look back
            output_dir (str): Directory to save results
            
        Returns:
            dict: Dictionary of coin symbols to sentiment DataFrames
        """
        return asyncio.run(self.process_all_coins_async(
            max_results_per_coin=max_results_per_coin,
            days_back=days_back,
            output_dir=output_dir
        ))


# Example usage
if __name__ == "__main__":
    # Define async test function
    async def test_async():
        analyzer = TwscrapeTwitterSentimentAnalyzer()
        
        # Optional: Add a Twitter account for better rate limits
        # await analyzer.add_twitter_account("your_username", "your_password")
        
        # Get Bitcoin tweets
        btc_tweets = await analyzer.get_crypto_sentiment_async('BTC', max_results=20)
        
        if not btc_tweets.empty:
            print(f"Retrieved {len(btc_tweets)} tweets for BTC")
            print(btc_tweets[['sentiment_score', 'text']].head())
            
            # Aggregate by day
            daily_sentiment = analyzer.aggregate_daily_sentiment(btc_tweets)
            print(daily_sentiment[['date', 'sentiment_score', 'tweet_volume']].head())
    
    # Run the async test
    asyncio.run(test_async())