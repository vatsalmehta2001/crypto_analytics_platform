import tweepy
import pandas as pd
import numpy as np
import time
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datetime
from tqdm import tqdm

class TwitterSentimentAnalyzer:
    """
    Class for analyzing cryptocurrency sentiment from Twitter data.
    Uses VADER for sentiment analysis and Twitter API v2 for data retrieval.
    """
    
    def __init__(self, bearer_token, raw_data_path='data/raw'):
        """
        Initialize the Twitter sentiment analyzer.
        
        Args:
            bearer_token (str): Twitter API v2 bearer token
            raw_data_path (str): Path to raw data directory containing coin CSV files
        """
        self.client = tweepy.Client(bearer_token=bearer_token)
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
                "ICP": {
                    'name': 'Internet Computer',
                    'keywords': ['internet computer', 'icp', 'dfinity']
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
                },
                "UNI": {
                    'name': 'Uniswap',
                    'keywords': ['uniswap', 'uni']
                },
                "SHIB": {
                    'name': 'Shiba Inu',
                    'keywords': ['shiba', 'shib']
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
    
    def _respect_rate_limit(self):
        """Simple rate limiting to avoid 429 errors"""
        # Twitter free tier allows ~300 requests per 15 min window = ~1 request per 3 seconds
        # We'll be more conservative with 1 request per 5 seconds
        if self.last_request_time:
            elapsed = time.time() - self.last_request_time
            if elapsed < 5:  # Wait at least 5 seconds between requests
                time.sleep(5 - elapsed)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def get_crypto_sentiment(self, coin_symbol, max_results=100, days_back=7):
        """
        Get sentiment for a cryptocurrency from Twitter.
        
        Args:
            coin_symbol (str): Cryptocurrency ticker symbol (e.g., 'BTC')
            max_results (int): Maximum number of tweets to retrieve
            days_back (int): Number of days to look back (max 7 for free tier)
            
        Returns:
            pd.DataFrame: DataFrame with tweets and sentiment data
        """
        # Create search query with coin-specific terms
        query = self._build_query(coin_symbol)
        print(f"Searching for: {query}")
        
        # Respect rate limits
        self._respect_rate_limit()
        
        try:
            # Set time window - Twitter only allows searching last 7 days
            end_time = datetime.datetime.utcnow()
            start_time = end_time - datetime.timedelta(days=min(days_back, 7))
            
            # Search tweets with pagination if needed
            all_tweets = []
            pagination_token = None
            max_pages = max_results // 100 + (1 if max_results % 100 > 0 else 0)
            
            for i in range(max_pages):
                # Respect rate limits between pagination requests
                if i > 0:
                    self._respect_rate_limit()
                
                # Search tweets
                response = self.client.search_recent_tweets(
                    query=query,
                    max_results=min(100, max_results - len(all_tweets)),  # Max 100 per request
                    tweet_fields=['created_at', 'public_metrics', 'lang'],
                    start_time=start_time,
                    end_time=end_time,
                    next_token=pagination_token
                )
                
                if response is None or not response.data:
                    break
                    
                all_tweets.extend(response.data)
                
                # Get next pagination token
                if hasattr(response, 'meta') and response.meta.get('next_token') and len(all_tweets) < max_results:
                    pagination_token = response.meta['next_token']
                else:
                    break
            
            if not all_tweets:
                print("No tweets found")
                return pd.DataFrame()
            
            # Process tweets
            results = []
            for tweet in tqdm(all_tweets, desc=f"Analyzing {coin_symbol} tweets"):
                # Get sentiment scores
                sentiment = self.analyzer.polarity_scores(tweet.text)
                
                # Add to results
                results.append({
                    'date': tweet.created_at,
                    'text': tweet.text[:280],  # Limit text length for storage
                    'sentiment_score': sentiment['compound'],
                    'positive_ratio': sentiment['pos'],
                    'negative_ratio': sentiment['neg'],
                    'neutral_ratio': sentiment['neu'],
                    'retweet_count': tweet.public_metrics['retweet_count'] if hasattr(tweet, 'public_metrics') else 0,
                    'like_count': tweet.public_metrics['like_count'] if hasattr(tweet, 'public_metrics') else 0,
                    'reply_count': tweet.public_metrics['reply_count'] if hasattr(tweet, 'public_metrics') else 0
                })
            
            # Create DataFrame
            df = pd.DataFrame(results)
            
            # Add engagement-weighted sentiment
            if not df.empty:
                df['engagement'] = df['retweet_count'] + df['like_count'] + df['reply_count'] + 1
                df['weighted_sentiment'] = df['sentiment_score'] * np.log1p(df['engagement'])  # Log-scale engagement for more balanced weighting
            
            return df
            
        except tweepy.errors.TooManyRequests:
            print("Rate limit exceeded. Waiting before trying again...")
            time.sleep(60)  # Wait 60 seconds before trying again
            return pd.DataFrame()  # Return empty DataFrame for now
            
        except Exception as e:
            print(f"Error fetching tweets: {e}")
            return pd.DataFrame()
    
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
            query = " OR ".join(query_parts)
        else:
            # Base query with hashtag and symbol mention
            query = f"#{coin_symbol} OR {coin_symbol}"
        
        # Add filters for quality
        query += " -is:retweet lang:en"
        
        # Limit query length (Twitter has a maximum)
        if len(query) > 500:
            query = query[:500]
        
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
    
    def process_all_coins(self, max_results_per_coin=50, days_back=7, output_dir='data/twitter_sentiment'):
        """
        Process Twitter sentiment for all coins in the raw data directory.
        
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
            tweets_df = self.get_crypto_sentiment(
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
            
            # Respect rate limits between coins
            time.sleep(5)
        
        return results

# Example usage
if __name__ == "__main__":
    # Replace with your actual bearer token
    BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAAhH0AEAAAAAJSLfUglW3uLvPhr8LgKPhF8nyjk%3D70hLqvwhycoakcbYtsb1caseZqSm07tOjkQ11iUcOOLKcl2ZJS"
    
    analyzer = TwitterSentimentAnalyzer(BEARER_TOKEN)
    
    # Process specific coins
    btc_sentiment = analyzer.get_crypto_sentiment('BTC', max_results=50)
    
    if not btc_sentiment.empty:
        print(f"Retrieved {len(btc_sentiment)} tweets for BTC")
        print(btc_sentiment[['sentiment_score', 'text']].head())
        
        # Aggregate by day
        daily_sentiment = analyzer.aggregate_daily_sentiment(btc_sentiment)

        print(daily_sentiment[['date', 'sentiment_score', 'tweet_volume']].head())
    
    # Uncomment to process all coins
    # all_coins = analyzer.process_all_coins(max_results_per_coin=50)