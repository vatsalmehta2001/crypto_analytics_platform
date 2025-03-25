import asyncio
import os
import pandas as pd
import praw
import datetime
from tqdm import tqdm

# Import the sentiment analyzer
from src.models.sentiment.sentiment_analyzer import EnhancedSentimentAnalyzer

def collect_reddit_data():
    """
    Collect and analyze Reddit data for cryptocurrency sentiment
    """
    print("Starting Reddit data collection and analysis...")
    
    # Create output directory
    output_dir = 'data/reddit_sentiment'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the enhanced sentiment analyzer
    analyzer = EnhancedSentimentAnalyzer(model_dir='models/saved')
    
    # Initialize Reddit API (you'll need to create a Reddit app at https://www.reddit.com/prefs/apps)
    reddit = praw.Reddit(
        client_id="YNnpGUsSVTH-94FqVOK6ug",
        client_secret="mXHF3zW4w3FdzO_Q66YEmB6De65dNQ",
        user_agent="crypto_sentiment_analyzer by /u/m0kiro"
    )
    
    # Get list of coins from processed data directory
    data_dir = 'data/processed'
    files = [f for f in os.listdir(data_dir) if f.endswith('_processed.csv')]
    coins = [f.split('_processed.csv')[0] for f in files]
    
    print(f"Found {len(coins)} coins to process")
    
    # Select subset of coins for testing or use all
    coins_to_process = coins[:10]  # Process first 10 coins (adjust as needed)
    
    # Subreddits to search
    crypto_subreddits = [
        "CryptoCurrency", "CryptoMarkets", "Bitcoin", "ethereum", "CryptoMoonShots",
        "altcoin", "binance", "defi", "CryptoTechnology", "SatoshiStreetBets"
    ]
    
    # Process each coin
    for coin in tqdm(coins_to_process, desc="Processing coins"):
        print(f"\nCollecting posts for {coin}...")
        
        try:
            # Create search query
            # Combine exact match and general search
            search_terms = [coin, f"${coin}"]
            
            # Common names for major coins
            common_names = {
                "BTC": ["Bitcoin", "BTC"],
                "ETH": ["Ethereum", "ETH"],
                "SOL": ["Solana", "SOL"],
                "ADA": ["Cardano", "ADA"],
                "DOT": ["Polkadot", "DOT"],
                "DOGE": ["Dogecoin", "DOGE"],
                "SHIB": ["Shiba Inu", "SHIB"],
                "XRP": ["Ripple", "XRP"],
                "MATIC": ["Polygon", "MATIC"],
                "LINK": ["Chainlink", "LINK"],
                "AVAX": ["Avalanche", "AVAX"],
                "LTC": ["Litecoin", "LTC"]
            }
            
            if coin in common_names:
                search_terms.extend(common_names[coin])
            
            # Collect posts and comments
            results = []
            
            # Set time limit (7 days ago)
            cutoff_date = datetime.datetime.utcnow() - datetime.timedelta(days=7)
            
            for subreddit_name in crypto_subreddits:
                try:
                    subreddit = reddit.subreddit(subreddit_name)
                    
                    # Search for posts
                    for term in search_terms:
                        for submission in subreddit.search(term, limit=20, time_filter="week"):
                            post_date = datetime.datetime.fromtimestamp(submission.created_utc)
                            if post_date < cutoff_date:
                                continue
                                
                            # Add post
                            results.append({
                                'date': post_date,
                                'text': submission.title + " " + (submission.selftext or ""),
                                'score': submission.score,
                                'comments': submission.num_comments,
                                'url': submission.url,
                                'subreddit': subreddit_name,
                                'id': submission.id,
                                'type': 'post'
                            })
                            
                            # Get top comments
                            submission.comments.replace_more(limit=0)  # Only fetch top-level comments
                            for comment in submission.comments[:10]:  # Get top 10 comments
                                comment_date = datetime.datetime.fromtimestamp(comment.created_utc)
                                if comment_date < cutoff_date:
                                    continue
                                    
                                results.append({
                                    'date': comment_date,
                                    'text': comment.body,
                                    'score': comment.score,
                                    'comments': 0,  # No comments on comments
                                    'url': f"https://www.reddit.com{comment.permalink}",
                                    'subreddit': subreddit_name,
                                    'id': comment.id,
                                    'type': 'comment'
                                })
                    
                except Exception as e:
                    print(f"Error processing subreddit {subreddit_name}: {e}")
            
            # Create DataFrame
            if not results:
                print(f"No Reddit posts found for {coin}")
                continue
                
            print(f"Found {len(results)} posts/comments for {coin}")
            
            posts_df = pd.DataFrame(results)
            
            # Analyze sentiment
            sentiment_df = analyzer.analyze_dataframe(posts_df)
            
            # Save to CSV
            posts_output_path = os.path.join(output_dir, f"{coin}_posts.csv")
            sentiment_df.to_csv(posts_output_path, index=False)
            
            # Calculate daily sentiment
            if 'date' in sentiment_df.columns:
                sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
                sentiment_df['date_day'] = sentiment_df['date'].dt.date
                
                # Aggregate by day
                daily_agg = sentiment_df.groupby('date_day').agg({
                    'sentiment_compound': 'mean',
                    'sentiment_positive': 'mean',
                    'sentiment_negative': 'mean',
                    'sentiment_neutral': 'mean',
                    'score': 'sum',
                    'text': 'count'
                }).reset_index()
                
                # Rename columns
                daily_agg.rename(columns={'text': 'post_volume', 'score': 'total_score'}, inplace=True)
                
                # Convert date_day back to datetime
                daily_agg['date'] = pd.to_datetime(daily_agg['date_day'])
                daily_agg.drop('date_day', axis=1, inplace=True)
                
                # Save daily sentiment
                daily_output_path = os.path.join(output_dir, f"{coin}_daily.csv")
                daily_agg.to_csv(daily_output_path, index=False)
            
        except Exception as e:
            print(f"Error processing {coin}: {e}")
    
    # Print summary
    print("\nReddit data collection complete!")
    print(f"Data saved to {output_dir}")
    print(f"Total coins processed: {len(coins_to_process)}")

if __name__ == "__main__":
    collect_reddit_data()