from src.models.sentiment.twitter_sentiment import TwitterSentimentAnalyzer

# Initialize the analyzer (no bearer token needed anymore)
analyzer = TwitterSentimentAnalyzer()

# Get sentiment data for Bitcoin
btc_tweets = analyzer.get_crypto_sentiment('BTC', max_results=50, days_back=14)

# Print results
if not btc_tweets.empty:
    print(f"Retrieved {len(btc_tweets)} tweets for BTC")
    print("\nSample tweets with sentiment:")
    print(btc_tweets[['sentiment_score', 'text']].head())
    
    # Aggregate by day
    daily_sentiment = analyzer.aggregate_daily_sentiment(btc_tweets)
    print("\nDaily aggregated sentiment:")
    print(daily_sentiment[['date', 'sentiment_score', 'tweet_volume']].head())
else:
    print("No tweets found or error occurred")