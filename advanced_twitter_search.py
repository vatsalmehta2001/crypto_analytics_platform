import asyncio
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from twscrape import AccountsPool, gather

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

async def main():
    print("Twitter Cryptocurrency Sentiment Analysis")
    
    # Create AccountsPool (not API)
    pool = AccountsPool("accounts.db")
    
    # Add Twitter account
    username = input("Twitter username: ")
    password = input("Twitter password: ")
    email = input("Email for verification: ")
    email_password = input("Email password: ")
    
    # Check if we have the correct method (add_accounts vs add_account)
    if hasattr(pool, "add_accounts"):
        await pool.add_accounts([(username, password, email, email_password)])
        print("Account added successfully")
    elif hasattr(pool, "add_account"):
        await pool.add_account(username, password, email, email_password)
        print("Account added successfully")
    else:
        print("Could not find a method to add accounts")
        return
    
    # Login
    await pool.login_all()
    print("Login successful")
    
    # Get crypto symbol
    coin = input("Enter cryptocurrency symbol (e.g., BTC): ").strip().upper() or "BTC"
    
    # Create search query
    query = f"{coin} OR #{coin} -filter:retweets lang:en"
    print(f"Searching for: {query}")
    
    # Create a client from the pool
    from twscrape import Client
    client = Client(pool)
    
    # Search tweets
    try:
        # Search for tweets (using the client)
        tweets = await client.search(query, limit=20)
        
        if tweets:
            print(f"Found {len(tweets)} tweets")
            
            # Process results
            results = []
            for tweet in tweets:
                # Check if we have the right property (text vs rawContent)
                content = tweet.text if hasattr(tweet, "text") else tweet.rawContent if hasattr(tweet, "rawContent") else "No content"
                
                # Analyze sentiment
                sentiment = analyzer.polarity_scores(content)
                
                # Print result
                label = "POSITIVE" if sentiment['compound'] > 0.05 else "NEGATIVE" if sentiment['compound'] < -0.05 else "NEUTRAL"
                print(f"\n[{label}] Score: {sentiment['compound']:.2f}")
                print(content[:200] + "..." if len(content) > 200 else content)
                
                # Add to results
                results.append({
                    'text': content,
                    'sentiment': sentiment['compound'],
                    'date': tweet.date if hasattr(tweet, "date") else None
                })
            
            # Save results
            df = pd.DataFrame(results)
            df.to_csv(f"{coin}_sentiment.csv", index=False)
            print(f"\nResults saved to {coin}_sentiment.csv")
        else:
            print("No tweets found")
            
    except Exception as e:
        print(f"Error during search: {e}")

if __name__ == "__main__":
    asyncio.run(main())