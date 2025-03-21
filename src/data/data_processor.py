import os
import pandas as pd
import json
import numpy as np
from datetime import datetime, timedelta
import re
import time
from tqdm import tqdm

class CryptoDataProcessor:
    """
    Class for processing cryptocurrency price data and news sentiment data.
    Handles data loading, cleaning, alignment, and feature generation.
    """
    
    def __init__(self, raw_data_path, processed_data_path):
        """
        Initialize the data processor with paths to raw and processed data.
        
        Args:
            raw_data_path (str): Path to raw data directory
            processed_data_path (str): Path to processed data directory
        """
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.price_data = {}
        self.news_data = None
        
        # Create processed directory if it doesn't exist
        os.makedirs(processed_data_path, exist_ok=True)
    
    def load_price_data(self, coin_list=None):
        """
        Load price data for specified coins or all available coins.
        
        Args:
            coin_list (list, optional): List of coin tickers to load. Defaults to None (all coins).
        """
        if coin_list is None:
            # Get all CSV files in the raw data directory
            coin_files = [f for f in os.listdir(self.raw_data_path) if f.endswith('.csv') and f != 'cryptonews.csv']
            coin_list = [os.path.splitext(f)[0] for f in coin_files]
        
        print(f"Loading price data for {len(coin_list)} coins...")
        for coin in tqdm(coin_list):
            try:
                file_path = os.path.join(self.raw_data_path, f"{coin}.csv")
                df = pd.read_csv(file_path)
                
                # Basic data cleaning
                df['date'] = pd.to_datetime(df['date'])
                df.sort_values('date', inplace=True)
                
                # Store in dictionary
                self.price_data[coin] = df
                
                print(f"Loaded {coin} data: {len(df)} records from {df['date'].min()} to {df['date'].max()}")
            except Exception as e:
                print(f"Error loading {coin} data: {e}")
    
    def load_news_data(self):
        """
        Load and process news sentiment data.
        """
        try:
            file_path = os.path.join(self.raw_data_path, "cryptonews.csv")
            df = pd.read_csv(file_path)
            
            # Convert date string to datetime
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Print a sample of sentiment data to debug
            print(f"Sample sentiment value: {df['sentiment'].iloc[0]}")
            
            # Process sentiment column which is stored as a string representation of dict
            df['sentiment_data'] = df['sentiment'].apply(self._parse_sentiment)
            
            # Create extracted features with careful error handling
            df['sentiment_class'] = df['sentiment_data'].apply(
                lambda x: x.get('class', 'neutral') if isinstance(x, dict) else 'neutral'
            )
            df['sentiment_polarity'] = df['sentiment_data'].apply(
                lambda x: float(x.get('polarity', 0)) if isinstance(x, dict) else 0.0
            )
            df['sentiment_subjectivity'] = df['sentiment_data'].apply(
                lambda x: float(x.get('subjectivity', 0)) if isinstance(x, dict) else 0.0
            )
            
            # Drop the original sentiment column and the parsed dictionary
            df.drop(['sentiment', 'sentiment_data'], axis=1, inplace=True)
            
            # Drop rows with NaT dates
            df = df.dropna(subset=['date'])
            
            # Sort by date
            df.sort_values('date', inplace=True)
            
            self.news_data = df
            
            print(f"Loaded news data: {len(df)} records from {df['date'].min()} to {df['date'].max()}")
        except Exception as e:
            print(f"Error loading news data: {e}")
            import traceback
            traceback.print_exc()
    
    def _parse_sentiment(self, sentiment_str):
        """
        Parse sentiment string into a dictionary.
        
        Args:
            sentiment_str (str): String representation of sentiment dictionary
            
        Returns:
            dict: Parsed sentiment dictionary
        """
        if not isinstance(sentiment_str, str):
            return {'class': 'neutral', 'polarity': 0.0, 'subjectivity': 0.0}
            
        try:
            # Method 1: Try direct JSON parsing after fixing quotes
            cleaned_str = sentiment_str.replace("'", '"')
            return json.loads(cleaned_str)
        except json.JSONDecodeError:
            # Method 2: Try regex pattern matching
            try:
                pattern = r"'class':\s*'(\w+)',\s*'polarity':\s*([-\d\.]+),\s*'subjectivity':\s*([-\d\.]+)"
                match = re.search(pattern, sentiment_str)
                if match:
                    return {
                        'class': match.group(1),
                        'polarity': float(match.group(2)),
                        'subjectivity': float(match.group(3))
                    }
            except Exception:
                pass
            
            # Method 3: Try individual regex patterns for each field
            try:
                class_match = re.search(r"'class':\s*'(\w+)'", sentiment_str)
                polarity_match = re.search(r"'polarity':\s*([-\d\.]+)", sentiment_str)
                subjectivity_match = re.search(r"'subjectivity':\s*([-\d\.]+)", sentiment_str)
                
                return {
                    'class': class_match.group(1) if class_match else 'neutral',
                    'polarity': float(polarity_match.group(1)) if polarity_match else 0.0,
                    'subjectivity': float(subjectivity_match.group(1)) if subjectivity_match else 0.0
                }
            except Exception:
                pass
        
        # If all parsing attempts fail, return default values
        return {'class': 'neutral', 'polarity': 0.0, 'subjectivity': 0.0}
    
    def generate_technical_indicators(self, coin):
        """
        Generate technical indicators for a specific coin.
        
        Args:
            coin (str): Coin ticker
            
        Returns:
            pd.DataFrame: DataFrame with technical indicators
        """
        if coin not in self.price_data:
            print(f"No data available for {coin}")
            return None
        
        df = self.price_data[coin].copy()
        
        # Simple Moving Averages
        df['sma_7'] = df['close'].rolling(window=7).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Relative Strength Index (RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        
        # Volatility
        df['daily_return'] = df['close'].pct_change()
        df['volatility_7'] = df['daily_return'].rolling(window=7).std()
        df['volatility_30'] = df['daily_return'].rolling(window=30).std()
        
        # Volume-based indicators could be added here if volume data is available
        
        return df
    
    def aggregate_daily_sentiment(self, coin_mentions=None):
        """
        Aggregate news sentiment on a daily basis, optionally filtering by coin mentions.
        
        Args:
            coin_mentions (list, optional): List of coins to filter news by. Defaults to None.
            
        Returns:
            pd.DataFrame: Daily aggregated sentiment
        """
        if self.news_data is None:
            print("News data not loaded")
            return None
        
        news_df = self.news_data.copy()
        
        # Filter by coin mentions if specified
        if coin_mentions:
            filtered_news = []
            for _, row in news_df.iterrows():
                text = str(row['text']).lower() + " " + str(row['title']).lower()
                if any(coin.lower() in text for coin in coin_mentions):
                    filtered_news.append(row)
            
            if filtered_news:
                news_df = pd.DataFrame(filtered_news)
            else:
                print(f"No news found mentioning {coin_mentions}")
                return pd.DataFrame()
        
        # Create date column without time for grouping by day
        news_df['date_day'] = news_df['date'].dt.date
        
        # Check if sentiment_class column exists
        if 'sentiment_class' not in news_df.columns:
            print("Warning: sentiment_class column not found. Creating a default neutral column.")
            news_df['sentiment_class'] = 'neutral'
        
        # Aggregate by day
        try:
            # Safely perform aggregation with error handling
            agg_dict = {
                'sentiment_polarity': ['mean', 'std', 'count'],
                'sentiment_subjectivity': ['mean', 'std']
            }
            
            # Add sentiment_class aggregation if available
            if 'sentiment_class' in news_df.columns:
                # First convert to category to ensure consistent handling
                news_df['sentiment_class'] = news_df['sentiment_class'].astype('category')
                
                # Safe aggregation function
                def safe_value_counts(x):
                    try:
                        return x.value_counts().to_dict()
                    except Exception:
                        return {'neutral': len(x)}
                
                agg_dict['sentiment_class'] = safe_value_counts
            
            daily_sentiment = news_df.groupby('date_day').agg(agg_dict).reset_index()
            
            # Flatten MultiIndex columns
            daily_sentiment.columns = ['_'.join(col).strip('_') for col in daily_sentiment.columns.values]
            
            # Convert date_day back to datetime
            daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date_day'])
            daily_sentiment.drop('date_day', axis=1, inplace=True)
            
            # Add sentiment ratio columns with error handling
            def calculate_ratio(x, class_name):
                try:
                    if not isinstance(x, dict):
                        return 0.0
                    total = sum(x.values())
                    return x.get(class_name, 0) / total if total > 0 else 0
                except Exception:
                    return 0.0
            
            if 'sentiment_class' in daily_sentiment.columns:
                daily_sentiment['positive_ratio'] = daily_sentiment['sentiment_class'].apply(
                    lambda x: calculate_ratio(x, 'positive')
                )
                
                daily_sentiment['negative_ratio'] = daily_sentiment['sentiment_class'].apply(
                    lambda x: calculate_ratio(x, 'negative')
                )
                
                daily_sentiment['neutral_ratio'] = daily_sentiment['sentiment_class'].apply(
                    lambda x: calculate_ratio(x, 'neutral')
                )
                
                # Add simple sentiment score (scaled -1 to 1)
                daily_sentiment['sentiment_score'] = daily_sentiment['positive_ratio'] - daily_sentiment['negative_ratio']
            else:
                # Create default columns if sentiment_class aggregation failed
                daily_sentiment['positive_ratio'] = 0.0
                daily_sentiment['negative_ratio'] = 0.0
                daily_sentiment['neutral_ratio'] = 1.0
                daily_sentiment['sentiment_score'] = 0.0
            
            return daily_sentiment
            
        except Exception as e:
            print(f"Error in aggregate_daily_sentiment: {e}")
            import traceback
            traceback.print_exc()
            
            # Return empty DataFrame with expected columns if aggregation fails
            return pd.DataFrame(columns=[
                'date', 'sentiment_polarity_mean', 'sentiment_polarity_std', 'sentiment_polarity_count',
                'sentiment_subjectivity_mean', 'sentiment_subjectivity_std',
                'positive_ratio', 'negative_ratio', 'neutral_ratio', 'sentiment_score'
            ])
    
    def align_price_sentiment_data(self, coin, sentiment_df=None):
        """
        Align price data with sentiment data for a specific coin.
        
        Args:
            coin (str): Coin ticker
            sentiment_df (pd.DataFrame, optional): Pre-filtered sentiment DataFrame. 
                                                  If None, all sentiment data will be used.
            
        Returns:
            pd.DataFrame: Combined price and sentiment data
        """
        if coin not in self.price_data:
            print(f"No price data available for {coin}")
            return None
        
        # Get price data with technical indicators
        price_df = self.generate_technical_indicators(coin)
        
        # If no sentiment data provided, get for this specific coin
        if sentiment_df is None or sentiment_df.empty:
            sentiment_df = self.aggregate_daily_sentiment(coin_mentions=[coin])
            
        # If still no sentiment data, return just price data with default sentiment values
        if sentiment_df is None or sentiment_df.empty:
            print(f"No sentiment data available for {coin}, returning price data only")
            # Add default sentiment columns
            price_df['sentiment_polarity_mean'] = 0.0
            price_df['sentiment_polarity_std'] = 0.0
            price_df['sentiment_polarity_count'] = 0
            price_df['sentiment_subjectivity_mean'] = 0.0
            price_df['sentiment_subjectivity_std'] = 0.0
            price_df['positive_ratio'] = 0.0
            price_df['negative_ratio'] = 0.0
            price_df['neutral_ratio'] = 1.0
            price_df['sentiment_score'] = 0.0
            return price_df
        
        # Prepare price data for merging
        price_df['date_day'] = price_df['date'].dt.date
        price_df['date_day'] = pd.to_datetime(price_df['date_day'])
        
        # Prepare sentiment data for merging
        if 'date_day' not in sentiment_df.columns:
            sentiment_df['date_day'] = pd.to_datetime(sentiment_df['date']).dt.date
            sentiment_df['date_day'] = pd.to_datetime(sentiment_df['date_day'])
        
        # Merge on date_day
        combined_df = pd.merge(
            price_df, 
            sentiment_df, 
            left_on='date_day', 
            right_on='date_day', 
            how='left'
        )
        
        # Forward fill missing sentiment data (carry forward last known sentiment)
        sentiment_columns = [col for col in combined_df.columns if 'sentiment' in col or 'ratio' in col]
        combined_df[sentiment_columns] = combined_df[sentiment_columns].fillna(method='ffill')
        
        # Fill any remaining NaN values with defaults
        for col in sentiment_columns:
            if 'polarity' in col or 'score' in col:
                combined_df[col] = combined_df[col].fillna(0)
            elif 'count' in col:
                combined_df[col] = combined_df[col].fillna(0)
            elif 'ratio' in col:
                combined_df[col] = combined_df[col].fillna(1/3 if 'neutral' in col else 0)
            else:
                combined_df[col] = combined_df[col].fillna(0)
        
        # Clean up date columns
        combined_df.drop('date_day', axis=1, inplace=True)
        if 'date_y' in combined_df.columns:
            combined_df.drop('date_y', axis=1, inplace=True)
            combined_df.rename(columns={'date_x': 'date'}, inplace=True)
        
        return combined_df
    
    def prepare_all_coins_data(self, coin_list=None):
        """
        Prepare aligned price and sentiment data for multiple coins.
        
        Args:
            coin_list (list, optional): List of coin tickers to process. If None, all loaded coins are processed.
            
        Returns:
            dict: Dictionary of coin tickers to aligned DataFrames
        """
        if coin_list is None:
            coin_list = list(self.price_data.keys())
        
        # Calculate market-wide sentiment once
        market_sentiment = self.aggregate_daily_sentiment()
        
        aligned_data = {}
        for coin in tqdm(coin_list):
            # Get coin-specific sentiment
            coin_sentiment = self.aggregate_daily_sentiment(coin_mentions=[coin])
            
            # If not enough coin-specific sentiment, use market sentiment
            if coin_sentiment is None or coin_sentiment.empty or len(coin_sentiment) < 30:  # Arbitrary threshold
                print(f"Not enough sentiment data for {coin}, using market-wide sentiment")
                combined_df = self.align_price_sentiment_data(coin, market_sentiment)
            else:
                combined_df = self.align_price_sentiment_data(coin, coin_sentiment)
            
            if combined_df is not None and len(combined_df) > 0:
                aligned_data[coin] = combined_df
                
                # Save to processed data folder
                output_path = os.path.join(self.processed_data_path, f"{coin}_processed.csv")
                combined_df.to_csv(output_path, index=False)
                print(f"Saved processed data for {coin} to {output_path}")
        
        return aligned_data
    
    def load_twitter_sentiment(self, coin_list=None, bearer_token=None):
        """
        Load sentiment data from Twitter instead of news data.
        
        Args:
            coin_list (list): List of coin tickers to analyze
            bearer_token (str): Twitter API bearer token
        """
        if bearer_token is None:
            print("No Twitter API bearer token provided")
            return
        
        # Import TwitterSentimentAnalyzer here to avoid importing when not used
        from src.models.sentiment.twitter_sentiment import TwitterSentimentAnalyzer
        
        # Initialize Twitter analyzer
        twitter_analyzer = TwitterSentimentAnalyzer(bearer_token, self.raw_data_path)
        
        if coin_list is None:
            coin_list = list(self.price_data.keys())
        
        # Store sentiment data
        self.twitter_sentiment = {}
        
        for coin in coin_list:
            print(f"Getting Twitter sentiment for {coin}...")
            try:
                # Get tweets with conservative rate limiting
                coin_sentiment = twitter_analyzer.get_crypto_sentiment(
                    coin, max_results=50, days_back=7
                )
                
                if not coin_sentiment.empty:
                    daily_sentiment = twitter_analyzer.aggregate_daily_sentiment(coin_sentiment)
                    self.twitter_sentiment[coin] = daily_sentiment
                    print(f"  Found {len(coin_sentiment)} tweets for {coin}, aggregated to {len(daily_sentiment)} days")
                else:
                    print(f"  No tweets found for {coin}")
                    
                # Respect rate limits
                time.sleep(5)
                
            except Exception as e:
                print(f"Error getting Twitter sentiment for {coin}: {e}")
        
        return self.twitter_sentiment
    
    def align_price_twitter_sentiment(self, coin):
        """
        Align price data with Twitter sentiment for a specific coin.
        
        Args:
            coin (str): Coin ticker
                
        Returns:
            pd.DataFrame: Combined price and Twitter sentiment data
        """
        if coin not in self.price_data:
            print(f"No price data available for {coin}")
            return None
        
        # Get price data with technical indicators
        price_df = self.generate_technical_indicators(coin)
        
        # Check if we have Twitter sentiment
        if hasattr(self, 'twitter_sentiment') and coin in self.twitter_sentiment:
            twitter_df = self.twitter_sentiment[coin]
            
            if not twitter_df.empty:
                # Prepare price data for merging
                price_df['date_day'] = price_df['date'].dt.date
                price_df['date_day'] = pd.to_datetime(price_df['date_day'])
                
                # Merge on date
                combined_df = pd.merge(
                    price_df, 
                    twitter_df, 
                    left_on='date_day', 
                    right_on='date', 
                    how='left',
                    suffixes=('', '_twitter')
                )
                
                # Clean up date columns
                combined_df.drop('date_day', axis=1, inplace=True)
                if 'date_twitter' in combined_df.columns:
                    combined_df.drop('date_twitter', axis=1, inplace=True)
                
                # Forward fill missing sentiment data
                sentiment_cols = [col for col in combined_df.columns if 'sentiment' in col or 'ratio' in col]
                combined_df[sentiment_cols] = combined_df[sentiment_cols].fillna(method='ffill')
                
                # Fill any remaining NaN values with defaults
                for col in sentiment_cols:
                    if 'score' in col:
                        combined_df[col] = combined_df[col].fillna(0)
                    elif 'positive' in col or 'negative' in col:
                        combined_df[col] = combined_df[col].fillna(0)
                    elif 'neutral' in col:
                        combined_df[col] = combined_df[col].fillna(1)
                    else:
                        combined_df[col] = combined_df[col].fillna(0)
                
                return combined_df
            
        # If no Twitter sentiment, return price data with default sentiment values
        print(f"No Twitter sentiment data for {coin}, returning price data with defaults")
        price_df['sentiment_score'] = 0.0
        price_df['positive_ratio'] = 0.0
        price_df['negative_ratio'] = 0.0
        price_df['neutral_ratio'] = 1.0
        price_df['tweet_volume'] = 0
        
        return price_df
    
    def prepare_all_coins_data_twitter(self, coin_list=None, bearer_token=None):
        """
        Prepare aligned price and Twitter sentiment data for multiple coins.
        
        Args:
            coin_list (list): List of coin tickers to process
            bearer_token (str): Twitter API bearer token
            
        Returns:
            dict: Dictionary of coin tickers to aligned DataFrames
        """
        if coin_list is None:
            coin_list = list(self.price_data.keys())
        
        # Load Twitter sentiment
        self.load_twitter_sentiment(coin_list, bearer_token)
        
        aligned_data = {}
        for coin in coin_list:
            # Align price and Twitter sentiment
            combined_df = self.align_price_twitter_sentiment(coin)
            
            if combined_df is not None and len(combined_df) > 0:
                aligned_data[coin] = combined_df
                
                # Save to processed data folder
                output_path = os.path.join(self.processed_data_path, f"{coin}_processed_twitter.csv")
                combined_df.to_csv(output_path, index=False)
                print(f"Saved processed data with Twitter sentiment for {coin} to {output_path}")
            
        return aligned_data

# Example usage
if __name__ == "__main__":
    processor = CryptoDataProcessor(
        raw_data_path="data/raw",
        processed_data_path="data/processed"
    )
    
    # Load data
    processor.load_price_data(coin_list=["BTC", "ETH", "ICP"])
    processor.load_news_data()
    
    # Process and align data for all loaded coins
    aligned_data = processor.prepare_all_coins_data()
    
    # Print sample of processed data
    for coin, df in aligned_data.items():
        print(f"\n{coin} sample data:")
        print(df.head())
        print(f"Shape: {df.shape}")