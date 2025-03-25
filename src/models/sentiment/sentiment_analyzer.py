import pandas as pd
import numpy as np
import asyncio
import pickle
import os
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re
from tqdm import tqdm
import joblib

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('vader_lexicon')

# Add crypto-specific terms to VADER lexicon
try:
    from VADER_lexicon import crypto_lexicon, emoji_lexicon
except ImportError:
    print("Warning: VADER_lexicon not found or error importing. Using default lexicon.")
    crypto_lexicon = {}
    emoji_lexicon = {}

class EnhancedSentimentAnalyzer:
    """
    Enhanced sentiment analyzer that combines VADER with a trained classifier model
    for better accuracy on cryptocurrency text data.
    """
    
    def __init__(self, model_dir='models/saved'):
        """
        Initialize the enhanced sentiment analyzer with VADER and ML models.
        
        Args:
            model_dir (str): Directory to save/load trained models
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize VADER with crypto lexicon
        self.vader = SentimentIntensityAnalyzer()
        # Add crypto-specific terms to lexicon
        for word, score in crypto_lexicon.items():
            self.vader.lexicon[word] = score
        
        # ML model components
        self.vectorizer = None
        self.classifier = None
        self.is_trained = False
        
        # Try to load pre-trained models
        self._load_models()
        
        # Text preprocessing
        self.stop_words = set(stopwords.words('english'))
        
    def _load_models(self):
        """Load pre-trained vectorizer and classifier models if they exist."""
        try:
            vectorizer_path = os.path.join(self.model_dir, 'tfidf_vectorizer.pkl')
            classifier_path = os.path.join(self.model_dir, 'sentiment_classifier.pkl')
            
            if os.path.exists(vectorizer_path) and os.path.exists(classifier_path):
                self.vectorizer = joblib.load(vectorizer_path)
                self.classifier = joblib.load(classifier_path)
                self.is_trained = True
                print("Loaded pre-trained sentiment models")
                return True
            
            print("No pre-trained models found. Models need to be trained.")
            return False
        
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def _save_models(self):
        """Save the trained vectorizer and classifier models."""
        try:
            if self.vectorizer is not None and self.classifier is not None:
                joblib.dump(self.vectorizer, os.path.join(self.model_dir, 'tfidf_vectorizer.pkl'))
                joblib.dump(self.classifier, os.path.join(self.model_dir, 'sentiment_classifier.pkl'))
                print("Saved sentiment models to disk")
                return True
            
            print("Models not initialized, cannot save")
            return False
        
        except Exception as e:
            print(f"Error saving models: {e}")
            return False
    
    def preprocess_text(self, text):
        """
        Preprocess text for sentiment analysis.
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove mentions and hashtags while keeping hashtag content
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words
        tokens = [word for word in tokens if word not in self.stop_words]
        
        # Join tokens back to string
        return ' '.join(tokens)
    
    def train_model(self, tweets_df, manual_labels=None, test_size=0.2):
        """
        Train the ML classifier for sentiment analysis.
        
        Args:
            tweets_df (pd.DataFrame): DataFrame with tweet data including text
            manual_labels (pd.DataFrame, optional): DataFrame with manual sentiment labels
            test_size (float): Proportion of data to use for testing
            
        Returns:
            dict: Training metrics
        """
        # Check if we have enough data
        if tweets_df is None or len(tweets_df) < 100:
            print("Not enough data to train model (need at least 100 examples)")
            return {"error": "Insufficient data"}
        
        try:
            # Prepare the data
            texts = tweets_df['text'].astype(str).values
            
            # Preprocess texts
            processed_texts = [self.preprocess_text(text) for text in tqdm(texts, desc="Preprocessing texts")]
            
            # Use VADER to create training labels if no manual labels provided
            if manual_labels is None:
                print("Using VADER to create training labels")
                labels = []
                
                for text in tqdm(texts, desc="Generating VADER labels"):
                    score = self.vader.polarity_scores(text)
                    # Convert to classification: -1 (negative), 0 (neutral), 1 (positive)
                    if score['compound'] >= 0.05:
                        labels.append(1)  # Positive
                    elif score['compound'] <= -0.05:
                        labels.append(-1)  # Negative
                    else:
                        labels.append(0)  # Neutral
            else:
                print("Using provided manual labels")
                if len(manual_labels) != len(tweets_df):
                    print("Warning: Number of labels doesn't match number of tweets")
                    return {"error": "Label count mismatch"}
                labels = manual_labels
            
            # Create TF-IDF features
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                min_df=5,
                max_df=0.8,
                ngram_range=(1, 2)
            )
            X = self.vectorizer.fit_transform(processed_texts)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, labels, test_size=test_size, random_state=42, stratify=labels
            )
            
            # Train model
            print("Training sentiment classifier...")
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            self.classifier.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            print(f"Model accuracy: {accuracy:.4f}")
            print(classification_report(y_test, y_pred))
            
            # Save models
            self.is_trained = True
            self._save_models()
            
            return {
                "accuracy": accuracy,
                "report": report,
                "n_samples": len(labels),
                "feature_count": X.shape[1]
            }
            
        except Exception as e:
            print(f"Error training model: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def analyze_text(self, text):
        """
        Analyze the sentiment of a single text using combined approach.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment scores and classification
        """
        if not isinstance(text, str) or not text.strip():
            return {
                "compound": 0,
                "pos": 0,
                "neg": 0,
                "neu": 1,
                "sentiment_class": 0,
                "confidence": 0
            }
            
        # Get VADER scores
        vader_scores = self.vader.polarity_scores(text)
        
        # Use ML model if it's trained
        if self.is_trained and self.vectorizer is not None and self.classifier is not None:
            try:
                # Preprocess text
                processed_text = self.preprocess_text(text)
                
                # Transform to feature vector
                features = self.vectorizer.transform([processed_text])
                
                # Predict sentiment class
                sentiment_class = self.classifier.predict(features)[0]
                
                # Get confidence scores
                confidence_scores = self.classifier.predict_proba(features)[0]
                confidence = max(confidence_scores)
                
                # Combine VADER and ML model
                # Adjust compound score based on ML prediction
                if sentiment_class == 1:  # Positive
                    compound_adjusted = (vader_scores['compound'] + confidence) / 2
                elif sentiment_class == -1:  # Negative
                    compound_adjusted = (vader_scores['compound'] - confidence) / 2
                else:  # Neutral
                    compound_adjusted = vader_scores['compound'] * 0.8  # Reduce intensity
                
                # Ensure in range [-1, 1]
                compound_adjusted = max(min(compound_adjusted, 1.0), -1.0)
                
                return {
                    "compound": compound_adjusted,
                    "pos": vader_scores['pos'],
                    "neg": vader_scores['neg'],
                    "neu": vader_scores['neu'],
                    "sentiment_class": sentiment_class,
                    "confidence": confidence
                }
                
            except Exception as e:
                print(f"Error in ML sentiment analysis: {e}. Falling back to VADER.")
                # Fall back to VADER
                pass
        
        # If ML model not available or error occurred, use VADER classification
        if vader_scores['compound'] >= 0.05:
            sentiment_class = 1  # Positive
        elif vader_scores['compound'] <= -0.05:
            sentiment_class = -1  # Negative
        else:
            sentiment_class = 0  # Neutral
            
        return {
            "compound": vader_scores['compound'],
            "pos": vader_scores['pos'],
            "neg": vader_scores['neg'],
            "neu": vader_scores['neu'],
            "sentiment_class": sentiment_class,
            "confidence": max(vader_scores['pos'], vader_scores['neg'], vader_scores['neu'])
        }
    
    def analyze_dataframe(self, df, text_column='text'):
        """
        Analyze sentiment for all texts in a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame with text data
            text_column (str): Column containing text to analyze
            
        Returns:
            pd.DataFrame: Original DataFrame with added sentiment columns
        """
        if df is None or df.empty or text_column not in df.columns:
            print("Invalid DataFrame or text column not found")
            return df
            
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Initialize columns
        result_df['sentiment_compound'] = 0.0
        result_df['sentiment_positive'] = 0.0
        result_df['sentiment_negative'] = 0.0
        result_df['sentiment_neutral'] = 0.0
        result_df['sentiment_class'] = 0
        result_df['sentiment_confidence'] = 0.0
        
        # Analyze each text
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing sentiment"):
            text = row[text_column]
            sentiment = self.analyze_text(text)
            
            result_df.at[i, 'sentiment_compound'] = sentiment['compound']
            result_df.at[i, 'sentiment_positive'] = sentiment['pos']
            result_df.at[i, 'sentiment_negative'] = sentiment['neg']
            result_df.at[i, 'sentiment_neutral'] = sentiment['neu']
            result_df.at[i, 'sentiment_class'] = sentiment['sentiment_class']
            result_df.at[i, 'sentiment_confidence'] = sentiment['confidence']
        
        return result_df
    
    def calculate_sentiment_score(self, df, 
                                weight_engagement=True,
                                engagement_cols=['retweet_count', 'like_count', 'reply_count'],
                                normalize=True):
        """
        Calculate an overall sentiment score (0-100) similar to Fear & Greed Index.
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment data
            weight_engagement (bool): Whether to weight by engagement metrics
            engagement_cols (list): Columns to use for engagement weighting
            normalize (bool): Whether to normalize to 0-100 scale
            
        Returns:
            float: Overall sentiment score (0-100)
        """
        if df is None or df.empty or 'sentiment_compound' not in df.columns:
            return 50.0  # Default neutral score
            
        try:
            if weight_engagement and all(col in df.columns for col in engagement_cols):
                # Create engagement score (sum of all metrics + 1 to avoid zeros)
                df['engagement'] = df[engagement_cols].sum(axis=1) + 1
                
                # Use log scale for more balanced weighting
                df['log_engagement'] = np.log1p(df['engagement'])
                
                # Calculate weighted sentiment
                weighted_sentiment = (df['sentiment_compound'] * df['log_engagement']).sum() / df['log_engagement'].sum()
            else:
                # Simple average
                weighted_sentiment = df['sentiment_compound'].mean()
                
            # Convert from [-1, 1] to [0, 100] scale if requested
            if normalize:
                score = (weighted_sentiment + 1) * 50
                # Ensure in range [0, 100]
                score = max(min(score, 100), 0)
            else:
                score = weighted_sentiment
                
            return score
            
        except Exception as e:
            print(f"Error calculating sentiment score: {e}")
            return 50.0  # Default neutral score
    
    def get_market_sentiment_level(self, score):
        """
        Convert numerical sentiment score to sentiment level category.
        
        Args:
            score (float): Sentiment score on 0-100 scale
            
        Returns:
            str: Sentiment level category
        """
        if score >= 90:
            return "Extreme Greed"
        elif score >= 75:
            return "Greed"
        elif score >= 60:
            return "Moderate Greed"
        elif score >= 40:
            return "Neutral"
        elif score >= 25:
            return "Moderate Fear"
        elif score >= 10:
            return "Fear"
        else:
            return "Extreme Fear"
    
    def analyze_twitter_trends(self, df, days_back=30):
        """
        Analyze sentiment trends over time for a cryptocurrency.
        
        Args:
            df (pd.DataFrame): DataFrame with tweet data including dates and sentiment
            days_back (int): Number of days to analyze
            
        Returns:
            dict: Trend metrics and insights
        """
        if df is None or df.empty or 'date' not in df.columns:
            return {"error": "Invalid data"}
            
        try:
            # Ensure date is datetime
            if not pd.api.types.is_datetime64_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
                
            # Filter for requested time period
            end_date = df['date'].max()
            start_date = end_date - timedelta(days=days_back)
            period_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            if period_df.empty:
                return {"error": "No data in specified time period"}
                
            # Group by day
            period_df['date_day'] = period_df['date'].dt.date
            daily = period_df.groupby('date_day').agg({
                'sentiment_compound': 'mean',
                'sentiment_positive': 'mean',
                'sentiment_negative': 'mean',
                'sentiment_neutral': 'mean',
                'date': 'count'  # Count as volume
            }).reset_index()
            
            daily.rename(columns={'date': 'volume'}, inplace=True)
            
            # Calculate overall sentiment
            overall_score = self.calculate_sentiment_score(period_df)
            sentiment_level = self.get_market_sentiment_level(overall_score)
            
            # Calculate sentiment change
            if len(daily) >= 2:
                # First half vs second half
                half_point = len(daily) // 2
                first_half = daily.iloc[:half_point]
                second_half = daily.iloc[half_point:]
                
                first_half_score = first_half['sentiment_compound'].mean()
                second_half_score = second_half['sentiment_compound'].mean()
                
                trend_direction = "improving" if second_half_score > first_half_score else "deteriorating"
                trend_strength = abs(second_half_score - first_half_score) * 100
                
                # Day-to-day momentum (average daily change)
                daily['change'] = daily['sentiment_compound'].diff()
                avg_momentum = daily['change'].mean()
                
                # Volatility (standard deviation of sentiment)
                volatility = daily['sentiment_compound'].std()
                
            else:
                trend_direction = "stable"
                trend_strength = 0
                avg_momentum = 0
                volatility = 0
            
            # Volume trend
            if len(daily) >= 2:
                volume_change_pct = ((daily['volume'].iloc[-1] / daily['volume'].iloc[0]) - 1) * 100
            else:
                volume_change_pct = 0
                
            return {
                "overall_score": overall_score,
                "sentiment_level": sentiment_level,
                "trend_direction": trend_direction,
                "trend_strength": trend_strength,
                "avg_momentum": avg_momentum,
                "volatility": volatility,
                "volume_change_pct": volume_change_pct,
                "days_analyzed": len(daily),
                "total_mentions": period_df.shape[0],
                "period_start": start_date.strftime("%Y-%m-%d"),
                "period_end": end_date.strftime("%Y-%m-%d"),
                "daily_data": daily.to_dict(orient='records')
            }
            
        except Exception as e:
            print(f"Error analyzing trends: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

# Example usage
if __name__ == "__main__":
    # Directory for storing trained models
    model_dir = 'models/saved'
    
    # Initialize enhanced sentiment analyzer
    analyzer = EnhancedSentimentAnalyzer(model_dir=model_dir)
    
    # Example text to analyze
    example_texts = [
        "Bitcoin is going to the moon! 🚀 Just bought more $BTC #bullish",
        "Crypto markets crashing again. Lost 50% of my portfolio. This is terrible.",
        "Ethereum transition to proof-of-stake could happen next month.",
        "Not sure about this new altcoin, seems risky but might have potential."
    ]
    
    print("Example Sentiment Analysis:")
    for text in example_texts:
        sentiment = analyzer.analyze_text(text)
        print(f"\nText: {text}")
        print(f"Compound Score: {sentiment['compound']:.3f}")
        print(f"Classification: {sentiment['sentiment_class']} (Confidence: {sentiment['confidence']:.3f})")
        print(f"Positive: {sentiment['pos']:.3f}, Negative: {sentiment['neg']:.3f}, Neutral: {sentiment['neu']:.3f}")
        
        # Convert to 0-100 score
        score = (sentiment['compound'] + 1) * 50
        level = analyzer.get_market_sentiment_level(score)
        print(f"Sentiment Score (0-100): {score:.1f} - {level}")