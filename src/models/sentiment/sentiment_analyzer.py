import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os

# NLP libraries
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from collections import Counter

class CryptoSentimentAnalyzer:
    """
    Advanced sentiment analysis for cryptocurrency news.
    Provides methods for analyzing sentiment, topics, and coin-specific mentions.
    """
    
    def __init__(self, model_type='vader'):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_type (str): Type of sentiment model to use. 
                             Options: 'vader', 'finbert', 'combined'
        """
        self.model_type = model_type
        self.coin_keywords = self._load_coin_keywords()
        
        # Initialize models based on type
        if model_type in ['vader', 'combined']:
            try:
                nltk.download('vader_lexicon', quiet=True)
                self.vader = SentimentIntensityAnalyzer()
            except Exception as e:
                print(f"Error loading VADER: {e}")
                self.vader = None
        
        if model_type in ['finbert', 'combined']:
            try:
                # Initialize FinBERT (financial BERT model)
                self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                self.finbert = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            except Exception as e:
                print(f"Error loading FinBERT: {e}")
                self.finbert = None
                self.tokenizer = None
        
        # Topic modeling
        self.topic_model = None
        self.vectorizer = None
    
    def _load_coin_keywords(self):
        """
        Load cryptocurrency keywords and their variations.
        
        Returns:
            dict: Dictionary mapping coin tickers to lists of keywords
        """
        # This would ideally load from a file, but for now we'll hardcode some examples
        coin_keywords = {
            "BTC": ["bitcoin", "btc", "satoshi", "nakamoto", "bitcoin core"],
            "ETH": ["ethereum", "eth", "vitalik", "buterin", "ether", "ethers"],
            "XRP": ["xrp", "ripple", "garlinghouse"],
            "SOL": ["solana", "sol"],
            "ADA": ["cardano", "ada", "hoskinson"],
            "DOT": ["polkadot", "dot", "gavin wood"],
            "AVAX": ["avalanche", "avax"],
            "BNB": ["binance", "bnb", "binance coin", "binance smart chain", "bsc"],
            "LINK": ["chainlink", "link"],
            "ICP": ["internet computer", "icp", "dfinity"],
            # Add more coins and their keywords as needed
        }
        return coin_keywords
    
    def preprocess_text(self, text):
        """
        Preprocess text for sentiment analysis.
        
        Args:
            text (str): Text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_vader_sentiment(self, text):
        """
        Analyze sentiment using VADER.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment scores
        """
        if not self.vader:
            return {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0}
        
        if not text or not isinstance(text, str):
            return {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0}
        
        return self.vader.polarity_scores(text)
    
    def analyze_finbert_sentiment(self, text):
        """
        Analyze sentiment using FinBERT.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Sentiment scores and label
        """
        if not self.finbert or not self.tokenizer:
            return {'positive': 0, 'negative': 0, 'neutral': 0, 'label': 'neutral'}
        
        if not text or not isinstance(text, str):
            return {'positive': 0, 'negative': 0, 'neutral': 0, 'label': 'neutral'}
        
        # Truncate long texts to fit BERT's maximum sequence length
        max_length = 512
        if len(text.split()) > max_length:
            text = ' '.join(text.split()[:max_length])
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self.finbert(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT has labels: positive (0), negative (1), neutral (2)
            probs = probabilities[0].tolist()
            label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
            label_idx = probs.index(max(probs))
            
            return {
                'positive': probs[0],
                'negative': probs[1],
                'neutral': probs[2],
                'label': label_map[label_idx]
            }
        except Exception as e:
            print(f"Error in FinBERT analysis: {e}")
            return {'positive': 0, 'negative': 0, 'neutral': 0, 'label': 'neutral'}
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment using the selected model(s).
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Combined sentiment analysis results
        """
        preprocessed_text = self.preprocess_text(text)
        
        if self.model_type == 'vader':
            vader_scores = self.analyze_vader_sentiment(preprocessed_text)
            
            # Map VADER compound score to sentiment class
            if vader_scores['compound'] >= 0.05:
                sentiment_class = 'positive'
            elif vader_scores['compound'] <= -0.05:
                sentiment_class = 'negative'
            else:
                sentiment_class = 'neutral'
            
            return {
                'compound': vader_scores['compound'],
                'positive': vader_scores['pos'],
                'negative': vader_scores['neg'],
                'neutral': vader_scores['neu'],
                'class': sentiment_class
            }
        
        elif self.model_type == 'finbert':
            finbert_scores = self.analyze_finbert_sentiment(preprocessed_text)
            
            return {
                'positive': finbert_scores['positive'],
                'negative': finbert_scores['negative'],
                'neutral': finbert_scores['neutral'],
                'class': finbert_scores['label']
            }
        
        elif self.model_type == 'combined':
            vader_scores = self.analyze_vader_sentiment(preprocessed_text)