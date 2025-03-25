import nltk
nltk.download('vader_lexicon')

# Cryptocurrency-specific lexicon for VADER sentiment analysis
# This extends the default VADER lexicon with crypto-specific terms and their sentiment values

# Format: word/phrase : sentiment_value
# Sentiment values range from -4 (extremely negative) to +4 (extremely positive)

crypto_lexicon = {
    # Positive terms
    "hodl": 2.0,
    "hodling": 2.0,
    "mooning": 3.5,
    "moon": 3.0,
    "to the moon": 3.5,
    "bullish": 3.0,
    "bull run": 3.0,
    "bull market": 2.5,
    "diamond hands": 2.5,
    "lambo": 2.5,
    "adoption": 2.0,
    "mainstream": 1.5,
    "institutional": 1.5,
    "accumulate": 2.0,
    "accumulating": 2.0,
    "dca": 1.5,  # Dollar-cost averaging
    "staking": 1.5,
    "passive income": 2.0,
    "yield": 1.5,
    "nft boom": 2.0,
    "fomo": 1.0,  # Could be negative in some contexts, but often signals upward momentum
    "all time high": 3.0,
    "ath": 3.0,
    "support": 1.0,
    "hold support": 1.5,
    "breakout": 2.5,
    "rally": 2.5,
    
    # Negative terms
    "bear market": -2.5,
    "bearish": -2.5,
    "crash": -3.5,
    "crashing": -3.5,
    "dump": -3.0,
    "dumping": -3.0,
    "paper hands": -1.5,
    "fud": -2.0,  # Fear, uncertainty, doubt
    "ponzi": -3.5,
    "scam": -3.5,
    "rugpull": -4.0,
    "rug pull": -4.0,
    "shitcoin": -2.5,
    "altcoin": -0.5,  # Slightly negative connotation for some
    "correction": -1.5,
    "resistance": -1.0,
    "sec": -0.5,  # Securities and Exchange Commission - often negative in crypto context
    "regulation": -1.0,
    "ban": -3.0,
    "hack": -3.5,
    "hacked": -3.5,
    "exploit": -3.0,
    "vulnerability": -2.0,
    "inflation": -1.0,
    "volatility": -1.0,
    "sell off": -2.5,
    "selloff": -2.5,
    "panic sell": -3.0,
    "bubble": -2.0,
    "overvalued": -2.0,
    
    # Neutral/context-dependent terms
    "whale": 0.0,
    "mining": 0.0,
    "miner": 0.0,
    "halving": 0.5,  # Slightly positive
    "fork": 0.0,
    "ico": 0.0,
    "airdrop": 0.5,  # Slightly positive
    "defi": 0.5,  # Slightly positive
    "decentralized": 1.0,
    "centralized": -0.5,  # Slightly negative in crypto context
    "exchange": 0.0,
    "wallet": 0.0,
    "nft": 0.5,
    "token": 0.0,
    "altseason": 1.0,
    "volume": 0.0,
    "liquidity": 0.0,
    "leverage": 0.0,
    "long": 0.5,
    "short": -0.5,
    "margin": 0.0,
    
    # Coin-specific sentiment
    "bitcoin": 0.5,  # Slightly positive baseline
    "btc": 0.5,
    "ethereum": 0.5,
    "eth": 0.5,
    "solana": 0.0,
    "sol": 0.0,
    "cardano": 0.0,
    "ada": 0.0,
    "ripple": 0.0,
    "xrp": 0.0,
    "doge": 0.0,
    "dogecoin": 0.0,
    "shib": 0.0,
    "bnb": 0.0,
    
    # Technical terms
    "resistance broken": 2.5,
    "support broken": -2.5,
    "golden cross": 3.0,
    "death cross": -3.0,
    "higher high": 2.0,
    "higher low": 1.5,
    "lower high": -1.5,
    "lower low": -2.0,
    "oversold": 1.5,  # Usually means buying opportunity
    "overbought": -1.5,
    "consolidation": 0.0,
}

# Additional emoji lexicon that is crypto-specific
emoji_lexicon = {
    "🚀": 3.0,      # Rocket - extremely positive
    "🌕": 3.0,      # Full moon - positive (to the moon)
    "💎": 2.5,      # Diamond - positive (diamond hands)
    "🙌": 2.0,      # Raised hands - positive
    "📈": 2.5,      # Chart increasing - positive
    "📉": -2.5,     # Chart decreasing - negative
    "🐂": 2.5,      # Bull - positive
    "🐻": -2.5,     # Bear - negative
    "💰": 2.0,      # Money bag - positive
    "💵": 1.5,      # Dollar - positive
    "🔥": 2.0,      # Fire - positive (hot)
    "❤️": 1.5,      # Heart - positive
    "😭": -2.0,     # Crying - negative
    "😢": -1.5,     # Sad - negative
    "🤡": -2.0,     # Clown - negative
    "🤮": -3.0,     # Vomiting - very negative
    "💩": -2.5,     # Pile of poo - negative
    "☠️": -3.0,     # Skull and crossbones - very negative
    "⚰️": -3.0,     # Coffin - very negative
}

# Merge emoji lexicon into main crypto lexicon
crypto_lexicon.update(emoji_lexicon)

# Check for some specific crypto slang phrases that should be treated as single terms
def update_vader_for_crypto(vader_analyzer):
    """
    Update a VADER SentimentIntensityAnalyzer with crypto-specific lexicon.
    
    Args:
        vader_analyzer: A VADER SentimentIntensityAnalyzer instance
        
    Returns:
        The updated analyzer
    """
    for term, score in crypto_lexicon.items():
        vader_analyzer.lexicon[term] = score
    
    return vader_analyzer