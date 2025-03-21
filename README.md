# Crypto Sentiment Analysis Platform

A data science platform for analyzing the relationship between cryptocurrency price movements and social media sentiment. This project combines real-time Twitter sentiment analysis, technical indicators, correlation analysis, and predictive modeling to extract actionable insights from cryptocurrency markets.

## Features

- **Real-time Sentiment Analysis**: Uses Twitter API and VADER sentiment analysis to capture current market sentiment rather than relying on outdated or biased datasets
- **Technical Analysis**: Generates comprehensive technical indicators (SMA, EMA, MACD, RSI, Bollinger Bands)
- **Correlation Analysis**: Examines relationships between sentiment signals and price movements using time-lagged correlation and Granger causality tests
- **Price Prediction**: Implements LSTM and XGBoost models for price forecasting incorporating sentiment features
- **Interactive Dashboard**: Visualizes insights through an intuitive web interface built with Dash and Plotly
- **Cross-Coin Comparison**: Compares sentiment patterns and price correlations across multiple cryptocurrencies

## Technology Stack

- **Data Processing**: Pandas, NumPy
- **Sentiment Analysis**: Twitter API, NLTK, VADER
- **Machine Learning**: TensorFlow/Keras (LSTM), XGBoost
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Dashboard**: Dash, Bootstrap
- **Statistical Analysis**: SciPy, StatsModels

## How It Works

1. **Data Collection**: Fetches historical cryptocurrency price data and recent Twitter sentiment
2. **Sentiment Analysis**: Analyzes tweets mentioning specific cryptocurrencies to extract sentiment scores
3. **Feature Engineering**: Generates technical indicators and aligns with sentiment data
4. **Correlation Analysis**: Identifies relationships between sentiment and price movements
5. **Predictive Modeling**: Trains machine learning models to forecast future price movements
6. **Visualization**: Presents insights through interactive charts and a web dashboard

## Project Structure

```
./
├── data/                       # Data storage
│   ├── processed/              # Processed data files
│   ├── raw/                    # Raw cryptocurrency and Twitter data 
├── dashboard/                  # Interactive dashboard
│   ├── app.py                  # Dash web application
│   └── assets/                 # Dashboard assets
├── models/                     # Machine learning models
│   └── saved/                  # Saved model files
├── reports/                    # Analysis reports
│   ├── correlation/            # Correlation analysis results
│   └── figures/                # Generated visualizations
├── src/                        # Source code
│   ├── data/                   # Data processing modules
│   │   └── data_processor.py   # Main data processing class
│   ├── main.py                 # Main application entry point
│   └── models/                 # Analysis and prediction modules
│       ├── correlation/        # Correlation analysis
│       ├── price/              # Price prediction models
│       ├── sentiment/          # Twitter sentiment analysis
│       └── visualization/      # Data visualization tools
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/vatsalmehta2001/crypto_analytics_platform.git
   cd crypto-sentiment-analysis
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up your Twitter API credentials:
   - Create a developer account at [developer.twitter.com](https://developer.twitter.com)
   - Create a project and an app to get your API credentials
   - Store your bearer token securely for use with the platform

## Running the Platform

1. Process data and generate analysis:
   ```
   python src/main.py --use-twitter --twitter-token YOUR_BEARER_TOKEN --coins BTC,ETH,ICP
   ```
   
   Options:
   - `--coins` - Specify comma-separated list of cryptocurrencies to analyze
   - `--use-twitter` - Use Twitter API for sentiment analysis
   - `--twitter-token` - Your Twitter API bearer token
   - `--skip-models` - Skip model training for faster processing

2. Launch the dashboard:
   ```
   python dashboard/app.py
   ```
   
   Then open a web browser and navigate to `http://127.0.0.1:8050/`

## Advantages Over Traditional Sentiment Analysis

This platform uses real-time Twitter data rather than pre-packaged sentiment datasets, offering several key advantages:

1. **Real-time insights**: Captures current market sentiment rather than historical snapshots
2. **Reduced bias**: Avoids biases inherent in curated datasets
3. **Greater specificity**: Analyzes sentiment for specific coins and events
4. **Customizable analysis**: Allows adjusting time windows and sentiment parameters
5. **Multi-source potential**: Framework can be extended to integrate additional social media sources

## Future Improvements

- Implement real-time streaming of Twitter data
- Add NLP-based entity recognition for better coin mention detection
- Incorporate on-chain metrics and exchange volume data
- Develop automated trading strategies based on sentiment signals
- Expand to more cryptocurrencies and alternative data sources

## Developer

Vatsal Gagankumar Mehta

*Developed as a showcase of end-to-end machine learning capabilities for data science portfolio*

## License

This project is licensed under the MIT License - see the LICENSE file for details.