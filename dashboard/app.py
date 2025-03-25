import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import plotly.io as pio

# Set default Plotly template
pio.templates.default = "plotly_dark"

# Suppress warnings
warnings.filterwarnings('ignore')

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import the enhanced sentiment analyzer
from src.models.sentiment.sentiment_analyzer import EnhancedSentimentAnalyzer

# Initialize sentiment analyzer
sentiment_analyzer = EnhancedSentimentAnalyzer(model_dir='models/saved')

# Create a Dash app with dark theme
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.CYBORG],  # Dark theme
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}]
)
app.title = 'Crypto Sentiment Analysis Dashboard'
server = app.server

# Cache data
DATA_CACHE = {}
REDDIT_DATA_CACHE = {}

# Helper functions
def load_processed_data(coin, data_dir='data/processed'):
    """Load processed data for a specific coin."""
    global DATA_CACHE
    
    # Check if data is already in cache
    if coin in DATA_CACHE:
        return DATA_CACHE[coin]
    
    try:
        file_path = os.path.join(data_dir, f"{coin}_processed.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Convert date to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            # Cache the data
            DATA_CACHE[coin] = df
            return df
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data for {coin}: {e}")
        return None

def load_reddit_data(coin, data_dir='data/reddit_sentiment'):
    """Load Reddit sentiment data for a specific coin."""
    global REDDIT_DATA_CACHE
    
    # Check if data is already in cache
    if coin in REDDIT_DATA_CACHE:
        return REDDIT_DATA_CACHE[coin]
    
    try:
        # Try to load daily aggregated data first
        daily_file_path = os.path.join(data_dir, f"{coin}_daily.csv")
        if os.path.exists(daily_file_path):
            df = pd.read_csv(daily_file_path)
            # Convert date to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            # Cache the data
            REDDIT_DATA_CACHE[coin] = df
            return df
        
        # If daily not found, try posts data
        posts_file_path = os.path.join(data_dir, f"{coin}_posts.csv")
        if os.path.exists(posts_file_path):
            df = pd.read_csv(posts_file_path)
            # Convert date to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            # Aggregate by day
            if 'date' in df.columns and 'sentiment_compound' in df.columns:
                df['date_day'] = df['date'].dt.date
                daily_agg = df.groupby('date_day').agg({
                    'sentiment_compound': 'mean',
                    'sentiment_positive': 'mean',
                    'sentiment_negative': 'mean',
                    'sentiment_neutral': 'mean',
                    'text': 'count',
                    'score': 'sum'
                }).reset_index()
                
                # Rename columns for consistency
                daily_agg.rename(columns={'text': 'post_volume', 'score': 'total_score'}, inplace=True)
                
                # Convert date_day back to datetime
                daily_agg['date'] = pd.to_datetime(daily_agg['date_day'])
                
                # Cache the data
                REDDIT_DATA_CACHE[coin] = daily_agg
                return daily_agg
            
            # If no date or sentiment columns, return raw data
            REDDIT_DATA_CACHE[coin] = df
            return df
            
        print(f"No Reddit data found for {coin}")
        return None
    except Exception as e:
        print(f"Error loading Reddit data for {coin}: {e}")
        return None

def get_available_coins(data_dir='data/processed'):
    """Get list of available coins from processed data directory."""
    try:
        if not os.path.exists(data_dir):
            print(f"Processed data directory not found: {data_dir}")
            return []
        files = [f for f in os.listdir(data_dir) if f.endswith('_processed.csv')]
        coins = [f.split('_processed.csv')[0] for f in files]
        return sorted(coins)
    except Exception as e:
        print(f"Error getting available coins: {e}")
        return []

def get_reddit_coins(data_dir='data/reddit_sentiment'):
    """Get list of coins with Reddit data."""
    try:
        if not os.path.exists(data_dir):
            print(f"Reddit data directory not found: {data_dir}")
            return []
        daily_files = [f for f in os.listdir(data_dir) if f.endswith('_daily.csv')]
        post_files = [f for f in os.listdir(data_dir) if f.endswith('_posts.csv')]
        
        daily_coins = [f.split('_daily.csv')[0] for f in daily_files]
        post_coins = [f.split('_posts.csv')[0] for f in post_files]
        
        # Combine and remove duplicates
        all_coins = list(set(daily_coins + post_coins))
        return sorted(all_coins)
    except Exception as e:
        print(f"Error getting Reddit coins: {e}")
        return []

def calculate_sentiment_gauge(df, window=14, sentiment_col='sentiment_compound'):
    """Calculate sentiment gauge value from recent data."""
    if df is None or df.empty or sentiment_col not in df.columns:
        return 50  # Neutral default
    
    # Sort by date and get the most recent window days
    df_sorted = df.sort_values('date', ascending=False)
    recent_df = df_sorted.head(window)
    
    # Calculate score using our sentiment analyzer
    score = sentiment_analyzer.calculate_sentiment_score(recent_df)
    return score

def get_sentiment_color(score):
    """Get color based on sentiment score (0-100)."""
    if score >= 80:
        return "#00CC00"  # Bright green - extreme greed
    elif score >= 60:
        return "#88CC00"  # Yellow-green - greed
    elif score >= 45:
        return "#CCCC00"  # Yellow - neutral leaning positive
    elif score >= 40:
        return "#DDDDDD"  # Light gray - neutral
    elif score >= 25:
        return "#CCAA00"  # Orange-yellow - fear
    elif score >= 10:
        return "#CC5500"  # Dark orange - high fear
    else:
        return "#CC0000"  # Red - extreme fear

def create_sentiment_gauge(score, coin_name):
    """Create a sentiment gauge visualization."""
    level = sentiment_analyzer.get_market_sentiment_level(score)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{coin_name} Sentiment Index", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': get_sentiment_color(score)},
            'bgcolor': "rgba(50, 50, 50, 0.8)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 10], 'color': "#CC0000"},    # Extreme Fear
                {'range': [10, 25], 'color': "#CC5500"},   # Fear
                {'range': [25, 40], 'color': "#CCAA00"},   # Moderate Fear
                {'range': [40, 60], 'color': "#DDDDDD"},   # Neutral
                {'range': [60, 75], 'color': "#88CC00"},   # Moderate Greed
                {'range': [75, 90], 'color': "#66CC00"},   # Greed
                {'range': [90, 100], 'color': "#00CC00"},  # Extreme Greed
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.add_annotation(
        x=0.5,
        y=0.25,
        text=level,
        font=dict(size=20, color="white"),
        showarrow=False
    )
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )
    
    return fig

def create_sentiment_trend(df, coin_name, window=7, sentiment_col='sentiment_compound', include_reddit=True):
    """Create a sentiment trend visualization over time."""
    if df is None or df.empty or sentiment_col not in df.columns:
        fig = go.Figure()
        fig.update_layout(
            title="No sentiment data available",
            annotations=[dict(text="No data available", showarrow=False, xref="paper", yref="paper", font=dict(size=20))]
        )
        return fig
    
    # Sort by date
    df = df.sort_values('date')
    
    # Calculate rolling sentiment score
    df['rolling_sentiment'] = df[sentiment_col].rolling(window=window).mean()
    
    # Convert to 0-100 scale
    df['sentiment_index'] = (df['rolling_sentiment'] + 1) * 50
    
    # Create figure
    fig = go.Figure()
    
    # Add sentiment line
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['sentiment_index'],
            name='Market Sentiment',
            line=dict(color='yellow', width=3),
            mode='lines'
        )
    )
    
    # Add Reddit sentiment if available and requested
    if include_reddit:
        reddit_df = load_reddit_data(coin_name.lower())
        if reddit_df is not None and not reddit_df.empty and 'sentiment_compound' in reddit_df.columns:
            # Sort by date
            reddit_df = reddit_df.sort_values('date')
            
            # Calculate rolling sentiment
            reddit_df['rolling_sentiment'] = reddit_df['sentiment_compound'].rolling(window=window).mean()
            
            # Convert to 0-100 scale
            reddit_df['sentiment_index'] = (reddit_df['rolling_sentiment'] + 1) * 50
            
            # Add to chart
            fig.add_trace(
                go.Scatter(
                    x=reddit_df['date'],
                    y=reddit_df['sentiment_index'],
                    name='Reddit Sentiment',
                    line=dict(color='#ff6600', width=2),
                    mode='lines'
                )
            )
    
    # Add horizontal reference lines
    levels = [
        (10, "Extreme Fear", "#CC0000"),
        (25, "Fear", "#CC5500"),
        (40, "Neutral", "#CCCCCC"),
        (60, "Greed", "#88CC00"),
        (90, "Extreme Greed", "#00CC00")
    ]
    
    for level, label, color in levels:
        fig.add_trace(
            go.Scatter(
                x=[df['date'].min(), df['date'].max()],
                y=[level, level],
                mode='lines',
                line=dict(color=color, width=1, dash='dash'),
                name=label,
                hoverinfo='text',
                text=label
            )
        )
    
    # Add layout details
    fig.update_layout(
        title=f"{coin_name} Sentiment Trend ({window}-day rolling average)",
        xaxis_title='Date',
        yaxis_title='Sentiment Index (0-100)',
        yaxis=dict(range=[0, 100]),
        hovermode='x unified',
        height=500,
        template='plotly_dark',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_price_sentiment_overlay(df, coin_name, include_reddit=True):
    """Create a price and sentiment overlay chart."""
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No data available",
            annotations=[dict(text="No data available", showarrow=False, xref="paper", yref="paper", font=dict(size=20))]
        )
        return fig
    
    # Create subplot with two y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['close'],
            name='Price',
            line=dict(color='#00ffff', width=2)
        ),
        secondary_y=False
    )
    
    # Add sentiment line (convert to 0-100 scale)
    df['sentiment_index'] = (df['sentiment_score'] + 1) * 50
    
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['sentiment_index'],
            name='Market Sentiment',
            line=dict(color='#ffcc00', width=2)
        ),
        secondary_y=True
    )
    
    # Add Reddit sentiment if available and requested
    if include_reddit:
        reddit_df = load_reddit_data(coin_name.lower())
        if reddit_df is not None and not reddit_df.empty and 'sentiment_compound' in reddit_df.columns:
            # Get date range that matches the price data
            min_date = df['date'].min()
            max_date = df['date'].max()
            
            # Filter Reddit data to match date range
            reddit_df = reddit_df[(reddit_df['date'] >= min_date) & (reddit_df['date'] <= max_date)]
            
            if not reddit_df.empty:
                # Convert to 0-100 scale
                reddit_df['sentiment_index'] = (reddit_df['sentiment_compound'] + 1) * 50
                
                # Add to chart
                fig.add_trace(
                    go.Scatter(
                        x=reddit_df['date'],
                        y=reddit_df['sentiment_index'],
                        name='Reddit Sentiment',
                        line=dict(color='#ff6600', width=2)
                    ),
                    secondary_y=True
                )
    
    # Add layout details
    fig.update_layout(
        title=f"{coin_name} Price and Sentiment Correlation",
        xaxis_title='Date',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified',
        height=500,
        template='plotly_dark'
    )
    
    fig.update_yaxes(title_text="Price", secondary_y=False)
    fig.update_yaxes(title_text="Sentiment Index (0-100)", secondary_y=True, range=[0, 100])
    
    return fig

def create_multi_coin_comparison(selected_coins, data_dir='data/processed', include_reddit=True):
    """Create a comparison of sentiment across multiple coins."""
    if not selected_coins:
        fig = go.Figure()
        fig.update_layout(
            title="Select coins to compare",
            annotations=[dict(text="No coins selected", showarrow=False, xref="paper", yref="paper", font=dict(size=20))]
        )
        return fig
    
    # Load data for each coin
    coin_data = {}
    
    for coin in selected_coins:
        # Get market data
        df = load_processed_data(coin, data_dir)
        if df is not None and not df.empty:
            # Calculate sentiment score for most recent 14 days
            df_sorted = df.sort_values('date', ascending=False)
            recent_df = df_sorted.head(14)
            score = sentiment_analyzer.calculate_sentiment_score(recent_df)
            coin_data[coin] = {
                'score': score,
                'level': sentiment_analyzer.get_market_sentiment_level(score),
                'type': 'Market'
            }
        
        # Get Reddit data if available
        if include_reddit:
            reddit_df = load_reddit_data(coin)
            if reddit_df is not None and not reddit_df.empty and 'sentiment_compound' in reddit_df.columns:
                # Calculate sentiment score for most recent 14 days
                reddit_sorted = reddit_df.sort_values('date', ascending=False)
                recent_reddit = reddit_sorted.head(14)
                reddit_score = sentiment_analyzer.calculate_sentiment_score(recent_reddit, 
                                                                           weight_engagement=False) 
                reddit_key = f"{coin}_reddit"
                coin_data[reddit_key] = {
                    'score': reddit_score,
                    'level': sentiment_analyzer.get_market_sentiment_level(reddit_score),
                    'type': 'Reddit'
                }
    
    if not coin_data:
        fig = go.Figure()
        fig.update_layout(
            title="No data available for selected coins",
            annotations=[dict(text="No data available", showarrow=False, xref="paper", yref="paper", font=dict(size=20))]
        )
        return fig
    
    # Sort coins by sentiment score
    sorted_coins = sorted(coin_data.items(), key=lambda x: x[1]['score'], reverse=True)
    
    # Prepare data for chart
    coins = []
    scores = []
    colors = []
    labels = []
    types = []
    
    for coin, data in sorted_coins:
        if "_reddit" in coin:
            display_name = f"{coin.split('_reddit')[0].upper()} (Reddit)"
        else:
            display_name = coin.upper()
            
        coins.append(display_name)
        scores.append(data['score'])
        colors.append(get_sentiment_color(data['score']))
        labels.append(f"{data['score']:.1f} - {data['level']}")
        types.append(data['type'])
    
    # Create bar chart
    fig = go.Figure()
    
    # Add bars with different colors by source type
    for source_type in ['Market', 'Reddit']:
        indices = [i for i, t in enumerate(types) if t == source_type]
        if indices:
            fig.add_trace(
                go.Bar(
                    x=[coins[i] for i in indices],
                    y=[scores[i] for i in indices],
                    marker_color=[colors[i] for i in indices],
                    text=[labels[i] for i in indices],
                    textposition='outside',
                    name=source_type
                )
            )
    
    # Add layout details
    fig.update_layout(
        title="Multi-Coin Sentiment Comparison",
        xaxis_title='Cryptocurrency',
        yaxis_title='Sentiment Index (0-100)',
        height=500,
        template='plotly_dark',
        yaxis=dict(range=[0, 100]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_sentiment_volume_scatter(df, coin_name, volume_col='volume', sentiment_col='sentiment_score'):
    """Create a scatter plot of sentiment vs. volume."""
    if df is None or df.empty or volume_col not in df.columns or sentiment_col not in df.columns:
        # Try to use Reddit data if available
        reddit_df = load_reddit_data(coin_name.lower())
        if reddit_df is not None and not reddit_df.empty and 'post_volume' in reddit_df.columns and 'sentiment_compound' in reddit_df.columns:
            df = reddit_df
            volume_col = 'post_volume'
            sentiment_col = 'sentiment_compound'
        else:
            fig = go.Figure()
            fig.update_layout(
                title="No data available",
                annotations=[dict(text="No data available", showarrow=False, xref="paper", yref="paper", font=dict(size=20))]
            )
            return fig
    
    # Create a copy of dataframe
    df_scatter = df.copy()
    
    # Calculate sentiment on 0-100 scale
    df_scatter['sentiment_index'] = (df_scatter[sentiment_col] + 1) * 50
    
    # Create scatter plot
    fig = px.scatter(
        df_scatter,
        x=volume_col,
        y='sentiment_index',
        color='date',
        color_continuous_scale='Viridis',
        size=volume_col,
        hover_data=['date'],
        title=f"{coin_name} Sentiment vs. Volume"
    )
    
    # Add layout details
    fig.update_layout(
        xaxis_title='Volume',
        yaxis_title='Sentiment Index (0-100)',
        height=500,
        template='plotly_dark',
        yaxis=dict(range=[0, 100])
    )
    
    return fig

def create_stats_summary(df, coin_name):
    """Create a summary of key statistics for the selected coin and date range."""
    if df is None or df.empty:
        # Try to use Reddit data if available
        reddit_df = load_reddit_data(coin_name.lower())
        if reddit_df is not None and not reddit_df.empty:
            return html.Div([
                html.H4(f"Reddit Sentiment Data Available for {coin_name.upper()}"),
                html.P(f"No price data available for this period, but Reddit sentiment data was found.")
            ])
        return html.Div("No data available for the selected period")
    
    try:
        # Price stats
        price_current = df['close'].iloc[-1]
        price_change = df['close'].iloc[-1] - df['close'].iloc[0]
        price_change_pct = (price_change / df['close'].iloc[0]) * 100
        price_high = df['high'].max()
        price_low = df['low'].min()
        
        # Sentiment stats
        sentiment_score = calculate_sentiment_gauge(df)
        sentiment_level = sentiment_analyzer.get_market_sentiment_level(sentiment_score)
        sentiment_color = get_sentiment_color(sentiment_score)
        
        # Reddit sentiment if available
        reddit_df = load_reddit_data(coin_name.lower())
        if reddit_df is not None and not reddit_df.empty and 'sentiment_compound' in reddit_df.columns:
            reddit_score = calculate_sentiment_gauge(reddit_df, sentiment_col='sentiment_compound')
            reddit_level = sentiment_analyzer.get_market_sentiment_level(reddit_score)
            reddit_color = get_sentiment_color(reddit_score)
            has_reddit = True
        else:
            has_reddit = False
        
        # Sentiment description based on level
        sentiment_descriptions = {
            "Extreme Greed": "Market is overheated with extreme bullish sentiment. Potential correction risk.",
            "Greed": "Strong bullish sentiment prevails. Traders are confident in further price increases.",
            "Moderate Greed": "Positive sentiment with cautious optimism. Market trending upward.",
            "Neutral": "Balanced market sentiment. No strong bias in either direction.",
            "Moderate Fear": "Slight bearish sentiment. Traders cautious but not panicking.",
            "Fear": "Significant bearish sentiment. Many traders pessimistic about price direction.",
            "Extreme Fear": "Market panic. Often a contrarian buying opportunity."
        }
        
        sentiment_description = sentiment_descriptions.get(sentiment_level, "")
        
        # Create stats cards
        price_card = dbc.Card([
            dbc.CardHeader(f"{coin_name.upper()} Price Statistics", className="bg-primary text-white"),
            dbc.CardBody([
                html.P(f"Current: ${price_current:.2f}", className="card-text"),
                html.P([
                    "Change: ",
                    html.Span(
                        f"${price_change:.2f} ({price_change_pct:.2f}%)",
                        style={'color': 'green' if price_change >= 0 else 'red', 'fontWeight': 'bold'}
                    )
                ], className="card-text"),
                html.P(f"High: ${price_high:.2f}", className="card-text"),
                html.P(f"Low: ${price_low:.2f}", className="card-text")
            ])
        ], className="mb-3")
        
        sentiment_card = dbc.Card([
            dbc.CardHeader("Sentiment Analysis", className="bg-info text-white"),
            dbc.CardBody([
                html.Div([
                    html.Span(f"Market Sentiment: ", className="me-2"),
                    html.Span(
                        f"{sentiment_score:.1f}/100 - {sentiment_level}", 
                        style={'fontWeight': 'bold', 'color': sentiment_color}
                    )
                ]),
                
                # Add Reddit sentiment if available
                html.Div([
                    html.Span(f"Reddit Sentiment: ", className="me-2"),
                    html.Span(
                        f"{reddit_score:.1f}/100 - {reddit_level}", 
                        style={'fontWeight': 'bold', 'color': reddit_color}
                    )
                ]) if has_reddit else None,
                
                html.P(sentiment_description, className="card-text text-muted small mt-2"),
                html.P(f"Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}", 
                      className="card-text mt-3 small"),
                html.P(f"Days Analyzed: {len(df)}", className="card-text small")
            ])
        ], className="mb-3")
        
        return dbc.Row([
            dbc.Col(price_card, md=6),
            dbc.Col(sentiment_card, md=6)
        ])
        
    except Exception as e:
        print(f"Error creating stats summary: {e}")
        return html.Div(f"Error creating statistics: {str(e)}", className="text-danger")

# Define app layout
def create_layout():
    available_coins = get_available_coins()
    reddit_coins = get_reddit_coins()
    
    print(f"Available coins: {len(available_coins)}")
    print(f"Reddit coins: {len(reddit_coins)}")
    
    if not available_coins:
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H2("No processed data found", className="text-center mt-5"),
                    html.P("Please run the data processing pipeline first:", className="text-center"),
                    html.Code("python train_sentiment_model.py", className="d-block text-center mt-3"),
                    html.P("Check if the data/processed directory exists and contains *_processed.csv files.", 
                          className="text-center mt-3"),
                    html.Div([
                        html.H4("Reddit Data Status:", className="mt-5 text-center"),
                        html.P(f"Found {len(reddit_coins)} coins with Reddit sentiment data" if reddit_coins else "No Reddit data found yet", 
                              className="text-center"),
                        html.Code("python collect_reddit_data.py", className="d-block text-center mt-3")
                    ])
                ])
            ])
        ], fluid=True)
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Crypto Sentiment Analysis Dashboard", className="text-center my-4 text-info"),
                html.P("Real-time sentiment analysis of cryptocurrency markets based on social media and market data", 
                      className="text-center mb-4 text-light")
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Label("Select Cryptocurrency:"),
                dcc.Dropdown(
                    id='coin-selector',
                    options=[{'label': coin.upper(), 'value': coin} for coin in available_coins],
                    value=available_coins[0] if available_coins else None,
                    clearable=False,
                    className="mb-3"
                )
            ], md=6),
            
            dbc.Col([
                html.Label("Date Range:"),
                dcc.DatePickerRange(
                    id='date-picker',
                    start_date_placeholder_text="Start Date",
                    end_date_placeholder_text="End Date",
                    className="mb-3"
                )
            ], md=6)
        ]),
        
        # Reddit data indicator
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Span("Reddit Data: ", className="me-2"),
                    html.Span(
                        f"Available for {len(reddit_coins)} coins" if reddit_coins else "Not available",
                        style={'color': '#ff6600' if reddit_coins else 'gray', 'fontWeight': 'bold'}
                    )
                ], className="text-right mb-2")
            ], md=12)
        ]),
        
        # Sentiment Gauge
        dbc.Row([
            dbc.Col([
                dcc.Loading(
                    id="loading-gauge",
                    type="default",
                    children=[dcc.Graph(id='sentiment-gauge')]
                )
            ], md=12)
        ], className="mb-4"),
        
        # Stats Summary
        dbc.Row([
            dbc.Col([
                html.Div(id='stats-container', className="mb-4")
            ])
        ]),
        
        # Tabs for different visualizations
        dbc.Row([
            dbc.Col([
                dbc.Tabs([
                    dbc.Tab(label="Sentiment Trend", tab_id="sentiment-trend"),
                    dbc.Tab(label="Price vs Sentiment", tab_id="price-sentiment"),
                    dbc.Tab(label="Coin Comparison", tab_id="coin-comparison"),
                    dbc.Tab(label="Sentiment vs Volume", tab_id="sentiment-volume")
                ], id="viz-tabs", active_tab="sentiment-trend")
            ])
        ], className="mb-3"),
        
        # Graph container
        dbc.Row([
            dbc.Col([
                dcc.Loading(
                    id="loading-graph",
                    type="default",
                    children=[html.Div(id="graph-container")]
                )
            ])
        ]),
        
        # Coin comparison selector (hidden by default)
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Label("Select Coins to Compare:"),
                    dcc.Dropdown(
                        id='comparison-selector',
                        options=[{'label': coin.upper(), 'value': coin} for coin in available_coins],
                        value=[available_coins[0]] if available_coins else [],
                        multi=True,
                        className="mb-3"
                    )
                ], id="comparison-container", style={"display": "none"})
            ])
        ]),
        
        # Reddit data toggle
        dbc.Row([
            dbc.Col([
                html.Div([
                    dbc.Checkbox(
                        id="include-reddit-toggle",
                        className="form-check-input",
                        checked=True
                    ),
                    dbc.Label(
                        "Include Reddit data in visualizations",
                        html_for="include-reddit-toggle",
                        className="form-check-label ms-2"
                    )
                ], className="form-check mb-3", style={"display": "none" if not reddit_coins else "block"})
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div(id="error-container", className="text-danger")
            ])
        ], className="mb-2"),
        
        dbc.Row([
            dbc.Col([
                html.Hr(),
                html.P("Crypto Sentiment Analysis Dashboard | © 2023", className="text-center text-muted")
            ])
        ], className="mt-4")
    ], fluid=True)

app.layout = create_layout()

# Define callbacks
@app.callback(
    [Output('sentiment-gauge', 'figure'),
     Output('date-picker', 'min_date_allowed'),
     Output('date-picker', 'max_date_allowed'),
     Output('date-picker', 'start_date'),
     Output('date-picker', 'end_date'),
     Output('stats-container', 'children'),
     Output('error-container', 'children')],
    [Input('coin-selector', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')]
)
def update_sentiment_gauge(coin, start_date, end_date):
    """Update the sentiment gauge and stats based on selected coin and date range."""
    print(f"Update gauge: coin={coin}, start_date={start_date}, end_date={end_date}")
    
    try:
        # Load data
        df = load_processed_data(coin)
        
        if df is None:
            # Try to use Reddit data if available
            reddit_df = load_reddit_data(coin)
            if reddit_df is not None and not reddit_df.empty and 'sentiment_compound' in reddit_df.columns:
                # Calculate sentiment from Reddit data
                sentiment_score = calculate_sentiment_gauge(reddit_df, sentiment_col='sentiment_compound')
                
                # Create sentiment gauge
                gauge_fig = create_sentiment_gauge(sentiment_score, coin.upper())
                
                # Get date range from Reddit data
                min_date = reddit_df['date'].min()
                max_date = reddit_df['date'].max()
                
                if start_date is None:
                    start_date = max_date - timedelta(days=30)
                    end_date = max_date
                
                # Filter Reddit data by date
                if start_date and end_date:
                    reddit_filtered = reddit_df[(reddit_df['date'] >= start_date) & (reddit_df['date'] <= end_date)]
                    if reddit_filtered.empty:
                        reddit_filtered = reddit_df
                else:
                    reddit_filtered = reddit_df
                
                # Create stats summary
                stats = html.Div([
                    html.H4(f"Reddit Sentiment Data for {coin.upper()}"),
                    html.P(f"Market data not available, showing Reddit sentiment only."),
                    html.P(f"Sentiment Score: {sentiment_score:.1f}/100 - {sentiment_analyzer.get_market_sentiment_level(sentiment_score)}"),
                    html.P(f"Date Range: {reddit_filtered['date'].min().strftime('%Y-%m-%d')} to {reddit_filtered['date'].max().strftime('%Y-%m-%d')}"),
                    html.P(f"Reddit posts analyzed: {len(reddit_filtered)}")
                ])
                
                return gauge_fig, min_date, max_date, start_date, end_date, stats, ""
            
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title=f"No data available for {coin}",
                annotations=[dict(text="No data available", showarrow=False, xref="paper", yref="paper", font=dict(size=20))]
            )
            return empty_fig, None, None, None, None, "No data available", f"Could not load data for {coin}. Check if data/processed/{coin}_processed.csv exists."
        
        # Set date range
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        # If no dates selected, use last 30 days
        if start_date is None:
            start_date = max_date - timedelta(days=30)
            end_date = max_date
        
        # Filter data by date if needed
        if start_date and end_date:
            df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            # If filtered dataframe is empty, use full dataframe
            if df_filtered.empty:
                print(f"Warning: No data in selected date range for {coin}")
                df_filtered = df
        else:
            df_filtered = df
        
        # Calculate sentiment gauge score
        sentiment_score = calculate_sentiment_gauge(df_filtered)
        
        # Create sentiment gauge
        gauge_fig = create_sentiment_gauge(sentiment_score, coin.upper())
        
        # Create stats summary
        stats = create_stats_summary(df_filtered, coin)
        
        return gauge_fig, min_date, max_date, start_date, end_date, stats, ""
    
    except Exception as e:
        import traceback
        print(f"Error in update_sentiment_gauge callback: {e}")
        traceback.print_exc()
        
        # Return empty figure with error message
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="An error occurred",
            annotations=[dict(text="Error loading data", showarrow=False, xref="paper", yref="paper")]
        )
        
        error_message = f"Error: {str(e)}"
        
        return empty_fig, None, None, None, None, "", error_message

@app.callback(
    [Output('graph-container', 'children'),
     Output('comparison-container', 'style')],
    [Input('viz-tabs', 'active_tab'),
     Input('coin-selector', 'value'),
     Input('comparison-selector', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date'),
     Input('include-reddit-toggle', 'checked')]
)
def update_visualization(active_tab, coin, compared_coins, start_date, end_date, include_reddit):
    """Update visualizations based on selected tab and options."""
    print(f"Update viz: tab={active_tab}, coin={coin}, compared={compared_coins}, reddit={include_reddit}")
    
    # Show or hide comparison selector
    comparison_style = {"display": "block"} if active_tab == "coin-comparison" else {"display": "none"}
    
    try:
        # Load data
        df = load_processed_data(coin)
        
        # If no data and not comparison tab, try Reddit data
        if df is None and active_tab != "coin-comparison":
            reddit_df = load_reddit_data(coin)
            if reddit_df is not None and not reddit_df.empty and 'sentiment_compound' in reddit_df.columns:
                # Filter Reddit data by date if needed
                if start_date and end_date:
                    reddit_filtered = reddit_df[(reddit_df['date'] >= start_date) & (reddit_df['date'] <= end_date)]
                    if reddit_filtered.empty:
                        reddit_filtered = reddit_df
                else:
                    reddit_filtered = reddit_df
                
                # Create visualization based on active tab
                if active_tab == "sentiment-trend":
                    # Use Reddit data for sentiment trend
                    fig = create_sentiment_trend(reddit_filtered, coin.upper(), sentiment_col='sentiment_compound', include_reddit=False)
                    fig.update_layout(title=f"{coin.upper()} Reddit Sentiment Trend")
                    graph = dcc.Graph(figure=fig, id="trend-graph", className="mb-4")
                    return [graph], comparison_style
                
                elif active_tab == "sentiment-volume":
                    # Use Reddit data for sentiment vs volume scatter
                    if 'post_volume' in reddit_filtered.columns:
                        fig = create_sentiment_volume_scatter(reddit_filtered, coin.upper(), 
                                                            volume_col='post_volume', 
                                                            sentiment_col='sentiment_compound')
                        fig.update_layout(title=f"{coin.upper()} Reddit Sentiment vs Post Volume")
                        graph = dcc.Graph(figure=fig, id="volume-graph", className="mb-4")
                        return [graph], comparison_style
                
                # If we can't create a specific visualization with Reddit data
                graph = dcc.Graph(
                    figure=go.Figure().update_layout(
                        title=f"No suitable market data available for {coin}",
                        annotations=[dict(text="Only Reddit data available for this coin", 
                                        showarrow=False, xref="paper", yref="paper", font=dict(size=20))]
                    )
                )
                return [graph], comparison_style
            
            # If no data at all
            graph = dcc.Graph(
                figure=go.Figure().update_layout(
                    title=f"No data available for {coin}",
                    annotations=[dict(text="No data available", showarrow=False, xref="paper", yref="paper", font=dict(size=20))]
                )
            )
            return [graph], comparison_style
        
        # If we have market data but it's the comparison tab
        if active_tab == "coin-comparison":
            fig = create_multi_coin_comparison(compared_coins, include_reddit=include_reddit)
            graph = dcc.Graph(figure=fig, id="comparison-graph", className="mb-4")
            return [graph], comparison_style
        
        # Filter data by date if needed for other tabs
        if start_date and end_date:
            df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            # If filtered dataframe is empty, use full dataframe
            if df_filtered.empty:
                print(f"Warning: No data in selected date range for {coin}")
                df_filtered = df
        else:
            df_filtered = df
        
        # Create visualization based on active tab
        if active_tab == "sentiment-trend":
            fig = create_sentiment_trend(df_filtered, coin.upper(), include_reddit=include_reddit)
            graph = dcc.Graph(figure=fig, id="trend-graph", className="mb-4")
            return [graph], comparison_style
            
        elif active_tab == "price-sentiment":
            fig = create_price_sentiment_overlay(df_filtered, coin.upper(), include_reddit=include_reddit)
            graph = dcc.Graph(figure=fig, id="overlay-graph", className="mb-4")
            return [graph], comparison_style
            
        elif active_tab == "sentiment-volume":
            fig = create_sentiment_volume_scatter(df_filtered, coin.upper())
            graph = dcc.Graph(figure=fig, id="volume-graph", className="mb-4")
            return [graph], comparison_style
            
        else:
            # Default case
            graph = dcc.Graph(
                figure=go.Figure().update_layout(
                    title="Select a visualization type",
                    annotations=[dict(text="Choose a visualization from the tabs above", showarrow=False, xref="paper", yref="paper")]
                )
            )
            return [graph], comparison_style
    
    except Exception as e:
        import traceback
        print(f"Error in update_visualization callback: {e}")
        traceback.print_exc()
        
        # Return empty figure with error message
        graph = dcc.Graph(
            figure=go.Figure().update_layout(
                title="An error occurred",
                annotations=[dict(text=f"Error: {str(e)}", showarrow=False, xref="paper", yref="paper")]
            )
        )
        
        return [graph], comparison_style

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)