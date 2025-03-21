import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Create a simplified visualizer class
class SimpleVisualizer:
    def __init__(self, output_dir='reports/figures'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print("Using simplified visualizer")
    
    def plot_price_sentiment_time_series(self, df, price_col='close', sentiment_col='sentiment_score',
                                       title=None, coin_name='', save_path=None):
        if title is None:
            title = f'{coin_name} Price and Sentiment Over Time'
        
        # Ensure date column is datetime
        if 'date' in df.columns and not pd.api.types.is_datetime64_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Create subplot with two y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add price line
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df[price_col],
                name=f'{price_col.capitalize()} Price',
                line=dict(color='blue')
            ),
            secondary_y=False
        )
        
        # Add sentiment line
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df[sentiment_col],
                name='Sentiment Score',
                line=dict(color='red')
            ),
            secondary_y=True
        )
        
        # Add layout details
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
            hovermode='x unified',
            height=600,
            template='plotly_white'
        )
        
        fig.update_yaxes(title_text=f"{price_col.capitalize()} Price", secondary_y=False)
        fig.update_yaxes(title_text="Sentiment Score", secondary_y=True)
        
        return fig
    
    def plot_candlestick_with_sentiment(self, df, sentiment_col='sentiment_score', coin_name='',
                                      title=None, save_path=None):
        if title is None:
            title = f'{coin_name} Price Candlestick with Sentiment'
        
        # Ensure date column is datetime
        if 'date' in df.columns and not pd.api.types.is_datetime64_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Create subplot with two y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ),
            secondary_y=False
        )
        
        # Add sentiment line
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df[sentiment_col],
                name='Sentiment Score',
                line=dict(color='purple', width=2)
            ),
            secondary_y=True
        )
        
        # Add layout details
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
            hovermode='x unified',
            height=700,
            template='plotly_white',
            xaxis_rangeslider_visible=False
        )
        
        fig.update_yaxes(title_text="Price", secondary_y=False)
        fig.update_yaxes(title_text="Sentiment Score", secondary_y=True)
        
        return fig
    
    def plot_technical_indicators(self, df, coin_name='', title=None, save_path=None):
        if title is None:
            title = f'{coin_name} Technical Indicators'
            
        # Ensure date column is datetime
        if 'date' in df.columns and not pd.api.types.is_datetime64_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
            
        # Create a simple figure as fallback
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['close'],
                name='Close Price',
                line=dict(color='blue', width=2)
            )
        )
        
        # Add SMA lines if available
        for ma_col, color in [('sma_7', 'green'), ('sma_20', 'orange'), ('sma_50', 'red')]:
            if ma_col in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['date'],
                        y=df[ma_col],
                        name=ma_col.upper(),
                        line=dict(color=color, width=1.5)
                    )
                )
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price',
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
            template='plotly_white',
            height=600
        )
        
        return fig

# Initialize visualizer
visualizer = SimpleVisualizer()

# Create the Dash app
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1'}]
)
app.title = 'Crypto Sentiment Analysis Dashboard'
server = app.server

# Helper functions
def load_processed_data(coin, data_dir='data/processed'):
    """Load processed data for a specific coin."""
    try:
        file_path = os.path.join(data_dir, f"{coin}_processed.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Convert date to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            return df
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data for {coin}: {e}")
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

def create_stats_summary(df, coin_name):
    """Create a summary of key statistics for the selected coin and date range."""
    if df is None or df.empty:
        return html.Div("No data available for the selected period")
    
    try:
        # Price stats
        price_current = df['close'].iloc[-1]
        price_change = df['close'].iloc[-1] - df['close'].iloc[0]
        price_change_pct = (price_change / df['close'].iloc[0]) * 100
        price_high = df['high'].max()
        price_low = df['low'].min()
        
        # Sentiment stats
        sentiment_avg = df['sentiment_score'].mean()
        
        # Create stats cards
        price_card = dbc.Card([
            dbc.CardHeader(f"{coin_name} Price Statistics", className="bg-primary text-white"),
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
            dbc.CardHeader("Sentiment Statistics", className="bg-info text-white"),
            dbc.CardBody([
                html.P(f"Average Sentiment: {sentiment_avg:.3f}", className="card-text"),
                html.P(f"Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}", 
                      className="card-text"),
                html.P(f"Days Analyzed: {len(df)}", className="card-text")
            ])
        ], className="mb-3")
        
        return dbc.Row([
            dbc.Col(price_card, width=6),
            dbc.Col(sentiment_card, width=6)
        ])
        
    except Exception as e:
        print(f"Error creating stats summary: {e}")
        return html.Div(f"Error creating statistics: {str(e)}", className="text-danger")

# Define app layout
def create_layout():
    available_coins = get_available_coins()
    print(f"Available coins: {available_coins}")
    
    if not available_coins:
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H2("No processed data found", className="text-center mt-5"),
                    html.P("Please run the data processing pipeline first:", className="text-center"),
                    html.Code("python src/main.py", className="d-block text-center mt-3"),
                    html.P("Check if the data/processed directory exists and contains *_processed.csv files.", 
                          className="text-center mt-3")
                ])
            ])
        ], fluid=True)
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Crypto Sentiment Analysis Dashboard", className="text-center my-4"),
                html.P("Explore the relationship between cryptocurrency prices and market sentiment", 
                      className="text-center mb-4")
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Label("Select Cryptocurrency:"),
                dcc.Dropdown(
                    id='coin-selector',
                    options=[{'label': coin.upper(), 'value': coin} for coin in available_coins],
                    value=available_coins[0] if available_coins else None,
                    clearable=False
                )
            ], width=4),
            
            dbc.Col([
                html.Label("Date Range:"),
                dcc.DatePickerRange(
                    id='date-picker',
                    start_date_placeholder_text="Start Date",
                    end_date_placeholder_text="End Date",
                    calendar_orientation='horizontal',
                    clearable=True
                )
            ], width=4),
            
            dbc.Col([
                html.Label("Visualization Type:"),
                dcc.Tabs(
                    id='viz-tabs',
                    value='price-sentiment',
                    children=[
                        dcc.Tab(label='Price & Sentiment', value='price-sentiment'),
                        dcc.Tab(label='Candlestick Chart', value='candlestick'),
                        dcc.Tab(label='Technical Indicators', value='technical')
                    ]
                )
            ], width=4)
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dcc.Loading(
                    id="loading-graphs",
                    type="default",
                    children=[dcc.Graph(id='main-graph', style={'height': '70vh'})]
                )
            ])
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div(id='stats-container', className="mt-4")
            ])
        ]),
        
        html.Hr(),
        
        dbc.Row([
            dbc.Col([
                html.Div(id="error-container", className="text-danger")
            ])
        ], className="mb-2"),
        
        dbc.Row([
            dbc.Col([
                html.P("Crypto Sentiment Analysis Platform | Created for GitHub Portfolio", className="text-center text-muted")
            ])
        ], className="mt-4")
    ], fluid=True)

app.layout = create_layout()

# Define callbacks
@app.callback(
    [Output('main-graph', 'figure'),
     Output('date-picker', 'min_date_allowed'),
     Output('date-picker', 'max_date_allowed'),
     Output('date-picker', 'start_date'),
     Output('date-picker', 'end_date'),
     Output('stats-container', 'children'),
     Output('error-container', 'children')],
    [Input('coin-selector', 'value'),
     Input('viz-tabs', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')]
)
def update_graph(coin, viz_type, start_date, end_date):
    print(f"Update graph: coin={coin}, viz_type={viz_type}, start_date={start_date}, end_date={end_date}")
    
    try:
        # Load data
        df = load_processed_data(coin)
        
        if df is None:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title=f"No data available for {coin}",
                annotations=[dict(text="No data available", showarrow=False, xref="paper", yref="paper", font=dict(size=20))]
            )
            return empty_fig, None, None, None, None, "No data available", f"Could not load data for {coin}. Check if data/processed/{coin}_processed.csv exists."
        
        # Set date range
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        # If no dates selected, use last 90 days
        if start_date is None:
            start_date = max_date - timedelta(days=90)
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
        
        # Create the appropriate visualization
        if viz_type == 'price-sentiment':
            fig = visualizer.plot_price_sentiment_time_series(df_filtered, coin_name=coin.upper())
        
        elif viz_type == 'candlestick':
            fig = visualizer.plot_candlestick_with_sentiment(df_filtered, coin_name=coin.upper())
            
        elif viz_type == 'technical':
            fig = visualizer.plot_technical_indicators(df_filtered, coin_name=coin.upper())
        
        else:
            fig = go.Figure()
            fig.update_layout(
                title="Select a visualization type",
                annotations=[dict(text="Choose a visualization from the tabs above", showarrow=False, xref="paper", yref="paper")]
            )
        
        # Create stats summary
        stats = create_stats_summary(df_filtered, coin.upper())
        
        return fig, min_date, max_date, start_date, end_date, stats, ""
    
    except Exception as e:
        import traceback
        print(f"Error in update_graph callback: {e}")
        traceback.print_exc()
        
        # Return empty figure with error message
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="An error occurred",
            annotations=[dict(text="Try a different visualization or date range", showarrow=False, xref="paper", yref="paper")]
        )
        
        error_message = f"Error: {str(e)}"
        
        return empty_fig, None, None, None, None, "", error_message

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8050)