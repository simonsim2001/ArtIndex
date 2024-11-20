import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.optimize import newton
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import re
import plotly.express as px
import plotly.graph_objs as go
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.optimize import minimize
import os
import io

# Set page configuration at the very top
st.set_page_config(layout="wide")

# Add the check_password function
def check_password():
    def password_entered():
        if st.session_state["password"] == "maxime":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.title("🔒 Enter Password")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.title("🔒 Enter Password")
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

# Load and preprocess the data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['Auction_Date', 'Price_USD', 'Artist', 'Dimensions'])
    df['Auction_Date'] = pd.to_datetime(df['Auction_Date'])
    df['Year'] = df['Auction_Date'].dt.year
    df['Price_USD'] = pd.to_numeric(df['Price_USD'], errors='coerce')
    df = df.dropna(subset=['Price_USD'])
    df[['Width', 'Height']] = df['Dimensions'].str.extract(r'(\d+\.?\d*)\s*x\s*(\d+\.?\d*)').astype(float)
    df['Area'] = df['Width'] * df['Height']
    df = df.dropna(subset=['Width', 'Height'])
    df['Normalized_Price'] = df['Price_USD'] / df['Area']

    # Correctly process 'Creation_Date' to extract the year
    def extract_creation_year(date_str):
        if pd.isnull(date_str):
            return None
        # Extract four-digit year from the string
        match = re.search(r'(1[7-9]\d{2}|20[0-2]\d)', str(date_str))
        if match:
            return int(match.group(0))
        else:
            return None

    df['Creation_Year'] = df['Creation_Date'].apply(extract_creation_year)

    # Since avg_red, avg_green, avg_blue are now in the main dataset, no need to merge
    # Function to map RGB to colour names
    def rgb_to_colour_name(r, g, b):
        if pd.isnull(r) or pd.isnull(g) or pd.isnull(b):
            return 'Unknown'
        # Normalize RGB values to [0, 1]
        r_norm = r / 255.0
        g_norm = g / 255.0
        b_norm = b / 255.0
        # Determine dominant colour
        if r_norm > g_norm and r_norm > b_norm:
            return 'Red'
        elif g_norm > r_norm and g_norm > b_norm:
            return 'Green'
        elif b_norm > r_norm and b_norm > g_norm:
            return 'Blue'
        elif abs(r_norm - g_norm) < 0.1 and b_norm < 0.2:
            return 'Yellow'
        elif r_norm > 0.8 and g_norm > 0.8 and b_norm > 0.8:
            return 'White'
        elif r_norm < 0.2 and g_norm < 0.2 and b_norm < 0.2:
            return 'Black'
        else:
            return 'Other'

    # Apply the function to create 'Dominant_Colour'
    df['Dominant_Colour'] = df.apply(lambda row: rgb_to_colour_name(row['avg_red'], row['avg_green'], row['avg_blue']), axis=1)

    return df

@st.cache_data
def load_data():
    data = pd.read_csv('https://raw.githubusercontent.com/simonsim2001/ArtIndex/refs/heads/main/BlueChip.csv')
    data['Auction_Date'] = pd.to_datetime(data['Auction_Date'], errors='coerce')
    data['Price_USD'] = pd.to_numeric(data['Price_USD'], errors='coerce')
    cleaned_data = data.dropna(subset=['Auction_Date', 'Price_USD'])
    cleaned_data.set_index('Auction_Date', inplace=True)
    cleaned_data.sort_index(inplace=True)
    return cleaned_data

def extract_dimensions(df):
    df['Width_cm'] = df['Dimensions'].apply(lambda x: float(re.search(r'(\d+\.?\d*)\s*cm', x).group(1)) if re.search(r'(\d+\.?\d*)\s*cm', x) else None)
    df['Height_cm'] = df['Dimensions'].apply(lambda x: float(re.search(r'x\s*(\d+\.?\d*)\s*cm', x).group(1)) if re.search(r'x\s*(\d+\.?\d*)\s*cm', x) else None)
    return df

def generate_html_table_with_images(data):
    # Create the HTML table string
    html = '<table style="width:100%; border-collapse: collapse;">'
    html += '<tr>'
    for column in data.columns:
        html += f'<th style="border: 1px solid black; padding: 8px;">{column}</th>'
    html += '</tr>'

    for _, row in data.iterrows():
        html += '<tr>'
        for column in data.columns:
            if 'image' in column.lower() or 'img' in column.lower():
                html += f'<td style="border: 1px solid black; padding: 8px;"><img src="{row[column]}" alt="{column}" style="width:100px; height:auto;"></td>'
            else:
                html += f'<td style="border: 1px solid black; padding: 8px;">{row[column]}</td>'
        html += '</tr>'
    html += '</table>'
    return html

# Helper functions
def calculate_irr(cash_flows):
    years = np.arange(len(cash_flows))
    def npv(rate):
        return np.sum(cash_flows / (1 + rate) ** years)
    try:
        return newton(npv, 0.1) * 100
    except RuntimeError:
        return np.nan

def calculate_moic(initial_investment, final_value):
    return final_value / initial_investment

def calculate_rolling_average(data, window):
    rolling_average = data['Price_USD'].rolling(window=window, min_periods=1).mean()
    rolling_average = rolling_average[rolling_average.index.year >= 1995]
    return rolling_average.resample('Y').last()  # Resample to yearly data

def load_market_data(selected_ticker_symbols, start_date, end_date):
    try:
        market_data = {}
        for symbol in selected_ticker_symbols:
            data = yf.download(symbol, start=start_date, end=end_date, interval="1mo")
            if not data.empty:
                market_data[symbol] = data['Adj Close'].resample('Y').last()  # Use last price of the year
        return market_data
    except Exception as e:
        st.error(f"Failed to download data for one or more symbols: {e}")
        return None



#Portfolio visualisation
    #'art': '#6a0dad',      # Dark purple for Art
    #'artprice': '#8a2be2', # Light purple for Artprice Index
    #'market': '#ff8c00',   # Orange for S&P500
    #'combined': '#808080', # Grey for Combined Portfolio
    #'reference': '#000000' # Black for reference or baseline (if needed)

# Define a unified color palette
colors = {
    'art': '#1f77b4',      # Standard blue for Art
    'artprice': '#2ca02c', # Standard green for Artprice Index
    'market': '#d62728',   # Standard red for S&P500 (Market)
    'combined': '#808080', # Grey for Combined Portfolio
    'reference': '#000000' # Black for reference or baseline (if needed)
}

def plot_cumulative_returns(metrics, artprice_index=None):
    fig = go.Figure()
    
    # Plot art cumulative returns
    art_cumulative = metrics['Art Portfolio']['cumulative_return']
    fig.add_trace(go.Scatter(x=art_cumulative.index, y=art_cumulative, mode='lines', name='Art Portfolio', 
                             line=dict(color=colors['art'])))
    
    # Plot Artprice Blue Chip Index if provided
    if artprice_index is not None:
        fig.add_trace(go.Scatter(x=artprice_index.index, y=artprice_index, mode='lines', 
                                 name='Artprice Blue Chip Index', line=dict(color=colors['artprice'], dash='dash')))
    
    # Plot market cumulative returns
    market_cumulative = metrics['Market Portfolio']['cumulative_return']
    fig.add_trace(go.Scatter(x=market_cumulative.index, y=market_cumulative, mode='lines', name='Market Portfolio', 
                             line=dict(color=colors['market'])))
    
    # Plot combined cumulative returns
    combined_cumulative = metrics['Combined Portfolio']['cumulative_return']
    fig.add_trace(go.Scatter(x=combined_cumulative.index, y=combined_cumulative, mode='lines', 
                             name='Combined Portfolio', line=dict(color=colors['combined'])))
    
    fig.update_layout(
        title='Cumulative Returns',
        xaxis_title='Year',
        yaxis_title='Cumulative Returns',
        legend_title='Portfolio',
        template='plotly_white',
        showlegend=True,
        xaxis=dict(showgrid=False),  # Remove gridlines
        yaxis=dict(showgrid=False)
    )
    
    return fig



def plot_annual_returns(art_annual_returns, market_annual_returns, combined_annual_returns):
    fig = go.Figure()
    
    annual_returns_df = pd.DataFrame({
        'Year': art_annual_returns.index.year,
        'Art Portfolio': art_annual_returns.values,
        'Market Portfolio': market_annual_returns.values,
        'Combined Portfolio': combined_annual_returns.values
    })
    annual_returns_df.set_index('Year', inplace=True)
    
    for column in ['Art Portfolio', 'Market Portfolio', 'Combined Portfolio']:
        # Apply the colors consistently across the bars
        if column == 'Art Portfolio':
            color = colors['art']
        elif column == 'Market Portfolio':
            color = colors['market']
        else:
            color = colors['combined']
        fig.add_trace(go.Bar(x=annual_returns_df.index, y=annual_returns_df[column], name=column, marker=dict(color=color)))
    
    fig.update_layout(
        title='Annual Returns',
        xaxis_title='Year',
        yaxis_title='Annual Return',
        barmode='group',
        legend_title='Portfolio',
        xaxis=dict(showgrid=False),  # Remove gridlines
        yaxis=dict(showgrid=False),
        template='plotly_white'
    )
    
    return fig




def plot_rolling_correlation(metrics, window=12):
    fig = go.Figure()
    
    art_returns = metrics['Art Portfolio']['returns']
    market_returns = metrics['Market Portfolio']['returns']
    
    # Ensure the returns are aligned
    art_returns, market_returns = art_returns.align(market_returns, join='inner')
    
    correlation = art_returns.rolling(window=window).corr(market_returns).dropna()
    
    fig.add_trace(go.Scatter(
        x=correlation.index, 
        y=correlation, 
        mode='lines', 
        name='Rolling Correlation',
        line=dict(width=2, color=colors['combined'])
    ))

    fig.add_shape(type="line", x0=correlation.index.min(), x1=correlation.index.max(), 
                  y0=0, y1=0, line=dict(color="gray", dash="dash"))

    fig.update_layout(
        title=f'{window}-Month Rolling Correlation between Art and Market',
        xaxis_title='Year',
        yaxis_title='Correlation',
        xaxis=dict(showgrid=False, showline=True, linecolor='black', mirror=True),
        yaxis=dict(showgrid=False, showline=True, linecolor='black', mirror=True, range=[-1, 1]),
        template='plotly_white',
        hovermode='x unified',
        legend_title='Correlation',
        height=600
    )
    
    return fig



def plot_efficient_frontier(metrics):
    fig = go.Figure()
    
    # Extract Art and Market returns
    art_returns = metrics['Art Portfolio']['returns']
    market_returns = metrics['Market Portfolio']['returns']
    
    # Ensure the returns are aligned
    art_returns, market_returns = art_returns.align(market_returns, join='inner')
    
    # Create a range of weights for Art and Market
    weights = np.arange(0, 1.01, 0.01)
    
    portfolio_metrics = []
    
    for w in weights:
        # Calculate portfolio returns
        portfolio_return = w * art_returns + (1 - w) * market_returns
        
        # Calculate expected annual return and annual volatility
        mean_return = np.mean(portfolio_return) * 12  # Annualized return
        volatility = np.std(portfolio_return) * np.sqrt(12)  # Annualized volatility
        
        portfolio_metrics.append({
            'Art_Weight': w,
            'Expected_Return': mean_return,
            'Volatility': volatility
        })
    
    # Create a DataFrame from the portfolio metrics
    portfolios = pd.DataFrame(portfolio_metrics)
    
    # Plot the Efficient Frontier
    fig.add_trace(go.Scatter(
        x=portfolios['Volatility'],
        y=portfolios['Expected_Return'],
        mode='markers',
        marker=dict(
            size=5,
            color=portfolios['Art_Weight'],
            colorscale='Purples',
            showscale=True,
            colorbar=dict(title='Art Weight')
        ),
        text=portfolios['Art_Weight'].apply(lambda x: f'Art Weight: {x:.2f}<br>Market Weight: {1 - x:.2f}'),
        hoverinfo='text+x+y',
        name='Efficient Frontier'
    ))
    
    fig.update_layout(
        title='Efficient Frontier',
        xaxis_title='Volatility',
        yaxis_title='Expected Annual Return',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False)
    )
    
    return fig





#Portfolio metrics

def calculate_combined_portfolio_metrics(
    art_returns, market_returns_dict, art_allocation, risk_free_rate
):
    metrics = {}

    # Initialize the combined market returns
    combined_market_returns = None

    # Calculate the combined market returns by summing the weighted returns of all tickers
    for ticker, returns in market_returns_dict.items():
        if combined_market_returns is None:
            combined_market_returns = returns
        else:
            combined_market_returns += returns

    # Average the combined market returns by the number of tickers
    combined_market_returns /= len(market_returns_dict)

    # Remaining allocation for market
    market_allocation = 1 - art_allocation

    # Calculate the combined returns
    combined_returns = (art_returns * art_allocation) + (combined_market_returns * market_allocation)

    # Ensure the returns are aligned
    art_returns, combined_returns = art_returns.align(combined_returns, join='inner')
    combined_market_returns = combined_market_returns.reindex(art_returns.index).fillna(0)

    # Calculate metrics for Art Portfolio
    art_excess_returns = art_returns - risk_free_rate / 12
    art_sharpe_ratio = np.mean(art_excess_returns) / np.std(art_excess_returns) * np.sqrt(12)
    art_volatility = np.std(art_returns) * np.sqrt(12)
    art_cumulative = (1 + art_returns).cumprod()
    art_annualized_return = art_cumulative.iloc[-1] ** (12 / len(art_cumulative)) - 1
    art_max_drawdown = (art_cumulative / art_cumulative.cummax() - 1).min()

    # Calculate metrics for Market Portfolio
    market_excess_returns = combined_market_returns - risk_free_rate / 12
    market_sharpe_ratio = np.mean(market_excess_returns) / np.std(combined_market_returns) * np.sqrt(12)
    market_volatility = np.std(combined_market_returns) * np.sqrt(12)
    market_cumulative = (1 + combined_market_returns).cumprod()
    market_annualized_return = market_cumulative.iloc[-1] ** (12 / len(market_cumulative)) - 1
    market_max_drawdown = (market_cumulative / market_cumulative.cummax() - 1).min()

    # Calculate metrics for Combined Portfolio
    excess_returns = combined_returns - risk_free_rate / 12
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(12)
    combined_volatility = np.std(combined_returns) * np.sqrt(12)
    combined_cumulative = (1 + combined_returns).cumprod()
    annualized_return = combined_cumulative.iloc[-1] ** (12 / len(combined_cumulative)) - 1
    max_drawdown = (combined_cumulative / combined_cumulative.cummax() - 1).min()

    # Store the calculated metrics
    metrics['Art Portfolio'] = {
        'sharpe_ratio': art_sharpe_ratio,
        'volatility': art_volatility,
        'cumulative_return': art_cumulative,
        'returns': art_returns,
        'annualized_return': art_annualized_return,
        'max_drawdown': art_max_drawdown
    }

    metrics['Market Portfolio'] = {
        'sharpe_ratio': market_sharpe_ratio,
        'volatility': market_volatility,
        'cumulative_return': market_cumulative,
        'returns': combined_market_returns,
        'annualized_return': market_annualized_return,
        'max_drawdown': market_max_drawdown
    }

    metrics['Combined Portfolio'] = {
        'sharpe_ratio': sharpe_ratio,
        'volatility': combined_volatility,
        'cumulative_return': combined_cumulative,
        'returns': combined_returns,
        'art_cumulative': art_cumulative,
        'market_cumulative': market_cumulative,
        'art_returns': art_returns,
        'market_returns': combined_market_returns,
        'combined_returns': combined_returns,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown
    }

    return metrics





#Portfolio parameters

def get_portfolio_parameters(df):
    st.sidebar.header('Portfolio Parameters')

    # Rolling window for art data
    art_rolling_window = st.sidebar.slider('Art Rolling Window (days)', 365, 1825, 680)

    # Explanatory note for rolling window
    st.sidebar.markdown(
        "<small>Note: Art is an illiquid asset class. To simulate a real investment scenario, "
        "we consider an entry and exit period using a rolling window. The selected window represents "
        "the time needed to buy and sell art, taking into account market conditions and liquidity constraints.</small>",
        unsafe_allow_html=True
    )

    # Allocation of art in the portfolio
    art_allocation = st.sidebar.slider('Art Allocation', 0.0, 1.0, 0.2)

    # Risk-free rate for Sharpe ratio calculation
    risk_free_rate = st.sidebar.number_input('Risk-Free Rate', 0.0, 0.1, 0.02, 0.01)

    # Date range for the data
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("1995-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-12-31"))
    
    # List of available ticker symbols with their real names
    ticker_list = [
    ("^GSPC", "S&P 500"),
    ("AAPL", "Apple Inc."),
    ("GOOGL", "Alphabet Inc."),
    ("AMZN", "Amazon.com Inc."),
    ("MSFT", "Microsoft Corporation"),
    ("META", "Meta Platforms Inc."),
    ("TSLA", "Tesla Inc."),
    ("NVDA", "NVIDIA Corporation"),
    ("JPM", "JPMorgan Chase & Co."),
    ("V", "Visa Inc."),
    ("JNJ", "Johnson & Johnson"),
    ("WMT", "Walmart Inc."),
    ("UNH", "UnitedHealth Group Inc."),
    ("BAC", "Bank of America Corporation"),
    ("PG", "Procter & Gamble Company"),
    ("MA", "Mastercard Incorporated"),
    ("HD", "The Home Depot Inc."),
    ("DIS", "The Walt Disney Company"),
    ("ADBE", "Adobe Inc."),
    ("NFLX", "Netflix Inc."),
    ("XOM", "Exxon Mobil Corporation"),
    ("CMCSA", "Comcast Corporation"),
    ("CSCO", "Cisco Systems Inc."),
    ("PFE", "Pfizer Inc."),
    ("INTC", "Intel Corporation"),
    ("VZ", "Verizon Communications Inc."),
    ("KO", "The Coca-Cola Company"),
    ("CVX", "Chevron Corporation"),
    ("PEP", "PepsiCo Inc."),
    ("ABT", "Abbott Laboratories"),
    ("MRK", "Merck & Co. Inc."),
    ("T", "AT&T Inc."),
    ("ORCL", "Oracle Corporation"),
    ("CRM", "Salesforce.com Inc."),
    ("COST", "Costco Wholesale Corporation"),
    ("ABBV", "AbbVie Inc."),
    ("NKE", "NIKE Inc."),
    ("AVGO", "Broadcom Inc."),
    ("ACN", "Accenture plc"),
    ("TMO", "Thermo Fisher Scientific Inc."),
    ("MDT", "Medtronic plc"),
    ("LLY", "Eli Lilly and Company"),
    ("NEE", "NextEra Energy Inc."),
    ("TXN", "Texas Instruments Incorporated"),
    ("UNP", "Union Pacific Corporation"),
    ("BMY", "Bristol-Myers Squibb Company"),
    ("LIN", "Linde plc"),
    ("QCOM", "QUALCOMM Incorporated"),
    ("DHR", "Danaher Corporation"),
    ("PM", "Philip Morris International Inc."),
    ("AMT", "American Tower Corporation"),
    ("IBM", "International Business Machines Corporation"),
    ("RTX", "Raytheon Technologies Corporation"),
    ("SBUX", "Starbucks Corporation"),
    ("HON", "Honeywell International Inc."),
    ("C", "Citigroup Inc."),
    ("INTU", "Intuit Inc."),
    ("BA", "The Boeing Company"),
    ("GE", "General Electric Company"),
    ("AMD", "Advanced Micro Devices Inc."),
    ("CAT", "Caterpillar Inc."),
    ("GS", "The Goldman Sachs Group Inc."),
    ("MMM", "3M Company"),
    ("BKNG", "Booking Holdings Inc."),
    ("GILD", "Gilead Sciences Inc."),
    ("AXP", "American Express Company"),
    ("LOW", "Lowe's Companies Inc."),
    ("CHTR", "Charter Communications Inc."),
    ("MDLZ", "Mondelez International Inc."),
    ("ISRG", "Intuitive Surgical Inc."),
    ("TGT", "Target Corporation"),
    ("SPGI", "S&P Global Inc."),
    ("BLK", "BlackRock Inc."),
    ("PYPL", "PayPal Holdings Inc."),
    ("SYK", "Stryker Corporation"),
    ("ZTS", "Zoetis Inc."),
    ("ANTM", "Anthem Inc."),
    ("CVS", "CVS Health Corporation"),
    ("FIS", "Fidelity National Information Services Inc."),
    ("TMUS", "T-Mobile US Inc."),
    ("MO", "Altria Group Inc."),
    ("SCHW", "The Charles Schwab Corporation"),
    ("CI", "Cigna Corporation"),
    ("COP", "ConocoPhillips"),
    ("CME", "CME Group Inc."),
    ("ATVI", "Activision Blizzard Inc."),
    ("DE", "Deere & Company"),
    ("DUK", "Duke Energy Corporation"),
    ("PLD", "Prologis Inc."),
    ("MS", "Morgan Stanley"),
    ("BDX", "Becton Dickinson and Company"),
    ("CB", "Chubb Limited"),
    ("EOG", "EOG Resources Inc."),
    ("SO", "The Southern Company"),
    ("MMC", "Marsh & McLennan Companies Inc."),
    ("USB", "U.S. Bancorp"),
    ("CL", "Colgate-Palmolive Company"),
    ("CSX", "CSX Corporation"),
    ("ICE", "Intercontinental Exchange Inc."),
    ("EL", "The Estée Lauder Companies Inc.")
]

    # Create a dictionary for ticker symbols and their names
    ticker_dict = {ticker: name for ticker, name in ticker_list}

    # Multiselect for users to choose the ticker symbols
    selected_ticker_names = st.sidebar.multiselect(
        "Select Ticker Symbols",
        options=[name for ticker, name in ticker_list],
        default=["S&P 500"]
    )

    # Map the selected names to their corresponding ticker symbols
    selected_ticker_symbols = [ticker for ticker, name in ticker_list if name in selected_ticker_names]

    # Add artist selection
    st.sidebar.subheader("Artist Selection")
    unique_artists = df['Artist'].unique().tolist()
    unique_artists.sort()  # Sort artists alphabetically

    # Initialize session state for selected artists if not present
    if 'selected_artists' not in st.session_state:
        st.session_state.selected_artists = unique_artists.copy()  # Default to all artists

    # Buttons to select/deselect all artists
    col1, col2 = st.sidebar.columns([1, 1])
    with col1:
        if st.button('Select All Artists'):
            st.session_state.selected_artists = unique_artists.copy()
    with col2:
        if st.button('Deselect All Artists'):
            st.session_state.selected_artists = []

    # Multiselect for artists
    selected_artists = st.sidebar.multiselect(
        "Select Artist(s)",
        options=unique_artists,
        default=st.session_state.selected_artists,
        key='selected_artists_multiselect'
    )

    # Update the session state with the current selection
    st.session_state.selected_artists = selected_artists

    # Return the selected parameters
    return (
        art_rolling_window, art_allocation,
        risk_free_rate, start_date, end_date, selected_ticker_symbols, selected_artists
    )


    #'art': '#6a0dad',      # Dark purple for Art
    #'artprice': '#8a2be2', # Light purple for Artprice Index
    #'market': '#ff8c00',   # Orange for S&P500
    #'combined': '#808080', # Grey for Combined Portfolio
    #'reference': '#000000' # Black for reference or baseline (if needed)

#Visualisation (only art and stock market)

def plot_projected_cumulative_returns(historical_metrics, projected_metrics):
    # Define a unified color palette
    colors = {
        'art': '#1f77b4',      # Standard blue for Art
        'artprice': '#2ca02c', # Standard green for Artprice Index
        'market': '#d62728',   # Standard red for S&P500 (Market)
        'combined': '#808080', # Grey for Combined Portfolio
        'reference': '#000000' # Black for reference or baseline (if needed)
    }

    fig = go.Figure()

    # Historical combined cumulative returns
    combined_cumulative_hist = historical_metrics['Combined Portfolio']['combined_cumulative']
    fig.add_trace(go.Scatter(
        x=combined_cumulative_hist.index,
        y=combined_cumulative_hist,
        mode='lines',
        name='Combined Portfolio (Historical)',
        line=dict(color=colors['combined'])  # Use color from the palette
    ))

    # Projected combined cumulative returns
    combined_cumulative_proj = projected_metrics['Combined Portfolio']['combined_cumulative']
    projected_year_index = combined_cumulative_proj.index[combined_cumulative_proj.index > pd.Timestamp('2024-06-30')]
    combined_cumulative_new = combined_cumulative_proj.loc[projected_year_index]

    fig.add_trace(go.Scatter(
        x=combined_cumulative_new.index,
        y=combined_cumulative_new,
        mode='lines',
        name='Combined Portfolio (Projected)',
        line=dict(color=colors['combined'], dash='dash')  # Dashed line for projection
    ))

    # Art cumulative returns (historical and projected)
    art_cumulative_hist = historical_metrics['Combined Portfolio']['art_cumulative']
    fig.add_trace(go.Scatter(
        x=art_cumulative_hist.index,
        y=art_cumulative_hist,
        mode='lines',
        name='Art Portfolio (Historical)',
        line=dict(color=colors['art'])  # Dark purple for Art
    ))

    art_cumulative_proj = projected_metrics['Combined Portfolio']['art_cumulative']
    art_projected_year_index = art_cumulative_proj.index[art_cumulative_proj.index > pd.Timestamp('2024-06-30')]
    art_cumulative_new = art_cumulative_proj.loc[art_projected_year_index]

    fig.add_trace(go.Scatter(
        x=art_cumulative_new.index,
        y=art_cumulative_new,
        mode='lines',
        name='Art Portfolio (Projected)',
        line=dict(color=colors['art'], dash='dash')  # Dashed line for projection
    ))

    # Market cumulative returns (historical and projected)
    market_cumulative_hist = historical_metrics['Combined Portfolio']['market_cumulative']
    fig.add_trace(go.Scatter(
        x=market_cumulative_hist.index,
        y=market_cumulative_hist,
        mode='lines',
        name='Market Portfolio (Historical)',
        line=dict(color=colors['market'])  # Orange for Market (S&P500)
    ))

    market_cumulative_proj = projected_metrics['Combined Portfolio']['market_cumulative']
    market_projected_year_index = market_cumulative_proj.index[market_cumulative_proj.index > pd.Timestamp('2024-06-30')]
    market_cumulative_new = market_cumulative_proj.loc[market_projected_year_index]

    fig.add_trace(go.Scatter(
        x=market_cumulative_new.index,
        y=market_cumulative_new,
        mode='lines',
        name='Market Portfolio (Projected)',
        line=dict(color=colors['market'], dash='dash')  # Dashed line for projection
    ))

    # Update layout
    fig.update_layout(
        title='Projected Cumulative Returns Including User Predictions',
        xaxis_title='Year',
        yaxis_title='Cumulative Returns',
        legend_title='Portfolio',
        template='plotly_white'
    )

    return fig






def plot_artist_profile(artist_data):
    if 'Artist' not in artist_data.columns:
        st.error("Artist data is not available.")
        return None

    fig = go.Figure()

    # Sentiment Score Plot
    fig.add_trace(go.Scatter(
        x=artist_data['Date of Publication'],
        y=artist_data['Sentiment Score'],
        mode='lines+markers',
        name='Sentiment Score',
        line=dict(color='blue', dash='solid'),
        marker=dict(symbol='circle'),
        yaxis="y1",
        hovertemplate="Date: %{x}<br>Sentiment Score: %{y}<extra></extra>"
    ))

    # Price Estimate Difference Plot
    fig.add_trace(go.Scatter(
        x=artist_data['Date of Publication'],
        y=artist_data['Price_Estimate_Difference'],
        mode='lines+markers',
        name='Price Estimate Difference (%)',
        line=dict(color='red', dash='dot'),
        marker=dict(symbol='triangle-up'),
        yaxis="y2",
        hovertemplate="Date: %{x}<br>Price Estimate Difference: %{y}%<extra></extra>"
    ))

    # Article Count Plot
    if 'Article Count' in artist_data.columns:
        fig.add_trace(go.Scatter(
            x=artist_data['Date of Publication'],
            y=artist_data['Article Count'],
            mode='lines+markers',
            name='Article Count',
            line=dict(color='green', dash='dash'),
            marker=dict(symbol='square'),
            yaxis="y3",
            hovertemplate="Date: %{x}<br>Article Count: %{y}<extra></extra>"
        ))

    # Price USD Plot
    if 'Price_USD' in artist_data.columns:
        fig.add_trace(go.Scatter(
            x=artist_data['Date of Publication'],
            y=artist_data['Price_USD'],
            mode='lines+markers',
            name='Price USD',
            line=dict(color='orange', dash='longdash'),
            marker=dict(symbol='diamond'),
            yaxis="y4",
            hovertemplate="Date: %{x}<br>Price USD: %{y}<extra></extra>"
        ))

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"Profile for {artist_data['Artist'].iloc[0]} (1990-2024)",
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        xaxis=dict(title='Date', showgrid=True, zeroline=False),
        yaxis=dict(
            title='Sentiment Score',
            titlefont=dict(color='blue'),
            tickfont=dict(color='blue'),
            showgrid=True,
            zeroline=False
        ),
        yaxis2=dict(
            title='Price Estimate Difference (%)',
            titlefont=dict(color='red'),
            tickfont=dict(color='red'),
            overlaying='y',
            side='right',
            showgrid=False,
            zeroline=False
        ),
        yaxis3=dict(
            title='Article Count',
            titlefont=dict(color='green'),
            tickfont=dict(color='green'),
            overlaying='y',
            side='left',
            position=0.15,
            showgrid=False,
            zeroline=False
        ),
        yaxis4=dict(
            title='Price USD',
            titlefont=dict(color='orange'),
            tickfont=dict(color='orange'),
            overlaying='y',
            side='right',
            position=0.85,
            showgrid=False,
            zeroline=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified",
        template="plotly_white"
    )

    return fig


######Artprice Index 
@st.cache_data
def load_artprice_index():
    # Assuming you have the Artprice index data in a DataFrame
    data = {
        'Year': list(range(2000, 2024)),
        'Artprice_Index': [1, 1.11, 1.11, 1.22, 1.35, 1.64, 1.79, 2.31, 3.20, 2.81, 2.35, 3.03, 3.55, 3.34, 3.85, 4.52, 4.89, 3.97, 4.60, 4.80, 4.96, 5.05, 6.89, 7.09]
    }
    artprice_index_df = pd.DataFrame(data)
    artprice_index_df['Year'] = pd.to_datetime(artprice_index_df['Year'], format='%Y')
    artprice_index_df.set_index('Year', inplace=True)
    return artprice_index_df['Artprice_Index']


def main():
    df = load_and_preprocess_data('https://raw.githubusercontent.com/simonsim2001/ArtIndex/refs/heads/main/BlueChip.csv')
    cleaned_data = load_data()
    df = extract_dimensions(df)

    st.title("Matis Index")


    # Sidebar for navigation
    view_mode = st.sidebar.selectbox("Select View Mode", ["Overview", "Search", "Portfolio", "Market Opportunities"])

    
    if view_mode == "Overview":

        # Display key summary metrics directly
        st.header("Summary Metrics")
        # Use st.columns to create a compact view
        col1, col2, col3 = st.columns(3)
        with col1:
            unique_artists = df['Artist'].nunique()
            st.metric("Unique Artists", unique_artists)
        with col2:
            unique_auction_houses = df['Auction_House'].nunique()
            st.metric("Unique Auction Houses", unique_auction_houses)
        with col3:
            total_artworks = df.shape[0]
            st.metric("Total Artworks Sold", total_artworks)

        # Price Statistics Across Currencies using st.metric()
        st.header("Price Statistics Across Currencies")
        currencies = ["GBP", "USD", "EUR"]
        for currency in currencies:
            min_price = df[f'Price_{currency}'].min()
            max_price = df[f'Price_{currency}'].max()
            avg_price = df[f'Price_{currency}'].mean()
            st.subheader(f"{currency} Prices")
            # Compact view using columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"Min Price ({currency})", f"${min_price:,.2f}")
            with col2:
                st.metric(f"Max Price ({currency})", f"${max_price:,.2f}")
            with col3:
                st.metric(f"Avg Price ({currency})", f"${avg_price:,.2f}")

        # Unique Artists and Auction Houses in an expander
        st.header("Additional Information")
        with st.expander("Unique Artists and Auction Houses"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Unique Artists")
                st.write(", ".join(df['Artist'].unique()))
            with col2:
                st.subheader("Unique Auction Houses")
                st.write(", ".join(df['Auction_House'].unique()))

        # Performance of Artists
        st.header("Performance of Artists")
        # Move date inputs to the main view
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=pd.to_datetime("2005-01-01"),
                                       min_value=pd.to_datetime("1985-01-01"),
                                       max_value=pd.to_datetime("2024-01-01"))
        with col2:
            end_date = st.date_input("End Date", value=pd.to_datetime("2015-01-01"),
                                     min_value=pd.to_datetime("1985-01-01"),
                                     max_value=pd.to_datetime("2024-01-01"))

        if start_date < end_date:
            start_year, end_year = start_date.year, end_date.year
            df_filtered = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]

            # Calculate average normalized prices
            avg_norm_prices = df_filtered.groupby(['Artist', 'Year'])['Normalized_Price'].mean().reset_index()
            artist_avg_norm_prices = avg_norm_prices.groupby('Artist')['Normalized_Price'].mean().reset_index()
            artist_avg_norm_prices.columns = ['Artist', 'Avg_Normalized_Price']
            avg_norm_prices = pd.merge(avg_norm_prices, artist_avg_norm_prices, on='Artist', how='left')
            avg_norm_prices['Normalized_Price'].fillna(avg_norm_prices['Avg_Normalized_Price'], inplace=True)
            avg_norm_prices_pivot = avg_norm_prices.pivot(index='Artist', columns='Year', values='Normalized_Price').reset_index()

            # Calculate average areas
            avg_areas = df_filtered.groupby(['Artist', 'Year'])['Area'].mean().reset_index()
            avg_areas_pivot = avg_areas.pivot(index='Artist', columns='Year', values='Area').reset_index()

            # Merge prices and areas
            merged_prices = pd.merge(
                avg_norm_prices_pivot.melt(id_vars='Artist', var_name='Year', value_name='Normalized_Price'),
                avg_areas_pivot.melt(id_vars='Artist', var_name='Year', value_name='Area'),
                on=['Artist', 'Year']
            )
            merged_prices['Price_USD'] = merged_prices['Normalized_Price'] * merged_prices['Area']

            # Calculate average prices
            avg_prices = merged_prices.groupby(['Artist', 'Year']).agg({'Price_USD': 'mean'}).reset_index()
            avg_prices_pivot = avg_prices.pivot(index='Artist', columns='Year', values='Price_USD').reset_index()

            # Compute performance metrics
            results = []
            for _, row in avg_prices_pivot.iterrows():
                artist = row['Artist']
                if start_year in row and end_year in row:
                    initial_price, final_price = row[start_year], row[end_year]
                    num_years = end_year - start_year
                    avg_yearly_return = (final_price / initial_price) ** (1/num_years) - 1
                    initial_investment, final_value = 1, 1 * (1 + avg_yearly_return) ** num_years
                    cash_flows = [-initial_investment] + [0] * (num_years - 1) + [final_value]
                    irr = calculate_irr(cash_flows)
                    moic = calculate_moic(initial_investment, final_value)
                    results.append({
                        'Artist': artist,
                        'Avg Initial Price ($K)': initial_price / 1000,
                        'Avg Final Price ($K)': final_price / 1000,
                       # 'Avg Yearly Return (%)': avg_yearly_return * 100,
                        'IRR (%)': irr,
                        'Avg MOIC': moic
                    })

            results_df = pd.DataFrame(results).dropna().sort_values(by='IRR (%)', ascending=False).reset_index(drop=True)
            # Format the percentages
           # results_df['Avg Yearly Return (%)'] = results_df['Avg Yearly Return (%)'].apply(lambda x: f"{x:.2f}%")
            results_df['IRR (%)'] = results_df['IRR (%)'].apply(lambda x: f"{x:.2f}%")
            results_df['Avg MOIC'] = results_df['Avg MOIC'].apply(lambda x: f"{x:.2f}")
            st.subheader("Artists Performance Metrics")
            st.dataframe(results_df)

            # Additional metrics
            df['Price_Estimate_Difference_Percent'] = df['Price_Estimate_Difference'].str.replace(r'[^\d.-]', '', regex=True).astype(float)
            artist_metrics = df.groupby('Artist').agg(
                Total_Sales_USD=('Price_USD', 'sum'),
                Avg_Sale_Price_USD=('Price_USD', 'mean'),
                Highest_Sale_Price_USD=('Price_USD', 'max'),
                Total_Sales_GBP=('Price_GBP', 'sum'),
                Avg_Sale_Price_GBP=('Price_GBP', 'mean'),
                Highest_Sale_Price_GBP=('Price_GBP', 'max'),
                Total_Sales_EUR=('Price_EUR', 'sum'),
                Avg_Sale_Price_EUR=('Price_EUR', 'mean'),
                Highest_Sale_Price_EUR=('Price_EUR', 'max'),
                Artworks_Sold=('Artwork_Title', 'count'),
                Avg_Price_Estimate_Difference=('Price_Estimate_Difference_Percent', 'mean'),
                Auction_Houses=('Auction_House', lambda x: x.unique().tolist())
            ).reset_index()

            # Currency conversion
            conversion_rates = {'GBP': 1.3, 'EUR': 1.1}
            for currency in ['GBP', 'EUR']:
                artist_metrics[f'Total_Sales_{currency}_to_USD'] = artist_metrics[f'Total_Sales_{currency}'] * conversion_rates[currency]
                artist_metrics[f'Avg_Sale_Price_{currency}_to_USD'] = artist_metrics[f'Avg_Sale_Price_{currency}'] * conversion_rates[currency]
                artist_metrics[f'Highest_Sale_Price_{currency}_to_USD'] = artist_metrics[f'Highest_Sale_Price_{currency}'] * conversion_rates[currency]

            artist_metrics['Combined_Total_Sales_USD'] = artist_metrics[['Total_Sales_USD', 'Total_Sales_GBP_to_USD', 'Total_Sales_EUR_to_USD']].sum(axis=1)
            artist_metrics['Combined_Avg_Sale_Price_USD'] = artist_metrics['Combined_Total_Sales_USD'] / artist_metrics['Artworks_Sold']
            artist_metrics['Combined_Highest_Sale_Price_USD'] = artist_metrics[['Highest_Sale_Price_USD', 'Highest_Sale_Price_GBP_to_USD', 'Highest_Sale_Price_EUR_to_USD']].max(axis=1)

            combined_report = artist_metrics[[
                'Artist', 'Combined_Total_Sales_USD', 'Combined_Avg_Sale_Price_USD',
                'Combined_Highest_Sale_Price_USD', 'Artworks_Sold', 'Avg_Price_Estimate_Difference', 'Auction_Houses'
            ]]
            # Format numbers with commas and two decimal places
            combined_report['Combined_Total_Sales_USD'] = combined_report['Combined_Total_Sales_USD'].apply(lambda x: f"${x:,.2f}")
            combined_report['Combined_Avg_Sale_Price_USD'] = combined_report['Combined_Avg_Sale_Price_USD'].apply(lambda x: f"${x:,.2f}")
            combined_report['Combined_Highest_Sale_Price_USD'] = combined_report['Combined_Highest_Sale_Price_USD'].apply(lambda x: f"${x:,.2f}")
            combined_report['Avg_Price_Estimate_Difference'] = combined_report['Avg_Price_Estimate_Difference'].apply(lambda x: f"{x:.2f}%")
            st.subheader("Ranking Report based on Average Sale Price (USD) YTD")
            st.dataframe(combined_report)

            # Transactions and Prices per Year
            # In the Additional Metrics section
            st.subheader("Additional Metrics")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Max Transactions Per Year**")
                transactions_per_artist_per_year = df.groupby(['Artist', 'Year']).size().reset_index(name='Transactions')
                max_transactions = transactions_per_artist_per_year.sort_values(by='Transactions', ascending=False).reset_index(drop=True)
                # Keep 'Year' as integer during merge
                st.dataframe(max_transactions.head(10))

            with col2:
                st.markdown("**Highest Average Price Per Year**")
                avg_price_per_artist_per_year = df.groupby(['Artist', 'Year'])['Price_USD'].mean().reset_index(name='Avg_Sale_Price_USD')
                highest_avg_price = avg_price_per_artist_per_year.sort_values(by='Avg_Sale_Price_USD', ascending=False).reset_index(drop=True)
                # Keep 'Year' as integer during merge
                highest_avg_price['Avg_Sale_Price_USD'] = highest_avg_price['Avg_Sale_Price_USD'].apply(lambda x: f"${x:,.2f}")
                st.dataframe(highest_avg_price.head(10))

            st.markdown("**Combined Metrics Yearly Sorted**")
            # Ensure 'Year' is integer in both DataFrames before merging
            # No need to convert 'Year' to int because it should already be int
            combined_yearly_metrics = pd.merge(
                max_transactions,
                avg_price_per_artist_per_year,
                on=['Artist', 'Year'],
                how='inner'
            ).sort_values(by=['Transactions', 'Avg_Sale_Price_USD'], ascending=[False, False])

            # Now, convert 'Year' to string to display without commas
            combined_yearly_metrics['Year'] = combined_yearly_metrics['Year'].astype(str)
            combined_yearly_metrics['Avg_Sale_Price_USD'] = combined_yearly_metrics['Avg_Sale_Price_USD'].apply(lambda x: f"${x:,.2f}")
            st.dataframe(combined_yearly_metrics.head(10))


        else:
            st.error("End Date must be after Start Date")


    elif view_mode == "Search":
        # Search Mode
        st.title("Artist Search and Analysis")

        # Create a list of unique artists for the autocomplete feature
        unique_artists = df['Artist'].unique().tolist()

        # Use st.selectbox for the artist name input with autocomplete
        artist_name = st.selectbox("Select Artist", unique_artists)

        if artist_name:
            # Data filtering based on artist name
            artist_data = df[df['Artist'] == artist_name]

            if not artist_data.empty:
                df_filtered = artist_data.copy()  # Initialize df_filtered

                # Sidebar inputs for date range and artwork parameters
                with st.sidebar:
                    st.header("Filters")
                    start_date = st.date_input(
                        "Start Date",
                        value=pd.to_datetime("2005-01-01"),
                        min_value=pd.to_datetime("1985-01-01"),
                        max_value=pd.to_datetime("2024-01-01")
                    )
                    end_date = st.date_input(
                        "End Date",
                        value=pd.to_datetime("2015-01-01"),
                        min_value=pd.to_datetime("1985-01-01"),
                        max_value=pd.to_datetime("2024-01-01")
                    )

                    if start_date >= end_date:
                        st.error("End Date must be after Start Date")

                    # Add input fields for artwork parameters
                    st.subheader("Artwork Parameters")
                    target_width = st.number_input("Target Width (cm)", value=40.5)
                    target_height = st.number_input("Target Height (cm)", value=40.5)
                    tolerance_cm = st.number_input("Tolerance (cm)", value=5.0)

                    # Add input fields for creation year range
                    st.subheader("Creation Year Range")
                    start_creation_year = st.number_input(
                        "Start Creation Year", value=1900, min_value=1700, max_value=2024
                    )
                    end_creation_year = st.number_input(
                        "End Creation Year", value=2024, min_value=1700, max_value=2024
                    )

                    # Add input fields for colours
                    st.subheader("Colour Parameters")
                    available_colours = df['Dominant_Colour'].dropna().unique().tolist()
                    if 'Unknown' in available_colours:
                        available_colours.remove('Unknown')
                    selected_colours = st.multiselect(
                        "Select Colours", options=available_colours, default=available_colours
                    )

                    # Apply colour filter only if colours are selected
                    if selected_colours:
                        df_filtered = df_filtered[df_filtered['Dominant_Colour'].isin(selected_colours)]

                # Apply creation year filter
                if start_creation_year <= end_creation_year:
                    df_filtered = df_filtered[
                        (df_filtered['Creation_Year'] >= start_creation_year) &
                        (df_filtered['Creation_Year'] <= end_creation_year)
                    ]
                else:
                    st.error("End Creation Year must be after Start Creation Year")

                if start_date < end_date:
                    df_filtered = df_filtered[
                        (df_filtered['Year'] >= start_date.year) &
                        (df_filtered['Year'] <= end_date.year)
                    ]

                    # Filter data based on artwork parameters
                    filtered_data = df_filtered[
                        (abs(df_filtered['Width_cm'] - target_width) <= tolerance_cm) &
                        (abs(df_filtered['Height_cm'] - target_height) <= tolerance_cm)
                    ]

                    # Organize content into tabs
                    tab1, tab2, tab3 = st.tabs([
                        "Artist Overview", "Price Analysis", "Comparable Artworks"
                    ])

                    with tab1:
                        st.subheader(f"Overview of {artist_name}")
                        # Compact view using columns
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            total_artworks_sold = artist_data.shape[0]
                            st.metric("Total Artworks Sold", total_artworks_sold)
                        with col2:
                            total_sales = artist_data['Price_USD'].sum()
                            st.metric("Total Sales (USD)", f"${total_sales:,.2f}")
                        with col3:
                            avg_price = artist_data['Price_USD'].mean()
                            st.metric("Average Sale Price (USD)", f"${avg_price:,.2f}")

                        st.subheader("Annual Turnover")
                        annual_turnover = artist_data.groupby(
                            artist_data['Auction_Date'].dt.year
                        )['Price_USD'].sum().reset_index(name='Annual_Turnover_USD')
                        annual_turnover.rename(columns={'Auction_Date': 'Year'}, inplace=True)
                        turnover_fig = px.bar(
                            annual_turnover,
                            x='Year',
                            y='Annual_Turnover_USD',
                            title='Annual Turnover (USD)'
                        )
                        st.plotly_chart(turnover_fig, use_container_width=True)

                    with tab2:
                        st.subheader("Price Analysis")
                        avg_price_per_year = df_filtered.groupby(
                            df_filtered['Auction_Date'].dt.year
                        )['Price_USD'].mean().reset_index(name='Average_Sale_Price_USD')
                        avg_price_per_year.rename(columns={'Auction_Date': 'Year'}, inplace=True)
                        highest_price_per_year = df_filtered.groupby(
                            df_filtered['Auction_Date'].dt.year
                        )['Price_USD'].max().reset_index(name='Highest_Sale_Price_USD')
                        highest_price_per_year.rename(columns={'Auction_Date': 'Year'}, inplace=True)

                        col1, col2 = st.columns(2)
                        with col1:
                            avg_price_fig = px.line(
                                avg_price_per_year,
                                x='Year',
                                y='Average_Sale_Price_USD',
                                title='Average Sale Price per Year'
                            )
                            st.plotly_chart(avg_price_fig, use_container_width=True)
                        with col2:
                            highest_price_fig = px.line(
                                highest_price_per_year,
                                x='Year',
                                y='Highest_Sale_Price_USD',
                                title='Highest Sale Price per Year'
                            )
                            st.plotly_chart(highest_price_fig, use_container_width=True)

                    with tab3:
                        st.subheader("Comparable Artworks")
                        if not filtered_data.empty:
                            # Select relevant columns
                            comparable_artworks = filtered_data[[
                                'Image_URL', 'Artwork_Title', 'Artist', 'Auction_Date',
                                'Creation_Year', 'Price_USD', 'Width_cm', 'Height_cm',
                                'Dominant_Colour', 'Auction_House'
                            ]].copy()

                            # Format date columns
                            comparable_artworks['Auction_Date'] = comparable_artworks['Auction_Date'].dt.date

                            # Format prices
                            comparable_artworks['Price_USD'] = comparable_artworks['Price_USD'].apply(
                                lambda x: f"${x:,.2f}"
                            )

                            # Display the table with images
                            def generate_html_table_with_images(data):
                                html = '''
                                <html>
                                <head>
                                    <meta charset="UTF-8">
                                    <style>
                                        body {
                                            font-family: Arial, sans-serif;
                                            color: #333333;
                                        }
                                        table {
                                            width: 100%;
                                            border-collapse: collapse;
                                        }
                                        th, td {
                                            border: 1px solid #dddddd;
                                            text-align: left;
                                            padding: 8px;
                                            vertical-align: top;
                                        }
                                        th {
                                            background-color: #f9f9f9;
                                            font-weight: normal;
                                        }
                                        img {
                                            max-width: 100px;
                                            height: auto;
                                            display: block;
                                            margin: 0 auto;
                                        }
                                        tr:nth-child(even) {
                                            background-color: #ffffff;
                                        }
                                        tr:nth-child(odd) {
                                            background-color: #f5f5f5;
                                        }
                                    </style>
                                </head>
                                <body>
                                    <table>
                                '''
                                # Add table header
                                html += '<tr>'
                                for column in data.columns:
                                    html += f'<th>{column}</th>'
                                html += '</tr>'

                                # Add table rows
                                for _, row in data.iterrows():
                                    html += '<tr>'
                                    for column in data.columns:
                                        if column == 'Image_URL':
                                            html += f'<td><img src="{row[column]}" alt="{row["Artwork_Title"]}"></td>'
                                        else:
                                            html += f'<td>{row[column]}</td>'
                                    html += '</tr>'
                                html += '''
                                    </table>
                                </body>
                                </html>
                                '''
                                return html

                            # Generate HTML table with images
                            html_content = generate_html_table_with_images(comparable_artworks)

                            # Display the table in Streamlit
                            st.components.v1.html(html_content, height=600, scrolling=True)

                            # Provide download button for the HTML table
                            st.download_button(
                                label="Download Table with Images (HTML)",
                                data=html_content,
                                file_name='comparable_artworks.html',
                                mime='text/html',
                            )
                        else:
                            st.write("No comparable artworks found with the specified parameters.")

                else:
                    st.error("End Date must be after Start Date")
            else:
                st.error(f"No data found for artist: {artist_name}")



    
    elif view_mode == "Market Opportunities":
        st.title("Market Opportunities")

        # Prepare the data
        df_ml = df.copy()

        # Data Preprocessing
        required_columns = ['Price_USD', 'Dimensions', 'Creation_Date', 'Artist', 'Auction_Date', 'Image_URL', 'Auction_House']
        df_ml = df_ml.dropna(subset=required_columns)

        # Convert 'Creation_Date' to integer year and 'Auction_Date' to datetime
        df_ml['Creation_Year'] = df_ml['Creation_Date'].astype(int)
        df_ml['Auction_Date'] = pd.to_datetime(df_ml['Auction_Date'], errors='coerce')

        # Extract numerical dimensions and calculate area
        df_ml[['Width_cm', 'Height_cm']] = df_ml['Dimensions'].str.extract(r'(\d+\.?\d*)\s*x\s*(\d+\.?\d*)').astype(float)
        df_ml['Area'] = df_ml['Width_cm'] * df_ml['Height_cm']
        df_ml = df_ml.dropna(subset=['Width_cm', 'Height_cm', 'Area', 'Price_USD', 'Creation_Year', 'Auction_Date'])

        # Define consistent number of categories (3 size ranges)
        num_categories = 3
        labels = ['Small', 'Medium', 'Large']

        # Select an artist
        artists = df_ml['Artist'].unique()
        selected_artist = st.selectbox("Select an artist to view market trends", artists)

        if selected_artist:
            # Filter data for the selected artist
            artist_filtered_data = df_ml[df_ml['Artist'] == selected_artist]

            # Define artist-specific size ranges based on average sizes
            min_area = artist_filtered_data['Area'].min()
            max_area = artist_filtered_data['Area'].max()
            bins = pd.cut([min_area, max_area], num_categories, retbins=True)[1]
            artist_filtered_data['Size_Category'] = pd.cut(artist_filtered_data['Area'], bins=bins, labels=labels, include_lowest=True)

            # Display size category ranges for the selected artist
            st.subheader(f"Size Category Ranges for {selected_artist}")
            size_ranges = artist_filtered_data.groupby('Size_Category')['Area'].agg(['min', 'max']).reindex(labels)
            size_ranges['min'] = size_ranges['min'].round(2)
            size_ranges['max'] = size_ranges['max'].round(2)
            st.table(size_ranges)

            # User selects size category
            selected_size = st.selectbox("Choose Size Category", labels)
            artist_filtered_data = artist_filtered_data[artist_filtered_data['Size_Category'] == selected_size]

            # Calculate average price per creation year and YoY % change
            artist_yearly_prices = artist_filtered_data.groupby('Creation_Year')['Price_USD'].mean().reset_index()
            artist_yearly_prices['Price_Change_Percent'] = artist_yearly_prices['Price_USD'].pct_change() * 100

            # Provide indicators on data recency
            st.subheader("Data Recency Indicator")
            latest_auction_date = artist_filtered_data['Auction_Date'].max()
            st.write(f"**Most Recent Auction Date for {selected_artist}:** {latest_auction_date.date()}")

            # Identify years with data in the last 5 years
            current_year = pd.Timestamp.now().year
            recent_data_threshold = pd.Timestamp.now() - pd.DateOffset(years=5)
            years_with_recent_data = artist_filtered_data[artist_filtered_data['Auction_Date'] >= recent_data_threshold]['Creation_Year'].unique()
            st.write(f"**Years with Data in the Last 5 Years:** {', '.join(map(str, sorted(years_with_recent_data)))}")

            # Allow user to select multiple years for comparison
            years_available = artist_yearly_prices['Creation_Year'].unique()
            selected_years = st.multiselect("Select years for comparison", years_available, default=years_available[:3])

            # Check if selected years have recent data
            for year in selected_years:
                year_data = artist_filtered_data[artist_filtered_data['Creation_Year'] == year]
                if not year_data.empty:
                    most_recent_transaction = year_data['Auction_Date'].max()
                    days_since_transaction = (pd.Timestamp.now() - most_recent_transaction).days
                    if days_since_transaction > 365 * 5:
                        st.warning(f"Data for year {year} may be outdated (last transaction on {most_recent_transaction.date()}).")
                else:
                    st.warning(f"No auction data available for year {year}.")

            if selected_years:
                # Proceed with comparison
                # Bar chart for year price comparison
                st.subheader("Average Price Comparison Across Selected Years")
                year_price_data = artist_yearly_prices[artist_yearly_prices['Creation_Year'].isin(selected_years)]
                import altair as alt
                year_price_chart = alt.Chart(year_price_data).mark_bar().encode(
                    x=alt.X('Creation_Year:O', title='Creation Year'),
                    y=alt.Y('Price_USD', title='Average Price (USD)')
                ).properties(
                    width=700,
                    height=400
                )
                st.altair_chart(year_price_chart)

                # Side-by-side comparison of selected years
                st.subheader("Artwork Comparison by Year")
                comparison_columns = st.columns(len(selected_years))

                for idx, year in enumerate(selected_years):
                    with comparison_columns[idx]:
                        st.write(f"### Year {year}")

                        # Retrieve examples for the selected year in the filtered data
                        examples_year = artist_filtered_data[artist_filtered_data['Creation_Year'] == year]
                        
                        if not examples_year.empty:
                            # Sort by recent auction dates and retrieve up to 3 examples
                            examples_year = examples_year.sort_values(by='Auction_Date', ascending=False).head(3)

                            # Get average price for the year from artist_yearly_prices
                            avg_price = artist_yearly_prices.loc[artist_yearly_prices['Creation_Year'] == year, 'Price_USD'].values[0]
                            st.write(f"**Average Price:** ${avg_price:,.2f}")


                            for _, row in examples_year.iterrows():
                                st.image(row['Image_URL'], caption=row['Artwork_Title'], width=150)
                                st.write(f"**Title:** {row['Artwork_Title']}")
                                st.write(f"**Auction Date:** {row['Auction_Date'].date()}")
                                st.write(f"**Auction House:** {row['Auction_House']}")
                                st.write(f"**Price USD:** ${row['Price_USD']:,.2f}")
                                st.write(f"**Dimensions:** {row['Width_cm']} x {row['Height_cm']} cm")
                                st.write("---")
                        else:
                            st.write("No data available for this year.")



    # Portfolio and Transparency Section
    elif view_mode == "Portfolio":
        # Retrieve the user-selected parameters
        (
            art_rolling_window, art_allocation,
            risk_free_rate, start_date, end_date, selected_ticker_symbols, selected_artists
        ) = get_portfolio_parameters(df)

        if not selected_artists:
            st.warning("Please select at least one artist to proceed.")
        else:
            # Filter the art data based on selected artists
            filtered_art_data = cleaned_data[cleaned_data['Artist'].isin(selected_artists)].copy()
            # Ensure index is sorted
            filtered_art_data.sort_index(inplace=True)

            if filtered_art_data.empty:
                st.error("No data available for the selected artist(s). Please select different artist(s).")
            else:
                # Proceed with calculations using filtered_art_data
                # Calculate rolling average for the filtered art data
                overall_rolling_average = calculate_rolling_average(filtered_art_data, art_rolling_window)

                # Calculate art returns
                art_returns = overall_rolling_average.pct_change().dropna()

                # Load the Artprice Blue Chip Index
                show_artprice_index = st.sidebar.checkbox("Include Artprice Blue Chip Index")
                artprice_index = load_artprice_index() if show_artprice_index else None

                # Load market data based on selected tickers and date range
                market_data = load_market_data(selected_ticker_symbols, start_date, end_date)

                if market_data:
                    # Ensure market data is sorted and calculate returns
                    market_returns = {}
                    for ticker, data in market_data.items():
                        data.sort_index(inplace=True)
                        market_returns[ticker] = data.pct_change().dropna()

                    # Align the data
                    common_index = art_returns.index
                    for ticker in selected_ticker_symbols:
                        common_index = common_index.intersection(market_returns[ticker].index)

                    art_returns = art_returns.loc[common_index]
                    market_returns = {ticker: returns.loc[common_index] for ticker, returns in market_returns.items()}

                    if not art_returns.empty:
                        # Calculate combined portfolio metrics
                        metrics = calculate_combined_portfolio_metrics(
                            art_returns, market_returns, art_allocation, risk_free_rate
                        )

                        # Calculate annual returns for each portfolio
                        art_annual_returns = metrics['Art Portfolio']['returns'].resample('Y').apply(lambda x: (1 + x).prod() - 1)
                        market_annual_returns = metrics['Market Portfolio']['returns'].resample('Y').apply(lambda x: (1 + x).prod() - 1)
                        combined_annual_returns = metrics['Combined Portfolio']['returns'].resample('Y').apply(lambda x: (1 + x).prod() - 1)

                        st.header('Portfolio Performance')

                        tab1, tab2, tab3, tab4 = st.tabs([
                            "Cumulative Returns", "Annual Returns", "Rolling Correlation", "Efficient Frontier"
                        ])

                        # **Tab 1: Cumulative Returns**
                        with tab1:
                            fig1 = plot_cumulative_returns(metrics, artprice_index=artprice_index)

                            # Adjust figure layout
                            fig1.update_layout(
                                xaxis_title='Date',
                                yaxis_title='Cumulative Returns',
                                xaxis=dict(
                                    showline=True,
                                    linewidth=1,  # Finer axis lines
                                    linecolor='black',
                                    mirror=False,
                                    showgrid=False     # Disable vertical grid lines
                                ),
                                yaxis=dict(
                                    showline=True,
                                    linewidth=1,  # Finer axis lines
                                    linecolor='black',
                                    mirror=False,
                                    showgrid=True,       # Add horizontal grid lines
                                    gridwidth=0.5,
                                    gridcolor='lightgrey'
                                ),
                                font=dict(size=18),
                                margin=dict(l=50, r=50, t=50, b=100),  # Increase bottom margin for legend
                                legend=dict(
                                    orientation='h',
                                    yanchor='top',
                                    y=-0.2,
                                    xanchor='center',
                                    x=0.5
                                )
                            )

                            st.plotly_chart(fig1, use_container_width=True)

                            # Add download button for cumulative returns graph
                            buffer = io.BytesIO()
                            fig1.write_image(buffer, format='pdf', width=1600, height=900, scale=2)
                            buffer.seek(0)
                            st.download_button(
                                label="Download Cumulative Returns as PDF",
                                data=buffer,
                                file_name='cumulative_returns.pdf',
                                mime='application/pdf'
                            )

                        # **Tab 2: Annual Returns**
                        with tab2:
                            fig2 = plot_annual_returns(art_annual_returns, market_annual_returns, combined_annual_returns)

                            # Adjust figure layout
                            fig2.update_layout(
                                xaxis_title='Year',
                                yaxis_title='Annual Returns (%)',
                                xaxis=dict(
                                    showline=True,
                                    linewidth=1,  # Finer axis lines
                                    linecolor='black',
                                    mirror=False,
                                    showgrid=False     # Disable vertical grid lines
                                ),
                                yaxis=dict(
                                    showline=True,
                                    linewidth=1,  # Finer axis lines
                                    linecolor='black',
                                    mirror=False,
                                    showgrid=True,       # Add horizontal grid lines
                                    gridwidth=0.5,
                                    gridcolor='lightgrey'
                                ),
                                font=dict(size=18),
                                margin=dict(l=50, r=50, t=50, b=100),
                                legend=dict(
                                    orientation='h',
                                    yanchor='top',
                                    y=-0.2,
                                    xanchor='center',
                                    x=0.5
                                )
                            )

                            st.plotly_chart(fig2, use_container_width=True)

                            # Add download button for annual returns graph
                            buffer = io.BytesIO()
                            fig2.write_image(buffer, format='pdf', width=1600, height=900, scale=2)
                            buffer.seek(0)
                            st.download_button(
                                label="Download Annual Returns as PDF",
                                data=buffer,
                                file_name='annual_returns.pdf',
                                mime='application/pdf'
                            )

                        # **Tab 3: Rolling Correlation**
                        with tab3:
                            st.subheader("Rolling Correlation")

                            # **Step 1: Extract art_returns and market_returns from metrics**
                            art_returns_series = metrics['Combined Portfolio']['art_returns']
                            market_returns_series = metrics['Combined Portfolio']['market_returns']

                            # **Step 2: Define the rolling window (make sure it matches the one used in plotting)**
                            window = 12  # Or use the window size you've defined elsewhere

                            # **Step 3: Calculate the rolling correlation**
                            rolling_correlation = art_returns_series.rolling(window=window).corr(market_returns_series).dropna()

                            # **Step 4: Add date inputs for the average correlation calculation**
                            st.markdown("### Average Correlation Calculation Period")
                            col1, col2 = st.columns(2)
                            with col1:
                                avg_corr_start_date = st.date_input(
                                    "Start Date",
                                    value=rolling_correlation.index.min().date(),
                                    min_value=rolling_correlation.index.min().date(),
                                    max_value=rolling_correlation.index.max().date(),
                                    key="avg_corr_start_date"
                                )
                            with col2:
                                avg_corr_end_date = st.date_input(
                                    "End Date",
                                    value=rolling_correlation.index.max().date(),
                                    min_value=rolling_correlation.index.min().date(),
                                    max_value=rolling_correlation.index.max().date(),
                                    key="avg_corr_end_date"
                                )

                            # **Step 5: Filter the rolling correlation series**
                            filtered_rolling_corr = rolling_correlation.loc[
                                (rolling_correlation.index >= pd.to_datetime(avg_corr_start_date)) &
                                (rolling_correlation.index <= pd.to_datetime(avg_corr_end_date))
                            ]

                            # **Step 6: Calculate the average correlation**
                            if not filtered_rolling_corr.empty:
                                average_correlation = filtered_rolling_corr.mean()
                                st.write(f"**Average Correlation from {avg_corr_start_date} to {avg_corr_end_date}:** {average_correlation:.4f}")
                            else:
                                st.warning("No data available in the selected date range.")

                            # **Plot the rolling correlation**
                            fig3 = plot_rolling_correlation(metrics)

                            # Adjust figure layout
                            fig3.update_layout(
                                xaxis_title='Date',
                                yaxis_title='Correlation',
                                xaxis=dict(
                                    showline=True,
                                    linewidth=1,  # Finer axis lines
                                    linecolor='black',
                                    mirror=False,
                                    showgrid=False     # Disable vertical grid lines
                                ),
                                yaxis=dict(
                                    showline=True,
                                    linewidth=1,  # Finer axis lines
                                    linecolor='black',
                                    mirror=False,
                                    showgrid=True,       # Add horizontal grid lines
                                    gridwidth=0.5,
                                    gridcolor='lightgrey'
                                ),
                                font=dict(size=18),
                                margin=dict(l=50, r=50, t=50, b=100),
                                legend=dict(
                                    orientation='h',
                                    yanchor='top',
                                    y=-0.2,
                                    xanchor='center',
                                    x=0.5
                                )
                            )

                            st.plotly_chart(fig3, use_container_width=True)

                            # Add download button for rolling correlation graph
                            buffer = io.BytesIO()
                            fig3.write_image(buffer, format='pdf', width=1600, height=900, scale=2)
                            buffer.seek(0)
                            st.download_button(
                                label="Download Rolling Correlation as PDF",
                                data=buffer,
                                file_name='rolling_correlation.pdf',
                                mime='application/pdf'
                            )

                        # **Tab 4: Efficient Frontier**
                        with tab4:
                            fig4 = plot_efficient_frontier(metrics)

                            # Adjust figure layout
                            fig4.update_layout(
                                xaxis_title='Portfolio Volatility (%)',
                                yaxis_title='Portfolio Return (%)',
                                xaxis=dict(
                                    showline=True,
                                    linewidth=1,  # Finer axis lines
                                    linecolor='black',
                                    mirror=False,
                                    showgrid=False     # Disable vertical grid lines
                                ),
                                yaxis=dict(
                                    showline=True,
                                    linewidth=1,  # Finer axis lines
                                    linecolor='black',
                                    mirror=False,
                                    showgrid=True,       # Add horizontal grid lines
                                    gridwidth=0.5,
                                    gridcolor='lightgrey'
                                ),
                                font=dict(size=18),
                                margin=dict(l=50, r=50, t=50, b=100),
                                legend=dict(
                                    orientation='h',
                                    yanchor='top',
                                    y=-0.2,
                                    xanchor='center',
                                    x=0.5
                                )
                            )

                            st.plotly_chart(fig4, use_container_width=True)

                            # Add download button for efficient frontier graph
                            buffer = io.BytesIO()
                            fig4.write_image(buffer, format='pdf', width=1600, height=900, scale=2)
                            buffer.seek(0)
                            st.download_button(
                                label="Download Efficient Frontier as PDF",
                                data=buffer,
                                file_name='efficient_frontier.pdf',
                                mime='application/pdf'
                            )

                        st.header('Performance Metrics')

                        # Create a DataFrame to display metrics for all portfolios
                        metrics_df = pd.DataFrame({
                            'Metric': [
                                'Final Cumulative Return',
                                'Annualized Return',
                                'Volatility',
                                'Sharpe Ratio',
                                'Max Drawdown'
                            ],
                            'Art Portfolio': [
                                f"{metrics['Art Portfolio']['cumulative_return'].iloc[-1]:.2%}",
                                f"{metrics['Art Portfolio']['annualized_return']:.2%}",
                                f"{metrics['Art Portfolio']['volatility']:.2%}",
                                f"{metrics['Art Portfolio']['sharpe_ratio']:.2f}",
                                f"{metrics['Art Portfolio']['max_drawdown']:.2%}"
                            ],
                            'Market Portfolio': [
                                f"{metrics['Market Portfolio']['cumulative_return'].iloc[-1]:.2%}",
                                f"{metrics['Market Portfolio']['annualized_return']:.2%}",
                                f"{metrics['Market Portfolio']['volatility']:.2%}",
                                f"{metrics['Market Portfolio']['sharpe_ratio']:.2f}",
                                f"{metrics['Market Portfolio']['max_drawdown']:.2%}"
                            ],
                            'Combined Portfolio': [
                                f"{metrics['Combined Portfolio']['cumulative_return'].iloc[-1]:.2%}",
                                f"{metrics['Combined Portfolio']['annualized_return']:.2%}",
                                f"{metrics['Combined Portfolio']['volatility']:.2%}",
                                f"{metrics['Combined Portfolio']['sharpe_ratio']:.2f}",
                                f"{metrics['Combined Portfolio']['max_drawdown']:.2%}"
                            ]
                        })

                        st.table(metrics_df)

                        # Get the last N years based on user selection or default to last 3 years
                        last_n_years = st.sidebar.number_input('Number of Recent Years for Comparison', min_value=1, max_value=10, value=3, step=1)

                        # Get the last N years of data
                        recent_years = art_annual_returns.index[-last_n_years:]

                        # Prepare DataFrame for comparison
                        comparison_df = pd.DataFrame({
                            'Year': recent_years.year,
                            'Art Annual Return': art_annual_returns.loc[recent_years].values,
                            'Market Annual Return': market_annual_returns.loc[recent_years].values,
                            'Combined Annual Return': combined_annual_returns.loc[recent_years].values
                        })

                        # Calculate outperformance
                        comparison_df['Art vs Market'] = art_annual_returns.loc[recent_years].values - market_annual_returns.loc[recent_years].values
                        comparison_df['Combined vs Market'] = combined_annual_returns.loc[recent_years].values - market_annual_returns.loc[recent_years].values

                        # Format percentages
                        for col in ['Art Annual Return', 'Market Annual Return', 'Combined Annual Return', 'Art vs Market', 'Combined vs Market']:
                            comparison_df[col] = comparison_df[col].apply(lambda x: f"{x:.2%}")

                        st.subheader(f'Outperformance Over the Last {last_n_years} Years')
                        st.dataframe(comparison_df)

                        # Calculate artist weights before using them
                        artist_weights = filtered_art_data.groupby(['Artist', filtered_art_data.index.year])['Price_USD'].sum() / filtered_art_data.groupby(filtered_art_data.index.year)['Price_USD'].sum()

                        # Additional Information Section
                        st.header('Additional Information')

                        tab1, tab2, tab3 = st.tabs(["Artist Weights", "Art Returns", "Debug Information"])

                        with tab1:
                            st.subheader('Artist Weights')
                            artist_weights_df = artist_weights.reset_index()
                            artist_weights_df.columns = ['Artist', 'Year', 'Weight']
                            # Convert 'Year' to string to prevent commas
                            artist_weights_df['Year'] = artist_weights_df['Year'].astype(str)
                            artist_weights_pivot = artist_weights_df.pivot(index='Year', columns='Artist', values='Weight').fillna(0).reset_index()
                            # Format all columns except 'Year' as percentages
                            st.dataframe(
                                artist_weights_pivot.style.format(
                                    {column: "{:.2%}" for column in artist_weights_pivot.columns if column != 'Year'}
                                )
                            )

                        with tab2:
                            st.subheader('Art Returns')
                            st.dataframe(art_returns.head())

                        with tab3:
                            st.subheader('Debug Information')
                            st.write("**Selected Artists:**", selected_artists)
                            st.write("**Combined Market Returns:**")
                            for ticker in selected_ticker_symbols:
                                st.write(f"**{ticker} Returns:**")
                                st.dataframe(market_returns[ticker].head())

                    else:
                        st.error("No data points to calculate returns. Please check the alignment of the series.")
                else:
                    st.error("Failed to load market data. Please try again later.")


                        

if __name__ == "__main__":
    if check_password():
        main()
