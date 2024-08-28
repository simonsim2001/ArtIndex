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

# Set page configuration at the very top
st.set_page_config(layout="wide")

# Load and preprocess the data
@st.cache_data
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
    return df

@st.cache_data
def load_data():
    data = pd.read_csv('https://raw.githubusercontent.com/simonsim2001/ArtIndex/main/BlueChip_Consolidated%20(2).csv')
    data['Auction_Date'] = pd.to_datetime(data['Auction_Date'], errors='coerce')
    data['Price_USD'] = pd.to_numeric(data['Price_USD'], errors='coerce')
    cleaned_data = data.dropna(subset=['Auction_Date', 'Price_USD'])
    cleaned_data.set_index('Auction_Date', inplace=True)
    cleaned_data.sort_index(inplace=True)
    return cleaned_data


def extract_dimensions(df):
    df['Width_cm'] = df['Dimensions'].apply(lambda x: float(re.search(r'(\d+\.\d+) cm', x).group(1)) if re.search(r'(\d+\.\d+) cm', x) else None)
    df['Height_cm'] = df['Dimensions'].apply(lambda x: float(re.search(r'x (\d+\.\d+) cm', x).group(1)) if re.search(r'x (\d+\.\d+) cm', x) else None)
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

def calculate_combined_portfolio_metrics(art_returns, market_returns_dict, art_allocation, risk_free_rate):
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
    
    # Now calculate the combined portfolio with art and the averaged market returns
    market_allocation = 1 - art_allocation
    combined_returns = (art_returns * art_allocation) + (combined_market_returns * market_allocation)
    
    # Ensure the returns are aligned (outer join and fill with zeros)
    art_returns, combined_returns = art_returns.align(combined_returns, join='outer', fill_value=0)

    excess_returns = combined_returns - risk_free_rate
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(12)  # Monthly data to annualize

    combined_volatility = np.std(combined_returns) * np.sqrt(12)  # Annualized volatility
    art_volatility = np.std(art_returns) * np.sqrt(12)
    market_volatility = np.std(combined_market_returns) * np.sqrt(12)

    art_cumulative = (1 + art_returns).cumprod()
    market_cumulative = (1 + combined_market_returns).cumprod()
    combined_cumulative = (1 + combined_returns).cumprod()

    annualized_return = (combined_cumulative.iloc[-1] ** (1 / (len(combined_cumulative) / 12))) - 1
    max_drawdown = (combined_cumulative.cummax() - combined_cumulative).max()
    
    metrics['Combined Portfolio'] = {
        'sharpe_ratio': sharpe_ratio,
        'combined_volatility': combined_volatility,
        'art_volatility': art_volatility,
        'market_volatility': market_volatility,
        'art_cumulative': art_cumulative,
        'market_cumulative': market_cumulative,
        'combined_cumulative': combined_cumulative,
        'art_returns': art_returns,
        'market_returns': combined_market_returns,
        'combined_returns': combined_returns,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown
    }
    
    return metrics

def plot_cumulative_returns(metrics, artprice_index=None):
    fig = go.Figure()
    
    # Plot art cumulative returns
    art_cumulative = metrics['Combined Portfolio']['art_cumulative']
    fig.add_trace(go.Scatter(x=art_cumulative.index, y=art_cumulative, mode='lines', name='Art', line=dict(color='blue')))

    # Plot Artprice Blue Chip Index if provided
    if artprice_index is not None:
        fig.add_trace(go.Scatter(x=artprice_index.index, y=artprice_index, mode='lines', name='Artprice Blue Chip Index', line=dict(color='purple', dash='dash')))
    
    # Plot combined market cumulative returns
    market_cumulative = metrics['Combined Portfolio']['market_cumulative']
    combined_cumulative = metrics['Combined Portfolio']['combined_cumulative']

    fig.add_trace(go.Scatter(x=market_cumulative.index, y=market_cumulative, mode='lines', name='Combined Market', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=combined_cumulative.index, y=combined_cumulative, mode='lines', name='Combined Portfolio', line=dict(color='green')))
    
    fig.update_layout(
        title='Cumulative Returns',
        xaxis_title='Year',
        yaxis_title='Cumulative Returns',
        legend_title='Portfolio',
        template='plotly_white'
    )
    
    return fig


def plot_annual_returns(metrics):
    fig = go.Figure()
    for ticker, data in metrics.items():
        annual_returns = pd.DataFrame({
            'Art': data['art_returns'].resample('Y').apply(lambda x: (1 + x).prod() - 1),
            ticker: data['market_returns'].resample('Y').apply(lambda x: (1 + x).prod() - 1),
            'Combined': data['combined_returns'].resample('Y').apply(lambda x: (1 + x).prod() - 1)
        })
        for column in annual_returns.columns:
            fig.add_trace(go.Bar(x=annual_returns.index, y=annual_returns[column], name=column))
    fig.update_layout(title='Annual Returns', xaxis_title='Year', yaxis_title='Annual Return', barmode='group', legend_title='Portfolio')
    return fig

def plot_rolling_correlation(metrics, window=12):
    fig = go.Figure()

    for ticker, data in metrics.items():
        # Calculate rolling correlation and handle NaN values
        correlation = data['art_returns'].rolling(window=window).corr(data['market_returns']).dropna()

        # Add a trace for each market asset's correlation with art
        fig.add_trace(go.Scatter(
            x=correlation.index, 
            y=correlation, 
            mode='lines', 
            name=f'{ticker} Correlation',
            line=dict(width=2)
        ))

    # Add a horizontal line at y=0 for reference
    fig.add_shape(type="line", x0=correlation.index.min(), x1=correlation.index.max(), 
                  y0=0, y1=0, line=dict(color="gray", dash="dash"))

    # Customize layout
    fig.update_layout(
        title=f'{window}-Month Rolling Correlation between Art and Markets',
        xaxis_title='Year',
        yaxis_title='Correlation',
        xaxis=dict(
            showgrid=True,
            showline=True,
            linecolor='black',
            mirror=True
        ),
        yaxis=dict(
            showgrid=True,
            showline=True,
            linecolor='black',
            mirror=True,
            range=[-1, 1]  # Setting the range to [-1, 1] for correlation
        ),
        template='plotly_white',
        hovermode='x unified',
        legend_title='Ticker',
        height=600
    )

    return fig


def plot_efficient_frontier(metrics):
    fig = go.Figure()
    for ticker, data in metrics.items():
        portfolios = pd.DataFrame({
            'Art_Weight': np.arange(0, 1.01, 0.01),
            'Market_Weight': np.arange(1, -0.01, -0.01)
        })
        
        portfolios['Returns'] = portfolios.apply(lambda row: np.mean(data['art_returns'] * row['Art_Weight'] + data['market_returns'] * row['Market_Weight']), axis=1)
        portfolios['Volatility'] = portfolios.apply(lambda row: np.std(data['art_returns'] * row['Art_Weight'] + data['market_returns'] * row['Market_Weight']), axis=1)
        
        fig.add_trace(go.Scatter(x=portfolios['Volatility'], y=portfolios['Returns'], mode='markers', 
                                 marker=dict(size=5, color=portfolios['Art_Weight'], colorscale='Viridis', showscale=True),
                                 text=portfolios['Art_Weight'].apply(lambda x: f'Art Weight: {x:.2f}<br>Market Weight: {1-x:.2f}'),
                                 hoverinfo='text+x+y', name=f'{ticker} Frontier'))
    fig.update_layout(title='Efficient Frontier', xaxis_title='Volatility', yaxis_title='Expected Return', coloraxis_colorbar=dict(title='Art Weight'))
    return fig


def get_portfolio_parameters():
    st.sidebar.header('Portfolio Parameters')
    
    # Rolling window for art data
    art_rolling_window = st.sidebar.slider('Art Rolling Window (days)', 365, 1825, 680)
    
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
    
    return art_rolling_window, art_allocation, risk_free_rate, start_date, end_date, selected_ticker_symbols

# Load and preprocess data
@st.cache_data
def load_sentiment_data(sentiment_file_path):
    sentiment_df = pd.read_csv(sentiment_file_path)
    sentiment_df['Date of Publication'] = pd.to_datetime(sentiment_df['Date of Publication'], errors='coerce', dayfirst=True)
    sentiment_df = sentiment_df[(sentiment_df['Date of Publication'].dt.year >= 1990) & (sentiment_df['Date of Publication'].dt.year <= 2024)]
    sentiment_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    sentiment_df['Sentiment Score'] = sentiment_df['Sentiment Article'].map(sentiment_mapping)
    return sentiment_df

@st.cache_data
def load_artwork_data(artwork_data_path):
    artwork_df = pd.read_csv(artwork_data_path)
    artwork_df['Auction_Date'] = pd.to_datetime(artwork_df['Auction_Date'], errors='coerce', dayfirst=True)
    artwork_df = artwork_df[(artwork_df['Auction_Date'].dt.year >= 1990) & (artwork_df['Auction_Date'].dt.year <= 2024)]
    artwork_df['Price_Estimate_Difference'] = artwork_df['Price_Estimate_Difference'].replace({'\+|% est|%':'', ' est':''}, regex=True).astype(float)
    return artwork_df

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


# Load data
artwork_data_path = 'https://raw.githubusercontent.com/simonsim2001/ArtIndex/main/Artwork_Data_with_RGB.csv'
sentiment_file_path = 'https://raw.githubusercontent.com/simonsim2001/ArtIndex/main/Cleaned_Artist_Sentiment_Data.csv'
sentiment_df = load_sentiment_data(sentiment_file_path)
artwork_df = load_artwork_data(artwork_data_path)


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
    df = load_and_preprocess_data('https://raw.githubusercontent.com/simonsim2001/ArtIndex/main/BlueChip_Consolidated%20(2).csv')
    cleaned_data = load_data()
    df = extract_dimensions(df)

    st.title("Arte Blue Chip Index")

    # Sidebar for navigation
    view_mode = st.sidebar.selectbox("Select View Mode", ["Overview", "Search", "Portfolio and Transparency", "Art Sentiment"])

    if view_mode == "Overview":
        # Overview Mode
        # Basic Information
        with st.expander("Statistical Summary"):
            st.write(df.describe(include='all'))

        st.header("Summary Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Unique Artists", df['Artist'].nunique())
        with col2:
            st.metric("Unique Auction Houses", df['Auction_House'].nunique())

        st.header("Price Statistics")
        currencies = ["GBP", "USD", "EUR"]
        for currency in currencies:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"Min Price ({currency})", f"{df[f'Price_{currency}'].min():.2f}")
            with col2:
                st.metric(f"Max Price ({currency})", f"{df[f'Price_{currency}'].max():.2f}")
            with col3:
                st.metric(f"Avg Price ({currency})", f"{df[f'Price_{currency}'].mean():.2f}")

        with st.expander("Unique Artists"):
            st.write(df['Artist'].unique())

        with st.expander("Unique Auction Houses"):
            st.write(df['Auction_House'].unique())

        # Performance of Artists
        st.header("Performance of Artists")
        start_date = st.date_input("Start Date", value=pd.to_datetime("2005-01-01"), min_value=pd.to_datetime("1985-01-01"), max_value=pd.to_datetime("2024-01-01"))
        end_date = st.date_input("End Date", value=pd.to_datetime("2015-01-01"), min_value=pd.to_datetime("1985-01-01"), max_value=pd.to_datetime("2024-01-01"))

        if start_date < end_date:
            start_year, end_year = start_date.year, end_date.year
            df_filtered = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]

            avg_norm_prices = df_filtered.groupby(['Artist', 'Year'])['Normalized_Price'].mean().reset_index()
            artist_avg_norm_prices = avg_norm_prices.groupby('Artist')['Normalized_Price'].mean().reset_index()
            artist_avg_norm_prices.columns = ['Artist', 'Avg_Normalized_Price']
            avg_norm_prices = pd.merge(avg_norm_prices, artist_avg_norm_prices, on='Artist', how='left')
            avg_norm_prices['Normalized_Price'].fillna(avg_norm_prices['Avg_Normalized_Price'], inplace=True)
            avg_norm_prices_pivot = avg_norm_prices.pivot(index='Artist', columns='Year', values='Normalized_Price').reset_index()

            avg_areas = df_filtered.groupby(['Artist', 'Year'])['Area'].mean().reset_index()
            avg_areas_pivot = avg_areas.pivot(index='Artist', columns='Year', values='Area').reset_index()

            merged_prices = pd.merge(avg_norm_prices_pivot.melt(id_vars='Artist', var_name='Year', value_name='Normalized_Price'),
                                     avg_areas_pivot.melt(id_vars='Artist', var_name='Year', value_name='Area'),
                                     on=['Artist', 'Year'])
            merged_prices['Price_USD'] = merged_prices['Normalized_Price'] * merged_prices['Area']

            avg_prices = merged_prices.groupby(['Artist', 'Year']).agg({'Price_USD': 'mean'}).reset_index()
            avg_prices_pivot = avg_prices.pivot(index='Artist', columns='Year', values='Price_USD').reset_index()

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
                    results.append({'Artist': artist, 'Avg Initial Price ($K)': initial_price / 1000, 'Avg Final Price ($K)': final_price / 1000,
                                    'Avg Yearly Return': avg_yearly_return, 'IRR (%)': irr, 'Avg MOIC': moic})

            results_df = pd.DataFrame(results).dropna().sort_values(by='IRR (%)', ascending=False).reset_index(drop=True)
            st.write(results_df)

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

            conversion_rates = {'GBP': 1.3, 'EUR': 1.1}
            for currency in ['GBP', 'EUR']:
                artist_metrics[f'Total_Sales_{currency}_to_USD'] = artist_metrics[f'Total_Sales_{currency}'] * conversion_rates[currency]
                artist_metrics[f'Avg_Sale_Price_{currency}_to_USD'] = artist_metrics[f'Avg_Sale_Price_{currency}'] * conversion_rates[currency]
                artist_metrics[f'Highest_Sale_Price_{currency}_to_USD'] = artist_metrics[f'Highest_Sale_Price_{currency}'] * conversion_rates[currency]

            artist_metrics['Combined_Total_Sales_USD'] = artist_metrics[['Total_Sales_USD', 'Total_Sales_GBP_to_USD', 'Total_Sales_EUR_to_USD']].sum(axis=1)
            artist_metrics['Combined_Avg_Sale_Price_USD'] = artist_metrics['Combined_Total_Sales_USD'] / artist_metrics['Artworks_Sold']
            artist_metrics['Combined_Highest_Sale_Price_USD'] = artist_metrics[['Highest_Sale_Price_USD', 'Highest_Sale_Price_GBP_to_USD', 'Highest_Sale_Price_EUR_to_USD']].max(axis=1)

            combined_report = artist_metrics[['Artist', 'Combined_Total_Sales_USD', 'Combined_Avg_Sale_Price_USD', 'Combined_Highest_Sale_Price_USD',
                                              'Artworks_Sold', 'Avg_Price_Estimate_Difference', 'Auction_Houses']]
            ranking_report = combined_report.sort_values(by='Combined_Avg_Sale_Price_USD', ascending=False).reset_index(drop=True)
            st.write("Ranking Report based on Average Sale Price (USD):")
            st.write(ranking_report)

            transactions_per_artist_per_year = df.groupby(['Artist', 'Year']).size().reset_index(name='Transactions')
            max_transactions = transactions_per_artist_per_year.sort_values(by='Transactions', ascending=False).reset_index(drop=True)
            st.write("Max Transactions Per Year:")
            st.write(max_transactions)

            avg_price_per_artist_per_year = df.groupby(['Artist', 'Year'])['Price_USD'].mean().reset_index(name='Avg_Sale_Price_USD')
            highest_avg_price = avg_price_per_artist_per_year.sort_values(by='Avg_Sale_Price_USD', ascending=False).reset_index(drop=True)
            st.write("Highest Average Price Per Year:")
            st.write(highest_avg_price)

            combined_yearly_metrics = pd.merge(max_transactions, highest_avg_price, on=['Artist', 'Year'], how='inner').sort_values(by=['Transactions', 'Avg_Sale_Price_USD'], ascending=[False, False])
            st.write("Combined Metrics Yearly Sorted:")
            st.write(combined_yearly_metrics)
        else:
            st.error("End Date must be after Start Date")

    elif view_mode == "Search":
        # Search Mode
        # Create a list of unique artists for the autocomplete feature
        unique_artists = df['Artist'].unique().tolist()

        # Use st.selectbox for the artist name input with autocomplete
        artist_name = st.selectbox("Select Artist", unique_artists)

        if artist_name:
            # Data filtering based on artist name
            artist_data = df[df['Artist'] == artist_name]

            if not artist_data.empty:
                # Filtering data based on user-selected date range
                start_date = st.date_input("Start Date", value=pd.to_datetime("2005-01-01"), min_value=pd.to_datetime("1985-01-01"), max_value=pd.to_datetime("2024-01-01"))
                end_date = st.date_input("End Date", value=pd.to_datetime("2015-01-01"), min_value=pd.to_datetime("1985-01-01"), max_value=pd.to_datetime("2024-01-01"))

                if start_date < end_date:
                    df_filtered = artist_data[(artist_data['Year'] >= start_date.year) & (artist_data['Year'] <= end_date.year)]

                    # Add input fields for artwork parameters
                    target_width = st.number_input("Target Width (cm)", value=40.5)
                    target_height = st.number_input("Target Height (cm)", value=40.5)
                    tolerance_cm = st.number_input("Tolerance (cm)", value=5.0)

                    # Filter data based on artwork parameters
                    filtered_data = df_filtered[
                        (abs(df_filtered['Width_cm'] - target_width) <= tolerance_cm) &
                        (abs(df_filtered['Height_cm'] - target_height) <= tolerance_cm)
                    ]

                    # Calculate average sale price per artist per year
                    avg_price_per_artist_per_year = df_filtered.groupby(['Artist', 'Year'])['Price_USD'].mean().reset_index(name='Average_Sale_Price_USD')
                    highest_price_per_artist_per_year = df_filtered.groupby(['Artist', 'Year'])['Price_USD'].max().reset_index(name='Highest_Sale_Price_USD')

                    # Interactive Plot for Highest Sale Price
                    artist_record_prices = highest_price_per_artist_per_year[highest_price_per_artist_per_year['Artist'] == artist_name]
                    fig = px.line(artist_record_prices, x='Year', y='Highest_Sale_Price_USD', title=f'Record Sale Price per Year for {artist_name}')
                    st.plotly_chart(fig)

                    # Interactive Plot for Average Sale Price
                    artist_average_prices = avg_price_per_artist_per_year[avg_price_per_artist_per_year['Artist'] == artist_name]
                    fig = px.line(artist_average_prices, x='Year', y='Average_Sale_Price_USD', title=f'Average Sale Price per Year for {artist_name}')
                    st.plotly_chart(fig)

                    # Calculate IRR and MOIC
                    avg_price_per_artist_per_year = avg_price_per_artist_per_year.sort_values(by='Year')
                    beginning_value = avg_price_per_artist_per_year.iloc[0]['Average_Sale_Price_USD']
                    ending_value = avg_price_per_artist_per_year.iloc[-1]['Average_Sale_Price_USD']
                    number_of_years = avg_price_per_artist_per_year['Year'].iloc[-1] - avg_price_per_artist_per_year['Year'].iloc[0]

                    annualized_return = ((ending_value / beginning_value) ** (1 / number_of_years)) - 1
                    annualized_return_percentage = annualized_return * 100

                    total_return_percentage = ((ending_value - beginning_value) / beginning_value) * 100
                    average_yearly_return_percentage = total_return_percentage / number_of_years

                    roi_percentage = ((ending_value - beginning_value) / beginning_value) * 100

                    st.write(f"The average return for {artist_name} over the available years is USD {avg_price_per_artist_per_year['Average_Sale_Price_USD'].mean():,.2f}")
                    st.write(f"The annualized return for {artist_name} over the period is {annualized_return_percentage:.2f}%")
                    st.write(f"The average yearly return for {artist_name} over the period is {average_yearly_return_percentage:.2f}%")
                    st.write(f"If a painting by {artist_name} was purchased at the beginning of the period and sold at the end, the return on investment would be {roi_percentage:.2f}%")

                    # Annual Turnover
                    annual_turnover = artist_data.groupby('Year')['Price_USD'].sum().reset_index(name='Annual_Turnover_USD')
                    st.write(f"Annual turnover for {artist_name}:")
                    st.write(annual_turnover)

                    # Filter data based on artwork parameters
                    filtered_data = df_filtered[
                        (abs(df_filtered['Width_cm'] - target_width) <= tolerance_cm) &
                        (abs(df_filtered['Height_cm'] - target_height) <= tolerance_cm)
                    ]

                    # Calculate average sale price per year for comparable artworks
                    average_price_per_year = filtered_data.groupby('Year')['Price_USD'].mean().reset_index(name='Average_Sale_Price_USD')

                    # Interactive Plot for Comparable Artworks
                    fig = px.line(average_price_per_year, x='Year', y='Average_Sale_Price_USD', title='Average Sale Price per Year for Comparable Artworks')
                    st.plotly_chart(fig)

                    # Display a table of comparable artworks with images
                    if st.button('Generate HTML Table with Images'):
                        html_content = generate_html_table_with_images(filtered_data.head(15))
                        st.markdown(html_content, unsafe_allow_html=True)

                else:
                    st.error("End Date must be after Start Date")
            else:
                st.error(f"No data found for artist: {artist_name}")


    # Machine Learning Section
    # Machine Learning Section
    elif view_mode == "Portfolio and Transparency":

        # Retrieve the user-selected parameters
        art_rolling_window, art_allocation, risk_free_rate, start_date, end_date, selected_ticker_symbols = get_portfolio_parameters()

        # Load the Artprice Blue Chip Index
        show_artprice_index = st.sidebar.checkbox("Include Artprice Blue Chip Index")
        artprice_index = load_artprice_index() if show_artprice_index else None

        # Calculate rolling average for art data
        overall_rolling_average = calculate_rolling_average(cleaned_data, art_rolling_window)

        # Load market data based on selected tickers and date range
        market_data = load_market_data(selected_ticker_symbols, start_date, end_date)

        if market_data:
            art_returns = overall_rolling_average.pct_change().dropna()
            market_returns = {ticker: data.pct_change().dropna() for ticker, data in market_data.items()}

            # Align the data
            common_index = art_returns.index.intersection(market_returns[selected_ticker_symbols[0]].index)
            art_returns = art_returns[common_index]
            market_returns = {ticker: returns[common_index] for ticker, returns in market_returns.items()}

            if not art_returns.empty:
                metrics = calculate_combined_portfolio_metrics(art_returns, market_returns, art_allocation, risk_free_rate)

                st.header('Portfolio Performance')

                tab1, tab2, tab3, tab4 = st.tabs(["Cumulative Returns", "Annual Returns", "Rolling Correlation", "Efficient Frontier"])

                with tab1:
                    st.plotly_chart(plot_cumulative_returns(metrics, artprice_index=artprice_index), use_container_width=True)

                with tab2:
                    st.plotly_chart(plot_annual_returns(metrics), use_container_width=True)

                with tab3:
                    st.plotly_chart(plot_rolling_correlation(metrics), use_container_width=True)

                with tab4:
                    st.plotly_chart(plot_efficient_frontier(metrics), use_container_width=True)

                st.header('Performance Metrics')
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Art Final Cumulative Return", f"{metrics['Combined Portfolio']['art_cumulative'].iloc[-1]:.2%}")
                    st.metric("Art Portfolio Volatility", f"{metrics['Combined Portfolio']['art_volatility']:.2%}")
                with col2:
                    st.metric("Combined Market Final Cumulative Return", f"{metrics['Combined Portfolio']['market_cumulative'].iloc[-1]:.2%}")
                    st.metric("Combined Market Volatility", f"{metrics['Combined Portfolio']['market_volatility']:.2%}")
                with col3:
                    st.metric("Combined Portfolio Final Cumulative Return", f"{metrics['Combined Portfolio']['combined_cumulative'].iloc[-1]:.2%}")
                    st.metric("Combined Portfolio Volatility", f"{metrics['Combined Portfolio']['combined_volatility']:.2%}")
                with col4:
                    st.metric("Sharpe Ratio", f"{metrics['Combined Portfolio']['sharpe_ratio']:.2f}")
                    st.metric("Annualized Return", f"{metrics['Combined Portfolio']['annualized_return']:.2%}")
                    st.metric("Max Drawdown", f"{metrics['Combined Portfolio']['max_drawdown']:.2%}")

                artist_weights = cleaned_data.groupby(['Artist', cleaned_data.index.year])['Price_USD'].sum() / cleaned_data.groupby(cleaned_data.index.year)['Price_USD'].sum()
                st.header('Artist Weights')
                st.dataframe(artist_weights)

                # Debug information
                st.header('Debug Information')
                st.write("Art Returns:")
                st.write(art_returns)
                st.write("Combined Market Returns:")
                st.write(metrics['Combined Portfolio']['market_returns'])

            else:
                st.error("No data points to calculate returns. Please check the alignment of the series.")
        else:
            st.error("Failed to load market data. Please try again later.")


    elif view_mode == "Art Sentiment":
        st.title("Art Sentiment Analysis (1990-2024)")

        # Artist selection
        artist_list = sentiment_df['Artist'].unique()
        selected_artist = st.sidebar.selectbox("Select Artist", options=artist_list)

        # Filter sentiment data for the selected artist
        artist_sentiment_df = sentiment_df[sentiment_df['Artist'] == selected_artist]

        # Search for articles by year
        start_year, end_year = st.sidebar.select_slider(
            "Filter by Year",
            options=list(range(1990, 2025)),
            value=(1990, 2024)
        )

        # Filter by selected year range
        artist_sentiment_df = artist_sentiment_df[
            (artist_sentiment_df['Date of Publication'].dt.year >= start_year) &
            (artist_sentiment_df['Date of Publication'].dt.year <= end_year)
        ]

        # Search for articles by keyword
        keyword = st.sidebar.text_input("Search by Keyword")
        if keyword:
            artist_sentiment_df = artist_sentiment_df[
                artist_sentiment_df['Full Article'].str.contains(keyword, case=False, na=False) | 
                artist_sentiment_df['Title'].str.contains(keyword, case=False, na=False)
            ]

        # Calculate the number of articles per year for the selected artist
        artist_sentiment_df['Year'] = artist_sentiment_df['Date of Publication'].dt.year
        articles_per_year = artist_sentiment_df.groupby('Year').size().reset_index(name='Article Count')

        # Merge with artwork data for the selected artist
        artist_data = pd.merge(
            artist_sentiment_df, 
            artwork_df[artwork_df['Artist'] == selected_artist], 
            left_on='Date of Publication', 
            right_on='Auction_Date', 
            how='left'
        )

        # Ensure the 'Artist' column is included
        artist_data['Artist'] = selected_artist

        # Add the 'Article Count' to the artist data
        artist_data = pd.merge(
            artist_data, 
            articles_per_year, 
            on='Year', 
            how='left'
        )

        # Plot the number of articles per year
        st.subheader("Number of Articles per Year")
        bar_fig = px.bar(articles_per_year, x='Year', y='Article Count', title='Number of Articles per Year', labels={'Article Count': 'Number of Articles'})
        st.plotly_chart(bar_fig, use_container_width=True)

        # Display detailed profile
        st.subheader(f"Profile for {selected_artist}")
        profile_fig = plot_artist_profile(artist_data)
        if profile_fig:
            st.plotly_chart(profile_fig, use_container_width=True)

        # Display specific articles
        st.subheader("Articles")
        for _, row in artist_sentiment_df.iterrows():
            with st.expander(row['Title']):
                st.write(f"**Publisher:** {row['Publisher']}")
                st.write(f"**Date of Publication:** {row['Date of Publication'].date()}")
                st.write(f"**Sentiment:** {row['Sentiment Article']}")
                st.write(row['Full Article'])

        # Show data table for deeper analysis if needed
        st.write("Data Table:")
        st.dataframe(artist_data[['Date of Publication', 'Sentiment Score', 'Price_Estimate_Difference', 'Price_USD', 'Article Count']])

if __name__ == "__main__":
    main()
