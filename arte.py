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
        if st.session_state["password"] == "francodanesi":
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
    art_cumulative = metrics['Combined Portfolio']['art_cumulative']
    fig.add_trace(go.Scatter(x=art_cumulative.index, y=art_cumulative, mode='lines', name='Arte Blue Chip Index', 
                             line=dict(color=colors['art'])))

    # Plot Artprice Blue Chip Index if provided
    if artprice_index is not None:
        fig.add_trace(go.Scatter(x=artprice_index.index, y=artprice_index, mode='lines', 
                                 name='Artprice Blue Chip Index', line=dict(color=colors['artprice'], dash='dash')))
    
    # Plot combined market cumulative returns
    market_cumulative = metrics['Combined Portfolio']['market_cumulative']
    combined_cumulative = metrics['Combined Portfolio']['combined_cumulative']

    fig.add_trace(go.Scatter(x=market_cumulative.index, y=market_cumulative, mode='lines', name='Stocks', 
                             line=dict(color=colors['market'])))
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


def plot_annual_returns(metrics):
    fig = go.Figure()
    for ticker, data in metrics.items():
        annual_returns = pd.DataFrame({
            'Art': data['art_returns'].resample('Y').apply(lambda x: (1 + x).prod() - 1),
            'Stocks': data['market_returns'].resample('Y').apply(lambda x: (1 + x).prod() - 1),
            'Combined Portfolio': data['combined_returns'].resample('Y').apply(lambda x: (1 + x).prod() - 1)
        })
        for column in annual_returns.columns:
            # Apply the colors consistently across the bars
            if column == 'Art':
                color = colors['art']
            elif column == 'Stocks':
                color = colors['market']
            else:
                color = colors['combined']
            fig.add_trace(go.Bar(x=annual_returns.index, y=annual_returns[column], name=column, marker=dict(color=color)))
    
    fig.update_layout(
        title='Annual Returns',
        xaxis_title='Year',
        yaxis_title='Annual Return',
        barmode='group',
        legend_title='Portfolio',
        xaxis=dict(showgrid=False),  # Remove gridlines
        yaxis=dict(showgrid=False)
    )
    
    return fig


def plot_rolling_correlation(metrics, window=12):
    fig = go.Figure()

    for ticker, data in metrics.items():
        correlation = data['art_returns'].rolling(window=window).corr(data['market_returns']).dropna()
        
        fig.add_trace(go.Scatter(
            x=correlation.index, 
            y=correlation, 
            mode='lines', 
            name=f'{ticker} Correlation',
            line=dict(width=2, color=colors['combined'])  # Use the 'combined' color for correlation
        ))

    fig.add_shape(type="line", x0=correlation.index.min(), x1=correlation.index.max(), 
                  y0=0, y1=0, line=dict(color="gray", dash="dash"))

    fig.update_layout(
        title=f'{window}-Month Rolling Correlation between Art and Markets',
        xaxis_title='Year',
        yaxis_title='Correlation',
        xaxis=dict(showgrid=False, showline=True, linecolor='black', mirror=True),
        yaxis=dict(showgrid=False, showline=True, linecolor='black', mirror=True, range=[-1, 1]),
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
                                 marker=dict(size=5, color=portfolios['Art_Weight'], colorscale='Purples', showscale=True),  # Purple colorscale
                                 text=portfolios['Art_Weight'].apply(lambda x: f'Art Weight: {x:.2f}<br>Market Weight: {1-x:.2f}'),
                                 hoverinfo='text+x+y', name=f'{ticker} Frontier'))
    
    fig.update_layout(
        title='Efficient Frontier',
        xaxis_title='Volatility',
        yaxis_title='Expected Return',
        coloraxis_colorbar=dict(title='Art Weight'),
        xaxis=dict(showgrid=False),  # Remove gridlines
        yaxis=dict(showgrid=False)
    )
    
    return fig



    # Customizing the layout for a minimalist design
    fig.update_layout(
        title=None,  # No title for minimalist look
        xaxis_title='Volatility',  # Simple axis title
        yaxis_title='Expected Return',
        legend=dict(
            orientation='h',  # Horizontal legend
            yanchor="bottom", y=-0.2, xanchor="center", x=0.5
        ),
        margin=dict(l=40, r=40, t=20, b=40),  # Tight margins for a cleaner view
        plot_bgcolor='white',  # Keep the background white
        xaxis=dict(
            showgrid=False,  # Remove gridlines for x-axis
            showline=True,  # Show baseline line for x-axis
            linecolor='black',
            ticks='outside',
            tickcolor='black'
        ),
        yaxis=dict(
            showgrid=True,  # Subtle gridlines on y-axis
            gridcolor='lightgray',
            showline=False,  # No baseline line for y-axis
            ticks='outside',
            tickcolor='black'
        ),
        font=dict(family="Arial", size=14, color="black"),  # Consistent font across all charts
        showlegend=True  # Keep the legend visible
    )
    
    return fig
    


#Portfolio metrics

def calculate_combined_portfolio_metrics(
    art_returns, market_returns_dict, art_allocation, risk_free_rate, 
    cash_allocation=0.0, pe_allocation=0.0, re_allocation=0.0, 
    pe_settings=None, re_settings=None
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

    # Calculate Cash returns (fixed at the risk-free rate)
    cash_returns = pd.Series([risk_free_rate] * len(art_returns), index=art_returns.index)

    # Calculate Private Equity returns based on user settings (IRR, investment period, etc.)
    pe_returns = model_private_equity_returns(art_returns.index, pe_settings) if pe_allocation > 0 else pd.Series([0] * len(art_returns), index=art_returns.index)

    # Calculate Real Estate returns based on user settings (cap rate, appreciation rate, etc.)
    re_returns = model_real_estate_returns(art_returns.index, re_settings) if re_allocation > 0 else pd.Series([0] * len(art_returns), index=art_returns.index)

    # Total allocation check
    total_allocation = art_allocation + cash_allocation + pe_allocation + re_allocation
    if total_allocation > 1:
        raise ValueError("Total allocation for all asset classes cannot exceed 100%.")

    # Remaining allocation for market
    market_allocation = 1 - total_allocation

    # Calculate the combined returns
    combined_returns = (
        (art_returns * art_allocation) + 
        (combined_market_returns * market_allocation) +
        (cash_returns * cash_allocation) +
        (pe_returns * pe_allocation) +
        (re_returns * re_allocation)
    )

    # Ensure the returns are aligned (outer join and fill with zeros)
    art_returns, combined_returns = art_returns.align(combined_returns, join='outer', fill_value=0)

    # Calculate the metrics
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

    # Store the calculated metrics
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

    # Allocation for Cash, Private Equity, and Real Estate
    cash_allocation = st.sidebar.slider('Cash Allocation - return based on risk free rate', 0.0, 1.0, 0.1)
    pe_allocation = st.sidebar.slider('Private Equity Allocation', 0.0, 1.0, 0.2)
    re_allocation = st.sidebar.slider('Real Estate Allocation', 0.0, 1.0, 0.2)

    # Small explanatory comment
    st.sidebar.markdown("<small>Note: Remaining allocation will be automatically assigned to the stock market.</small>", unsafe_allow_html=True)

    # Risk-free rate for Sharpe ratio calculation
    risk_free_rate = st.sidebar.number_input('Risk-Free Rate', 0.0, 0.1, 0.02, 0.01)

    # Private Equity settings
    st.sidebar.subheader('Private Equity Settings')
    pe_irr = st.sidebar.number_input('Private Equity IRR (%)', 0.0, 100.0, 15.0, 0.1)
    pe_investment_period = st.sidebar.slider('Investment Period (years)', 1, 10, 5)
    pe_fund_call_rate = st.sidebar.slider('Fund Call Rate (%)', 0.0, 100.0, 25.0)

    private_equity_settings = {
        'IRR': pe_irr / 100,
        'investment_period': pe_investment_period,
        'fund_call_rate': pe_fund_call_rate / 100
    }

    # Real Estate settings
    st.sidebar.subheader('Real Estate Settings')
    re_cap_rate = st.sidebar.number_input('Real Estate Cap Rate (%)', 0.0, 20.0, 5.0, 0.1)
    re_appreciation_rate = st.sidebar.number_input('Appreciation Rate (%)', 0.0, 10.0, 3.0, 0.1)

    real_estate_settings = {
        'cap_rate': re_cap_rate / 100,
        'appreciation_rate': re_appreciation_rate / 100
    }

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
    unique_artists = df['Artist'].unique().tolist()
    selected_artists = st.sidebar.multiselect(
        "Select Artist(s)",
        options=unique_artists,
        default=unique_artists  # You can set a default selection if desired
    )

    # Return the selected parameters
    return (
        art_rolling_window, art_allocation, cash_allocation, pe_allocation, re_allocation,
        risk_free_rate, start_date, end_date, selected_ticker_symbols, selected_artists,
        private_equity_settings, real_estate_settings
    )




def model_private_equity_returns(dates, settings):
    """
    Simulate Private Equity returns based on IRR, investment period, and fund call rate.
    This is a simplified model and may need adjustment based on your specific requirements.
    
    Args:
        dates (pd.Index): The index of dates for the returns.
        settings (dict): A dictionary containing 'IRR', 'investment_period', and 'fund_call_rate'.
    
    Returns:
        pd.Series: A time series of simulated Private Equity returns.
    """
    irr = settings['IRR']
    investment_period = settings['investment_period']
    fund_call_rate = settings['fund_call_rate']
    
    # Generate a simple return model based on IRR and investment period
    pe_returns = pd.Series([irr / investment_period] * len(dates), index=dates)

    # You can adjust this to simulate fund calls or apply more complex logic if needed.
    
    return pe_returns


def model_real_estate_returns(dates, settings):
    """
    Simulate Real Estate returns based on cap rate and appreciation rate.
    
    Args:
        dates (pd.Index): The index of dates for the returns.
        settings (dict): A dictionary containing 'cap_rate' and 'appreciation_rate'.
    
    Returns:
        pd.Series: A time series of simulated Real Estate returns.
    """
    cap_rate = settings['cap_rate']
    appreciation_rate = settings['appreciation_rate']
    
    # Generate a simple return model combining cap rate and appreciation rate
    re_returns = pd.Series([cap_rate + appreciation_rate] * len(dates), index=dates)
    
    return re_returns


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

        # Display key summary metrics directly
        st.header("Summary Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Unique Artists", df['Artist'].nunique())
        with col2:
            st.metric("Unique Auction Houses", df['Auction_House'].nunique())
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
                        'Avg Yearly Return (%)': avg_yearly_return * 100,
                        'IRR (%)': irr,
                        'Avg MOIC': moic
                    })

            results_df = pd.DataFrame(results).dropna().sort_values(by='IRR (%)', ascending=False).reset_index(drop=True)
            # Format the percentages
            results_df['Avg Yearly Return (%)'] = results_df['Avg Yearly Return (%)'].apply(lambda x: f"{x:.2f}%")
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
                # Sidebar inputs for date range and artwork parameters
                with st.sidebar:
                    st.header("Filters")
                    start_date = st.date_input("Start Date", value=pd.to_datetime("2005-01-01"), min_value=pd.to_datetime("1985-01-01"), max_value=pd.to_datetime("2024-01-01"))
                    end_date = st.date_input("End Date", value=pd.to_datetime("2015-01-01"), min_value=pd.to_datetime("1985-01-01"), max_value=pd.to_datetime("2024-01-01"))

                    if start_date >= end_date:
                        st.error("End Date must be after Start Date")

                    # Add input fields for artwork parameters
                    st.subheader("Artwork Parameters")
                    target_width = st.number_input("Target Width (cm)", value=40.5)
                    target_height = st.number_input("Target Height (cm)", value=40.5)
                    tolerance_cm = st.number_input("Tolerance (cm)", value=5.0)

                if start_date < end_date:
                    df_filtered = artist_data[(artist_data['Year'] >= start_date.year) & (artist_data['Year'] <= end_date.year)]

                    # Filter data based on artwork parameters
                    filtered_data = df_filtered[
                        (abs(df_filtered['Width_cm'] - target_width) <= tolerance_cm) &
                        (abs(df_filtered['Height_cm'] - target_height) <= tolerance_cm)
                    ]

                    # Organize content into tabs
                    tab1, tab2, tab3 = st.tabs(["Artist Overview", "Price Analysis", "Comparable Artworks"])

                    with tab1:
                        st.subheader(f"Overview of {artist_name}")

                        # Display key metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Artworks Sold", artist_data.shape[0])
                        with col2:
                            total_sales = artist_data['Price_USD'].sum()
                            st.metric("Total Sales (USD)", f"${total_sales:,.2f}")
                        with col3:
                            avg_price = artist_data['Price_USD'].mean()
                            st.metric("Average Sale Price (USD)", f"${avg_price:,.2f}")

                        st.markdown("---")

                        # Annual Turnover
                        st.subheader("Annual Turnover")
                        annual_turnover = artist_data.groupby('Year')['Price_USD'].sum().reset_index(name='Annual_Turnover_USD')
                        turnover_fig = px.bar(annual_turnover, x='Year', y='Annual_Turnover_USD', title='Annual Turnover (USD)')
                        st.plotly_chart(turnover_fig, use_container_width=True)

                    with tab2:
                        st.subheader("Price Analysis")

                        # Average and Highest Sale Prices per Year
                        avg_price_per_year = df_filtered.groupby('Year')['Price_USD'].mean().reset_index(name='Average_Sale_Price_USD')
                        highest_price_per_year = df_filtered.groupby('Year')['Price_USD'].max().reset_index(name='Highest_Sale_Price_USD')

                        # Display plots side by side
                        col1, col2 = st.columns(2)
                        with col1:
                            # Interactive Plot for Average Sale Price
                            avg_price_fig = px.line(avg_price_per_year, x='Year', y='Average_Sale_Price_USD', title='Average Sale Price per Year')
                            st.plotly_chart(avg_price_fig, use_container_width=True)
                        with col2:
                            # Interactive Plot for Highest Sale Price
                            highest_price_fig = px.line(highest_price_per_year, x='Year', y='Highest_Sale_Price_USD', title='Highest Sale Price per Year')
                            st.plotly_chart(highest_price_fig, use_container_width=True)

                        # Calculate IRR and MOIC
                        if not avg_price_per_year.empty:
                            avg_price_per_year = avg_price_per_year.sort_values(by='Year')
                            beginning_value = avg_price_per_year.iloc[0]['Average_Sale_Price_USD']
                            ending_value = avg_price_per_year.iloc[-1]['Average_Sale_Price_USD']
                            number_of_years = avg_price_per_year['Year'].iloc[-1] - avg_price_per_year['Year'].iloc[0]

                            if number_of_years > 0:
                                annualized_return = ((ending_value / beginning_value) ** (1 / number_of_years)) - 1
                                annualized_return_percentage = annualized_return * 100

                                total_return_percentage = ((ending_value - beginning_value) / beginning_value) * 100
                                average_yearly_return_percentage = total_return_percentage / number_of_years

                                roi_percentage = ((ending_value - beginning_value) / beginning_value) * 100

                                st.markdown("---")
                                st.subheader("Investment Analysis")
                                st.write(f"The average return over the available years is **${avg_price_per_year['Average_Sale_Price_USD'].mean():,.2f}**")
                                st.write(f"The annualized return over the period is **{annualized_return_percentage:.2f}%**")
                                st.write(f"The average yearly return over the period is **{average_yearly_return_percentage:.2f}%**")
                                st.write(f"Return on investment over the period would be **{roi_percentage:.2f}%**")
                            else:
                                st.warning("Not enough data to calculate investment returns.")
                        else:
                            st.warning("No data available for the selected date range.")

                    with tab3:
                        st.subheader("Comparable Artworks")

                        if not filtered_data.empty:
                            # Calculate average sale price per year for comparable artworks
                            average_price_per_year = filtered_data.groupby('Year')['Price_USD'].mean().reset_index(name='Average_Sale_Price_USD')

                            # Interactive Plot for Comparable Artworks
                            comp_avg_price_fig = px.line(
                                average_price_per_year,
                                x='Year',
                                y='Average_Sale_Price_USD',
                                title='Average Sale Price per Year for Comparable Artworks'
                            )
                            st.plotly_chart(comp_avg_price_fig, use_container_width=True)

                            # Display a table of comparable artworks
                            st.subheader("List of Comparable Artworks")
                            # Select relevant columns and copy to avoid SettingWithCopyWarning
                            comparable_artworks = filtered_data[['Artwork_Title', 'Year', 'Price_USD', 'Width_cm', 'Height_cm']].copy()
                            # Option 1: Convert 'Year' to string to prevent commas
                            comparable_artworks['Year'] = comparable_artworks['Year'].astype(str)
                            # Optionally format 'Price_USD' and other columns
                            comparable_artworks['Price_USD'] = comparable_artworks['Price_USD'].apply(lambda x: f"${x:,.2f}")
                            # Display the dataframe
                            st.dataframe(comparable_artworks)

                            # Optionally display images in an expander
                            with st.expander("Show Artwork Images"):
                                html_content = generate_html_table_with_images(filtered_data.head(15))
                                st.markdown(html_content, unsafe_allow_html=True)
                        else:
                            st.warning("No comparable artworks found with the specified dimensions and tolerance.")

            else:
                st.error(f"No data found for artist: {artist_name}")

    # Portfolio and Transparency Section
    elif view_mode == "Portfolio and Transparency":
        # Retrieve the user-selected parameters, including new asset classes
        (
            art_rolling_window, art_allocation, cash_allocation, pe_allocation, re_allocation,
            risk_free_rate, start_date, end_date, selected_ticker_symbols, selected_artists,
            private_equity_settings, real_estate_settings
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
                        # Calculate combined portfolio metrics with new asset classes
                        metrics = calculate_combined_portfolio_metrics(
                            art_returns, market_returns, art_allocation, risk_free_rate, 
                            cash_allocation, pe_allocation, re_allocation, 
                            private_equity_settings, real_estate_settings
                        )

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
                            fig2 = plot_annual_returns(metrics)

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

                        # **Add new elements under existing elements**
                        st.header('User Market Predictions for Next Year')

                        # User input for predicted art market return
                        predicted_art_return = st.number_input(
                            'Predicted Art Market Return (%) for Next Year',
                            min_value=-100.0, max_value=100.0, value=5.0, step=0.1
                        ) / 100  # Convert percentage to decimal

                        # User inputs for predicted returns of selected tickers
                        predicted_market_returns = {}
                        for ticker in selected_ticker_symbols:
                            predicted_return = st.number_input(
                                f'Predicted Return (%) for {ticker} Next Year',
                                min_value=-100.0, max_value=100.0, value=5.0, step=0.1
                            ) / 100  # Convert percentage to decimal
                            predicted_market_returns[ticker] = predicted_return

                        # Calculate the projected returns including the user predictions

                        # Extend the art_returns with the predicted return
                        projected_art_returns = art_returns.copy()
                        last_date = projected_art_returns.index[-1]  # Get the last date of the historical data

                        # Ensure the last date is after June 2024
                        if last_date < pd.Timestamp('2024-06-30'):
                            next_year = last_date + pd.DateOffset(months=6)  # Add 6 months for the projection
                        else:
                            next_year = last_date + pd.DateOffset(years=1)  # Add one year if it's before June

                        # Add the predicted return for next year
                        projected_art_returns.loc[next_year] = predicted_art_return

                        # Extend the market_returns with the predicted returns
                        projected_market_returns = {}
                        for ticker in selected_ticker_symbols:
                            returns = market_returns[ticker].copy()
                            returns.loc[next_year] = predicted_market_returns[ticker]
                            projected_market_returns[ticker] = returns

                        # Calculate projected combined portfolio metrics
                        projected_metrics = calculate_combined_portfolio_metrics(
                            projected_art_returns,
                            projected_market_returns,
                            art_allocation,
                            risk_free_rate
                        )

                        # Visualize the projected performance
                        st.header('Projected Portfolio Performance Including User Predictions')

                        # Create tabs for the projected performance
                        proj_tab1, proj_tab2 = st.tabs(["Projected Cumulative Returns", "Projected Performance Metrics"])

                        # **Projected Tab 1: Projected Cumulative Returns**
                        with proj_tab1:
                            fig_proj = plot_projected_cumulative_returns(metrics, projected_metrics)

                            # Adjust figure layout
                            fig_proj.update_layout(
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
                                margin=dict(l=50, r=50, t=50, b=100),
                                legend=dict(
                                    orientation='h',
                                    yanchor='top',
                                    y=-0.2,
                                    xanchor='center',
                                    x=0.5
                                )
                            )

                            st.plotly_chart(fig_proj, use_container_width=True)

                            # Add download button for projected cumulative returns graph
                            buffer = io.BytesIO()
                            fig_proj.write_image(buffer, format='pdf', width=1600, height=900, scale=2)
                            buffer.seek(0)
                            st.download_button(
                                label="Download Projected Cumulative Returns as PDF",
                                data=buffer,
                                file_name='projected_cumulative_returns.pdf',
                                mime='application/pdf'
                            )

                        # **Projected Tab 2: Projected Performance Metrics**
                        with proj_tab2:
                            # Display projected performance metrics
                            st.subheader('Projected Performance Metrics for Next Year')
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(
                                    "Projected Combined Portfolio Return Next Year",
                                    f"{projected_metrics['Combined Portfolio']['combined_cumulative'].iloc[-1]:.2%}"
                                )
                            with col2:
                                st.metric(
                                    "Projected Sharpe Ratio Next Year",
                                    f"{projected_metrics['Combined Portfolio']['sharpe_ratio']:.2f}"
                                )

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
    if check_password():
        main()
