import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os, json

# Constant to control table refresh
REFRESH_TICKER_DATA = False

# -------------------------------- Exercice 1 --------------------------------
# Universe of tickers
fr_tickers=[
    "EN.PA",
    "SAN.PA",
    "AI.PA",
    "CA.PA",
    "ORA.PA",
    "LR.PA",
    "KER.PA",
    "AIR.PA",
    "DG.PA",
    "MC.PA",
    "SGO.PA",
    "SW.PA",
    "ENGI.PA",
    "OR.PA",
    "VIE.PA",
    "RI.PA",
    "GLE.PA",
    "BNP.PA",
    "HO.PA",
    "BN.PA",
    "ATO.PA",
    "AC.PA",
    "ML.PA",
    "ACA.PA",
    "SU.PA",
    "WLN.PA",
    "VIV.PA",
    "CAP.PA"
]

us_tickers=[
    "NVDA",
    "GOOG",
    "G",
    "GOOGL",
    "AAPL",
    "MSFT",
    "AMZN",
    "M",
    "META",
    "T",
    "TSM",
    "TSLA",
    "AVGO",
    "B",
    "BRK-B",
    "BRK-A",
    "J",
    "JPM",
    "WMT",
    "LLY",
    "TCTZF",
    "TCEHY",
    "V",
    "XOM",
    "ASML",
    "ASMLF",
    "JNJ",
    "BAC"
]

indices_tickers=[
    "^FCHI",
    "^GSPC",
    "^DJI",
    "^IXIC",
    "^RUT",
    "^VIX",
    "^FTSE"
]

ccy_tickers=[
    "EURUSD=X",
    "EURGBP=X"
]

tickers=fr_tickers+us_tickers+indices_tickers+ccy_tickers

# -------------------------------- Exercice 2 --------------------------------

# Function to download ticker descriptions
def download_ticker_descriptions(tickers, refresh=REFRESH_TICKER_DATA):
    file_path = 'ticker_descriptions.parquet'

    if not refresh and os.path.exists(file_path):
        print("Ticker descriptions already exist. Skipping download.")
        return pd.read_parquet(file_path)

    print("Downloading ticker descriptions from yfinance...")
    ticker_data = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info

            ticker_data.append({
                'ticker': ticker,
                'name': info.get('longName', ''),
                'industry': info.get('industry', ''),
                'sector': info.get('sector', ''),
                'country': info.get('country', ''),
                'currency': info.get('currency', ''),
                'marketCap': info.get('marketCap', 0)
            })
        except Exception as e:
            print(f"Failed to download data for {ticker}: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(ticker_data)

    # Save to parquet file    
    df.to_parquet(file_path)
    print(f"Ticker descriptions saved to {file_path}")

    return df

# Function to download ticker prices
def download_ticker_prices(tickers, refresh=REFRESH_TICKER_DATA):
    file_path = 'ticker_prices.parquet'

    if not refresh and os.path.exists(file_path):
        print("Ticker prices already exist. Skipping download.")
        return pd.read_parquet(file_path)

    print("Downloading ticker prices from yfinance...")
    all_prices = []
    for ticker in tickers:
        try:
            data = yf.download(ticker, start="2019-01-01")
            data.reset_index(inplace=True)
            data['Ticker'] = ticker
            data.columns = [[col[0] for col in data.columns]]
            all_prices.append(data)
        except Exception as e:
            print(f"Failed to download prices for {ticker}: {e}")

    # Combine all prices into a single DataFrame
    if all_prices:
        prices_df = pd.concat(all_prices, ignore_index=True)

        # Save to parquet file
        prices_df.to_parquet(file_path)
        print(f"Ticker prices saved to {file_path}")

        return pd.read_parquet(file_path)
    else:
        print("No prices were downloaded.")
        return pd.DataFrame()

# Example usage
descriptions = download_ticker_descriptions(tickers)
prices = download_ticker_prices(tickers)



# -------------------------------- Exercice 3 --------------------------------

# Rename columns if from parquet
prices.rename(columns={
    "('Close',)": 'Close', 
    "('Ticker',)": 'Ticker', 
    "('Date',)": 'Date'
}, inplace=True)

# Transform to matrix format
prices=pd.pivot_table(
    prices,
    index='Date',
    columns='Ticker',
    values="Close",
    aggfunc='last'
)

# Retreat data
prices.ffill(inplace=True) # Forward fill
prices=prices.iloc[1:] # Drop first line (1st january)

# Filter for reproduction purposes
prices = prices.loc[:'2026-01-01']

# Convert to EUR
fx=prices[ccy_tickers]
prices.drop(columns=ccy_tickers, inplace=True)
fx["EUREUR=X"]=1.0
currencies=descriptions.set_index('ticker')['currency'].to_dict()
eur_prices=prices/fx[[f"EUR{currencies[ticker]}=X" for ticker in prices.columns]].values

# -------------------------------- Exercice 4 --------------------------------

# Function to calculate performance summary
def calculate_performance_summary(eur_prices, descriptions, perf_category):
    res={}

    # Loop on categories & compute total performance
    for cat, (start_date, end_date) in perf_category.items():
        # Convert to datetime
        start_date=pd.to_datetime(start_date)
        if end_date is not None:
            end_date=pd.to_datetime(end_date)

        # Get data for performance calculation
        if end_date is not None:
            if end_date in eur_prices.index:
                end_perf=eur_prices.loc[end_date]
            else:
                end_perf=eur_prices.loc[:end_date].iloc[-1]  # Use the closest available date before end_date
        else:
            end_perf=eur_prices.iloc[-1]
        if start_date in eur_prices.index:
            start_perf=eur_prices.loc[start_date]
        else:
            start_perf=eur_prices.loc[:start_date].iloc[-1]

        # Compute performance
        res[cat]=(end_perf/start_perf)-1

    # Create a DataFrame for performance
    perf_summary = pd.DataFrame(res)

    # Merge with descriptive data
    perf_summary = perf_summary.merge(descriptions, left_index=True, right_on='ticker', how='left')

    return perf_summary.set_index('ticker')

perf_category={
    "Perf 2024": ("2024-01-01", "2025-01-01"),
    "Perf 2025": ("2025-01-01", "2026-01-01"),
    "Perf 3Y": (eur_prices.iloc[-1].name - pd.DateOffset(years=3), None)
}

# Calculate performance summary
synthese=calculate_performance_summary(eur_prices, descriptions, perf_category)

# Chart of VL for each assets since 2024
names=descriptions[['ticker', 'name']].set_index('ticker').to_dict()['name']
vl=eur_prices.loc["2024-01-01":, indices_tickers]
vl=vl/vl.iloc[0]*100.0
vl.drop(columns=["^VIX"], inplace=True) # Drop VIX for better visualization
vl.rename(columns=names, inplace=True)
plt.figure(figsize=(10, 6))
for column in vl.columns:
    plt.plot(vl.index, vl[column], label=column)
plt.title('VL (indexes)')
plt.xlabel('Date')
plt.ylabel('VL (Normalized to 100)')
plt.legend()
plt.grid(True)
plt.savefig('1.vl_indexes.png')
plt.close()

# Create a bar chart for monthly performance comparison
monthly_perf=eur_prices[["^FCHI", "^GSPC"]].loc["2025-01-01":"2025-12-31"].resample('M').last().pct_change().dropna().rename(columns=names)

monthly_perf.index = monthly_perf.index.strftime('%Y-%m')
monthly_perf.plot(kind='bar', figsize=(10, 6))
plt.title('Monthly Performance Comparison (2025)')
plt.xlabel('Month')
plt.ylabel('Performance')
plt.xticks(rotation=45)
plt.legend(title='Assets')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('2.monthly_performance.png')
plt.close()

# -------------------------------- Exercice 5 --------------------------------

# Compute log returns
log_returns = np.log(eur_prices / eur_prices.shift(1)).dropna()

# Compute annualized volatility
annualized_coefficient = np.sqrt(252)  # Assuming 252 trading days in a year
synthese['Vol(3M)'] = log_returns.tail(int(252/4)).std() * annualized_coefficient
synthese['Vol(1Y)'] = log_returns.tail(252).std() * annualized_coefficient
synthese['Vol(3Y)'] = log_returns.tail(252*3).std() * annualized_coefficient

# -------------------------------- Exercice 6 --------------------------------

ptf_date="2024-12-31"

# Creation of portfolio 1 : market weighted on France
ptf1 = descriptions[descriptions['country'] == 'France']
ptf1["weight"] = ptf1["marketCap"] / ptf1["marketCap"].sum()
ptf1=ptf1[['ticker', 'weight']]
ptf1["name"]="Ptf1"

# Creation of portfolio 2 : equal weighted on France
ptf2 = descriptions[descriptions['country'] == 'France']
ptf2["weight"] = 1.0 / len(ptf2)
ptf2=ptf2[['ticker', 'weight']]
ptf2["name"]="Ptf2"

# Creation of portfolio 3 : user defined
ptf_3 = pd.DataFrame({
    'ticker': ['GLE.PA', 'SAN.PA', 'AI.PA', 'AC.PA', 'MC.PA'],
    'valo': [65000, 55000, 56000, 52000, 62000]
})
ptf_3['weight'] = ptf_3['valo'] / ptf_3['valo'].sum()
ptf_3=ptf_3[['ticker', 'weight']]
ptf_3["name"]="Ptf3"

# Merge portfolios
ptfs=pd.concat([ptf1, ptf2, ptf_3], ignore_index=True)
ptfs["date"]=ptf_date

# Save to cache
if not os.path.exists("3.portfolios.csv"):
    ptfs.to_csv("3.portfolios.csv", index=False, sep=";")
ptfs=pd.read_csv("3.portfolios.csv", sep=";")

# -------------------------------- Exercice 8 --------------------------------

# Compute VL for each portfolio, we suppose 1000 EUR invested
nominal=1000.0
start_price=eur_prices.loc[ptf_date].T
end_price=eur_prices.iloc[-1].T
vl_prices=eur_prices.loc[ptf_date:]

ptfs["nominal"]=nominal
ptfs["start_price"]=ptfs["ticker"].map(start_price)
ptfs["end_price"]=ptfs["ticker"].map(end_price)
ptfs["nb_shares"]=ptfs["nominal"]*ptfs["weight"]/ptfs["start_price"]
ptfs["P&L (EUR)"]=(ptfs["end_price"]-ptfs["start_price"])*ptfs["nb_shares"]

ptf_vls=pd.DataFrame({
    ptf_name: (vl_prices[ptf_data['ticker']].mul(ptf_data['nb_shares'].values, axis=1).sum(axis=1))
    for ptf_name, ptf_data in ptfs.groupby('name')
})

# Plot VL for each portfolio
plt.figure(figsize=(10, 6))
for column in ptf_vls.columns:
    plt.plot(ptf_vls.index, ptf_vls[column], label=column)
plt.title('VL of Portfolios')
plt.xlabel('Date')
plt.ylabel('VL (EUR)')
plt.legend(title='Portfolios')
plt.grid(True)
plt.savefig('4.vl_portfolios.png')
plt.close()

# Sort ptfs for better readability + merge with descriptions
ptfs=ptfs.sort_values(by=['name', "P&L (EUR)"], ascending=[True, False]).reset_index(drop=True)
ptfs=pd.merge(
    ptfs,
    descriptions[['ticker', 'name', 'sector', 'industry']].rename(columns={'name': 'asset_name'}),
    on='ticker',
    how='left'
)

# Show graphical P&l by category for a given portfolio
ptf="Ptf1"
category="sector"

ptf_data=ptfs[ptfs['name']==ptf]
ptf_data=ptf_data.groupby(category).agg({
    'P&L (EUR)': 'sum'
}).reset_index().sort_values(by='P&L (EUR)', ascending=False)
plt.figure(figsize=(10, 6))
plt.bar(ptf_data[category], ptf_data['P&L (EUR)'])
plt.title(f'P&L by {category.capitalize()} for {ptf}')
plt.xlabel(category.capitalize())
plt.ylabel('P&L (EUR)')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(f'5.pnl_by_{category}_{ptf}.png')
plt.close()

# -------------------------------- Exercice 9 --------------------------------

ptf="Ptf3"
bmk="Ptf1"

# Comparaison of the VL of the two portfolios + relative performance
ptf_vl=ptf_vls[ptf]
bmk_vl=ptf_vls[bmk]
relative_perf=(ptf_vl-bmk_vl)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

for column in ptf_vls.columns:
    ax1.plot(ptf_vls.index, ptf_vls[column], label=column)
ax1.set_title('VL of Portfolios')
ax1.set_xlabel('Date')
ax1.set_ylabel('VL (EUR)')
ax1.legend(title='Portfolios')
ax1.grid(True)

ax2.plot(relative_perf.index, relative_perf, label=f'{ptf} vs {bmk}')
ax2.set_title(f'Relative Performance of {ptf} vs {bmk}')
ax2.set_xlabel('Date')
ax2.set_ylabel('Relative Performance (EUR)')
ax2.axhline(0, color='red', linestyle='--')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('6.ptf_bmk_performance.png')
plt.close()

# Beta, correlation and tracking error (ex post)
ptf_returns = ptf_vl.pct_change().dropna()
bmk_returns = bmk_vl.pct_change().dropna()
cov_matrix = np.cov(ptf_returns, bmk_returns)

ptf_indicators={
    "ptf": ptf,
    "bmk": bmk,
    "ptf vol_expost": float(np.std(ptf_returns) * np.sqrt(252)),
    "bmk vol_expost": float(np.std(bmk_returns) * np.sqrt(252)),
    "beta_expost": float(cov_matrix[0, 1] / cov_matrix[1, 1]),
    "correlation_expost": float(np.corrcoef(ptf_returns, bmk_returns)[0, 1]),
    "tracking_error_expost": float(np.std(ptf_returns - bmk_returns) * np.sqrt(252))  
}

# Ex ante calculations
weights=ptfs[ptfs['name'].isin([ptf, bmk])]
weights=weights.pivot(index='ticker', columns='name', values='weight').fillna(0)
weights.rename(columns={ptf: 'ptf_weight', bmk: 'bmk_weight'}, inplace=True)
weights["active_weight"]=weights['ptf_weight']-weights['bmk_weight']
cov_matrix=log_returns.cov()*252  
ptf_indicators["ptf vol_exante"]=float(weights['ptf_weight'].T @ cov_matrix.loc[weights.index, weights.index] @ weights['ptf_weight'])**0.5
ptf_indicators["bmk vol_exante"]=float(weights['bmk_weight'].T @ cov_matrix.loc[weights.index, weights.index] @ weights['bmk_weight'])**0.5
ptf_indicators["tracking_error_exante"]=float(weights['active_weight'].T @ cov_matrix.loc[weights.index, weights.index] @ weights['active_weight'])**0.5

# MCTR / CTR todo

# Save synthesis to CSV
synthese.to_csv("0.synthese.csv", index=True, sep=";")
ptfs.to_csv("0.portfolios_detailed.csv", index=False, sep=";")
with open("0.ptf_indicators.json", "w") as f:
    json.dump(ptf_indicators, f, indent=4)