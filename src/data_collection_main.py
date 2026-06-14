import yfinance as yf
import pandas as pd
import os

# NIFTY 50 stocks

stocks = [
    "ADANIPORTS.NS",
    "ASIANPAINT.NS",
    "AXISBANK.NS",
    "BAJAJ-AUTO.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "BEL.NS",
    "BHARTIARTL.NS",
    "CIPLA.NS",
    "COALINDIA.NS",
    "DRREDDY.NS",
    "EICHERMOT.NS",
    "ETERNAL.NS",      
    "GRASIM.NS",
    "HCLTECH.NS",
    "HDFCBANK.NS",
    "HDFCLIFE.NS",
    "HEROMOTOCO.NS",
    "HINDALCO.NS",
    "HINDUNILVR.NS",
    "ICICIBANK.NS",
    "INDUSINDBK.NS",
    "INFY.NS",
    "ITC.NS",
    "JIOFIN.NS",
    "JSWSTEEL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "M&M.NS",
    "MARUTI.NS",
    "NESTLEIND.NS",
    "NTPC.NS",
    "ONGC.NS",
    "POWERGRID.NS",
    "RELIANCE.NS",
    "SBILIFE.NS",
    "SBIN.NS",
    "SHRIRAMFIN.NS",
    "SUNPHARMA.NS",
    "TATACONSUM.NS",
    "TATAMOTORS.NS",
    "TATASTEEL.NS",
    "TCS.NS",
    "TECHM.NS",
    "TITAN.NS",
    "TRENT.NS",
    "ULTRACEMCO.NS",
    "WIPRO.NS"
]



os.makedirs("data/raw", exist_ok=True)   #Creates a raw data folder

successful = []
failed = []

for ticker in stocks:
    try:                                    #try-except block the entire script from crashing if a single stock fails
        print(f"Downloading {ticker}...")

        data = yf.download(
            ticker,
            start="2018-01-01",
            end="2025-01-01",
            auto_adjust=True,           #helps like if the stock price is split so it helps us to compare by showing proper values
            progress=False,             #disables the animated progress bar in the terminal to keep the output clean
            group_by = "column"         #groups the downloaded dataframe structure by column names (Open, High, Low, Close etc)
        )

        if isinstance(data.columns, pd.MultiIndex):         #sometime yfinance returns multiple indices
            data.columns = data.columns.get_level_values(0) #If they are there then data.columns.get_level_values(0) flattens them

        data = data.reset_index()   #by default yfinance gives dates and this reset_index() make a standard column of date

        if data.empty:
            failed.append(ticker)
            print(f"No data for {ticker}")
            continue

        filename = f"data/raw/{ticker.replace('.NS', '')}.csv"

        data.to_csv(filename, index = False)

        successful.append(ticker)

        print(f"Saved {filename}")

    except Exception as e:
        failed.append(ticker)
        print(f"Error with {ticker}: {e}")

print("\nDOWNLOAD SUMMARY")
print("-" * 40)
print(f"Successful: {len(successful)}")
print(f"Failed: {len(failed)}")

if failed:
    print("\nFailed tickers:")
    for ticker in failed:
        print(ticker)