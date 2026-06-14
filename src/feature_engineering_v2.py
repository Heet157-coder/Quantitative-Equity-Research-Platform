import pandas as pd
import numpy as np

print("STEP 1: Loading dataset...")

df = pd.read_csv(
    "data/processed/final_dataset.csv"
)

print("Loaded successfully")
print("Shape:", df.shape)



print("\nSTEP 2: Processing dates...")

df["Date"] = pd.to_datetime(df["Date"])   #changes the string format of the date to date time format

df = df.sort_values(
    ["Stock", "Date"]            #first categorise into stock group and the filter a acc to dates
)

print("Dates processed")



print("\nSTEP 3: SMA...")   #simple moving averages 

df["SMA_20"] = (
    df.groupby("Stock")["Close"]   #isolates each stock group and the closing prices column
      .transform(lambda x: x.rolling(20).mean())  #.transform() calculates a single step for all rows keeping the structure intact
)                                  #it looks at the current row's closing price and the 19 rows before it and calculates average

df["SMA_50"] = (
    df.groupby("Stock")["Close"]
      .transform(lambda x: x.rolling(50).mean())
)

df["SMA_200"] = (
    df.groupby("Stock")["Close"]
      .transform(lambda x: x.rolling(200).mean())
)

print("SMA complete")



print("\nSTEP 4: Volatility...")

daily_returns = (
    df.groupby("Stock")["Close"]
      .pct_change()      #calculates % changee wrt the previous day
)

df["Volatility_30D"] = (
    daily_returns.groupby(df["Stock"])
                 .transform(   #takes a rolling window of the last 30 daily returns and calculates the standard deviation
                     lambda x: x.rolling(30).std()
                 )
)

print("Volatility complete")



print("\nSTEP 5: RSI...")          #relative strength index

delta = (
    df.groupby("Stock")["Close"]   
      .diff()                    #calculates the difference between todays price and the day before
)

gain = delta.clip(lower=0)   #calculates all the positive deltas

loss = -delta.clip(upper=0)  #calculates all the negative deltas

avg_gain = (
    gain.groupby(df["Stock"])
        .transform(
            lambda x: x.rolling(14).mean()
        )
)                   #calculates a simple 14-day rolling average of the daily gains and daily losses

avg_loss = (
    loss.groupby(df["Stock"])
        .transform(
            lambda x: x.rolling(14).mean()
        )
)

rs = avg_gain / avg_loss    #relative strenght ratio

df["RSI_14"] = (
    100 -
    (100 / (1 + rs))
)

print("RSI complete")


print("\nSTEP 6: Price/SMA ratios...")

df["Price_to_SMA20"] = (
    df["Close"] / df["SMA_20"]
)

df["Price_to_SMA50"] = (
    df["Close"] / df["SMA_50"]
)

df["Price_to_SMA200"] = (
    df["Close"] / df["SMA_200"]
)

print("Price/SMA complete")


print("\nSTEP 7: MACD...")     #moving average convergence divergence

df["EMA12"] = (
    df.groupby("Stock")["Close"]
      .transform(
          lambda x: x.ewm(
              span=12,          #calculates a 12-day Exponential Moving Average
              adjust=False
          ).mean()
      )
)

df["EMA26"] = (
    df.groupby("Stock")["Close"]
      .transform(
          lambda x: x.ewm(
              span=26,
              adjust=False
          ).mean()
      )
)

df["MACD"] = (
    df["EMA12"] -    #Calculated by subtracting the slower 26-day EMA from the faster 12-day EMA
    df["EMA26"]
)

df["MACD_Signal"] = (
    df.groupby("Stock")["MACD"]  
      .transform(      #Takes the newly calculated MACD line and runs a 9-day EMA smooth filter over it
          lambda x: x.ewm(
              span=9,
              adjust=False
          ).mean()
      )
)

df["MACD_Hist"] = (
    df["MACD"] -
    df["MACD_Signal"]
)

print("MACD complete")


print("\nSTEP 8: Bollinger Bands...")  #Bollinger Bands plot lines above and below a moving average based on volatility

rolling_std = (
    df.groupby("Stock")["Close"]
      .transform(
          lambda x: x.rolling(20).std()
      )
)

df["BB_Upper"] = (
    df["SMA_20"] +
    2 * rolling_std
)

df["BB_Lower"] = (
    df["SMA_20"] -
    2 * rolling_std
)

df["BB_Width"] = (
    (df["BB_Upper"] - df["BB_Lower"])
    / df["SMA_20"]
)

print("Bollinger complete")


print("\nSTEP 9: ATR...")   #Average true range

high_low = (
    df["High"] -
    df["Low"]
)

high_prev_close = abs(
    df["High"] -
    df.groupby("Stock")["Close"].shift(1)
)

low_prev_close = abs(
    df["Low"] -
    df.groupby("Stock")["Close"].shift(1)
)

df["TR"] = pd.concat(
    [
        high_low,
        high_prev_close,
        low_prev_close
    ],
    axis=1
).max(axis=1)

df["ATR_14"] = (
    df.groupby("Stock")["TR"]
      .transform(
          lambda x: x.rolling(14).mean()
      )
)

print("ATR complete")


print("\nSTEP 10: Relative Volume...")

df["AvgVolume20"] = (
    df.groupby("Stock")["Volume"]
      .transform(
          lambda x: x.rolling(20).mean()
      )
)

df["RelativeVolume"] = (
    df["Volume"] /
    df["AvgVolume20"]
)

print("Relative Volume complete")


print("\nSTEP 11: 52 Week High...")

df["High_252"] = (
    df.groupby("Stock")["High"]
      .transform(
          lambda x: x.rolling(252).max()
      )
)

df["Distance_52W_High"] = (
    df["Close"] /
    df["High_252"]
)

print("52 Week High complete")



print("\nSTEP 12: Final validation...")

print("Shape:")
print(df.shape)

print("\nColumns:")
print(len(df.columns))

print("\nMissing Values:")
print(df.isna().sum().sum())   #Counts total missing values (NaN)

print("\nFirst 5 rows:")
print(df.head())


print("\nSTEP 13: Saving file...")

df.to_csv(
    "data/processed/final_dataset_v2.csv",
    index=False
)

print("\nSUCCESS")
print("Saved: data/processed/final_dataset_v2.csv")