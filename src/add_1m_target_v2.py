import pandas as pd

df = pd.read_csv(
    "data/processed/final_dataset_v2.csv"
)

df["Date"] = pd.to_datetime(
    df["Date"]
)

df = df.sort_values(
    ["Stock", "Date"]
)


df["Forward_1M_Return"] = (
    df.groupby("Stock")["Close"]
      .shift(-21)
      .div(df["Close"])
      .sub(1)
)



df["Target_1M"] = (
    df["Forward_1M_Return"] > 0
).astype(int)

print(df[
    [
        "Stock",
        "Date",
        "Forward_1M_Return",
        "Target_1M"
    ]
].tail())

df.to_csv(
    "data/processed/final_dataset_v3.csv",
    index=False
)

print(
    "\nSaved: final_dataset_v3.csv"
)