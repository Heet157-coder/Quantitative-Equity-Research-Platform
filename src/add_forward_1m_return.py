import pandas as pd
import os


print("Files in data/processed:\n")

for file in os.listdir("data/processed"):
    print(file)



INPUT_FILE = "data/processed/clean_dataset_v2.csv"



df = pd.read_csv(INPUT_FILE)

print("\nLoaded:")
print(INPUT_FILE)

print("\nColumns:")
print(df.columns)



df["Date"] = pd.to_datetime(df["Date"])



df = df.sort_values(
    ["Stock", "Date"]
)


TRADING_DAYS = 21

df["Forward_1M_Return"] = (
    df.groupby("Stock")["Close"]
      .shift(-TRADING_DAYS)     #shift(-21) leads you to 21 day ahead of the current day
      .div(df["Close"])
      .sub(1)
)


df["Target_1M"] = (
    df["Forward_1M_Return"] > 0
).astype(int)    #gives output 1 if true and 0 if false


print("\nForward Return Preview:\n")

print(
    df[
        [
            "Date",
            "Stock",
            "Close",
            "Forward_1M_Return",
            "Target_1M"
        ]
    ].head(10)
)

print("\nMissing Forward Returns:")
print(
    df["Forward_1M_Return"]
    .isna()
    .sum()
)


OUTPUT_FILE = "data/processed/dataset_with_forward_1m.csv"

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)

print("\nFinal Shape:")
print(df.shape)