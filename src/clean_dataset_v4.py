import pandas as pd

df = pd.read_csv(
    "data/processed/final_dataset_v3.csv"
)

print("Original Shape:")
print(df.shape)

# Remove rows without future target

df = df.dropna(
    subset=[
        "Forward_1M_Return"
    ]
)

# Fundamental columns

fundamental_cols = [
    "PE",
    "PB",
    "ROE",
    "ROA",
    "Debt_Equity",
    "Current_Ratio",
    "Revenue_Growth",
    "Earnings_Growth",
    "EV_EBITDA",
    "Market_Cap"
]

for col in fundamental_cols:

    if col in df.columns:

        df[col] = df[col].fillna(
            df[col].median()     #Fills blank spaces with the median of the values
        )



numeric_cols = df.select_dtypes(
    include="number"   #scans the entire dataset and grabs a list of all column names that contain numeric data
).columns

df[numeric_cols] = (
    df.groupby("Stock")[numeric_cols]
      .transform(   #ffill = If it hits an empty value, it copies the value from the previous day to fill it
          lambda x: x.ffill().bfill()  #bfill = If it hits an empty value, it copies the next available valid day's value backward to fill it
      )
)

print("\nShape After Cleaning:")
print(df.shape)

print("\nRemaining Missing Values:")
print(
    df.isna().sum().sum()   #to check if any empty row is left or not
)

df.to_csv(
    "data/processed/final_dataset_clean_v4.csv",
    index=False
)

print(
    "\nSaved: final_dataset_clean_v4.csv"
)