import pandas as pd


df = pd.read_csv(
    "data/processed/scored_stocks_1m.csv"
)

df["Date"] = pd.to_datetime(df["Date"])


TOP_N = 10
#This determines the number of stocks your backtest will select for the portfolio each month

df["Month"] = df["Date"].dt.to_period("M")

portfolio_returns = []

for month, group in df.groupby("Month"):

    group = group.dropna( #drops the rows without any values
        subset=["Forward_1M_Return"]
    )

    if len(group) == 0:
        continue

    top = group.nlargest(
        TOP_N,  #Finds the top rows inside the group that have the highest numerical values in the "probability" column
        "probability"
    )

    monthly_return = (
        top["Forward_1M_Return"]
        .mean()
    )

    portfolio_returns.append({
        "Month": month,
        "Return": monthly_return
    })


portfolio_df = pd.DataFrame(
    portfolio_returns
)

portfolio_df = portfolio_df.sort_values(
    "Month"  #sorts chronologically according to month
)

portfolio_df["Cumulative_Return"] = (
    1 + portfolio_df["Return"]
).cumprod()
#Calculates the compound growth of the portfolio, it adds 1 to the percentage return (e.g., a 5% return becomes 1.05)


market = (
    df.groupby("Month")["Forward_1M_Return"]
      .mean()
      .reset_index()
)

market.columns = [
    "Month",
    "Market_Return"
]

market["Cumulative_Market"] = (
    1 + market["Market_Return"]
).cumprod()


results = pd.merge(
    portfolio_df,
    market,
    on="Month"
)


total_return = (
    results["Cumulative_Return"]
    .iloc[-1]
)

market_return = (
    results["Cumulative_Market"]
    .iloc[-1]   #Extracts the very last value (.iloc[-1])
)

months = len(results)

cagr = (
    total_return ** (12 / months)
) - 1

print("\n===== BACKTEST RESULTS =====")
print(results.tail())

print("\nPortfolio Final Value:")
print(total_return)

print("\nMarket Final Value:")
print(market_return)

print("\nApprox CAGR:")
print(cagr)


results.to_csv(
    "data/processed/backtest_results_1m.csv",
    index=False
)

print("\nSaved:")
print("data/processed/backtest_results_1m.csv")