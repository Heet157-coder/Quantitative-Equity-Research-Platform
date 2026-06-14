import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
 

st.set_page_config(
    page_title="Quantitative Equity Research Platform",  #sets the cover page and title
    layout="wide"
)

# @ is a decorator
@st.cache_data  #This prevents Streamlit from re-reading the CSV file from disk every time the user interacts with a widget
def load_data():
    df = pd.read_csv(
        "data/processed/final_dataset_clean_v4.csv"
    )

    df["Date"] = pd.to_datetime(df["Date"])

    return df


@st.cache_data
def load_backtest():

    bt = pd.read_csv(
        "data/processed/backtest_results_1m.csv"
    )

    return bt


df = load_data()
backtest = load_backtest()


try:

    model = joblib.load(     #load model from the dump
        "models/random_forest_1m.pkl"
    )

    model_loaded = True

except Exception as e:

    st.error(f"Model loading failed: {e}")

    model_loaded = False



latest = (
    df.sort_values("Date")    #gives you the latest data we have in the data set after sorting according to the dates
      .groupby("Stock")
      .tail(1)
      .reset_index(drop=True)
)


if model_loaded:

    expected_features = list(
        model.feature_names_in_
    )

    missing_features = [
        col
        for col in expected_features
        if col not in latest.columns
    ]

    if len(missing_features) > 0:

        st.error(
            f"Missing Features: {missing_features}"
        )

        latest["Buy_Probability"] = 0.50 #If features are missing, it throws a warning UI message and assigns a baseline buy probability of 50% to all stocks as a fallback strategy

    else:

        X_app = (  #If all features are present, it isolates just those feature columns into a new DataFrame (X_app)
            latest[
                expected_features
            ]
            .copy()
        )

        X_app = X_app.fillna(  #Fills any missing (NaN) values with the median of that specific feature column to avoid evaluation errors.
            X_app.median(
                numeric_only=True
            )
        )

        probs = model.predict_proba(
            X_app #predict_proba returns the classification probability pairs [probability_of_0, probability_of_1]
        )[:, 1]  #[:, 1] slices the output to keep only the probability of the positive class (the probability that the stock is a buy)

        latest["Buy_Probability"] = probs

else:

    latest["Buy_Probability"] = 0.50 #if the model failed to load in step 3 entirely, every stock gets assigned a default 0.50 probability


latest = latest.sort_values(
    "Buy_Probability", #Sorts the entire latest stock list from highest probability to lowest so the best-performing candidates appear at the top
    ascending=False
)

st.sidebar.title(  #creates the sidebar
    "Navigation"
)

page = st.sidebar.radio( #The selected page name is stored in the string variable page
    "Select Page",
    [
        "Dashboard",
        "Stock Screener",
        "Stock Research",
        "Portfolio Builder",
        "Backtest",
        "Risk Metrics",
        "Model Diagnostics"
    ]
)


if page == "Dashboard":

    st.title(
        "Quantitative Equity Research Platform"
    )

    col1, col2, col3 = st.columns(3)  #Uses st.columns(3) to divide the viewport horizontally

    col1.metric(
        "Stocks Covered",
        latest["Stock"].nunique()
    )

    col2.metric(
        "Latest Date",
        str(df["Date"].max().date())
    )

    col3.metric(
        "Observations",
        len(df)
    )

    st.subheader(
        "Top 10 Ranked Stocks"
    )

    top10 = latest[
        [
            "Stock",
            "Close",
            "Buy_Probability"
        ]
    ].head(10)

    top10["Buy_Probability"] = (
        top10["Buy_Probability"] * 100
    ).round(2)   #Rounds off to the 2nd decimal

    st.dataframe(
        top10,
        use_container_width=True
    )


elif page == "Stock Screener":

    st.title("Stock Screener")

    min_prob = st.slider( #generates a dynamic adjustment slider ranging from 0% to 100%, defaulting to 50%
        "Minimum Probability (%)",
        0,
        100,
        50
    )

    screener = latest[  #Filters out rows where the model prediction is lower than the user's slider threshold (min_prob/100)
        latest["Buy_Probability"] >= min_prob/100
    ]

    display = screener[
        [
        "Stock",
        "Close",
        "PE",
        "PB",
        "ROE",
        "Market_Cap",
        "Buy_Probability"
        ]
    ].copy()

    display["Buy_Probability"] = (
        display["Buy_Probability"] * 100
    ).round(2)

    st.dataframe(
        display,
        use_container_width=True
    )  #Picks fundamental metrics (Price-to-Earnings, Price-to-Book, Return on Equity, etc.) along with the model score and renders them inside an expandable, sortable interactive data frame grid


elif page == "Stock Research":

    st.title("Stock Research")

    stock = st.selectbox(  #Creates a searchable dropdown menu component listing all unique stock tickers alphabetically
        "Select Stock",
        sorted(df["Stock"].unique())
    )

    stock_df = df[
        df["Stock"] == stock
    ]

    latest_row = stock_df.iloc[-1] #Filters the master history down to that chosen entity, capturing its final individual entry (.iloc[-1])

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Price",
        f"₹{latest_row['Close']:.2f}"
    )

    c2.metric(
        "PE",
        round(latest_row["PE"],2)
    )

    c3.metric(
        "ROE",
        round(latest_row["ROE"],2)
    )

    c4.metric(
        "PB",
        round(latest_row["PB"],2)
    )

    fig = px.line( #draws a Plotly interactive line chart mapping out the selected asset's temporal Close price trajectory across the full timeframe data
        stock_df,
        x="Date",
        y="Close",
        title=f"{stock} Price History"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "Technical Indicators"
    )

    tech = pd.DataFrame({

        "Metric":[
            "Return_1M",
            "Return_3M",
            "Return_6M",
            "RSI",
            "MACD",
            "Volatility",
            "Distance_52W_High"
        ],

        "Value":[
            latest_row["Return_1M"],
            latest_row["Return_3M"],
            latest_row["Return_6M"],
            latest_row["RSI_14"],
            latest_row["MACD"],
            latest_row["Volatility_30D"],
            latest_row["Distance_52W_High"]
        ]
    })

    st.dataframe(
        tech,
        use_container_width=True
    )


elif page == "Portfolio Builder":

    st.title(
        "Portfolio Builder"
    )

    n = st.slider(
        "Number of Stocks",
        5,
        20,
        10
    )

    portfolio = latest.head(n).copy()

    st.dataframe(
        portfolio[
            [
                "Stock",
                "Close",
                "Buy_Probability"
            ]
        ],
        use_container_width=True
    )

    st.metric(
        "Average Probability",
        round(
            portfolio[
                "Buy_Probability"
            ].mean()*100,
            2
        )  #Calculates and shows the mean buy probability of the chosen basket as a combined portfolio quality score
    )


elif page == "Backtest":

    st.title(
        "Backtest Results"
    )

    fig = px.line(
        backtest,
        x="Month",
        y=[
            "Cumulative_Return",
            "Cumulative_Market"
        ]
    )  #Generates a dual-line chart comparing your algorithmic strategy performance (Cumulative_Return) against a standard benchmark reference point (Cumulative_Market) over time

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        backtest.tail(),
        use_container_width=True
    )


elif page == "Risk Metrics":

    st.title(
        "Risk Metrics"
    )
    #Extracts the non-cumulative monthly returns vector and estimates the total time span in years
    monthly_returns = backtest["Return"]

    years = len(backtest)/12
    #Calculates the Compound Annual Growth Rate (CAGR). It takes the final terminal value, raises it to the inverse power of the total years, and subtracts 1.
    cagr = (
        backtest[
            "Cumulative_Return"
        ].iloc[-1]
        ** (1/years)
    ) - 1
    #Calculates the annualized Sharpe Ratio (assuming a 0% risk-free rate) by dividing the average monthly return by the standard deviation, then multiplying by root 12 to convert it from monthly to annual scale
    sharpe = (
        monthly_returns.mean()
        /
        monthly_returns.std()
    ) * np.sqrt(12)
#Calculates the peak-to-trough decline (drawdown) profile by comparing each period's return to its historical maximum up to that point (.cummax())
    drawdown = (
        backtest[
            "Cumulative_Return"
        ]
        /
        backtest[
            "Cumulative_Return"
        ].cummax()
    ) - 1
    #Extracts the worst drop value to find the Maximum Drawdown (max_dd)
    max_dd = drawdown.min()

    c1,c2,c3 = st.columns(3)

    c1.metric(
        "CAGR",
        f"{cagr:.2%}"
    )

    c2.metric(
        "Sharpe Ratio",
        round(sharpe,2)
    )

    c3.metric(
        "Max Drawdown",
        f"{max_dd:.2%}"
    )


elif page == "Model Diagnostics":

    st.title(
        "Model Diagnostics"
    )

    st.markdown("""
### Model

Random Forest Classifier

### Inputs

- Momentum Features
- Trend Features
- Volatility Features
- Volume Features

### Important Features

- Return_6M
- SMA_200
- SMA_50
- SMA_20
- Volatility_30D
- MACD
- ATR_14
- Distance_52W_High

### Purpose

Rank stocks based on probability
of positive future performance.
""")

    importance = pd.DataFrame({

        "Feature":[
            "Return_6M",
            "SMA_200",
            "SMA_50",
            "SMA_20",
            "Volatility_30D",
            "MACD",
            "ATR_14",
            "Distance_52W_High"
        ],

        "Importance":[
            0.11,
            0.10,
            0.10,
            0.09,
            0.08,
            0.04,
            0.05,
            0.07
        ]
    })

    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )