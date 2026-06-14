# Quantitative Equity Research Platform

## Overview

This project is an end-to-end quantitative equity research platform built using Python and Machine Learning. The objective is to identify and rank stocks based on technical and financial characteristics, construct systematic portfolios, and evaluate performance through historical backtesting.

The platform covers the complete research workflow:

* Data Collection and Processing
* Feature Engineering
* Machine Learning Model Development
* Stock Ranking and Screening
* Portfolio Construction
* Backtesting and Performance Evaluation
* Interactive Streamlit Dashboard

Rather than predicting exact stock prices, the system ranks stocks according to their expected future performance using historical market patterns and quantitative factors.

---

## Features

### Market Data Processing

* Historical stock price collection
* Data cleaning and preprocessing
* Missing value handling
* Multi-stock dataset construction

### Feature Engineering

Technical indicators:

* 1M, 3M, and 6M Momentum
* SMA 20, SMA 50, SMA 200
* RSI (Relative Strength Index)
* Volatility Measures
* MACD
* Bollinger Bands
* ATR (Average True Range)
* Relative Volume
* Distance from 52-Week High

Fundamental factors:

* PE Ratio
* PB Ratio
* ROE
* ROA
* Debt-to-Equity Ratio
* Current Ratio
* Revenue Growth
* Earnings Growth
* EV/EBITDA
* Market Capitalization

### Machine Learning Models

* Logistic Regression
* Random Forest Classification
* Random Forest Regression
* Probability-Based Stock Ranking

### Portfolio Analytics

* Monthly Portfolio Rebalancing
* Benchmark Comparison
* CAGR Calculation
* Sharpe Ratio
* Maximum Drawdown Analysis

### Interactive Dashboard

Built using Streamlit for:

* Stock Rankings
* Portfolio Performance
* Risk Analytics
* Model Insights
* Historical Backtest Results

---

## Technology Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Matplotlib
* Streamlit
* Yahoo Finance API

---

## Project Workflow

Raw Market Data
↓
Data Cleaning
↓
Feature Engineering
↓
Machine Learning Models
↓
Stock Ranking
↓
Portfolio Construction
↓
Backtesting
↓
Performance Evaluation
↓
Interactive Dashboard

---

## Key Learning Outcomes

This project provided hands-on experience in:

* Quantitative Finance
* Financial Data Analysis
* Feature Engineering
* Machine Learning for Finance
* Portfolio Construction
* Backtesting Framework Design
* Risk Analysis
* Dashboard Development

---

## Disclaimer

This project is intended for educational and research purposes only. It does not constitute financial or investment advice. Historical performance does not guarantee future results.
