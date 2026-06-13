# Stock Price Forecasting — SPY ETF
 
## Project Overview
End-to-end time series forecasting project predicting next-day returns for the S&P 500 ETF (SPY) using machine learning. Built as part of a 5-project ML portfolio targeting junior AI/ML roles in fintech.
 
## Problem Statement
Predict the direction and magnitude of SPY's next-day return using historical price data and engineered technical features. The target variable is the next trading day's percentage return.
 
## Dataset
- **Source:** Yahoo Finance via `yfinance` Python library
- **Asset:** SPY (S&P 500 ETF)
- **Period:** January 2015 – December 2025
- **Frequency:** Daily (trading days only)
- **Size:** 2,765 rows, 5 raw columns (Open, High, Low, Close, Volume)
- **Missing values:** None
## Project Structure
```
stock-price-forecasting/
├── data/
│   ├── spy_raw.csv              ← Raw data from yfinance
│   ├── spy_processed.csv        ← Feature-engineered dataset
│   ├── spy_train.csv            ← Training set (2015–2023)
│   └── spy_test.csv             ← Test set (2023–2025)
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_evaluation.ipynb
├── models/
├── requirements.txt
└── README.md
```
 
## Notebook Summary
 
### 01 — Data Collection
- Downloaded 10 years of SPY daily OHLCV data using `yfinance`
- Saved raw data to CSV for reproducibility
### 02 — Exploratory Data Analysis
Four key findings:
1. **Strong upward trend** — raw price is non-stationary; statistical properties change over time
2. **Volatility clustering** — extreme return spikes cluster around crisis periods (COVID 2020, rate hikes 2022, tariff shock 2025)
3. **Heteroskedasticity** — market variance is not constant; rolling volatility captures regime changes
4. **Autocorrelation decay** — recent price lags are strongly correlated; correlation decays over longer horizons
### 03 — Preprocessing & Feature Engineering
Target variable: next day's daily return (`Daily_Return.shift(-1)`)
 
| Feature | Description |
|---|---|
| `Daily_Return` | Percentage change from previous close — ensures stationarity |
| `Close_Lag_1` | Yesterday's closing price |
| `Close_Lag_5` | Closing price 5 days ago (last week) |
| `Close_Lag_20` | Closing price 20 days ago (last month) |
| `Volatility_30` | 30-day rolling standard deviation of returns |
| `MA_20` | 20-day moving average (short-term trend) |
| `MA_50` | 50-day moving average (medium-term trend) |
 
**Train/test split:** time-based (no shuffling)
- Train: March 2015 – October 2023 (2,172 rows)
- Test: October 2023 – December 2025 (543 rows)
### 04 — Modeling
Two models evaluated:
 
| Model | Description |
|---|---|
| Naive Baseline | Predict zero return every day (no change) |
| XGBoost | Gradient boosting regressor (500 estimators, lr=0.05, max_depth=4) |
 
### 05 — Evaluation
 
| Metric | Baseline | XGBoost |
|---|---|---|
| MAE | 0.0066 | 0.0169 |
| RMSE | 0.0101 | 0.0194 |
| Directional Accuracy | — | 42.36% |
 
## Key Findings & Honest Assessment
The XGBoost model **failed to outperform the naive baseline**. Directional accuracy of 42.36% is below the 50% coin-flip threshold, and the model-driven trading strategy turned $1 into $0.69 while buy-and-hold returned $1.69 over the same period.
 
**Root cause:** The model relied heavily on price-level features (MA_20, Close_Lag_5, Close_Lag_20) rather than return-predictive signals. Daily returns have an extremely low signal-to-noise ratio, and tree-based models fit spurious patterns in training data that do not generalize to future observations.
 
## Proposed Improvements
To improve predictive performance, future iterations would incorporate:
- **News sentiment scores** (FinBERT-based) as an additional signal source
- **Volume-based momentum indicators** (e.g. OBV, volume change rate)
- **Return-based features** instead of price-level lag features
- **Walk-forward validation** for more robust out-of-sample evaluation
## Tech Stack
- Python 3.10
- `yfinance` — market data
- `pandas`, `numpy` — data manipulation
- `scikit-learn` — metrics and preprocessing
- `xgboost` — gradient boosting model
- `matplotlib` — visualization
## Key Concepts Demonstrated
- Time series stationarity vs non-stationarity
- Temporal leakage prevention (time-based train/test split)
- Feature engineering for financial time series
- Walk-forward validation logic
- Honest model evaluation including failure analysis
 