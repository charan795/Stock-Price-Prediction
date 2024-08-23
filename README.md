# Stock-Price-Prediction
This Python script is designed to predict future stock prices by leveraging a combination of technical analysis, feature engineering, 
and advanced forecasting techniques. The process starts with data acquisition, where historical stock prices and benchmark indices data 
are obtained from Yahoo Finance. This data is supplemented with calculated technical indicators, such as simple moving averages (SMA), 
exponential moving averages (EMA), and relative strength index (RSI). Additionally, financial metrics like returns, volatility, and 
Sharpe ratios are computed to provide a comprehensive view of the stock's performance.

The script then performs feature engineering by integrating technical indicators with fundamental financial data from quarterly reports. 
To handle quarterly data granularity, interpolation techniques are employed to generate daily values, ensuring a continuous dataset.

Feature selection is achieved using a Random Forest Regressor, which identifies significant predictors among the technical indicators and 
financial metrics. The features are then standardized and subjected to Principal Component Analysis (PCA), reducing dimensionality and 
enhancing model efficiency. An Ordinary Least Squares (OLS) regression model assesses the importance of these PCA components.

Forecasting is approached in a two-stage process: First, the script uses SARIMA (Seasonal AutoRegressive Integrated Moving Average) to 
forecast benchmark indices and other exogenous variables based on historical data. These forecasts provide the input for the subsequent 
stock price prediction model. In the second stage, SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors) 
is employed to predict stock prices using the forecasted values of benchmark indices and PCA components as exogenous variables.

The model's performance is evaluated using statistical metrics, including Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), 
and R-squared (R²). The script generates forecasts for the next 30 days and visualizes both historical and predicted stock prices, 
offering insights into future trends and model accuracy.

This comprehensive approach integrates multiple analytical methods and forecasting techniques to deliver robust and data-driven stock price 
predictions.
