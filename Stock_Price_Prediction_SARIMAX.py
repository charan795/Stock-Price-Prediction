#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 18:49:41 2024

@author: charanmakkina

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

"""
# Importing required libraries
import pandas as pd # For data manipulation and analysis
import numpy as np # For numerical operations
import yfinance as yf # For fetching financial data from yahoo finance
from datetime import date # For working with dates
import ta # For technical analysis indicators
import statsmodels.api as sm # For statistical models
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler # For feature scaling
from sklearn.decomposition import PCA # For dimensionality reduction
from statsmodels.tsa.stattools import adfuller # For checking stationarity in time series
from statsmodels.tsa.statespace.sarimax import SARIMAX # For SARIMAX time series forecasting
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,root_mean_squared_error # For model evaluation metrics
import matplotlib.pyplot as plt # For plotting graphs
from fastapi import FastAPI, HTTPException # For creating a web API
from pydantic import BaseModel # For data validation in API requests

# Initialize FastAPI application
app=FastAPI()

# Define a pydantic model for validating incoming stock requests
class StockRequest(BaseModel):
    ticker:str

# Define the endpoint for forecasting stock prices
@app.post("/forecast")
async def forecast_stock(request: StockRequest):
    # Create a yfinance Ticker object for the specified stock
    ticker=yf.Ticker(request.ticker)
    
    # List of index symbols for additional data, such as benchmark indices S&P 500, Russell 1000, Russell 2000, S&P 500 Information 
    #Technology. Along with Interest Rate Tickers CBOE Interest Rate 10 Year, 13 Week Treasury Bill, Treasury Yield 30 Years, 
    #Treasury Yield 5 Years, US Treasury 3 month bill ETF
    indices=['^GSPC','^RUI','^RUT','^SP500-45','^TNX','^IRX','^TYX','^FVX','TBIL']
    
    # Fetch the last two years of historical data for the specified stock
    stock_data=ticker.history(period='2y')
    
    # Remove timezone information from the index
    stock_data.index=stock_data.index.tz_localize(None)
    
    # Intialize an empty Dataframe for closing prices of Indices
    close_index=pd.DataFrame()
    
    # Loop through each index symbol to fetch and prepare index data and fetch historical data for each index and concenate each 
    # index to close_index Dataframe
    for index in indices:
        index_ticker=yf.Ticker(index)
        index_data=index_ticker.history(period='2y')
        index_data=index_data[['Close']]
        index_data.rename(columns={'Close':index},inplace=True)
        index_data.index=index_data.index.tz_localize(None)
        close_index=pd.concat([close_index,index_data],axis=1)
    
    # Combine the index data with the stock data
    stock_data=pd.concat([close_index,stock_data],axis=1)   
    
    # Compute daily returns of the stock
    stock_data['Stock_Return']=stock_data['Close'].pct_change()
    
    # Compute daily returns of the benchmark indices
    stock_data['GSPC_Return']=stock_data['^GSPC'].pct_change()
    stock_data['RUI_Return']=stock_data['^RUI'].pct_change()
    stock_data['RUT_Return']=stock_data['^RUT'].pct_change()
    stock_data['SP500-45_Return']=stock_data['^SP500-45'].pct_change()
    
    
    # Compute the average return of the benchmark indices
    stock_data['Benchmark_Return']=stock_data[['GSPC_Return',
    'RUI_Return', 'RUT_Return', 'SP500-45_Return']].mean(axis=1)
    
    # Compute Risk Free Rate using 3-month Treasury Bill in US
    stock_data['Risk_Free_Rate']=stock_data['TBIL']/100/360
    
    # Compute Excess Return for the stock and the benchmark
    stock_data['Stock_Excess_Return']=stock_data['Stock_Return']-stock_data['Risk_Free_Rate']
    stock_data['Benchmark_Excess_Return']=stock_data['Benchmark_Return']-stock_data['Risk_Free_Rate']
    
    
    rolling_window=15
    
    # Compute 15-day volatility of the stock and benchmark on a rolling basis for a window of 15 days.
    stock_data['Stock_Volatility']=stock_data['Stock_Return'].rolling(window=rolling_window).std()
    stock_data['Benchmark_Volatility']=stock_data['Benchmark_Return'].rolling(window=rolling_window).std()
    
    # Compute Sharpe ratio for the stock and the benchmark
    stock_data['Stock_Sharpe_Ratio']=stock_data['Stock_Excess_Return']/stock_data['Stock_Volatility']
    stock_data['Benchmark_Sharpe_Ratio']=stock_data['Benchmark_Excess_Return']/stock_data['Benchmark_Volatility']
    
    rolling_window=5
    
    # Compute Moving Average of the Stock Return, Stock Excess Return, Benchmark Returns, Benchmark Excess Returns on a rolling basis 
    #for a window of 5 days.
    stock_data['Stock_Return_MA']=stock_data['Stock_Return'].rolling(window=rolling_window).mean()
    stock_data['GSPC_Return_MA']=stock_data['GSPC_Return'].rolling(window=rolling_window).mean()
    stock_data['RUI_Return_MA']=stock_data['RUI_Return'].rolling(window=rolling_window).mean()
    stock_data['RUT_Return_MA']=stock_data['RUT_Return'].rolling(window=rolling_window).mean()
    stock_data['SP500-45_Return_MA']=stock_data['SP500-45_Return'].rolling(window=rolling_window).mean()
    stock_data['Benchmark_Return_MA']=stock_data['Benchmark_Return'].rolling(window=rolling_window).mean()
    stock_data['Stock_Excess_Return_MA']=stock_data['Stock_Excess_Return'].rolling(window=rolling_window).mean()
    stock_data['Benchmark_Excess_Return_MA']=stock_data['Benchmark_Excess_Return'].rolling(window=rolling_window).mean()
    
    # Compute the Cumulative Stock Returns, Cumulative Benchmark Returns, based on the moving averages.
    stock_data['Stock_Return_CMA']=(1+stock_data['Stock_Return_MA']).cumprod()-1
    stock_data['GSPC_Return_CMA']=(1+stock_data['GSPC_Return_MA']).cumprod()-1
    stock_data['RUI_Return_CMA']=(1+stock_data['RUI_Return']).cumprod()-1
    stock_data['RUT_Return_CMA']=(1+stock_data['RUT_Return']).cumprod()-1
    stock_data['SP500-45_Return_CMA']=(1+stock_data['SP500-45_Return']).cumprod()-1
    stock_data['Benchmark_Return_CMA']=(1+stock_data['Benchmark_Return']).cumprod()-1
    
    # Compute the Cumulative Stock Excess Returns, Cumulative Benchmark Excess Returns, based on the moving averages.
    stock_data['Stock_Excess_Return_CMA']=(1+stock_data['Stock_Excess_Return']).cumprod()-1
    stock_data['Benchmark_Excess_Return_CMA']=(1+stock_data['Benchmark_Excess_Return']).cumprod()-1
    
    # Compute the Moving average of the stock volatility, behchmark volatility on a rolling basis for a window of 5 days.
    stock_data['Stock_Volatility_MA']=stock_data['Stock_Volatility'].rolling(window=rolling_window).mean()
    stock_data['Benchmark_Volatility_MA']=stock_data['Benchmark_Volatility'].rolling(window=rolling_window).mean()
    
    # Compute the Moving Average of the stock Sharpe Ratio, Benchmark Sharpe Ratio on a rolling basis for a window of 5 days.
    stock_data['Stock_Sharpe_Ratio_MA']=stock_data['Stock_Sharpe_Ratio'].rolling(window=rolling_window).mean()
    stock_data['Benchmark_Sharpe_Ratio_MA']=stock_data['Benchmark_Sharpe_Ratio'].rolling(window=rolling_window).mean()
    
    stock_data.columns
    
    # Prepare a list of the statistical indicators for the model
    statistical_indicators=['Stock_Volatility',
    'Benchmark_Volatility', 'Stock_Sharpe_Ratio', 'Benchmark_Sharpe_Ratio',
    'Stock_Return_CMA', 'GSPC_Return_CMA', 'RUI_Return_CMA', 'RUT_Return_CMA',
    'SP500-45_Return_CMA', 'Benchmark_Return_CMA', 'Stock_Excess_Return_CMA',
    'Benchmark_Excess_Return_CMA', 'Stock_Volatility_MA',
    'Benchmark_Volatility_MA', 'Stock_Sharpe_Ratio_MA',
    'Benchmark_Sharpe_Ratio_MA']
    info=ticker.info
    
    # Calculate technical indicators
    
    # Compute 20-day Simple Moving Average of the Stock Price.
    stock_data['SMA_20']=ta.trend.sma_indicator(stock_data['Close'],window=20)
    
    # Compute 20-day Exponential Moving Average of the Stock Price.
    stock_data['EMA_20']=ta.trend.ema_indicator(stock_data['Close'],window=20)
    
    # Compute Relative Strenght Index
    stock_data['RSI']=ta.momentum.RSIIndicator(stock_data['Close']).rsi()
    
    # Compute Bollinger Upper and Lower Bands
    stock_data['Bollinger_Upper']=ta.volatility.BollingerBands(stock_data['Close']).bollinger_hband()
    stock_data['Bollinger_Lower']=ta.volatility.BollingerBands(stock_data['Close']).bollinger_lband()
    
    # Compute Moving Average Converage Divergence, MACD Signal line
    stock_data['MACD'], stock_data['MACD_Signal'] = ta.trend.MACD(stock_data['Close']).macd(), \
                                                    ta.trend.MACD(stock_data['Close']).macd_signal()
    
    # Compute Stochastic Oscillator
    stock_data['Stochastic_Oscillator'] = ta.momentum.StochasticOscillator(stock_data['High'], stock_data['Low'], 
                                                                           stock_data['Close']).stoch()
    
    # Compute Average True Range
    stock_data['ATR'] = ta.volatility.AverageTrueRange(stock_data['High'], stock_data['Low'], stock_data['Close']).average_true_range()
    
    # Compute Chaikin Money Flow
    stock_data['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(stock_data['High'], stock_data['Low'], stock_data['Close'], 
                                                            stock_data['Volume']).chaikin_money_flow()
    
    # Compute On-Balance Volumne
    stock_data['OBV'] = ta.volume.OnBalanceVolumeIndicator(stock_data['Close'], stock_data['Volume']).on_balance_volume()
    
    # Compute Commodity Channel Index
    stock_data['CCI'] = ta.trend.cci(stock_data['High'], stock_data['Low'], stock_data['Close'])
    
    
    #stock_data['Parabolic_SAR'] = ta.trend.PSARIndicator(stock_data['High'], stock_data['Low'], stock_data['Close']).psar()
    # Compute Williams %R
    stock_data['Williams_%R'] = ta.momentum.WilliamsRIndicator(stock_data['High'], stock_data['Low'], stock_data['Close']).williams_r()
    
    # Compute Volumne Weighted Average Price
    stock_data['VWAP'] = ta.volume.VolumeWeightedAveragePrice(stock_data['High'], stock_data['Low'], stock_data['Close'], 
                                                              stock_data['Volume']).volume_weighted_average_price()
    
    # Fetch quarterly financial data
    quarterly_financials = ticker.quarterly_financials.T
    quarterly_balance_sheet = ticker.quarterly_balance_sheet.T
    quarterly_cashflow = ticker.quarterly_cashflow.T
    
    # Compute additional fundamental metrics
    fundamental_data = pd.DataFrame({
        'EPS': quarterly_financials['Net Income'] / quarterly_financials['Basic Average Shares'], # Earnings Per Share
        'Revenue': quarterly_financials['Total Revenue'], # Total Revenue
        'Operating_Income': quarterly_financials['Operating Income'], # Operating Income
        'Net_Income': quarterly_financials['Net Income'], # Net Income
        'Gross_Profit': quarterly_financials['Gross Profit'], # Gross Profit
        'Total_Assets': quarterly_balance_sheet['Total Assets'], # Total Assets
        'Total_Liabilities': quarterly_balance_sheet['Total Liabilities Net Minority Interest'], # Total Liabilities
        'Debt_to_Equity': quarterly_balance_sheet['Total Liabilities Net Minority Interest'] / \
                            quarterly_balance_sheet['Stockholders Equity'], # Debt-to-Equity Ratio
        'Gross_Margin': quarterly_financials['Gross Profit'] / quarterly_financials['Total Revenue'], # Gross Margin
        'Operating_Margin': quarterly_financials['Operating Income'] / quarterly_financials['Total Revenue'], # Operating Margin
        'Net_Profit_Margin': quarterly_financials['Net Income'] / quarterly_financials['Total Revenue'], # Net Profit Margin
        'ROA': quarterly_financials['Net Income'] / quarterly_balance_sheet['Total Assets'], # Return on Assets
        'ROE': quarterly_financials['Net Income'] / quarterly_balance_sheet['Stockholders Equity'], # Return on Equity
        'Current_Ratio': quarterly_balance_sheet['Current Assets'] / quarterly_balance_sheet['Current Liabilities'], # Current Ratio
        'Quick_Ratio': (quarterly_balance_sheet['Current Assets'] - quarterly_balance_sheet['Inventory']) / \
                        quarterly_balance_sheet['Current Liabilities'], # Quick Ratio
        #'Dividend_Payout_Ratio': quarterly_financials['Dividends Paid'] / quarterly_financials['Net Income'],
        #'Interest_Coverage_Ratio': quarterly_financials['Operating Income'] / quarterly_cashflow['Interest Expense'],
        'Price_to_Book': info.get('marketCap', None) / info.get('bookValue', None) # Price to Book Ratio
    }, index=quarterly_financials.index).apply(pd.to_numeric,errors='coerce')
    
    # Function to perform segment-wise log-linear interpolation since fundamental data is available quarterly and 
    #we need daily fundamental data to fit our model.
    def segment_log_linear_interpolation(dates,values,new_dates):
        #new_dates=pd.to_datetime(new_dates)
        interpolated_values=pd.Series(index=new_dates)
        
        # Loop through each date to calculate interpolated values
        for new_date in new_dates[:-1]:
            
            # For dates before the first known date, use the first known value
            if new_date<dates[0]:
                interpolated_values[new_date]=values[0]
                
            # For dates after the last known date, use the last known value
            elif new_date>dates[-1]:
                interpolated_values[new_date]=values[-1]
            
            else:
                #print(new_date)
                # Find the close previous and next date
                previous_date=dates[dates<=new_date].max()
                next_date=dates[dates>new_date].min()
                
                # Get the values corresponding to these dates
                start_value=values[dates==previous_date].values[0]
                #print(start_value)
                end_value=values[dates==next_date].values[0]
                #print(end_value)
                
                # Calculate the number of days elapsed and the total days between the known dates
                days_elapsed=(new_date-previous_date).days
                total_days=(next_date-previous_date).days
                
                # Perform log-linear interpolation
                interpolated_values[new_date]=np.exp(np.log(start_value)+(np.log(end_value/start_value)*days_elapsed/total_days))
                #print(interpolated_values[new_date])
        
        # Set the known values at their original dates
        interpolated_values[dates]=values
        return interpolated_values
    
    # Drop rows with missing values from fundamental data 
    fundamental_data=fundamental_data.dropna()
    
    # Sort the fundamental data by index (data)
    fundamental_data.sort_index(inplace=True)
    
    # Create a date range from the minimum to maximum date in fundamental data with daily frequency
    date_range=pd.date_range(start=fundamental_data.index.min(),end=fundamental_data.index.max(),freq='D')
    
    fundamental_data_daily=pd.DataFrame(index=date_range,columns=fundamental_data.columns)
    
    # Interpolate fundamental data to get daily values
    for column in fundamental_data.columns:
        interpolated_values=segment_log_linear_interpolation(
        fundamental_data.index,
        fundamental_data[column],date_range.to_list())
        fundamental_data_daily[column]=interpolated_values
        
    # Merge stock data with interpolated fundamental data
    merged_data = stock_data.join(fundamental_data_daily, how='left')
    
    # Drop rows with missing values from the merged data
    merged_data = merged_data.dropna()
    
    # List of technical indicators to include in the model
    technical_indicators=['SMA_20', 'EMA_20', 'RSI', 'Bollinger_Upper', 'Bollinger_Lower', 'MACD', 'MACD_Signal', 
                          'Stochastic_Oscillator', 'ATR', 'CMF', 'OBV', 'CCI', 'Williams_%R', 'VWAP']
    
    # Create feature matrix X with statistical and technical indicators, fundamental data and benchmark index prices 
    # along with some additional index level like interest rate, volatility etc for differences periods.
    X = merged_data[statistical_indicators+technical_indicators + list(fundamental_data.columns)+indices].copy()
    #X = merged_data[list(fundamental_data.columns)+indices].copy()
    
    # Create target vector y (stock prices)
    y = merged_data['Close'].copy()
    X.dtypes
    
    # Initialize and train the random forecast regressor
    rf = RandomForestRegressor(n_estimators=500)
    rf.fit(X, y)
    
    # Compute feature importances
    feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
    print("\nFeature Importance:")
    print(feature_importances.sort_values(ascending=False))
    
    # Select top n features with importance greater than 0.005
    top_features=feature_importances[feature_importances>0.005]
    X_selected=X[top_features.index]
    
    # Standardize the selected features
    scaler=StandardScaler()
    X_standardized=scaler.fit_transform(X_selected)
    
    # Perform PCA (Pricipal Components Analysis) to reduce dimensionality
    pca=PCA(n_components=None)
    
    # Take the differencing between variables, since we need to align the dependent variables according to the independent 
    # variables after ADF test is performed and differenced fields are used as target.
    X_standardized=pd.DataFrame(data=X_standardized,index=X_selected.index)
    
    # Drop the rows with missing values.
	X_standardized=X_standardized.diff().dropna()
    
    # Align target variable y with the index of standardized features
    y=y[y.index.isin(X_standardized.index)].copy()
    
    # Fit PCA on standardized features after differencing is done.
    X_pca=pca.fit_transform(X_standardized)
    
    # Calculate explained_variance ratio and cumulative variance
    explained_variance=pca.explained_variance_ratio_
    cumulative_variance=explained_variance.cumsum()
    
    # Determine the number of PCA components that explain 99% of the variance
    n_components=next(i for i,total in enumerate(cumulative_variance) if total>0.99)+1
    pca=PCA(n_components=n_components)
    X_pca_reduced=pca.fit_transform(X_standardized)
    
    # Create a dataframe for PCA loadings (coefficients of each feature on the principal components)
    loadings=pd.DataFrame(pca.components_.T,
                          columns=[f'PC{i+1}' for i in range(n_components)],
                          index=X_selected.columns)
    
    # Add a constant to the PCA features for the OLS model
    X_pca_with_constant=sm.add_constant(X_pca_reduced)
    
    # Fit an OLS (Ordinary Least Sqaures) model to predict y using PCA features so as to understand the significane of each feature
    y_OLS=y.diff().dropna()
    X_OLS=pd.DataFrame(data=X_pca_with_constant,index=X_standardized.index)

    X_OLS=X_OLS[X_OLS.index.isin(y_OLS.index)].copy()
    
    model=sm.OLS(y_OLS,X_OLS).fit()
    coefficients=model.params[1:]
    t_statistics=model.tvalues[1:]
    p_values=model.pvalues[1:]
    
    # Prepare PCA Dataframe for model fitting
    pca_df=pd.DataFrame(data=X_pca_reduced,index=X_standardized.index)
    
    # Standardize the target variable y
    y_standardized=scaler.fit_transform(pd.DataFrame(y))
    y_standardized=pd.DataFrame(data=y_standardized,index=X_standardized.index)
    
    # Create a copy of the standardized target variable y
    y_s=y_standardized.copy()
    
    # Function to check the stationarity of a time series
    def check_stationarity(series):
        result=adfuller(series)
        print(f"ADF statistic: {result[0]}")
        print(f"p-value: {result[1]}")
        return result[1]>0.05
    
    # Check if the target variable is stationary
    if check_stationarity(y_standardized[0]):
        print('Data is Stationary')
        
        # If stationary, take the difference of the series
        y_standardized[0]=y_standardized[0].diff().dropna()
    
    # Prepare PCA DataFrame and target variable for modelling
    y_standardized=y_standardized.dropna()
    pca_df=pca_df[pca_df.index.isin(y_standardized.index)].copy()
    y_standardized=y_standardized[y_standardized.index.isin(pca_df.index)].copy()
    
    # Split data into training and test sets (80% training, 20% testing)
    train_size=int(len(y)*0.8)
    
    # Define target and features
    train_y,test_y=y_standardized[:train_size],y_standardized[train_size:]
    train_X,test_X=pca_df[:train_size],pca_df[train_size:]
    
    # Define SARIMAX model parameters
    order=(1,1,1)
    seasonal_order=(1,1,1,12)
    trend='c'
    
    # Fit SARIMAX model to training data
    model=SARIMAX(train_y,order=order,seasonal_order=seasonal_order,trend=trend,exog=train_X)
    model_fitted=model.fit()
    model_fitted.summary()
    
    # Forecast test values using the SARIMAX model
    forecast=model_fitted.predict(start=len(train_y),end=len(train_y)+len(test_y)-1,exog=test_X)
    
    # Evaluate the forecast
    rmse=root_mean_squared_error(test_y,forecast)
    mae=mean_absolute_error(test_y,forecast)
    mse=mean_squared_error(test_y,forecast)
    r2=r2_score(test_y,forecast)

    #last_observed=y_s[0].iloc[len(train_y)]
    #forecast_stock_prices=forecast.cumsum()+last_observed
    #forecast_stock_prices=scaler.inverse_transform(pd.DataFrame(forecast_stock_prices))
    #forecast_stock_prices=pd.DataFrame(data=forecast_stock_prices,index=test_y.index)
    
    #plot the results
    plt.figure(figsize=(12,6))
    plt.plot(train_y.index,train_y,label='Training Data')
    plt.plot(test_y.index,test_y,label='Test Data')
    plt.plot(test_y.index,forecast,label='Predicted',color='red',linestyle='--')
    plt.title('SARIMAX model Forecast vs Actual')
    plt.legend()
    plt.show()
    
    # Prepare the forecast for the next 30 days
    X_c=X_standardized
    next_30_days=pd.bdate_range(start=y_standardized.index[-1],periods=30)
    
    # For forecasting the stock prices, we first forecast the exogenous variables using SARIMAX model for individual exogenous factors.
    #Forecast each exogenous factor present in X_c using SARIMAX
    forecast_values={}
    for column in X_c.columns:
        model_exog=SARIMAX(X_c[column],order=(1,1,1),seasonal_order=(1,1,1,12),trend='c')
        model_fitted_exog=model_exog.fit()
        #print(model_fitted.summary())
        n_forecast=30
        forecast=model_fitted_exog.get_forecast(steps=n_forecast)
        forecast_values[column]=forecast.predicted_mean
        
        # Plot the results of the forecast along with the original values for each of the exogenous variables
        plt.figure(figsize=(12,6))
        plt.plot(X_c[column].index,X_c[column],label='Training Data')
        plt.plot(next_30_days,forecast_values[column],label='Predicted',color='red',linestyle='--')
        plt.title('SARIMAX model Original and Forecast values for '+top_features.index[column])
        plt.legend()
        plt.show()
    
    # Create a DataFrame for the forecasted exogenous variables
    exog_values=pd.DataFrame(data=forecast_values)
    
    # Apply PCA to the forecasted exogenous variables
    X_pca_exog=pca.fit_transform(exog_values)
    pca_df_exog=pd.DataFrame(data=X_pca_exog,index=next_30_days)
    pca_df_exog=pca_df_exog.dropna()
    
    # Forecast using the SARIMAX model with the PCA-transformed exogenous variables
    n_forecast=30
    forecast=model_fitted.get_forecast(steps=n_forecast,exog=X_pca_exog)
    forecast_values=forecast.predicted_mean
    
    # Compute the forecasted stock prices
    last_observed=y_s[0].iloc[-1]
    forecast_stock_prices=forecast_values.cumsum()+last_observed
    forecast_stock_prices=scaler.inverse_transform(pd.DataFrame(forecast_stock_prices))
    forecast_stock_prices=pd.DataFrame(data=forecast_stock_prices,index=next_30_days)
    forecast_stock_prices[0].to_list()
    
    # Retrieve the original target variable for comparison
    y_orig=scaler.inverse_transform(pd.DataFrame(y_s))
    y_orig=pd.DataFrame(data=y_orig,index=y_s.index)

    # Plot the results
    plt.figure(figsize=(12,6))
    plt.plot(y_orig.index,y_orig[0],label='Training Data')
    plt.plot(next_30_days,forecast_stock_prices[0],label='Predicted',color='red',linestyle='--')
    plt.title('SARIMAX model Original and Forecast Stock Prices')
    plt.legend()
    plt.show()
    
    # Return forecast results and evaluation metrics
    try:
        return {"forecast":forecast_stock_prices,"rmse":rmse,"mae":mae,"mse":mse,"r2_score":r2}
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))

# Entry point for running the FastAPI app
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host='0.0.0.0',port=8000)
   
