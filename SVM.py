# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 10:57:29 2022

@author: Abhay
"""

import yfinance as yf
from stocktrends import Renko
import numpy as np
import pandas as pd
import copy
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import ta
import matplotlib.pyplot as  plt

'''
Download required data from Yahoo Finance
'''
# tickers = ['^NSEI','HEROMOTOCO.NS','CIPLA.NS','ICICIBANK.NS','BAJFINANCE.NS','ULTRACEMCO.NS','NESTLEIND.NS','BAJAJ-AUTO.NS','ONGC.NS','GRASIM.NS','BHARTIARTL.NS','BRITANNIA.NS','HDFCLIFE.NS','BAJAJFINSV.NS','ITC.NS','INDUSINDBK.NS','TATACONSUM.NS','TITAN.NS','HINDALCO.NS','KOTAKBANK.NS','MARUTI.NS','LT.NS','TATASTEEL.NS','SHREECEM.NS','TCS.NS','COALINDIA.NS','M&M.NS','NTPC.NS','TECHM.NS','WIPRO.NS','RELIANCE.NS']
tickers = ['^NSEI']
ohlcv_data = []
data_app = {}

'''
Loop over the data and storing them as ohlcv data in dictionary
'''
for ticker in tickers:
    temp = yf.download(ticker, period='5y',interval='1d')
    temp.dropna(how='any',inplace=True)
    temp = temp[temp['Volume'] != 0]
    ohlcv_data = temp
    ohlcv_data['Return'] = ohlcv_data['Close'].pct_change(1)
    ohlcv_data['ADX'] = ta.trend.adx(ohlcv_data['High'],ohlcv_data['Low'],ohlcv_data['Close'],window=14,fillna=True)
    ohlcv_data['RSI'] = ta.momentum.rsi(ohlcv_data['Close'],window=14,fillna=True)
    final_data = ohlcv_data[27:]
    
# Let's Create Predictor variable X.
    X = final_data[['ADX','RSI']]
    X.head()

# Target variables
    y = np.where(final_data['Return'].shift(-1) > final_data['Return'], 1, 0)

# Split the dataset for training and testing.
    split_percentage = 0.8
    split = int(split_percentage*len(final_data))
  
# Train data set
    X_train = X[:split]
    y_train = y[:split]
  
# Test data set
    X_test = X[split:]
    y_test = y[split:]
    
# Support vector classifier
    classi = SVC().fit(X_train, y_train)
    
# Predict the response for dataset.
    final_data['Predicted_Signal'] = classi.predict(X)

# Accuracy Score of Classifiers.
    test_prediction  = classi.predict(X_test)
    test_accuracy = accuracy_score(y_test,test_prediction)