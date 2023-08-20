# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 20:42:00 2023

@author: chaeh
"""

import yfinance as yf
import pandas as pd

sp500 = yf.Ticker("^GSPC")  # Create a Ticker object for the S&P 500 index
sp500 = sp500.history(period="max")  # Fetch historical data

sp500.plot.line(y="Close", use_index=True) #plot the close price of sp 500 

#delete columns not needed
del sp500["Dividends"]
del sp500["Stock Splits"]

#Create boolean of whether or not the stock when up or down and change it to an int value
sp500["Tomorrow"] = sp500["Close"].shift(-1) #Create new columns and shift close value 

sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)

#Delete stock prices before 1980
sp500 = sp500.loc["1980-01-01":].copy()

#Use random forest to make basic model
from sklearn.ensemble import RandomForestClassifier 

model = RandomForestClassifier(n_estimators= 100, min_samples_split=100, random_state=1)
#split data into train and test data
train = sp500.iloc[:-100]
test = sp500.iloc[-100:]

predictors = ["Close", "Open", "High", "Low", "Volume"]
model.fit(train[predictors], train["Target"])

from sklearn.metrics import precision_score

preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index) #change to panda series
precision_score(test["Target"], preds) #0.559322033898305 55% precision score. pretty good.

combined = pd.concat([test["Target"],preds], axis = 1) #combine predicted and actual 
combined.plot()
 
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    
    return pd.concat(all_predictions)

predictions = backtest(sp500, model, predictors)
predictions["Predictions"].value_counts()
        
precision_score(predictions["Target"], predictions["Predictions"]) #0.5309791332263243 worse than random forest class
#%%
horizons = [2,5,60,250,1000] #last two day, week, 3 months, one year and four year of trading days
new_predictors = []

for horizon in horizons: #average of two day, week, 3 months, one year and four year of trading days and use that to predict
    rolling_averages = sp500.rolling(horizon).mean()
    
    ratio_column = f"Close_Ratio_{horizon}"
    sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]
    
    trend_column = f"Trend_{horizon}"
    sp500[trend_column] = sp500.shift(1).rolling(horizon).sum()["Target"]
    
    new_predictors+= [ratio_column, trend_column]
sp500 = sp500.dropna(subset=sp500.columns[sp500.columns != "Tomorrow"])

model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >=.6] = 1
    preds[preds <.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined
    
predictions = backtest(sp500, model, new_predictors)
predictions["Predictions"].value_counts()       
precision_score(predictions["Target"], predictions["Predictions"]) #0.5684474123539232 best model so far
        
        
        