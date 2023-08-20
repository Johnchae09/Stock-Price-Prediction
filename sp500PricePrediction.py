# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 19:04:45 2023

@author: chaeh
"""

import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

sp500 = yf.Ticker("^GSPC")  # Create a Ticker object for the S&P 500 index
sp500 = sp500.history(period="max")  # Fetch historical data
sp500 = sp500.loc["2010-01-01":].copy() #only data after 1990
print(sp500.shape) #Number of rows and columns

plt.figure(figsize=(18,8))
plt.title("Close Price history")
plt.plot(sp500["Close"])
plt.xlabel("Date", fontsize = 18)
plt.ylabel("Close Price USD", fontsize = 18)

#Create data frame with only close price
data = sp500.filter(["Close"])
#Convert to numpy arr
dataset = data.values
#Get number of rows to train the model
training_data_len = math.ceil(len(dataset) * .8)

#Scale data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

#Create training data set
#Create the scaled training data set
train_data = scaled_data[0:training_data_len,:]
#Split the data into x and y train
x_train = []
y_train = []

for i in range(100,len(train_data)):
    x_train.append(train_data[i-100:i,0])
    y_train.append(train_data[i,0])

#Convert the x and y train to nummpy array
x_train,y_train = np.array(x_train), np.array(y_train)

#Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

#Build the LSTM model
model = Sequential()
model.add(LSTM(50,return_sequences=True, input_shape = (x_train.shape[1],1)))
model.add(LSTM(50,return_sequences= False))
model.add(Dense(25))
model.add(Dense(1))

#Run the model
model.compile(optimizer ='adam', loss = 'mean_squared_error')

#Train model
model.fit(x_train,y_train, batch_size=1,epochs=10)

#Create test data set
#Create new array containing scaled value
test_data = scaled_data[training_data_len - 100:,:]

#Create data set x and y test
x_test = []
y_test = dataset[training_data_len:,:]

for i in range(100,len(test_data)):
    x_test.append(test_data[i-100:i,0])
    
#convert data to a numpy array
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1], 1))

#get the models predicted price value
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#Get RMSE
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
print(rmse)

#Plot
train = data[:training_data_len]
valid = data[training_data_len:]
valid["Predictions"] = predictions
plt.figure(figsize=(16,8))
plt.title("Model")
plt.xlabel("Date")
plt.ylabel("Close Price USD")
plt.plot(train["Close"])
plt.plot(valid[["Close", "Predictions"]])
plt.legend(["Train", "Valid","Predictions"], loc = "lower right")
plt.show()






















