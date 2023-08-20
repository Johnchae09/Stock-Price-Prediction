# Stock-Price-Prediction Using Machine Learning

This repo contains Python code to predict the direction of the stock price movement(up or down) and stock prices using historical data of the S&P 500 index. The prediction is made using a Random Forest classifier, LSTM, and a combination of technical indicators.

![CHEESE!](https://github.com/Johnchae09/Stock-Price-Prediction/blob/main/closing_price2.png?raw=true)

### Code and Resources
**Python Version**:3.9

**Packages**: yfinance, pandas, numpy, sklearn.preprocessing, keras.models, keras.layers, matplotlib.pyplot

### Data and Data Cleaning
- Usage of yfinance library which contains S&P 500 historical data and use that to train models.
- Create a boolean column to determine whether or not the closing price went up or down from the pervious day.
- Only use stock price data after 2000-01-01

### Training using Random Forst Classifier and Precision Score
- The model calculates and displays the previsions score for Random Forest Classifier for different time horizons.

  - **Random Forst Classifier**: 0.5373134328358209
  
  - **Using Horizon of two days, week, 3 months, one year, and four years**: 0.5373134328358209
  
![CHEESE!](https://github.com/Johnchae09/Stock-Price-Prediction/blob/main/prediction.png?raw=true)

### Training using LSTM
- The model predicts stock price using ```adam``` optimizer and loss of ```mean squared error```
- Use 80% of the data set as the training set, two LSTM layers, and two dense layers to the model.

![CHEESE!](https://github.com/Johnchae09/Stock-Price-Prediction/blob/main/LSTM_predictions.png?raw=true)
