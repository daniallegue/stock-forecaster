# -*- coding: utf-8 -*-
"""
Created on 14/08/2023

@author: daniallegue
"""

''' 

This code uses the Meta's Prophet to predict stock prices from a time series. 

'''


import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

SQ_FILE = 'data/SQ.csv'
VOO_FILE = 'data/VOO.csv'

#Load data
df = pd.read_csv(SQ_FILE)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
voo_data = pd.read_csv(VOO_FILE)
# Take market reference into the data
df['Market'] = voo_data['Adj Close']

#Add lag features to the dataframe
df = df.sort_index()
target_map = df['Adj Close'].to_dict()
df['lag1'] = (df.index - pd.Timedelta('364 days')).map(target_map)
df['lag2'] = (df.index - pd.Timedelta('728 days')).map(target_map)
df['lag3'] = (df.index - pd.Timedelta('1092 days')).map(target_map)

#Take into account the multiplicative seasonality of data
analysis_target = df[['Adj Close']].copy()
seasonal_decompose = seasonal_decompose(analysis_target, period=15, model = 'multiplicative')
trend = seasonal_decompose.trend
seasonality = seasonal_decompose.seasonal

seasonal_decompose.plot()
df['Seasonality'] = seasonality

#Create a time series 6-fold split
split = TimeSeriesSplit(n_splits = 6, test_size=100, gap=28)
fold = 0
predictions = []
scores = []

for train_i, test_i in split.split(df):
  train = df.iloc[train_i].copy()
  test = df.iloc[test_i].copy()

  FEATURES = ['High', 'Low', 'Volume', 'Market', 'Seasonality', 'lag1', 'lag2', 'lag3']
  TARGET = ['Adj Close']

  X_train = train[FEATURES]
  y_train = train[TARGET]

  X_test = test[FEATURES]
  y_test = test[TARGET]

  xgb_param = {'base_score' : 0.5, 'booster' : 'gbtree', 'random_state': 42,
               'max_depth': 6, 'n_estimators': 7300, 'learning_rate': 0.0015,
               'objective' : 'reg:squarederror',
               'min_child_weight': 80
               }

  model = XGBRegressor(**xgb_param)
  model.fit(X_train, y_train,
          eval_set = [(X_train, y_train), (X_test, y_test)],
          verbose = True,
          eval_metric = 'rmse')

  y_pred = model.predict(X_test)
  predictions.append(y_pred)
  scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))

print(f'Mean score accross folds {np.mean(scores)}')
print(f'Fold scores {scores}')

#Dive deeper into the test data
test['Prediction'] = model.predict(X_test)
df = df.merge(test[['Prediction']], how='left', left_index = True, right_index = True)

ax = df[['Adj Close']].plot()
df['Prediction'].plot(ax = ax, style='.')
plt.legend(['Truth Data', 'Prediction'])
ax.set_title('Actual Data & Prediction')
plt.show()

#Save the model for backtesting later
import joblib
joblib.dump(model, 'models/xgbregressor')

