# -*- coding: utf-8 -*-
"""
Created on 14/08/2023

@author: daniallegue
"""

''' 

This code uses the XGBoost Regressor interface predict labels from a time series. 
It is a supervised learning task to use features and make a regression to predict prices of stocks.

'''

import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

PYPL_FILE = 'data/PYPL.csv'
VOO_FILE = 'data/VOO-2.csv'

#Load the data
df = pd.read_csv(PYPL_FILE)
df['Date'] = pd.to_datetime(df['Date'])
voo_data = pd.read_csv(VOO_FILE)
# Take market reference into the data
df['Market'] = voo_data['Adj Close']
df = df.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'], axis = 1)
df = df.rename(columns={'Date': 'ds', 'Adj Close':'y'})


#Create the model
model = Prophet(mcmc_samples=1200,
                changepoint_prior_scale=0.1,
                seasonality_mode='additive',
                yearly_seasonality=8,
                weekly_seasonality=False,
                daily_seasonality=False, changepoint_range=0.9)
#model.add_seasonality('quarterly', period=91.25, fourier_order=10, mode='additive')
model.add_regressor('Market') #Coefficient ~ 0.42
model.fit(df)


from prophet.plot import plot_cross_validation_metric
from prophet.utilities import regressor_coefficients
print(regressor_coefficients(model))

from prophet.diagnostics import cross_validation, performance_metrics
data_cross_validation = cross_validation(model, initial='500 days', period='180 days',
                                         horizon='365 days')
#Initial cutoff at at 500 days, making predictions in a period of 180 days over a horizon of 365 days
print(data_cross_validation.head())

#Evaluate performance of the model
cv_metrics = performance_metrics(data_cross_validation)
rmse = cv_metrics['rmse']
fig = plot_cross_validation_metric(data_cross_validation, metric = 'rmse')

#Save model
from prophet.serialize import model_to_json

with open('models/prophet.json', 'w') as fout:
    fout.write(model_to_json(model))



