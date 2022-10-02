# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
import numpy as np
import pandas as pd
# Requires all tensorflow dependencies
try:
    from tensorflow import keras
    # import tensorflow.keras as keras
except:
    print("Error: Tensorflow import failed")
    exit(0)

# import datetime
from datetime import *
import math
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import rdb2csv
from getJson import getData

userLatitude = 0.0
userLongitude = 0.0
requestYear = 2020
requestMonth = 5
requestDay = 1
predIntervalLength = 7
# df = pd.DataFrame()

# Convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # Input sequence
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # Forecast sequence
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # Combine
    agg = concat(cols, axis=1)
    agg.columns = names
    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    agg
    return agg

def init(latitude, longitude, year, month, day):
    userLatitude = latitude
    userLongitude = longitude


def loadDataFrame():
    # TODO: parameters --> url --> load resulting data from USGS url to tsv --> pandas df
    print()

def makePredictions():
    # Load dataset
    dataset = pd.read_csv("trainingData.json")
    # datetime
    dataset['DateTime'] = pd.to_datetime(dataset['DateTime'], errors='coerce')

    # numeric
    dataset['Latitude'] = pd.to_numeric(dataset['Latitude'])
    dataset['Longitude'] = pd.to_numeric(dataset['Longitude'])
    dataset['Temperature'] = pd.to_numeric(dataset['Temperature'])
    dataset['Conductance'] = pd.to_numeric(dataset['Conductance'])
    dataset['Dissolved_oxygen'] = pd.to_numeric(dataset['Dissolved_oxygen'])
    dataset['PH'] = pd.to_numeric(dataset['PH'])
    dataset['Turbidity'] = pd.to_numeric(dataset['Turbidity'])

    # idk what's going on here?????
    # Add columns for water quality metrics +7 days into future (repeat for +14, +21, +28)
    # Original dataset has 96 rows of data per day
    # shiftN = 30 if (scope == "30day") else (365 if (scope == "year") else 7)
    # df_validation = dataset.tail(shiftN * 96).copy()

    # dataset["Gage_height_shift"] = (dataset.copy())["Gage_height"]
    # dataset["Gage_height_shift"] = dataset.Gage_height_shift.shift(
    #     -shiftN * 96)

    # last_date = pd.to_datetime(
    #     dataset['datetime'].dt.date.iloc[-1], errors='coerce')


    
    
    # Replace all NaNs with value from previous row, the exception being Gage_height;
    # Only consider rows with valid Gage_height values
    dataset = dataset[dataset['Gage_height'].notna()]

    for col in dataset:
        dataset[col].fillna(method='interpolate', inplace=True)

    # Remove any NaNs or infinite values
    dataset = dataset[~dataset.isin([np.nan, np.inf, -np.inf]).any(1)]

    
    # Validation data
    last_date = date(requestYear, requestMonth, requestDay)
    df_validation = dataset.copy()
    d1 = last_date
    d2 = last_date + timedelta(predIntervalLength)
    df_validation = df_validation.drop(
        df_validation[df_validation['datetime'].dt.date < d1].index)

    df_validation = df_validation.drop(
        df_validation[df_validation['datetime'].dt.date > d2].index)



    # Specify columns to plot
    groups = [0, 1, 2, 3, 4, 5, 6, 7]
    i = 1
    # Plot each column
    pyplot.figure()
    for group in groups:
        pyplot.subplot(len(groups), 1, i)
        pyplot.plot(dataset.values[:, group])
        pyplot.title(dataset.columns[group], y=0.5, loc='right')
        i += 1
    pyplot.show()

    values = values.astype('float32')

    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)

    # Repeat for validation data
    df_validation_relevant = df_validation.copy()
    df_validation_relevant = df_validation_relevant.drop('datetime', 1)
    # df_validation_relevant = df_validation_relevant.drop('fld_stg', 1)
    validation_vals = df_validation_relevant.values
    validation_vals = validation_vals.astype('float32')
    validation_scaled = scaler.fit_transform(validation_vals)
    validation_reframed = series_to_supervised(validation_scaled, 1, 1)
    
    print("VALIDATION MIN AND MAX")
    min_conduct_valid = df_validation['Conductance'].min()
    print(min_conduct_valid)
    mean_conduct_valid = df_validation['Conductance'].mean()
    print(mean_conduct_valid)
    max_conduct_valid = df_validation['Conductance'].max()
    print(max_conduct_valid)

    
    df_levels_valid = pd.DataFrame(
        {'Conductance': [min_conduct_valid, mean_conduct_valid, max_conduct_valid]})

    levels_valid_scaled = scaler.fit_transform(df_levels_valid.values)
    levels_valid_scaled = levels_valid_scaled[1][0]

    levels_valid_scaled = [levels_valid_scaled, (
        0.75 * levels_valid_scaled), (0.5 * levels_valid_scaled), (0.25 * levels_valid_scaled)]
    stg_colors = ['r', 'tab:orange', 'y', 'g']
    stg_labels = ['Flood Stage', '75%', '50%', '25%']
