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

userLatitude = 0.0
userLongitude = 0.0
requestYear, requestMonth, requestDay = 0
df = pd.DataFrame()

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

def processDataFrame():
# def makePredictions():
    # Load dataset
    dataset = pd.read_json("trainingData.json")

    
    # Process data:
    # TODO: replace column names

    # Inputs
    inputs = ['site_no', 'datetime', 'latitude', 'longitude']
    # Site_no to numeric
    dataset['site_no'] = pd.to_numeric(dataset['site_no'])
    # Datetime to datetime
    dataset['datetime'] = pd.to_datetime(dataset['datetime'], errors='coerce')
    # Latitude to numeric
    dataset['latitude'] = pd.to_numeric(dataset['latitude'])
    # Longitude to numeric
    dataset['longitude'] = pd.to_numeric(dataset['longitude'])
    # ... other inputs (climate? weather? season? idk)


    # Outputs (water quality metrics)
    outputs = ['Water_temp', 'Conductance', 'Dissolved_oxygen', 'Turbidity', 'pH']
    # Water_temp
    dataset.rename(columns={'Temperature_water_C': 'Water_temp'}, inplace=True)
    dataset['Water_temp'] = pd.to_numeric(dataset['Water_temp'])
    
    # Conductance
    dataset.rename(
        columns={'Specific_conductance_water_uScm': 'Conductance'}, inplace=True)
    dataset['Conductance'] = pd.to_numeric(dataset['Conductance'])

    # Dissolved_oxygen
    dataset.rename(
        columns={'Dissolved_oxygen_water_mgL': 'Dissolved_oxygen'}, inplace=True)
    dataset['Dissolved_oxygen'] = pd.to_numeric(dataset['Dissolved_oxygen'])

    # Turbidity
    dataset.rename(columns={'Turbidity_water_FNU': 'Turbidity'}, inplace=True)
    dataset['Turbidity'] = pd.to_numeric(dataset['Turbidity'])

    # pH
    dataset.rename(columsn={'pH_water_unfiltered': 'pH'}, inplace=True)
    dataset['pH'] = pd.to_numeric(dataset['pH'])


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



    # Skip TRIM section

    
    # Filter data
    # ...

    # Unnecessary columns
    dataset = dataset.drop('site_no',1, errors='ignore')

    # Invalid data
    dataset.drop(dataset[dataset['pH'] < 0 or dataset['pH'] > 14].index, inplace=True)
    dataset.drop(dataset[dataset['Dissolved_oxygen'] < 0].index, inplace=True)

    dataset = dataset.dropna()
    dataset.reset_index(drop=True)
    dataset = dataset[~dataset.isin([np.nan, np.inf, -np.inf]).any(1)]


    # # Move inputs before outputs
    # dataset = [c for c in dataset if c in inputs] + [c for c in dataset if c in outputs]

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


    # TODO: Handle NaN/empty values 
    

    # normalize features
    values = dataset.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    # drop columns we don't want to predict
    # reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
    print(reframed.head())
    
    # Validation data
    

    