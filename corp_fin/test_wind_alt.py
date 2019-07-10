# test ml & neural
# project wind & pv

# import data - ensoe
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime
import pandas as pd
import sys
import datetime
import numpy as np


sys.path.insert(0,'C:/Users/tanel.joon/OneDrive - Energia.ee/Documents_OneDrive/Python/for_import')
from holidays import holidays

import pickle
import copy

with open('df_total.obj', 'rb') as file:
    df_total = pickle.load(file)

#df_total['Start_time'] = df_total.index.values
folder = 'C:/Users/tanel.joon/OneDrive - Energia.ee/Documents_OneDrive/Python/data/'
file_cap = 'DE_Installed Capacity per Production Type_201501010000-202001010000.csv'

df_capacity = pd.read_csv(folder + file_cap)
print(df_capacity.head())

df_total['Year'] = df_total.index.year

df_capacity = df_capacity.set_index('Production Type')

print(df_total.columns)

def capacity(year, string, df_capacity):
    
    capacity_value = df_capacity[str(year) +' [MW]'][string]
    
    return capacity_value

df_total['Wind Onshore Capacity'] = df_total['Year'].apply(lambda x: capacity(x, 'Wind Onshore', df_capacity))
df_total['Wind Offshore Capacity'] = df_total['Year'].apply(lambda x: capacity(x, 'Wind Offshore', df_capacity))
df_total['Solar Capacity'] = df_total['Year'].apply(lambda x: capacity(x, 'Solar', df_capacity))

print(df_total['Wind Onshore Capacity'].head())
print(df_total['Wind Offshore Capacity'].head())
print(df_total['Solar Capacity'].head())


# _____________________________________________________________________________
# set time limit
time_lim_start = datetime.datetime(2015,1,1)
time_lim_end = datetime.datetime(2018,9,30)

# filter out unneeded columns
df_total = df_total.filter(['Solar  - Actual Aggregated [MW]','Solar Capacity'])
#df_total = df_total.filter(['Wind Onshore  - Actual Aggregated [MW]','Wind Onshore Capacity'])

# filter time
df_total = df_total[(time_lim_start<=df_total.index) & (df_total.index<= time_lim_end)]

# test values all ok
print(df_total.isnull().values.any())

# if solar hour is needed
df_total['Hour'] = df_total.index.hour
df_total['Month'] = df_total.index.month    


print(df_total.columns)
print(df_total.head())
#sys.exit()
# __________________________________________________________________
# read wind data

def import_wind_pv(file, folder):
    df = pd.read_csv(folder+ file, skiprows = 33, sep = '\t', header = 0)
    print(df.columns)
    print(df.head())
    df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'], dayfirst=True)
    df = df.set_index('Unnamed: 0')

    df[file + '_wind'] = df['[m/s]'].astype(float)
    df[file + '_pv'] = df['[W/m²]'].astype(float)
    
    df = df.filter([file + '_wind', file + '_pv'])
    
    return df

df_data_total = pd.DataFrame()
df = import_wind_pv('DE_Hamburg.txt', folder)
df_data_total = pd.concat([df_data_total, df], axis = 1, sort = False)

df = import_wind_pv('DE_Berlin.txt', folder)
df_data_total = pd.concat([df_data_total, df], axis = 1, sort = False)


df = import_wind_pv('DE_Frankfurt.txt', folder)
df_data_total = pd.concat([df_data_total, df], axis = 1, sort = False)


df = import_wind_pv('DE_Hannover.txt', folder)
df_data_total = pd.concat([df_data_total, df], axis = 1, sort = False)

df = import_wind_pv('DE_Munchen.txt', folder)
df_data_total = pd.concat([df_data_total, df], axis = 1, sort = False)

print(df)
print(df_data_total.head())

# time lim wind
df_data_total = df_data_total[(time_lim_start<=df_data_total.index) & (df_data_total.index<= time_lim_end)]

# only wind now
name_list = []
for ii in df_data_total.columns:
    #if 'wind' in ii:
    if 'pv' in ii:
        name_list.append(ii)

print(name_list)
df_data_total = df_data_total.filter(name_list)

# for wind collect wind data and month
df_total = pd.concat([df_total, df_data_total], axis = 1, sort = False)

print(df_total.head())
print(df_total.columns)

# _____________________________________________________________________________
# import nn libraries
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# _____________________________________________________________________________
# reload X & Y
Y = df_total.iloc[:,0].values
X = df_total.iloc[:,1:].values

# _____________________________________________________________________________
# categorise the data - we use the sklearn library
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()

print(X[0])
print(len(X[0]))
print(len(X))
print(X[1])

# __________________________________________________________________
# categorize 1st time - month

X[:,-1] = labelencoder_X.fit_transform(X[:,-1])
# we will use dummy encoding - 0 or 1
onehotencoder = OneHotEncoder(categorical_features = [2])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
X = X[:,1:]

print(X[0])
print(X[12])

print(len(X[0]))
print(len(X))
print(X)

#sys.exit()

# __________________________________________________________________
# categorize 2rd time - hour
X[:,-1] = labelencoder_X.fit_transform(X[:,-1])
# we will use dummy encoding - 0 or 1
onehotencoder = OneHotEncoder(categorical_features = [12])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
X = X[:,1:]

print(X[0])
print(X[12])

print(len(X[0]))
print(len(X))
print(X)

# length of input in nn
input_dim_number = len(X[0])

# reshape Y
# _____________________________________________________________________________
Y = np.reshape(Y, (-1,1))

# scale x & y
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

print(scaler_x.fit(X))
xscale = scaler_x.transform(X)
print(scaler_y.fit(Y))
yscale = scaler_y.transform(Y)

# train test split
X_train, X_test, y_train, y_test = train_test_split(xscale, yscale)

# create model
model = Sequential()
#model.add(Dense(8, input_dim=23, kernel_initializer='normal', activation='relu'))
#model.add(Dense(15, input_dim=15, kernel_initializer='normal', activation='relu'))
model.add(Dense(input_dim_number, input_dim=input_dim_number, kernel_initializer='normal', activation='relu'))
#model.add(Dense(3, kernel_initializer='normal', activation='relu'))
model.add(Dense(8, kernel_initializer='normal', activation='relu'))
model.add(Dense(8, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.summary()
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse','mae'])

history = model.fit(X_train, y_train, epochs=150, batch_size=50,  verbose=1, validation_split=0.2)

y_plot_nn_scaled = model.predict(xscale)
y_plot_nn = scaler_y.inverse_transform(y_plot_nn_scaled) 

base_prediction = y_plot_nn

"""
# __________________________________________________________________
#compute the RMSE value
#error = mean_squared_error(base_prediction, y_test) ** 0.5
#error = mean_squared_error(base_prediction, Y) ** 0.5

validation_error = (base_prediction - Y) ** 2


# create model
error_model = Sequential()
#model.add(Dense(8, input_dim=23, kernel_initializer='normal', activation='relu'))
error_model.add(Dense(35, input_dim=35, kernel_initializer='normal', activation='relu'))
#model.add(Dense(3, kernel_initializer='normal', activation='relu'))
error_model.add(Dense(18, kernel_initializer='normal', activation='relu'))
error_model.add(Dense(1, kernel_initializer='normal'))
error_model.summary()
# Compile model
error_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse','mae'])

error_history = error_model.fit(X, validation_error, epochs=15, batch_size=50,  verbose=1, validation_split=0.2)


y_plot_nn_scaled_std = error_model.predict(xscale)
y_plot_nn_std = scaler_y.inverse_transform(y_plot_nn_scaled_std) 

y_plot_nn_std = y_plot_nn_std ** 0.5

y_plot_con_95 = y_plot_nn+ 2 * y_plot_nn_std
y_plot_con_5 = y_plot_nn - 2 * y_plot_nn_std

"""

# _____________________________________________________________________________
# plotting
fig, axs = plt.subplots(3,1,sharex=True)

axs[0].plot(df_total.index, df_total['Solar  - Actual Aggregated [MW]'], color ='blue')
#axs[0].plot(df_total.index, df_total['Wind Onshore  - Actual Aggregated [MW]'], color ='blue')
axs[0].plot(df_total.index, y_plot_nn, color = 'red')
#axs[0].plot(df_total.index, y_plot_con_95, color = 'red',linestyle='-.')
#axs[0].plot(df_total.index, y_plot_con_5, color = 'red',linestyle='-.')

axs[1].plot(df_total.index, df_total['Month'])
#axs[2].plot(df_total.index, df_total['Hour'])

#axs[3].plot(df_total.index, y_plot_nn_std)
#axs[2].plot(df_total.index, df_total['Wind Onshore Capacity'])
axs[2].plot(df_total.index, df_total['Solar Capacity'])

plt.show()
sys.exit()
"""
# __________________________________________________________________
# predict future

date_rng = pd.date_range(start='1.1.2018', end='31.12.2025', freq='H')
df_future = pd.DataFrame(date_rng, columns = ['Start_time'])
df_future['Data'] = np.zeros(len(date_rng))

df_future['Date'] = pd.to_datetime(df_future['Start_time']).apply(lambda x: x.date())
df_future = df_future.set_index('Start_time')

df_future['Weekday'] = df_future.index.dayofweek
df_future['Holiday'] =(df_future['Date']).apply(lambda x: is_holiday(x, holidays_dict))

df_future['Hour'] = df_future.index.hour
df_future['Month'] = df_future.index.month
df_future['Year'] = df_future.index.year


df_future = df_future.filter(['Year','Holiday','Month','Hour','Weekday'])
print(df_future.head())

# __________________________________________________________________
# NN apply
X = df_future.iloc[:,0:].values

# __________________________________________________________________
# categorize 1st time - weekday
X[:,-1] = labelencoder_X.fit_transform(X[:,-1])
# we will use dummy encoding - 0 or 1
onehotencoder = OneHotEncoder(categorical_features = [4])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
X = X[:,1:]

# __________________________________________________________________
# categorize 2nd time - hour

X[:,-1] = labelencoder_X.fit_transform(X[:,-1])
# we will use dummy encoding - 0 or 1
onehotencoder = OneHotEncoder(categorical_features = [9])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
X = X[:,1:]

# __________________________________________________________________
# categorize 3rd time - month
X[:,-1] = labelencoder_X.fit_transform(X[:,-1])
# we will use dummy encoding - 0 or 1
onehotencoder = OneHotEncoder(categorical_features = [31])
X = onehotencoder.fit_transform(X).toarray()

# categorise again
# Avoiding the dummy variable trap
X = X[:,1:]

xscale2 = scaler_x.transform(X)


y_plot_nn_scaled2 = model.predict(xscale2)
y_plot_nn2 = scaler_y.inverse_transform(y_plot_nn_scaled2) 

axs[0].plot(df_future.index, y_plot_nn2, color = 'green',linestyle='-.')
plt.show()
sys.exit()
"""