# test ml & neural
# project demand


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

# _____________________________________________________________________________
# set time limit
time_lim_start = datetime.datetime(2015,1,1)
time_lim_end = datetime.datetime(2018,9,30)

# filter time
df_total = df_total[(time_lim_start<=df_total.index) & (df_total.index<= time_lim_end)]

# import de holidays
holidays_dict = holidays.DE()

# filter out unneeded columns
df_total = df_total.filter(['Actual Total Load [MW] - BZN|DE-AT-LU'])

# test values all ok
print(df_total.isnull().values.any())


# create workday & hour columns
def is_workday(weekday, is_holiday):
    if weekday < 5 and (not is_holiday):
        return 1
    else:
        return 0

def is_holiday(date, holidays_dict):
    if date in holidays_dict:
        return 1
    else:
        return 0
     
#         
df_total['Start_time'] = df_total.index
df_total['Date'] = pd.to_datetime(df_total['Start_time']).apply(lambda x: x.date())

df_total['Weekday'] = df_total.index.dayofweek

#df_total['Holiday'] =(df_total['Date']).apply(lambda x: x in de_holidays)
#df_total['Workday'] = df_total.apply(lambda x: is_workday(x['Weekday'], x['Holiday']), axis=1)
df_total['Holiday'] =(df_total['Date']).apply(lambda x: is_holiday(x, holidays_dict))

df_total['Hour'] = df_total.index.hour
df_total['Month'] = df_total.index.month
df_total['Year'] = df_total.index.year

#print(df_total['Hour'].head())
#print(df_total['Month'].head())
#print(df_total['Month'][0])
#print(type(df_total['Month'][0]))
#df_temp = df_total['Holiday'].resample('D').mean()
# if next day holiday
#df_temp['Next_day_holiday'] = df_temp['Holiday']
#for ii in len(df_total):
#    try:
#        if df_temp['Holiday'][ii+1].value ==1:
#        df_total['Next_day_holiday'][ii] = !
#    except:
#        df_total['Next_day_holiday'][ii] = 0
        

df_total = df_total.filter(['Actual Total Load [MW] - BZN|DE-AT-LU','Year','Holiday','Month','Hour','Weekday'])
print(df_total.head())

# _____________________________________________________________________________


# filter out unneeded
#df_total = df_total.filter(['Actual Total Load [MW] - BZN|DE-AT-LU'])

# _____________________________________________________________________________
#print(df_total['Date'])
#print(df_total['Holiday'])
#print(df_total['Workday'])
#df_total.to_csv('test_day.csv')
#print(df_total.head())

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
# categorize 1st time - weekday
X[:,-1] = labelencoder_X.fit_transform(X[:,-1])
# we will use dummy encoding - 0 or 1
onehotencoder = OneHotEncoder(categorical_features = [4])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
X = X[:,1:]

print(X[0])
print(X[12])

print(len(X[0]))
print(len(X))
print(X)

# __________________________________________________________________
# categorize 2nd time - hour

X[:,-1] = labelencoder_X.fit_transform(X[:,-1])
# we will use dummy encoding - 0 or 1
onehotencoder = OneHotEncoder(categorical_features = [9])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
X = X[:,1:]

print(X[0])
print(X[12])

print(len(X[0]))
print(len(X))
print(X)

# __________________________________________________________________
# categorize 3rd time - month
X[:,-1] = labelencoder_X.fit_transform(X[:,-1])
# we will use dummy encoding - 0 or 1
onehotencoder = OneHotEncoder(categorical_features = [31])
X = onehotencoder.fit_transform(X).toarray()

# categorise again
# Avoiding the dummy variable trap
X = X[:,1:]

print(X[0])
print(X[12])

print(len(X[0]))
print(len(X))
print(X)
#sys.exit()

# _____________________________________________________________________________
Y = np.reshape(Y, (-1,1))

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

print(scaler_x.fit(X))
xscale = scaler_x.transform(X)
print(scaler_y.fit(Y))
yscale = scaler_y.transform(Y)

X_train, X_test, y_train, y_test = train_test_split(xscale, yscale)

# create model
model = Sequential()
#model.add(Dense(8, input_dim=23, kernel_initializer='normal', activation='relu'))
model.add(Dense(42, input_dim=42, kernel_initializer='normal', activation='relu'))
#model.add(Dense(3, kernel_initializer='normal', activation='relu'))
model.add(Dense(20, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.summary()
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse','mae'])

history = model.fit(X_train, y_train, epochs=150, batch_size=50,  verbose=1, validation_split=0.2)

y_plot_nn_scaled = model.predict(xscale)
y_plot_nn = scaler_y.inverse_transform(y_plot_nn_scaled) 

# _____________________________________________________________________________
# plotting
fig, axs = plt.subplots(5,1,sharex=True)

axs[0].plot(df_total.index, df_total['Actual Total Load [MW] - BZN|DE-AT-LU'], color ='blue')
axs[0].plot(df_total.index, y_plot_nn, color = 'red',linestyle='-.')

axs[1].plot(df_total.index, df_total['Month'])
axs[2].plot(df_total.index, df_total['Hour'])
axs[3].plot(df_total.index, df_total['Weekday'])
axs[4].plot(df_total.index, df_total['Holiday'])

#plt.show()

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
