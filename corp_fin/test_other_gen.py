# test ml & neural

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


# import gen & price & consumption

with open('df_total.obj', 'rb') as file:
    df_total = pickle.load(file)


# C:\Users\tanel.joon\OneDrive - Energia.ee\Data_for_sharing\import_export\interim_results_imex
folder = 'C:/Users/tanel.joon/OneDrive - Energia.ee/Data_for_sharing/import_export/interim_results_imex/'
filename = 'DE_df_import_export.obj'

# import net gen
with open(folder + filename, 'rb') as file:
    df_imex = pickle.load(file)


# import C02 price
folder = 'C:/Users/tanel.joon/OneDrive - Energia.ee/Data_for_sharing/'
filename = 'EUA_price.csv'

df_co2 = pd.read_csv(folder + filename, delimiter = ';')

df_co2['Date'] = pd.to_datetime(df_co2['Date'], dayfirst=True)
df_co2 = df_co2.set_index('Date')

time_lim_start = datetime.datetime(2015,1,6)
time_lim_end = datetime.datetime(2018,9,30)

df_co2 = df_co2[(time_lim_start<=df_co2.index) & (df_co2.index<= time_lim_end)]
df_co2 = df_co2.resample('H').pad() 
#df_co2 = df_co2.drop(['Date'])
#print(df_co2)
#sys.exit()
   
# _____________________________________________________________________________
# import de holidays
holidays_dict = holidays.DE()

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
df_total['Holiday'] =(df_total['Date']).apply(lambda x: is_holiday(x, holidays_dict))

df_total['Hour'] = df_total.index.hour
df_total['Month'] = df_total.index.month
df_total['Year'] = df_total.index.year

# _____________________________________________________________________________

# means 

#df_total['Wind_3h'] = df_total.groupby('Wind Onshore  - Actual Aggregated [MW]').apply(lambda x: x.resample('3H').first())
#print(df_total['Wind_3h'])

df_total['Temp_72_wind'] = df_total['Wind Onshore  - Actual Aggregated [MW]'].rolling(72).mean()
df_total['Temp_48_wind'] = df_total['Wind Onshore  - Actual Aggregated [MW]'].rolling(48).mean()
df_total['Temp_24_wind'] = df_total['Wind Onshore  - Actual Aggregated [MW]'].rolling(24).mean()
df_total['Temp_12_wind'] = df_total['Wind Onshore  - Actual Aggregated [MW]'].rolling(12).mean()
df_total['Temp_6_wind'] = df_total['Wind Onshore  - Actual Aggregated [MW]'].rolling(6).mean()
df_total['Temp_3_wind'] = df_total['Wind Onshore  - Actual Aggregated [MW]'].rolling(3).mean()

df_total['Temp_24_wind_off'] = df_total['Wind Offshore  - Actual Aggregated [MW]'].rolling(24).mean()
df_total['Temp_12_wind_off'] = df_total['Wind Offshore  - Actual Aggregated [MW]'].rolling(12).mean()
df_total['Temp_6_wind_off'] = df_total['Wind Offshore  - Actual Aggregated [MW]'].rolling(6).mean()
df_total['Temp_3_wind_off'] = df_total['Wind Offshore  - Actual Aggregated [MW]'].rolling(3).mean()

df_total['Temp_24_pv'] = df_total['Solar  - Actual Aggregated [MW]'].rolling(24).mean()
df_total['Temp_12_pv'] = df_total['Solar  - Actual Aggregated [MW]'].rolling(12).mean()
df_total['Temp_6_pv'] = df_total['Solar  - Actual Aggregated [MW]'].rolling(6).mean()
df_total['Temp_3_pv'] = df_total['Solar  - Actual Aggregated [MW]'].rolling(3).mean()
df_total['Temp_1_pv'] = df_total['Solar  - Actual Aggregated [MW]'].rolling(1).mean()

print(df_total['Temp_24_wind'])
print(df_total['Temp_24_wind'].head())
print(df_total['Wind Onshore  - Actual Aggregated [MW]'])
print(df_total['Wind Onshore  - Actual Aggregated [MW]'].head())

#sys.exit()

# _____________________________________________________________________________
# set time limit


#time_lim_start = datetime.datetime(2015,1,6)
#time_lim_end = datetime.datetime(2018,9,30)

df_total = df_total[(time_lim_start<=df_total.index) & (df_total.index<= time_lim_end)]
df_total = pd.concat([df_total, df_co2], axis = 1, sort = False)
    

df_total['Gen_other'] = (df_total['Actual Total Load [MW] - BZN|DE-AT-LU'] -
    df_total['Wind Offshore  - Actual Aggregated [MW]'] -
    df_total['Wind Onshore  - Actual Aggregated [MW]'] -
    df_total['Solar  - Actual Aggregated [MW]'])
    
list_columns_filter = (['Day-ahead Price [EUR/MWh]',
    'Wind Offshore  - Actual Aggregated [MW]',
    'Wind Onshore  - Actual Aggregated [MW]',
    'Solar  - Actual Aggregated [MW]',
    'Actual Total Load [MW] - BZN|DE-AT-LU',
    'EUA',
    'Gen_other',
    'Holiday',
    'Temp_72_wind',
    'Temp_48_wind',
    'Temp_24_wind',
    'Temp_12_wind',
    'Temp_6_wind',
    'Temp_3_wind',
    'Temp_24_wind_off',
    'Temp_12_wind_off',
    'Temp_6_wind_off',
    'Temp_3_wind_off',
    'Temp_24_pv',
    'Temp_12_pv',
    'Temp_6_pv',
    'Temp_3_pv'])

#,
#'Fossil Gas  - Actual Aggregated [MW]'
#    , 

list_columns = df_total.columns.tolist()
print(list_columns)
list_columns.remove('Day-ahead Price [EUR/MWh]')
list_columns.insert(0,'Day-ahead Price [EUR/MWh]')
df_total = df_total[list_columns]

df_total = df_total.filter(list_columns_filter)

df_total = df_total.fillna(method ='ffill')

# _____________________________________________________________________________
# ml setup

Y = df_total.iloc[:,0].values
X = df_total.iloc[:,1:].values


print(len(X))
print(len(X[0]))
print(df_total.columns)
answer = input('Do you want to proceed? (y/n) ').lower()

if answer != 'y':
    sys.exit()


X_base = X
Y_base = Y

# _____________________________________________________________________________
# multilinear regression implement
"""
# categorise the data - we use the sklearn library
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()

X[:,3] = labelencoder_X.fit_transform(X[:,3])
# we will use dummy encoding - 0 or 1
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
X = X[:,1:]
"""


# splitting the dataset to the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# y plot
y_plot = regressor.predict(X)


# Building the optimal model using Backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((len(Y),1)).astype(int), values = X,axis=1)
# we are taking out columns that are not statistically significat
X_opt = X
#X_opt = X[:,[0,1,2,3]]

# select a significance level - we choose  SL = 0.05
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
report = regressor_OLS.summary() # summary - very good

print(report)

#sys.exit()
print(y_pred)
print(Y_test)

answer = input('Do you want to proceed with NN? (y/n) ').lower()

if answer != 'y':
    sys.exit()
# _____________________________________________________________________________
# nn implement
import numpy as np
#import pandas

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# reload X & Y
#Y = df_total.iloc[:,1].values
#X = df_total.iloc[:,2:].values

Y = df_total.iloc[:,0].values
X = df_total.iloc[:,1:].values

Y = np.reshape(Y, (-1,1))
# scale data

print(X[0])
print(len(X))
print(len(X[0]))
# length of input in nn
input_dim_number = len(X[0])

#sys.exit()

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
print(scaler_x.fit(X))
xscale = scaler_x.transform(X)
print(scaler_y.fit(Y))
yscale = scaler_y.transform(Y)

X_train, X_test, y_train, y_test = train_test_split(xscale, yscale, test_size=0.2, random_state = 0)

# create model
model = Sequential()

model.add(Dense(input_dim_number, input_dim=input_dim_number, kernel_initializer='normal', activation='relu'))
model.add(Dense(int(input_dim_number*2), kernel_initializer='normal', activation='relu'))
model.add(Dense(input_dim_number, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.summary()
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse','mae'])

history = model.fit(X_train, y_train, epochs=150, batch_size=50,  verbose=1, validation_split=0.2)

y_plot_nn_scaled = model.predict(xscale)
y_plot_nn = scaler_y.inverse_transform(y_plot_nn_scaled) 

# _____________________________________________________________________________
# plotting
fig, axs = plt.subplots(8,1,sharex=True)

axs[0].plot(df_total.index, df_total['Day-ahead Price [EUR/MWh]'], color ='blue')
axs[0].plot(df_total.index, y_plot, color = 'red',linestyle='-.')
axs[0].plot(df_total.index, y_plot_nn, color = 'green',linestyle='-.')

try:
    axs[1].plot(df_total.index, df_total['Wind Onshore  - Actual Aggregated [MW]'])
except:
    print('xxx')

try:
    axs[1].plot(df_total.index, df_total['Wind Offshore  - Actual Aggregated [MW]'])
except:
    print('xxx')

try:
    axs[1].plot(df_total.index, df_total['Solar  - Actual Aggregated [MW]'])
except:
    print('xxx')

try:
    axs[2].plot(df_total.index, df_total['Actual Total Load [MW] - BZN|DE-AT-LU'])
except:
    print('xxx')

try:
    axs[3].plot(df_total.index, df_total['Fossil Gas  - Actual Aggregated [MW]'])
except:
    print('xxx')

try:
    axs[4].plot(df_total.index, df_total['Fossil Coal-derived gas  - Actual Aggregated [MW]'])
    axs[5].plot(df_total.index, df_total['Fossil Hard coal  - Actual Aggregated [MW]'])
    axs[6].plot(df_total.index, df_total['Fossil Brown coal/Lignite  - Actual Aggregated [MW]'])
    axs[7].plot(df_total.index, df_total['Nuclear  - Actual Aggregated [MW]'])

except:
    print('xxx')

plt.show()


sys.exit()

