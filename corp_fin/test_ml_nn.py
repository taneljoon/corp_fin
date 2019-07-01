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
import holidays

import pickle


with open('df_total.obj', 'rb') as file:
    df_total = pickle.load(file)


df_total = df_total.fillna(method ='ffill')

print(df_total.isnull().values.any())

Y = df_total.iloc[:,1].values

X = df_total.iloc[:,3:].values

# save values
import random
for ii in range(0,len(Y)):
    df_total.iloc[ii,3] = random.random()

X_base = X
Y_base = Y


# _____________________________________________________________________________
# multilinear regression
"""
# categorise the data - we use the sklearn library
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()

X[:,3] = labelencoder_X.fit_transform(X[:,3])
# we will use dummy encoding - 0 or 1
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
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
X_opt = X[:,[0,1,2,3]]

# select a significance level - we choose  SL = 0.05
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
report = regressor_OLS.summary() # summary - very good

print(report)

#sys.exit()
print(y_pred)
print(Y_test)



# _____________________________________________________________________________

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
Y = df_total.iloc[:,1].values
X = df_total.iloc[:,3:].values

Y = np.reshape(Y, (-1,1))
# scale data

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
print(scaler_x.fit(X))
xscale = scaler_x.transform(X)
print(scaler_y.fit(Y))
yscale = scaler_y.transform(Y)

X_train, X_test, y_train, y_test = train_test_split(xscale, yscale)

# create model
model = Sequential()
model.add(Dense(8, input_dim=3, kernel_initializer='normal', activation='relu'))
model.add(Dense(3, kernel_initializer='normal', activation='relu'))
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

axs[0].plot(df_total['Start_time'], df_total['Day-ahead Price [EUR/MWh]'], color ='blue')
axs[0].plot(df_total['Start_time'], y_plot, color = 'red')
axs[0].plot(df_total['Start_time'], y_plot_nn, color = 'green')
axs[1].plot(df_total['Start_time'], df_total['Wind Onshore  - Actual Aggregated [MW]'])
axs[2].plot(df_total['Start_time'], df_total['Wind Offshore  - Actual Aggregated [MW]'])
axs[3].plot(df_total['Start_time'], df_total['Solar  - Actual Aggregated [MW]'])
axs[4].plot(df_total['Start_time'], df_total['Actual Total Load [MW] - BZN|DE-AT-LU'])
plt.show()
sys.exit()


# make ml prediction from inputs 
# Inputs
# - Wind onshore
# - Wind offshore
# - Solar
# - demand

# Output
# - price

