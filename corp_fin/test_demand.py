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

#df_total = df_total[(time_lim_start<df_total['Start_time']) & (df_total['Start_time']< time_lim_end)]
df_total = df_total[(time_lim_start<=df_total.index) & (df_total.index<= time_lim_end)]
print(df_total.head())
#sys.exit()

de_holidays = holidays.DE()


df_total = df_total.filter(['Actual Total Load [MW] - BZN|DE-AT-LU'])
print(df_total.columns)
print(df_total.isnull().values.any())
temp = df_total.index[0]
print(type(temp))


def is_workday(weekday, is_holiday):
    if weekday < 5 and (not is_holiday):
        return 1
    else:
        return 0
        
df_total['Start_time'] = df_total.index
df_total['Date'] = pd.to_datetime(df_total['Start_time']).apply(lambda x: x.date())


df_total['Weekday'] = df_total['Start_time'].dt.dayofweek

df_total['Holiday'] =(df_total['Date']).apply(lambda x: x in de_holidays)
df_total['Workday'] = df_total.apply(lambda x: is_workday(x['Weekday'], x['Holiday']), axis=1)

print(df_total['Date'])
print(df_total['Holiday'])
print(df_total['Workday'])

df_total.to_csv('test_day.csv')
#print(df_total)
sys.exit()

# plotting
fig, axs = plt.subplots(2,1,sharex=True)

axs[0].plot(df_total.index, df_total['Day-ahead Price [EUR/MWh]'], color ='blue')
axs[1].plot(df_total.index, df_total['Actual Total Load [MW] - BZN|DE-AT-LU'])
plt.show()