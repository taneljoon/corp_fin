# import and pickle into good format

# _____________________________________________________________________________
# import libraries
import datetime
import pandas as pd
import sys
import datetime

sys.path.insert(0,'C:/Users/tanel.joon/OneDrive - Energia.ee/Documents_OneDrive/Python/for_import')
import holidays

# _____________________________________________________________________________
# import data - ensoe
# load data
file_gen = 'DE_Actual Generation per Production Type_201801010000-201901010000.csv'
file_price = 'DE_Day-ahead Prices_201801010000-201901010000.csv'
file_load = 'DE_Total Load - Day Ahead _ Actual_201801010000-201901010000.csv'
folder = 'C:/Users/tanel.joon/OneDrive - Energia.ee/Documents_OneDrive/Python/data/'
df_gen = pd.read_csv(folder + file_gen)
df_price = pd.read_csv(folder + file_price)
df_load = pd.read_csv(folder + file_load)


# _____________________________________________________________________________
# edit data for good format 
# split time series to start time and end time
df_gen = df_gen.replace(' (CET)',' ')

temp = df_gen['MTU'].str.split(pat='-', n = 1, expand = True)
df_gen['Start_time'] = pd.to_datetime(temp[0], dayfirst=True)
#df_gen['Start_time'] = pd.to_datetime(temp[0], format='%d/%m/%Y, %H:%M')
#df_gen['End_time'] = pd.to_datetime(temp[1])

# split time series to start time and end time
temp = df_price['MTU (CET)'].str.split(pat='-', n = 1, expand = True)
df_price['Start_time'] = pd.to_datetime(temp[0], dayfirst=True)
#df_price['Start_time'] = pd.to_datetime(temp[0], format='%d/%m/%Y, %H:%M')
#df_price['End_time'] = pd.to_datetime(temp[1])

temp = df_load['Time (CET)'].str.split(pat='-', n = 1, expand = True)
df_load['Start_time'] = pd.to_datetime(temp[0], dayfirst=True)
#df_price['Start_time'] = pd.to_datetime(temp[0], format='%d/%m/%Y, %H:%M')
#df_price['End_time'] = pd.to_datetime(temp[1])

# _____________________________________________________________________________
# set time limit
time_lim = datetime.datetime(2018,9,30)
df_price = df_price[df_price['Start_time'] < time_lim]
df_gen = df_gen[df_gen['Start_time'] < time_lim]
df_load = df_load[df_load['Start_time'] < time_lim]

df_price = df_price.replace('-','0')
df_price['Day-ahead Price [EUR/MWh]'] = df_price['Day-ahead Price [EUR/MWh]'].astype(float)
df_price.fillna(0)

df_gen = df_gen.replace('-','0')
df_gen['Wind Onshore  - Actual Aggregated [MW]'] =  df_gen['Wind Onshore  - Actual Aggregated [MW]'].astype(float)
df_gen['Wind Offshore  - Actual Aggregated [MW]'] =  df_gen['Wind Offshore  - Actual Aggregated [MW]'].astype(float)
df_gen['Solar  - Actual Aggregated [MW]'] =  df_gen['Solar  - Actual Aggregated [MW]'].astype(float)
df_gen.fillna(0)

df_load['Actual Total Load [MW] - BZN|DE-AT-LU'] = df_load['Actual Total Load [MW] - BZN|DE-AT-LU'].astype(float)
df_load = df_load.replace('-','0')
df_load.fillna(0)

# _____________________________________________________________________________
# resample gen dataframe

df_gen = df_gen.set_index('Start_time', drop=False)
df_gen = df_gen.resample('H').mean()

df_gen['Start_time'] = df_gen.index


df_load = df_load.set_index('Start_time', drop=False)
df_price = df_price.set_index('Start_time', drop=False)

# _____________________________________________________________________________
# pickle
# collect data into one dataframe and store with pickle
import pickle


df_total = pd.DataFrame()
df_total['Start_time'] = df_price['Start_time']
df_total['Day-ahead Price [EUR/MWh]'] = df_price['Day-ahead Price [EUR/MWh]']
df_total['Wind Onshore  - Actual Aggregated [MW]'] = df_gen['Wind Onshore  - Actual Aggregated [MW]']
df_total['Wind Offshore  - Actual Aggregated [MW]'] = df_gen['Wind Offshore  - Actual Aggregated [MW]']
df_total['Solar  - Actual Aggregated [MW]'] = df_gen['Solar  - Actual Aggregated [MW]']
df_total['Actual Total Load [MW] - BZN|DE-AT-LU'] = df_load['Actual Total Load [MW] - BZN|DE-AT-LU']

with open('df_total.obj', 'wb') as file:
    pickle.dump(df_total, file)