# import and pickle into good format

# _____________________________________________________________________________
# import libraries
import datetime
import pandas as pd
import sys
import datetime
import pickle
sys.path.insert(0,'C:/Users/tanel.joon/OneDrive - Energia.ee/Documents_OneDrive/Python/for_import')
import holidays

# _____________________________________________________________________________
# import data - ensoe
# load data

year_begin = 2015
year_end = 2019
state = 'DE'
folder = 'C:/Users/tanel.joon/OneDrive - Energia.ee/Documents_OneDrive/Python/data/'

df_gen = pd.DataFrame()
df_price = pd.DataFrame()
df_load = pd.DataFrame()

for ii in range(year_begin,year_end):
    temp_str = str(ii) +'01010000-' + str(ii+1) + '01010000.csv'
    #temp_str2 = str(ii) +'01010000-' + str(ii+1) + '01010000.csv'
    temp_file_gen = state +'_Actual Generation per Production Type_' + temp_str
    #print(temp_file_gen)
    temp_file_price = state + '_Day-ahead Prices_' + temp_str
    temp_file_load = state + '_Total Load - Day Ahead _ Actual_' + temp_str
    
    temp_df_gen = pd.read_csv(folder + temp_file_gen)
    temp_df_price = pd.read_csv(folder + temp_file_price)
    temp_df_load = pd.read_csv(folder + temp_file_load)
    
    df_gen = df_gen.append(temp_df_gen)
    df_price = df_price.append(temp_df_price)
    df_load = df_load.append(temp_df_load)
    
#print(df_gen)
#df_gen.to_csv('test_gen.csv')
#df_load.to_csv('test_load.csv')
print(df_gen.shape)
print(df_price.shape)
print(df_load.shape)

#print(df_gen.columns)
#print(df_gen.columns.values)
#sys.exit()

# _____________________________________________________________________________
# edit data for good format 
# split time series to start time and end time
df_gen = df_gen.replace(' (CET)',' ')
temp = df_gen['MTU'].str.split(pat='-', n = 1, expand = True)
df_gen['Start_time'] = pd.to_datetime(temp[0], dayfirst=True)

temp = df_load['Time (CET)'].str.split(pat='-', n = 1, expand = True)
df_load['Start_time'] = pd.to_datetime(temp[0], dayfirst=True)

temp = df_price['MTU (CET)'].str.split(pat='-', n = 1, expand = True)
df_price['Start_time'] = pd.to_datetime(temp[0], dayfirst=True)

# _____________________________________________________________________________
# datetime to index
df_gen = df_gen.set_index('Start_time', drop=False)
df_load = df_load.set_index('Start_time', drop=False)
df_price = df_price.set_index('Start_time', drop=False)

# _____________________________________________________________________________
# replace - & n/e
replace_dict = {'N/A':0,'n/e':0, '-':0, 'NaN':0}
df_gen = df_gen.replace('-',0)
df_gen = df_gen.replace('n/e',0)
df_gen = df_gen.replace('N/A',0)

#df_gen = df_gen.fillna(method ='ffill')
#df_gen = df_gen.fillna(value=replace_dict)
#df_gen = df_gen.fillna(method ='ffill')

df_load = df_load.fillna(method ='ffill')
df_price = df_price.fillna(method ='ffill')

df_load = df_load.fillna(method ='bfill')
df_price = df_price.fillna(method ='bfill')

df_load = df_load.replace('-',0)
df_load = df_load.replace('n/e',0)
df_load = df_load.replace('N/A',0)

df_price = df_price.replace('-',0)
df_price = df_price.replace('n/e',0)
df_price = df_price.replace('N/A',0)

# _____________________________________________________________________________
# make columns into floats - that are not datetime

for ii in df_price.columns.values:
    print(ii)
    if ii != 'Start_time' and ii!='MTU (CET)':
        df_price[ii] = df_price[ii].astype(float)
    
for ii in df_gen.columns.values:
    print(ii)
    if ii != 'Start_time' and ii!='MTU' and ii!='Area':
        df_gen[ii] = df_gen[ii].astype(float)

for ii in df_load.columns.values:
    print(ii)
    if ii !='Start_time' and ii!= 'Time (CET)':
        df_load[ii] = df_load[ii].astype(float)
        
# _____________________________________________________________________________
# resample
        
df_gen = df_gen.resample('H').mean()
df_load = df_load.resample('H').mean()
df_price = df_price.resample('H').mean()

# _____________________________________________________________________________
# fillna
#df_price.fillna(0)
#df_gen.fillna(0)
#df_load.fillna(0)



print(df_gen.shape)
print(df_price.shape)
print(df_load.shape)

# _____________________________________________________________________________
# pickle 
#with open('df_temp.obj', 'wb') as file:
#    pickle.dump((df_gen, df_price, df_load), file)
    

#sys.exit()
#with open('df_temp.obj', 'rb') as file:
#    (df_gen, df_price, df_load) = pickle.load(file)
# _____________________________________________________________________________
# collect into df_total
#df_total = pd.DataFrame()

df_total = pd.concat([df_gen, df_load, df_price], axis = 1, sort = False)
# forward fill
#df_total = df_total.fillna(method ='ffill')
df_total.to_csv('df_total.csv')

with open('df_total.obj', 'wb') as file:
    pickle.dump(df_total, file)

sys.exit()