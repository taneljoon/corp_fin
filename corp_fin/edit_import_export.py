# python import export edit

# import libraries

import datetime
import pandas as pd
import sys
import datetime
import pickle
import os

sys.path.insert(0,'C:/Users/tanel.joon/OneDrive - Energia.ee/Documents_OneDrive/Python/for_import')

# sim main imputs
year_begin = 2015
year_end = 2019
state = 'EE'
folder = 'C:/Users/tanel.joon/OneDrive - Energia.ee/Documents_OneDrive/Python/data/'

list_states = ['DE'] # ['DE', 'EE']
   
def import_import_export(target_string):
    df_temp = pd.read_csv(target_string)
    #df_temp = df_temp.replace(' (CET)',' ')
    temp = df_temp['Time (CET)'].str.split(pat='-', n = 1, expand = True)
    df_temp['Start_time'] = pd.to_datetime(temp[0], dayfirst=True)
    df_temp = df_temp.set_index(df_temp['Start_time'])
    df_temp = df_temp.drop(['Time (CET)', 'Start_time'], axis = 1)

    return df_temp

for state in list_states:
    folder_sub = state + '_import_export/'
    file_list = os.listdir(folder + folder_sub)

    df_data_total = pd.DataFrame()

    for ii in range(year_begin, year_end):
        temp_string = str(ii) + '01010000-'
        
        df_one_year = pd.DataFrame()
        for jj in file_list:
            
            if temp_string in jj:
                target_string = folder + folder_sub + jj
                df = import_import_export(target_string)
                df_one_year = pd.concat([df_one_year, df], axis = 1, sort = False)
        
        df_data_total = pd.concat([df_data_total, df_one_year], axis = 0, sort = False)
            
    #df_data_total = df_data_total.replace('-',0)
    #df_data_total = df_data_total.replace('n/e',0)
    #df_data_total = df_data_total.replace('N/A',0)

    df_data_total = df_data_total.fillna(method ='ffill')

    jj = 0
    for ii in df_data_total.columns:
        temp =  str(jj//2)
        if jj%2 == 0:
            df_data_total['IN_'+ temp] = df_data_total[ii]
        else:
            df_data_total['OUT_'+ temp] = df_data_total[ii]
        
        jj = jj + 1
    #for ii in file_list:

    list_in = []
    list_out = []
    for ii in df_data_total.columns:
        if 'IN_' in ii:
            list_in.append(ii)
        if 'OUT_' in ii:
            list_out.append(ii)

    df_data_total['Net'] = df_data_total[list_in].sum(axis=1) - df_data_total[list_out].sum(axis=1)
    df_data_total.to_csv('test_import_export.csv')

    with open(state + '_df_import_export.obj', 'wb') as file:
        pickle.dump(df_data_total, file)

sys.exit()

# 