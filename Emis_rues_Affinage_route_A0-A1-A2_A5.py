#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 15:49:08 2024

@author: meganepourtois
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 11:01:28 2023

@author: meganepourtois
"""
#%% Import package
import pandas as pd
import matplotlib.pyplot as plt
#%% Process data
sir_segm=pd.read_excel('C:/SiraneDocument/Sirane2_segmenté.xlsx')
sir_segm['Hierarchie'] = sir_segm['Hierarchie'].apply(lambda x: 'A5' if pd.isna(x) or x == ' ' else x)
sir_segm.head()

no2=pd.read_excel('C:/SiraneDocument/Copert2022/no2.xlsx', index_col=0)
no=pd.read_excel('C:/SiraneDocument/Copert2022/no.xlsx', index_col=0)
nox=pd.read_excel('C:/SiraneDocument/Copert2022/NOx.xlsx', index_col=0)
pm10=pd.read_excel('C:/SiraneDocument/Copert2022/pm10.xlsx', index_col=0)
pm25=pd.read_excel('C:/SiraneDocument/Copert2022/pm25.xlsx', index_col=0)
# Renommer les colonnes créées vu que no2 dfs ont des colonnes différentes
for df in [no2, no, nox, pm10, pm25]:
    df.columns = ['vl_tot', 'vu_tot', 'c2_tot', 'c3_tot', 'ca_tot']
    
types_de_route = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']
types_de_vehicules = ['vl_tot', 'vu_tot', 'c2_tot', 'c3_tot', 'ca_tot']
polluants=['no','no2','nox','pm10','pm25']

A0 = {
    "no": no.loc[['Highway']],
    "no2": no2.loc[['Highway']],
    "nox": nox.loc[['Highway']],
    "pm10": pm10.loc[['Highway']],
    "pm25": pm25.loc[['Highway']]
}

A1 = {
    "no": no.loc[['Rural']],
    "no2": no2.loc[['Rural']],
    "nox": nox.loc[['Rural']],
    "pm10": pm10.loc[['Rural']],
    "pm25": pm25.loc[['Rural']]
}

A2_5 = {
    "no": no.loc[['Reste']],
    "no2": no2.loc[['Reste']],
    "nox": nox.loc[['Reste']],
    "pm10": pm10.loc[['Reste']],
    "pm25": pm25.loc[['Reste']]
}

sir_A0 = sir_segm[sir_segm['Hierarchie'] == 'A0']
sir_A1 = sir_segm[sir_segm['Hierarchie'] == 'A1']
sir_A2 = sir_segm[sir_segm['Hierarchie'] == 'A2']
sir_A3 = sir_segm[sir_segm['Hierarchie'] == 'A3']
sir_A4 = sir_segm[sir_segm['Hierarchie'] == 'A4']
sir_A5 = sir_segm[sir_segm['Hierarchie'] == 'A5']
sir_A2_5 = sir_segm[sir_segm['Hierarchie'].isin(['A2', 'A3', 'A4', 'A5'])]
#%%Proportion A2 à A5 pour chaque véhicule

#Facteurs de répartition
repartA2_A5 = pd.DataFrame(index=types_de_vehicules, columns=['A2', 'A3', 'A4', 'A5'])
vehicle_type_sums = sir_A2_5[types_de_vehicules].sum()
for vehicle_type in types_de_vehicules:
    for route in ['A2', 'A3', 'A4', 'A5']:
        repartA2_A5.at[vehicle_type, route] = sir_A2_5.loc[sir_A2_5['Hierarchie'] == route, vehicle_type].sum() / vehicle_type_sums[vehicle_type]

#Répartition (multiplication par le facteur)
# Dictionnaire par route par polluant par vehicule emission totale. 

A2_3_4_5 = {route: {'no': None, 'no2': None, 'nox': None, 'pm10': None, 'pm25': None} for route in ['A2', 'A3', 'A4', 'A5']}
for vehicle_type in types_de_vehicules:
    for route in ['A2', 'A3', 'A4', 'A5']:
        proportion = repartA2_A5.loc[vehicle_type, route]
        for pollutant in polluants:
            df = globals()[pollutant]  

            A2_3_4_5[route][pollutant] = df.loc[['Reste']] * proportion
            





#%% Emis rues 
# Dictionnaires: route --> polluants --> vehicules pour chaque segment
results_A0 = {pollutant: {} for pollutant in polluants}
results_A1 = {pollutant: {} for pollutant in polluants}
results_A2 = {pollutant: {} for pollutant in polluants}
results_A3 = {pollutant: {} for pollutant in polluants}
results_A4 = {pollutant: {} for pollutant in polluants}
results_A5 = {pollutant: {} for pollutant in polluants}
results_A2_5 = {pollutant: {} for pollutant in polluants}

for pollutant in polluants:
    df_A0 = A0[pollutant]
    df_A1 = A1[pollutant]
    df_A2 = A2_3_4_5['A2'][pollutant]
    df_A3 = A2_3_4_5['A3'][pollutant]
    df_A4 = A2_3_4_5['A4'][pollutant]
    df_A5 = A2_3_4_5['A5'][pollutant]
    df_A2_5 = A2_5[pollutant]
    for vehicle_type in types_de_vehicules:
        results_A0[pollutant][vehicle_type] = (sir_A0[vehicle_type] * df_A0[vehicle_type].sum()) / sir_A0[vehicle_type].sum()
        results_A1[pollutant][vehicle_type] = (sir_A1[vehicle_type] * df_A1[vehicle_type].sum()) / sir_A1[vehicle_type].sum()
        results_A2[pollutant][vehicle_type] = (sir_A2[vehicle_type] * df_A2[vehicle_type].sum()) / sir_A2[vehicle_type].sum()
        results_A3[pollutant][vehicle_type] = (sir_A3[vehicle_type] * df_A3[vehicle_type].sum()) / sir_A3[vehicle_type].sum()
        results_A4[pollutant][vehicle_type] = (sir_A4[vehicle_type] * df_A4[vehicle_type].sum()) / sir_A4[vehicle_type].sum()
        results_A5[pollutant][vehicle_type] = (sir_A5[vehicle_type] * df_A5[vehicle_type].sum()) / sir_A5[vehicle_type].sum()
        results_A2_5[pollutant][vehicle_type] = (sir_A2_5[vehicle_type] * df_A2_5[vehicle_type].sum()) / sir_A2_5[vehicle_type].sum()

# Dictionnaires: route --> polluants --> total pour chaque segment
totals_A0= pd.DataFrame()
totals_A1= pd.DataFrame()
totals_A2= pd.DataFrame()
totals_A3= pd.DataFrame()
totals_A4= pd.DataFrame()
totals_A5= pd.DataFrame()
totals_A2_5= pd.DataFrame()

for pollutant in polluants:
    totals_A0[pollutant] = pd.DataFrame(results_A0[pollutant]).sum(axis=1)
    totals_A1[pollutant] = pd.DataFrame(results_A1[pollutant]).sum(axis=1)
    totals_A2[pollutant] = pd.DataFrame(results_A2[pollutant]).sum(axis=1)
    totals_A3[pollutant] = pd.DataFrame(results_A3[pollutant]).sum(axis=1)
    totals_A4[pollutant] = pd.DataFrame(results_A4[pollutant]).sum(axis=1)
    totals_A5[pollutant] = pd.DataFrame(results_A5[pollutant]).sum(axis=1)
    totals_A2_5[pollutant] = pd.DataFrame(results_A2_5[pollutant]).sum(axis=1)

totals_A0['O3'] = pd.Series(0, index=totals_A0['no2'].index)
totals_A1['O3'] = pd.Series(0, index=totals_A1['no2'].index)
totals_A2['O3'] = pd.Series(0, index=totals_A2['no2'].index)
totals_A3['O3'] = pd.Series(0, index=totals_A3['no2'].index)
totals_A4['O3'] = pd.Series(0, index=totals_A4['no2'].index)
totals_A5['O3'] = pd.Series(0, index=totals_A5['no2'].index)
totals_A2_5['O3'] = pd.Series(0, index=totals_A2_5['no2'].index)

dfs = [totals_A0, totals_A1, totals_A2, totals_A3, totals_A4, totals_A5]
Emis_rues = pd.concat(dfs).sort_index()
Emis_rues.drop(columns='nox', inplace=True)

#%%
print(Emis_rues['no2'].sum())
print(Emis_rues['no'].sum())

facteur = 1000000/31536000
Emis_rues_new = Emis_rues*facteur
Emis_rues_new['no'] = Emis_rues_new['no'] * 0.75
print(Emis_rues_new['no2'].sum())
print(Emis_rues_new['no'].sum())
Emis_rues_new.index.name = 'Id'

# Renaming the columns as specified
Emis_rues_new.rename(columns={
    'no': 'NO',
    'no2': 'NO2',
    'pm10': 'PM',
    'pm25': 'PM25'
}, inplace=True)

# Display the updated dataframe
Emis_rues_new
Emis_rues_new.to_csv(path_or_buf='C:/SiraneDocument/SortieEmis_rues/Emis_rues.dat', sep='\t')
