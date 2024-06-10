# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:26:59 2024

@author: cleme
"""

#%% Import package

import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

Annee= 2022

#%% data_vl NO

df = pd.read_excel('C:/SiraneDocument/WEM_NO_NO2_2019-2040.xlsx', sheet_name=1)

df_filtré = df[df["Category"].isin(['Passenger Cars', 'L-Category'])]
df = pd.DataFrame(df_filtré.sum(numeric_only=True)).transpose()

annees = list(range(2019, 2041))  
categories = ['PK', 'OP', 'R', 'H']  
data_vl = pd.DataFrame(index=categories, columns=annees)

for categorie in categories:
    for annee in annees:
        nom_colonne = f"{annee}_{categorie}"
        if nom_colonne in df.columns:
            data_vl.at[categorie, annee] = df[nom_colonne].iloc[0]  


data_vl.loc['OP_PK'] = 0  

for annee in annees:
    op_value = pd.to_numeric(data_vl.at['OP', annee], errors='coerce')
    pk_value = pd.to_numeric(data_vl.at['PK', annee], errors='coerce')
    
    data_vl.at['OP_PK', annee] = op_value + pk_value

del nom_colonne, annee, categorie, op_value, pk_value, categories, annees

#%% data_vu NO

df = pd.read_excel('C:/SiraneDocument/WEM_NO_NO2_2019-2040.xlsx', sheet_name=1)

df_filtré = df[df["Category"].isin(['Light Commercial Vehicles'])]
df = pd.DataFrame(df_filtré.sum(numeric_only=True)).transpose()

annees = list(range(2019, 2041))  
categories = ['PK', 'OP', 'R', 'H']  
data_vu = pd.DataFrame(index=categories, columns=annees)

for categorie in categories:
    for annee in annees:
        nom_colonne = f"{annee}_{categorie}"
        if nom_colonne in df.columns:
            data_vu.at[categorie, annee] = df[nom_colonne].iloc[0]  



data_vu.loc['OP_PK'] = 0  


for annee in annees:
    op_value = pd.to_numeric(data_vu.at['OP', annee], errors='coerce')
    pk_value = pd.to_numeric(data_vu.at['PK', annee], errors='coerce')
    
    data_vu.at['OP_PK', annee] = op_value + pk_value

del nom_colonne, annee, categorie, op_value, pk_value, categories, annees

#%% data_c2 NO

df = pd.read_excel('C:/SiraneDocument/WEM_NO_NO2_2019-2040.xlsx', sheet_name=1)

df_filtré = df[df["Segment"].isin(['>3,5 t','Rigid <=7,5 t','Rigid 7,5 - 12 t','Rigid 12 - 14 t','Rigid 14 - 20 t','Rigid 20 - 26 t','Rigid 26 - 28 t','Rigid 28 - 32 t','Rigid >32 t'])]
df = pd.DataFrame(df_filtré.sum(numeric_only=True)).transpose()

annees = list(range(2019, 2041))  
categories = ['PK', 'OP', 'R', 'H']  
data_c2 = pd.DataFrame(index=categories, columns=annees)

for categorie in categories:
    for annee in annees:
        nom_colonne = f"{annee}_{categorie}"
        if nom_colonne in df.columns:
            data_c2.at[categorie, annee] = df[nom_colonne].iloc[0]  



data_c2.loc['OP_PK'] = 0  


for annee in annees:
    op_value = pd.to_numeric(data_c2.at['OP', annee], errors='coerce')
    pk_value = pd.to_numeric(data_c2.at['PK', annee], errors='coerce')
    
    data_c2.at['OP_PK', annee] = op_value + pk_value

del nom_colonne, annee, categorie, op_value, pk_value, categories, annees

#%% data_c3 NO

df = pd.read_excel('C:/SiraneDocument/WEM_NO_NO2_2019-2040.xlsx', sheet_name=1)

df_filtré = df[df["Segment"].isin(['Articulated 14 - 20 t','Articulated 20 - 28 t','Articulated 28 - 34 t','Articulated 34 - 40 t','Articulated 40 - 50 t','Articulated 50 - 60 t'])]
df = pd.DataFrame(df_filtré.sum(numeric_only=True)).transpose()

annees = list(range(2019, 2041))  
categories = ['PK', 'OP', 'R', 'H']  
data_c3 = pd.DataFrame(index=categories, columns=annees)

for categorie in categories:
    for annee in annees:
        nom_colonne = f"{annee}_{categorie}"
        if nom_colonne in df.columns:
            data_c3.at[categorie, annee] = df[nom_colonne].iloc[0]  



data_c3.loc['OP_PK'] = 0  

for annee in annees:
    op_value = pd.to_numeric(data_c3.at['OP', annee], errors='coerce')
    pk_value = pd.to_numeric(data_c3.at['PK', annee], errors='coerce')
    
    data_c3.at['OP_PK', annee] = op_value + pk_value

del nom_colonne, annee, categorie, op_value, pk_value, categories, annees

#%% data_ca NO

df = pd.read_excel('C:/SiraneDocument/WEM_NO_NO2_2019-2040.xlsx', sheet_name=1)

df_filtré = df[df["Category"].isin(['Buses'])]
df = pd.DataFrame(df_filtré.sum(numeric_only=True)).transpose()

annees = list(range(2019, 2041))  
categories = ['PK', 'OP', 'R', 'H']  
data_ca = pd.DataFrame(index=categories, columns=annees)

for categorie in categories:
    for annee in annees:
        nom_colonne = f"{annee}_{categorie}"
        if nom_colonne in df.columns:
            data_ca.at[categorie, annee] = df[nom_colonne].iloc[0]  



data_ca.loc['OP_PK'] = 0  

for annee in annees:
    op_value = pd.to_numeric(data_ca.at['OP', annee], errors='coerce')
    pk_value = pd.to_numeric(data_ca.at['PK', annee], errors='coerce')
    
    data_ca.at['OP_PK', annee] = op_value + pk_value

del nom_colonne, annee, categorie, op_value, pk_value, categories, annees
#%% Création du dataframe NO (Reste, Highway, Rural)

data_vl.columns = data_vl.columns.map(lambda x: str(int(x)) if x == int(x) else str(x))
data_vu.columns = data_vu.columns.map(str)
data_c2.columns = data_c2.columns.map(str)
data_c3.columns = data_c3.columns.map(str)
data_ca.columns = data_ca.columns.map(str)

df_global = pd.DataFrame(data_vl[f'{Annee}'])
df_global.columns = ['total vl']
df_global['total vu'] = data_vu[f'{Annee}']
df_global['total c2'] = data_c2[f'{Annee}']
df_global['total c3'] = data_c3[f'{Annee}']
df_global['total ca'] = data_ca[f'{Annee}']
df_global.drop(['PK', 'OP'], axis=0, inplace=True)
df_global.rename(index={'OP_PK': 'Reste', 'R': 'Rural', 'H': 'Highway'}, inplace=True)

df_NO=df_global
#%% data_vl NO2

df = pd.read_excel('C:/SiraneDocument/WEM_NO_NO2_2019-2040.xlsx', sheet_name=2)

df_filtré = df[df["Category"].isin(['Passenger Cars', 'L-Category'])]
df = pd.DataFrame(df_filtré.sum(numeric_only=True)).transpose()

annees = list(range(2019, 2041))  
categories = ['PK', 'OP', 'R', 'H']  
data_NO2_vl = pd.DataFrame(index=categories, columns=annees)

for categorie in categories:
    for annee in annees:
        nom_colonne = f"{annee}_{categorie}"
        if nom_colonne in df.columns:
            data_NO2_vl.at[categorie, annee] = df[nom_colonne].iloc[0]  


data_NO2_vl.loc['OP_PK'] = 0  

for annee in annees:
    op_value = pd.to_numeric(data_NO2_vl.at['OP', annee], errors='coerce')
    pk_value = pd.to_numeric(data_NO2_vl.at['PK', annee], errors='coerce')
    
    data_NO2_vl.at['OP_PK', annee] = op_value + pk_value

del nom_colonne, annee, categorie, op_value, pk_value, categories, annees

#%% data_vu NO2

df = pd.read_excel('C:/SiraneDocument/WEM_NO_NO2_2019-2040.xlsx', sheet_name=2)

df_filtré = df[df["Category"].isin(['Light Commercial Vehicles'])]
df = pd.DataFrame(df_filtré.sum(numeric_only=True)).transpose()

annees = list(range(2019, 2041))  # Années de 2019 à 2040
categories = ['PK', 'OP', 'R', 'H']  # Catégories
data_NO2_vu = pd.DataFrame(index=categories, columns=annees)

for categorie in categories:
    for annee in annees:
        nom_colonne = f"{annee}_{categorie}"
        if nom_colonne in df.columns:
            data_NO2_vu.at[categorie, annee] = df[nom_colonne].iloc[0] 



data_NO2_vu.loc['OP_PK'] = 0  

for annee in annees:
    op_value = pd.to_numeric(data_NO2_vu.at['OP', annee], errors='coerce')
    pk_value = pd.to_numeric(data_NO2_vu.at['PK', annee], errors='coerce')
    
    data_NO2_vu.at['OP_PK', annee] = op_value + pk_value

del nom_colonne, annee, categorie, op_value, pk_value, categories, annees

#%% data_c2 NO2

df = pd.read_excel('C:/SiraneDocument/WEM_NO_NO2_2019-2040.xlsx', sheet_name=2)

df_filtré = df[df["Segment"].isin(['>3,5 t','Rigid <=7,5 t','Rigid 7,5 - 12 t','Rigid 12 - 14 t','Rigid 14 - 20 t','Rigid 20 - 26 t','Rigid 26 - 28 t','Rigid 28 - 32 t','Rigid >32 t'])]
df = pd.DataFrame(df_filtré.sum(numeric_only=True)).transpose()

annees = list(range(2019, 2041))  
categories = ['PK', 'OP', 'R', 'H']  
data_NO2_c2 = pd.DataFrame(index=categories, columns=annees)

for categorie in categories:
    for annee in annees:
        nom_colonne = f"{annee}_{categorie}"
        if nom_colonne in df.columns:
            data_NO2_c2.at[categorie, annee] = df[nom_colonne].iloc[0]  



data_NO2_c2.loc['OP_PK'] = 0  

for annee in annees:
    op_value = pd.to_numeric(data_NO2_c2.at['OP', annee], errors='coerce')
    pk_value = pd.to_numeric(data_NO2_c2.at['PK', annee], errors='coerce')
    
    data_NO2_c2.at['OP_PK', annee] = op_value + pk_value

del nom_colonne, annee, categorie, op_value, pk_value, categories, annees

#%% data_c3 NO2

df = pd.read_excel('C:/SiraneDocument/WEM_NO_NO2_2019-2040.xlsx', sheet_name=2)

df_filtré = df[df["Segment"].isin(['Articulated 14 - 20 t','Articulated 20 - 28 t','Articulated 28 - 34 t','Articulated 34 - 40 t','Articulated 40 - 50 t','Articulated 50 - 60 t'])]
df = pd.DataFrame(df_filtré.sum(numeric_only=True)).transpose()

annees = list(range(2019, 2041))  
categories = ['PK', 'OP', 'R', 'H']  
data_NO2_c3 = pd.DataFrame(index=categories, columns=annees)

for categorie in categories:
    for annee in annees:
        nom_colonne = f"{annee}_{categorie}"
        if nom_colonne in df.columns:
            data_NO2_c3.at[categorie, annee] = df[nom_colonne].iloc[0]  



data_NO2_c3.loc['OP_PK'] = 0  

for annee in annees:
    op_value = pd.to_numeric(data_NO2_c3.at['OP', annee], errors='coerce')
    pk_value = pd.to_numeric(data_NO2_c3.at['PK', annee], errors='coerce')
    
    data_NO2_c3.at['OP_PK', annee] = op_value + pk_value

del nom_colonne, annee, categorie, op_value, pk_value, categories, annees

#%% data_ca NO2

df = pd.read_excel('C:/SiraneDocument/WEM_NO_NO2_2019-2040.xlsx', sheet_name=2)

df_filtré = df[df["Category"].isin(['Buses'])]
df = pd.DataFrame(df_filtré.sum(numeric_only=True)).transpose()

annees = list(range(2019, 2041))  
categories = ['PK', 'OP', 'R', 'H']  
data_NO2_ca = pd.DataFrame(index=categories, columns=annees)

for categorie in categories:
    for annee in annees:
        nom_colonne = f"{annee}_{categorie}"
        if nom_colonne in df.columns:
            data_NO2_ca.at[categorie, annee] = df[nom_colonne].iloc[0]  


data_NO2_ca.loc['OP_PK'] = 0  

for annee in annees:
    op_value = pd.to_numeric(data_NO2_ca.at['OP', annee], errors='coerce')
    pk_value = pd.to_numeric(data_NO2_ca.at['PK', annee], errors='coerce')
    
    data_NO2_ca.at['OP_PK', annee] = op_value + pk_value

del nom_colonne, annee, categorie, op_value, pk_value, categories, annees

#%% Création du dataframe NO2 (Reste, Highway, Rural)

data_NO2_vl.columns = data_NO2_vl.columns.map(lambda x: str(int(x)) if x == int(x) else str(x))
data_NO2_vu.columns = data_NO2_vu.columns.map(str)
data_NO2_c2.columns = data_NO2_c2.columns.map(str)
data_NO2_c3.columns = data_NO2_c3.columns.map(str)
data_NO2_ca.columns = data_NO2_ca.columns.map(str)

df_global = pd.DataFrame(data_NO2_vl[f'{Annee}'])
df_global.columns = ['total vl']
df_global['total vu'] = data_NO2_vu[f'{Annee}']
df_global['total c2'] = data_NO2_c2[f'{Annee}']
df_global['total c3'] = data_NO2_c3[f'{Annee}']
df_global['total ca'] = data_NO2_ca[f'{Annee}']
df_global.drop(['PK', 'OP'], axis=0, inplace=True)
df_global.rename(index={'OP_PK': 'Reste', 'R': 'Rural', 'H': 'Highway'}, inplace=True)

df_NO2=df_global

#%% Création du dataframe NOx

df_NOx= df_NO2 + df_NO
sommes_des_lignes = df_NOx.sum(axis=1)

print("Les sommes des valeurs de chaque ligne sont :\n", sommes_des_lignes)
#%% Process data

sir_segm=pd.read_excel('C:/SiraneDocument/Sirane2_segmenté.xlsx')
sir_segm['Hierarchie'] = sir_segm['Hierarchie'].apply(lambda x: 'A5' if pd.isna(x) or x == ' ' else x)
sir_segm['Hierarchie'] = sir_segm['Hierarchie'].apply(lambda x: 'A1' if pd.isna(x) or x == '0' else x)

no2=df_NO2
no=df_NO
nox=df_NOx
pm10=pd.read_excel('C:/SiraneDocument/Copert2022/pm10.xlsx', index_col=0)
pm25=pd.read_excel('C:/SiraneDocument/Copert2022/pm25.xlsx', index_col=0)


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

repartA2_A5 = pd.DataFrame(index=types_de_vehicules, columns=['A2', 'A3', 'A4', 'A5'])
vehicle_type_sums = sir_A2_5[types_de_vehicules].sum()
for vehicle_type in types_de_vehicules:
    for route in ['A2', 'A3', 'A4', 'A5']:
        repartA2_A5.at[vehicle_type, route] = sir_A2_5.loc[sir_A2_5['Hierarchie'] == route, vehicle_type].sum() / vehicle_type_sums[vehicle_type]


A2_3_4_5 = {route: {'no': None, 'no2': None, 'nox': None, 'pm10': None, 'pm25': None} for route in ['A2', 'A3', 'A4', 'A5']}
for vehicle_type in types_de_vehicules:
    for route in ['A2', 'A3', 'A4', 'A5']:
        proportion = repartA2_A5.loc[vehicle_type, route]
        for pollutant in polluants:
            df = globals()[pollutant]  

            A2_3_4_5[route][pollutant] = df.loc[['Reste']] * proportion

                
                
A2 = A2_3_4_5["A2"]
A3 = A2_3_4_5["A3"]
A4 = A2_3_4_5["A4"]
A5 = A2_3_4_5["A5"]



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

print(Emis_rues['no2'].sum())
print(Emis_rues['no'].sum())


facteur = 1000000/31536000
Emis_rues_new = Emis_rues*facteur
Emis_rues_new['no'] = Emis_rues_new['no'] * 0.75
print(Emis_rues_new['no2'].sum())
print(Emis_rues_new['no'].sum())
Emis_rues_new.index.name = 'Id'

Emis_rues_new.rename(columns={
    'no': 'NO',
    'no2': 'NO2',
    'pm10': 'PM',
    'pm25': 'PM25'
}, inplace=True)

Emis_rues_new
Emis_rues_new.to_csv(path_or_buf='C:/SiraneDocument/SortieEmis_rues/Emis_rues.dat', sep='\t')

