#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:21:08 2024

@author: meganepourtois
"""

#%% Import packages

import os
import pandas as pd
import matplotlib.pyplot as plt

#%%
def read_excel_files(directory):
    results = {}
    for filename in os.listdir(directory):
        if filename.endswith('.xlsx'):  
            file_path = os.path.join(directory, filename)
            try:
                df = pd.read_excel(file_path, engine='openpyxl')  
                df_name = os.path.splitext(filename)[0]
                results[df_name] = df
                print(f"Loaded {filename} into DataFrame.")
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
    return results

directory = '/Users/meganepourtois/Desktop/MapResults/Conc'
dataframes = read_excel_files(directory)
results = dataframes

sirane2segm = pd.read_excel('/Users/meganepourtois/Desktop/Affinage des routes/Sirane_segmenteupdated.xlsx')[['FID', 'length_2']]
tunnel = pd.read_excel('/Users/meganepourtois/Desktop/MapResults/RuesTunnel_Table.xlsx')

hierarchies=['A0', 'A1', 'A2', 'A3', 'A4', 'A5']

for key, df in results.items():
    df['IndexColumn'] = df.index
    mask = ~df['IndexColumn'].isin(tunnel['ID2'])
    df = df[mask]
    results[key] = df
    df.drop(columns=['IndexColumn'], inplace=True)
    df = df.drop(df[~df['Hierarchie'].isin(hierarchies)].index)

mask = ~sirane2segm['FID'].isin(tunnel['ID2'])
sirane2segm = sirane2segm[mask]
#%%
"""#%%
import pandas as pd
import os
import matplotlib.pyplot as plt
#%% Process
results = {}
for filename in os.listdir('/Users/meganepourtois/Desktop/MapResults'):
    #if filename.endswith('.xlsx'):  # Check for Excel files
    file_path = os.path.join('/Users/meganepourtois/Desktop/MapResults', filename)
    df = pd.read_excel(file_path,  engine='openpyxl')
    df_name = os.path.splitext(filename)[0]
    results[df_name] = df"""

mean_hierarchy = pd.DataFrame(index=results.keys(), columns=hierarchies)
for name, df in results.items():
    df['length']=sirane2segm['length_2']
for name, df in results.items():
    for hierarchy in hierarchies:
        filtered_df = df[df['Hierarchie'] == hierarchy]

        if not filtered_df.empty:
            weighted_sum = (filtered_df['length'] * filtered_df['mean']).sum()
            total_length = filtered_df['length'].sum()
            
            if total_length > 0:
                mean = weighted_sum / total_length
                mean_hierarchy.loc[name, hierarchy] = mean
            else:
                mean_hierarchy.loc[name, hierarchy] = pd.NA
        else:
            mean_hierarchy.loc[name, hierarchy] = pd.NA

mean_hierarchy = mean_hierarchy.sort_values(by=mean_hierarchy.columns[0], ascending=False)
mean_hierarchy.to_excel('/Users/meganepourtois/Desktop/MapResults/Data/mean_hierarchy_pond_tunnel.xlsx')
#%%

depassement_an = pd.DataFrame(index=results.keys(), columns=hierarchies)
for hierarchy in hierarchies:
    depassement_an[['Proportion' + hierarchy]]=0
for hierarchy in hierarchies:
    depassement_an['Percent' + hierarchy]=0
for name, df in results.items():
    for hierarchy in hierarchies:
        filtered_df = df[(df['Hierarchie'] == hierarchy) & (df['mean'] >= 40)]
        depassement_an.loc[name, hierarchy] = filtered_df['length'].sum()/1000
    for hierarchy in hierarchies:
        depassement_an.loc[name, 'Proportion' + hierarchy] = depassement_an.loc[name, hierarchy] / depassement_an.loc[name].sum() 
    
    for hierarchy in hierarchies:
        filtered_df = df[(df['Hierarchie'] == hierarchy)]
        total_length_for_hierarchy = filtered_df['length'].sum()
        depassement_an.loc[name,'Percent' + hierarchy] = depassement_an.loc[name, hierarchy] / (total_length_for_hierarchy/1000) 

depassement_an['Total'] = depassement_an[hierarchies].sum(axis=1)
depassement_an['Proportion'] = pd.NA  


input_dep=pd.read_excel('/Users/meganepourtois/Desktop/MapResults/Data/depassement_an_input.xlsx', sheet_name='Feuil1')
df = results['Conc_NO2_Moy_2025BASE_output']
mask = df['Hierarchie'].isin(hierarchies)
total_length = df[mask]['length'].sum()

input_dep.index = depassement_an.index

depassement_an.update(input_dep)

depassement_an['Proportion'] = depassement_an['Total'] / total_length

#%%
import seaborn as sns
import numpy as np
data = {
    'BASE': results['Conc_NO2_Moy_2022BASE_output']['mean'],
    'WEM': results['Conc_NO2_Moy_2022WEM_output']['mean'],
    'Real Data': results['Conc_NO2_Moy2022AXELAFFINAGE_output']['mean'],
    'WAM': results['Conc_NO2_Moy_2022WAM_output']['mean']
}
df = pd.DataFrame(data)

def custom_whiskers(x):
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

plt.figure(figsize=(13, 8))
boxplot = sns.boxplot(data=df, palette="Set2", showmeans=True, meanprops={"marker":"o", "markerfacecolor":"black", "markeredgecolor":"black"},
                      whis=[5, 95], showfliers=False)

plt.ylabel('NO₂ Concentration [µg/m³]', fontsize=19)
plt.xlabel('Scenario', fontsize=19)
plt.ylim(14, 35)  
plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(1))
plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(1))
plt.xticks( fontsize=16)
plt.yticks(fontsize=16)
file_name = 'Boxplot2022.svg'
folder_path = '/Users/meganepourtois/Desktop/MapResults/FiguresFinales'
file_path = os.path.join(folder_path, file_name)
plt.savefig(file_path, format='svg')
plt.show()
#%% Mean C

average_concentration = pd.DataFrame(index=results.keys(), columns=['average_concentration'])

for name, df in results.items():
    weighted_sum = (df['length'] * df['mean']).sum()
    print(name)
    print((df['mean'].iloc[15000:]*df['length'].iloc[15000:]).sum()/100000)
    print((df['mean'].iloc[15000:]).sum()/100000)

    total_length = df['length'].sum()
    average_concentration.loc[name, 'average_concentration'] = weighted_sum / total_length

average_concentration.to_excel('/Users/meganepourtois/Desktop/MapResults/Data/AverageC_pond_tunnel.xlsx')
#%%Bar
mean_hierarchy.plot(kind='bar', figsize=(12, 8))
plt.title('Bar Plot of DataFrame')
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()

mean_hierarchy.T.plot(kind='bar', figsize=(12, 8))
plt.title('Bar Plot of DataFrame')
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()

#%%Line
plt.figure(figsize=(10, 6))
for column in mean_hierarchy.columns:
    plt.plot(mean_hierarchy[column], label=column)
plt.xticks(rotation=90)
plt.title('Line Plot of DataFrame')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.show()
#%%Scatter
plt.figure(figsize=(8, 5))
plt.scatter(mean_hierarchy['A0'], mean_hierarchy['A5'])
plt.title('Scatter Plot between Column 1 and Column 2')
plt.xlabel('A0')
plt.ylabel('A5')
plt.show()

#%% Box

mean_hierarchy.plot(kind='box', figsize=(12, 8))
plt.title('Box Plot of DataFrame')
plt.ylabel('Value')
plt.show()

#%% Area
mean_hierarchy.plot(kind='area', figsize=(12, 8), stacked=True)
plt.title('Area Plot of DataFrame')
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()











