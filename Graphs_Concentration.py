#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 11:00:26 2024

@author: meganepourtois
"""
#%% Import packages

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import os
import re  
from matplotlib.ticker import PercentFormatter

#%% Proocess
averagec = pd.read_excel('/Users/meganepourtois/Desktop/MapResults/Data/AverageC_pond_tunnel.xlsx')
"""
averagec = averagec.drop([0, 1])
averagec.columns = ['Year', 'Average Concentration']
scenarios = ['WAM', 'WEM', 'BASE']
averagec.index=averagec['Year']
averagec = averagec.drop('Year', axis=1)
averagec['BASE']=averagec['Average Concentration'].iloc[-7:]
averagec['WEM']=averagec['Average Concentration'].iloc[-15:-7]
averagec['WAM']=averagec['Average Concentration'].iloc[:8]
averagec = averagec.drop('Average Concentration', axis=1)
averagec = averagec.drop(averagec.index[0])
averagec = averagec.iloc[:7]"""

depassement_an = pd.read_excel('/Users/meganepourtois/Desktop/MapResults/Data/depassement_an_pond_tunnel.xlsx')#, sheet_name='Feuil1')
depassement_an=depassement_an[:23]
mean_hierarchy = pd.read_excel('/Users/meganepourtois/Desktop/MapResults/Data/mean_hierarchy_pond_tunnel.xlsx')


def filter_by_scenario(df):
    scenarios = {'BASE': pd.DataFrame(), 'WEM': pd.DataFrame(), 'WAM': pd.DataFrame()}
    
    for scenario in scenarios:
        scenarios[scenario] = df[df.iloc[:, 0].str.contains(scenario, na=False)]
        
    return scenarios



def filter_by_scenario_and_set_index(df):
    scenarios = {'BASE': pd.DataFrame(), 'WEM': pd.DataFrame(), 'WAM': pd.DataFrame()}
    for scenario in scenarios:
        filtered_df = df[df.iloc[:, 0].str.contains(scenario, na=False)]
        
        filtered_df['Year'] = filtered_df.iloc[:, 0].apply(lambda x: int(re.findall(r'\d{4}', x)[0]) if re.findall(r'\d{4}', x) else None)
        filtered_df.set_index('Year', inplace=True)
        filtered_df.sort_index(inplace=True)

        scenarios[scenario] = filtered_df.drop(columns=df.columns[0])  
        
    return scenarios
depassement_an.astype(str)
depassement_an_scenarios = filter_by_scenario_and_set_index(depassement_an)
mean_hierarchy_scenarios = filter_by_scenario_and_set_index(mean_hierarchy)
averagec =filter_by_scenario_and_set_index(averagec)

mean_hierarchy_scenarios['BASE'].to_excel('/Users/meganepourtois/Desktop/MapResults/Data/mean_hierarchy_scenariosBASE.xlsx')#%%Process emissions
mean_hierarchy_scenarios['WEM'].to_excel('/Users/meganepourtois/Desktop/MapResults/Data/mean_hierarchy_scenariosWEM.xlsx')#%%Process emissions
mean_hierarchy_scenarios['WAM'].to_excel('/Users/meganepourtois/Desktop/MapResults/Data/mean_hierarchy_scenariosWAM.xlsx')#%%Process emissions


BASE= pd.read_excel('/Users/meganepourtois/Desktop/Projection_scénario/BASE_NO_NO2_2019-2040.xlsx', sheet_name=None)
WAM=  pd.read_excel('/Users/meganepourtois/Desktop/Projection_scénario/WAM_NO_NO2_2019-2040.xlsx', sheet_name=None)
WEM = pd.read_excel('/Users/meganepourtois/Desktop/Projection_scénario/WEM_NO_NO2_2019-2040.xlsx', sheet_name=None)
TAB = pd.read_excel('/Users/meganepourtois/Desktop/Projection_scénario/TAB20230614_ProjectionsPACE2023.xlsx', sheet_name=None)

def sum_columns_by_year(data_dict):
    years = range(2019, 2041)
    suffixes = ['_OP', '_PK', '_R', '_H']
    
    for sheet_name, df in data_dict.items():
        for year in years:
            columns_to_sum = [f'{year}{suffix}' for suffix in suffixes if f'{year}{suffix}' in df.columns]
            if columns_to_sum:  
                df[str(year)] = df[columns_to_sum].sum(axis=1)
                
        data_dict[sheet_name] = df

sum_columns_by_year(BASE)
sum_columns_by_year(WAM)
sum_columns_by_year(WEM)


def extract_summed_data(data_dict, sheet_name, start_year=2019, end_year=2040):
    columns_to_keep = data_dict.get(sheet_name).columns[:6].tolist() + [str(year) for year in range(start_year, end_year + 1)]
    df_filtered = data_dict[sheet_name][columns_to_keep]
    
    return df_filtered

BASE_NO2_tot = extract_summed_data(BASE, 'NO2_TOTAL')
BASE_NO2_tot.set_index(BASE_NO2_tot.columns[:6].tolist(), inplace=True)
WAM_NO2_tot = extract_summed_data(WAM, 'NO2_TOTAL')
WAM_NO2_tot.set_index(WAM_NO2_tot.columns[:6].tolist(), inplace=True)
WEM_NO2_tot = extract_summed_data(WEM, 'NO2_TOTAL_WEM')
WEM_NO2_tot.set_index(WEM_NO2_tot.columns[:6].tolist(), inplace=True)
BASE_NO_tot = extract_summed_data(BASE, 'NO_TOTAL')
BASE_NO_tot.set_index(BASE_NO_tot.columns[:6].tolist(), inplace=True)
WAM_NO_tot = extract_summed_data(WAM, 'NO_TOTAL')
WAM_NO_tot.set_index(WAM_NO_tot.columns[:6].tolist(), inplace=True)
WEM_NO_tot = extract_summed_data(WEM, 'NO_TOTAL_WEM')
WEM_NO_tot.set_index(WEM_NO_tot.columns[:6].tolist(), inplace=True)
#%%EVolution emissions
plt.figure(figsize=(12, 6))
plt.plot(BASE_NO2_tot.columns, BASE_NO2_tot.sum().values+BASE_NO_tot.sum().values, marker='o', linestyle='-', color='#007BA7', label = 'BASE')
plt.plot(WEM_NO2_tot.columns, WEM_NO2_tot.sum().values+WEM_NO_tot.sum().values, marker='o', linestyle='-', color='#DC143C', label = 'WEM')
plt.plot(WAM_NO2_tot.columns, WAM_NO2_tot.sum().values+WAM_NO_tot.sum().values, marker='o', linestyle='-', color='#228B22', label = 'WAM')
#plt.title('NO Total Emissions From Road Transport Over Time')
plt.xlabel('Year', fontsize=16)
plt.ylabel('Emissions of NO [t]', fontsize=16)
plt.grid(True)
plt.xticks(rotation=45, fontsize=13)
plt.yticks(fontsize=13)
plt.legend(loc='best', fontsize=16)
file_name = 'Emissions_NO2.svg'
folder_path = '/Users/meganepourtois/Desktop/Projection_scénario/FiguresMai'
file_path = os.path.join(folder_path, file_name)
plt.show()
#%% Evolution NO2 Concentration
plt.figure(figsize=(12, 6))
plt.plot(averagec['BASE'].index, averagec['BASE'], marker='o', linestyle='-', color='#007BA7', label='BASE')
plt.plot(averagec['BASE'].index, averagec['WEM'], marker='o', linestyle='-', color='#DC143C', label='WEM')
plt.plot(averagec['BASE'].index, averagec['WAM'], marker='o', linestyle='-', color='#228B22', label='WAM')
#plt.title(r'Projected Average Concentration of $NO_2$ Over Time in Brussels', fontsize=23)
plt.xlabel('Year', fontsize=19)
plt.ylabel(r'$NO_2$ concentration [$\mu g/m^3$]', fontsize=19)
plt.grid(True)
plt.xticks(rotation=45, fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='best', fontsize=19)
file_name = 'EvolutionCNO2.svg'
folder_path = '/Users/meganepourtois/Desktop/MapResults/FiguresFinales'
file_path = os.path.join(folder_path, file_name)
plt.show()
#%% Reduction in NO2 concentration

min_year = 2022
max_year = 2035
year_range = list(range(min_year, max_year + 1))
for key, df in averagec.items():
    df = df.reindex(year_range)
    df.interpolate(method='linear', inplace=True)
    averagec[key] = df


diff_base_wem = averagec['BASE'] - averagec['WEM']
diff_wem_wam = averagec['WEM'] - averagec['WAM']

years = np.arange(len(averagec['BASE'].index))  



plt.figure(figsize=(12, 6))
bar_width = 0.35
plt.bar(years , diff_base_wem['average_concentration'].values, bar_width, label='Difference BASE-WEM', color='#6495ED', alpha=0.7)
plt.bar(years, diff_wem_wam['average_concentration'].values, bar_width, bottom=diff_base_wem['average_concentration'].values, label='Difference WEM-WAM', color='#FFD700', alpha=0.7)
plt.xlabel('Year', fontsize=19)
plt.ylabel(r'$NO_2$ concentration reduction [$\mu g/m^3$]', fontsize=19)
plt.grid(True)
ymax = plt.ylim()[1]
plt.ylim(0, ymax * 1.1)
plt.xticks(years, averagec['BASE'].index, rotation=45, fontsize=16)  
plt.yticks(fontsize=16)  
plt.legend(loc='best', fontsize=16)
file_name = 'ReductionCNO2.svg'
folder_path = '/Users/meganepourtois/Desktop/MapResults/FiguresFinales'
file_path = os.path.join(folder_path, file_name)
plt.show()

#%%Emissions vs concentrations

BASE_NO2_tot = BASE_NO2_tot.loc[:, '2022':'2035']
WEM_NO2_tot =WEM_NO2_tot.loc[:, '2022':'2035']
WAM_NO2_tot = WAM_NO2_tot.loc[:, '2022':'2035'] 
BASE_NO_tot = BASE_NO_tot.loc[:, '2022':'2035']
WEM_NO_tot =WEM_NO_tot.loc[:, '2022':'2035']
WAM_NO_tot = WAM_NO_tot.loc[:, '2022':'2035'] 

fig, ax1 = plt.subplots(figsize=(12, 6))

emissions_color = '#800080' 
ax1.plot(BASE_NO2_tot.columns, BASE_NO2_tot.sum().values + BASE_NO_tot.sum().values, 
         marker='o', linestyle='--', color='#00008b', label='BASE (Emissions)')
ax1.plot(WEM_NO2_tot.columns, WEM_NO2_tot.sum().values + WEM_NO_tot.sum().values, 
         marker='s', linestyle='--', color='#FF6347', label='WEM (Emissions)')  
ax1.plot(WAM_NO2_tot.columns, WAM_NO2_tot.sum().values + WAM_NO_tot.sum().values, 
         marker='', linestyle='--', color='#013220', label='WAM (Emissions)')  
ax1.set_xlabel('Year', fontsize=16)
ax1.set_ylabel('Emissions of NO [t]', fontsize=19)
ax1.tick_params(axis='y', labelsize=16)
ax1.tick_params(axis='x', labelsize=15)
ax1.grid(True)

ax2 = ax1.twinx()
concentration_color = '#008000'  
ax2.plot(BASE_NO2_tot.columns, averagec['BASE'], marker='o', linestyle='-', color='#007BA7', label='BASE (Concentration)')
ax2.plot(BASE_NO2_tot.columns, averagec['WEM'], marker='s', linestyle='-', color='#DC143C', label='WEM (Concentration)')
ax2.plot(BASE_NO2_tot.columns, averagec['WAM'], marker='^', linestyle='-', color='#228B22', label='WAM (Concentration)')
ax2.set_ylabel(r'$NO_2$ concentration [$\mu g/m^3$]', fontsize=19)
ax2.tick_params(axis='y', labelsize=16)

# Legend
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best', fontsize=16)

file_name = 'Emissionsvsconcentrations.svg'
folder_path = '/Users/meganepourtois/Desktop/MapResults/FiguresFinales'
file_path = os.path.join(folder_path, file_name)
plt.show()
#%%Emissions vs concentrations autre essai



emissions_color = '#800080'  
emission_linewidth = 2  

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(BASE_NO2_tot.columns, BASE_NO2_tot.sum().values + BASE_NO_tot.sum().values,
         marker='o', linestyle='--', color='#00008b', label='BASE (Emissions)', linewidth=emission_linewidth)
ax1.plot(WEM_NO2_tot.columns, WEM_NO2_tot.sum().values + WEM_NO_tot.sum().values,
         marker='s', linestyle='--', color='#FF6347', label='WEM (Emissions)', linewidth=emission_linewidth)  
ax1.plot(WAM_NO2_tot.columns, WAM_NO2_tot.sum().values + WAM_NO_tot.sum().values,
         marker='^', linestyle='--', color='#013220', label='WAM (Emissions)', linewidth=emission_linewidth)  
ax1.set_xlabel('Year', fontsize=16)
ax1.set_ylabel('Emissions of NO [t]', fontsize=16)
ax1.tick_params(axis='both', which='major', labelsize=13)
ax1.grid(True)

ax2 = ax1.twinx()
concentration_color = '#008000'  
concentration_linewidth = 1  

ax2.plot(BASE_NO2_tot.columns, averagec['BASE'], marker='o', linestyle='-', color='#007BA7', label='BASE (Concentration)', linewidth=concentration_linewidth)
ax2.plot(BASE_NO2_tot.columns, averagec['WEM'], marker='s', linestyle='-', color='#DC143C', label='WEM (Concentration)', linewidth=concentration_linewidth)
ax2.plot(BASE_NO2_tot.columns, averagec['WAM'], marker='^', linestyle='-', color='#228B22', label='WAM (Concentration)', linewidth=concentration_linewidth)
ax2.set_ylim(12, 19.5)  
ax2.set_ylabel(r'$NO_2$ concentration [$\mu g/m^3$]', fontsize=19)
ax2.tick_params(axis='y', labelsize=16)

# Legend
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best', fontsize=16)
file_name = 'Emissionsvsconcentrationsautreessai.svg'
folder_path = '/Users/meganepourtois/Desktop/MapResults/FiguresFinales'
file_path = os.path.join(folder_path, file_name)
plt.show()



#%% % Reduction in NO2 concentration

percent_base_wem = (averagec['BASE'] - averagec['WEM']) * 100 / averagec['BASE']
percent_wem_wam = (averagec['WEM'] - averagec['WAM']) * 100 / averagec['WEM']

plt.figure(figsize=(12, 6))

bar_width = 0.35

plt.bar(years, percent_base_wem['average_concentration'].values, bar_width, label='Percent difference BASE-WEM', color='#6495ED', alpha=0.7)

plt.bar(years , percent_wem_wam['average_concentration'].values, bar_width, bottom=percent_base_wem['average_concentration'].values, label='Percent difference WEM-WAM', color='#FFD700', alpha=0.7)

plt.xlabel('Year', fontsize=19)
plt.ylabel(r'Reduction in $NO_2$ concentration [%]', fontsize=19)
plt.grid(True)
ymax = plt.ylim()[1]
plt.ylim(0, ymax * 1.1)
plt.xticks(years, averagec.index, rotation=45, fontsize=16)  
plt.yticks(fontsize=16)    
plt.legend(loc='best', fontsize=16)
file_name = 'PercentageReduction.svg'
folder_path = '/Users/meganepourtois/Desktop/MapResults/FiguresFinales'
file_path = os.path.join(folder_path, file_name)
plt.show()
#%% Evolution totale
# Recalculate percentages after cleaning data
data = {
    'Year': [2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035],
    'BASE': [19.10, 18.45, 17.83, 17.39, 16.85, 16.50, 16.12, 15.66, 15.40, 15.03, 14.78, 14.60, 14.35, 14.21],
    'WEM': [18.53, 17.74, 17.11, 15.82, 15.41, 15.22, 14.54, 14.31, 13.88, 13.77, 13.69, 13.46, 13.39, 12.93],
    'WAM': [18.44, 17.61, 16.95, 15.64, 15.21, 15.02, 14.32, 14.11, 13.70, 13.62, 13.55, 13.33, 13.29, 12.91]
}

df = pd.DataFrame(data)
df.set_index('Year', inplace=True)

averagec = {
    'BASE': df[['BASE']],
    'WEM': df[['WEM']],
    'WAM': df[['WAM']]
}
df_averagec = pd.concat(averagec.values(), axis=1)

percent_base_wem = (df_averagec['BASE'] - df_averagec['WEM'])* 100 / (df_averagec['BASE']-10.7969)
print(percent_base_wem)
percent_wem_wam = ((df_averagec['BASE'] - df_averagec['WAM']) * 100 / (df_averagec['BASE']-10.7969))-((df_averagec['BASE'] - df_averagec['WEM']) * 100 / (df_averagec['BASE']-10.7969))


fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(years, averagec['BASE'], marker='o', linestyle='-', color='#007BA7', label='BASE', linewidth=2, zorder=5)
ax1.plot(years, averagec['WEM'], marker='o', linestyle='-', color='#DC143C', label='WEM', linewidth=2, zorder=5)
ax1.plot(years, averagec['WAM'], marker='o', linestyle='-', color='#228B22', label='WAM', linewidth=2, zorder=5)
ax1.set_ylabel(r'$NO_2$ concentration [$\mu g/m^3$]', fontsize=19)

ax2 = ax1.twinx()
ax2.bar(years, percent_base_wem.values, bar_width, label='Percentage Reduction from BASE to WEM', color='#8B0000', alpha=0.4)
ax2.bar(years, percent_wem_wam.values, bar_width, bottom=percent_base_wem.values, label='Additional Percentage Reduction from BASE to WAM', color='#228B22', alpha=0.4)
ax2.set_ylabel('Reduction in $NO_2$ concentration [%]', fontsize=19)
ax2.set_ylim(0, 43)  
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x)}%'))  

ax2.set_xlabel('Year', fontsize=19)
ax2.set_xticks(years)
ax2.set_xticklabels(averagec['BASE'].index, rotation=45, fontsize=16)

ax1.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=17)#, 
ax2.legend(loc='lower right', bbox_to_anchor=(1, 1), fontsize=17)
ax1.tick_params(axis='y', labelsize=17)
ax2.tick_params(axis='y', labelsize=17)
ax1.tick_params(axis='x', labelsize=17)

plt.grid(False)
plt.tight_layout()
file_name = 'EvolutionCNO2Finalonycroit.svg'
folder_path = '/Users/meganepourtois/Desktop/MapResults/FiguresFinales'
file_path = os.path.join(folder_path, file_name)
plt.savefig(file_path, format='svg')
plt.show()

#%%Number of street segment depassement in {scenario} scenario
hierarchies = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']
Proportion = ['ProportionA0', 'ProportionA1', 'ProportionA2', 'ProportionA3', 'ProportionA4', 'ProportionA5']
Percent = ['PercentA0', 'PercentA1', 'PercentA2', 'PercentA3', 'PercentA4', 'PercentA5']

# Setup parameters
scenarios = ['BASE', 'WEM', 'WAM']  # The names of your scenarios
n_groups = len(depassement_an_scenarios[scenarios[0]])
n_bars = len(hierarchies)
bar_width = 30 
opacity = 0.7  

edge_colors = {'BASE': 'black', 'WEM': 'black', 'WAM': 'black'}
edge_styles = {'BASE': '-', 'WEM': '--', 'WAM': ':'}
cmap = plt.cm.YlGnBu  
norm = plt.Normalize(vmin=0, vmax=n_bars-1)  
spacing = bar_width * len(scenarios) * 0.45
index = np.arange(n_groups) * (len(scenarios) + 1) * spacing  
offset = np.linspace(-bar_width * len(scenarios) / 2, bar_width * len(scenarios) / 2, num=len(scenarios))

fig, ax = plt.subplots(figsize=(14, 8))
for i, scenario in enumerate(scenarios):
    df = depassement_an_scenarios[scenario]
    bottom = np.zeros(n_groups)
    for j, col in enumerate(hierarchies):
        bar_color = cmap(norm(j))  
        bars = ax.bar(index + offset[i], df[col], bar_width, bottom=bottom, alpha=opacity,
                label=f'{col}' if i == 0 else "", color=bar_color,
                edgecolor=edge_colors[scenario], linewidth=2,  
                linestyle=edge_styles[scenario])
        bottom += df[col].values
    if scenario == "WEM" : 
        label_y_position = max(bottom) + 50
    else : # Buffer above the highest bar
        label_y_position = max(bottom) +15
    ax.text(index[0] + offset[i], label_y_position, scenario, ha='center', va='bottom', fontsize=16,
            color=edge_colors[scenario])

ax.set_xlabel('Year', fontsize=19)
ax.set_ylabel('Distance exceeding 20 $\mu g/m^3$ [km]', fontsize=18)
ax.set_xticks(index + offset[len(scenarios) // 2])
ax.set_xticklabels(depassement_an_scenarios[scenarios[0]].index, fontsize=17)
ax.legend(title='Road hierarchies', loc='upper right', fontsize=16, title_fontsize=19)
ax.tick_params(axis='y', labelsize=17)
file_name = 'DistancesOver195.svg'
folder_path = '/Users/meganepourtois/Desktop/MapResults/FiguresFinales'
file_path = os.path.join(folder_path, file_name)
plt.savefig(file_path, format='svg')
plt.show()
#%%Proportion of street segment depassement in {scenario} scenario
# Setup parameters
scenarios = ['BASE', 'WEM', 'WAM']  # The names of your scenarios
n_groups = len(depassement_an_scenarios[scenarios[0]])
n_bars = len(Proportion)
bar_width = 30 
opacity = 0.7  

edge_colors = {'BASE': 'black', 'WEM': 'black', 'WAM': 'black'}
edge_styles = {'BASE': '-', 'WEM': '--', 'WAM': ':'}
cmap = plt.cm.YlGnBu  
norm = plt.Normalize(vmin=0, vmax=n_bars-1)  
spacing = bar_width * len(scenarios) * 0.42
index = np.arange(n_groups) * (len(scenarios) + 1) * spacing  
offset = np.linspace(-bar_width * len(scenarios) / 2, bar_width * len(scenarios) / 2, num=len(scenarios))

fig, ax = plt.subplots(figsize=(14, 8))
for i, scenario in enumerate(scenarios):
    df = depassement_an_scenarios[scenario]
    bottom = np.zeros(n_groups)
    for j, col in enumerate(Proportion):
        bar_color = cmap(norm(j))  
        bars = ax.bar(index + offset[i], df[col], bar_width, bottom=bottom, alpha=opacity,
                  label=f'{col}' if i == 0 else "", color=bar_color,
                  edgecolor=edge_colors[scenario], linewidth=2,  
                  linestyle=edge_styles[scenario])
        bottom += df[col].values

    if scenario == "WEM" : 
        label_y_position = bottom[-1] +0.25 
    elif scenario == "WAM" : 
        label_y_position = bottom[-1] +0.32  
    else : 
        label_y_position = label_y_position = bottom[-1] +0.03   

    ax.text(index[0] + offset[i], label_y_position, scenario, ha='center', va='bottom', fontsize=16,
            color=edge_colors[scenario], rotation = 45)


ax.set_xlabel('Year', fontsize=19)
ax.set_ylabel('Number of street segments', fontsize=19)
ax.set_xticks(index + offset[len(scenarios) // 2])
ax.set_xticklabels(depassement_an_scenarios[scenarios[0]].index, fontsize=17)
ax.legend(title='Proportion', loc='upper right', fontsize=16, title_fontsize=19)
ax.set_ylim([0, max(bottom) * 1.2])

ax.tick_params(axis='y', labelsize=17)
file_name = 'StreetSegmentsOver195_try.svg'
folder_path = '/Users/meganepourtois/Desktop/MapResults/FiguresFinales'
file_path = os.path.join(folder_path, file_name)
plt.savefig(file_path, format='svg')
plt.show()

#%% All in data
depassement_an_scenarios = filter_by_scenario_and_set_index(depassement_an)
mean_hierarchy_scenarios = filter_by_scenario_and_set_index(mean_hierarchy)
basic = pd.DataFrame(mean_hierarchy_scenarios['BASE'])
wemic = pd.DataFrame(mean_hierarchy_scenarios['WEM'])
wamic = pd.DataFrame(mean_hierarchy_scenarios['WAM'])
basic.to_excel('/Users/meganepourtois/Desktop/MapResults/Data/mean_hierarchy_scenarios_BASE.xlsx')
wemic.to_excel('/Users/meganepourtois/Desktop/MapResults/Data/mean_hierarchy_scenarios_WEM.xlsx')
wamic.to_excel('/Users/meganepourtois/Desktop/MapResults/Data/mean_hierarchy_scenarios_WAM.xlsx')

basic = pd.DataFrame(depassement_an_scenarios['BASE'])
wemic = pd.DataFrame(depassement_an_scenarios['WEM'])
wamic = pd.DataFrame(depassement_an_scenarios['WAM'])
basic.to_excel('/Users/meganepourtois/Desktop/MapResults/Data/depassement_an_scenarios_BASE.xlsx')
wemic.to_excel('/Users/meganepourtois/Desktop/MapResults/Data/depassement_an_scenarios_WEM.xlsx')
wamic.to_excel('/Users/meganepourtois/Desktop/MapResults/Data/depassement_an_scenarios_WAM.xlsx')

basic = pd.DataFrame(averagec['BASE'])
wemic = pd.DataFrame(averagec['WEM'])
wamic = pd.DataFrame(averagec['WAM'])
basic.to_excel('/Users/meganepourtois/Desktop/MapResults/Data/averagec_BASE.xlsx')
wemic.to_excel('/Users/meganepourtois/Desktop/MapResults/Data/averagec_WEM.xlsx')
wamic.to_excel('/Users/meganepourtois/Desktop/MapResults/Data/averagec_WAM.xlsx')
#%% Real proportion
for scenario, df in depassement_an_scenarios.items():
    
    depassement_an_scenarios[scenario] = df.fillna(0)
scenarios = ['BASE', 'WEM', 'WAM']  
n_groups = len(depassement_an_scenarios[scenarios[0]])  
n_bars = len(Proportion)  
bar_width = 0.2  
opacity = 0.7  

edge_colors = {'BASE': 'black', 'WEM': 'black', 'WAM': 'black'}
edge_styles = {'BASE': '-', 'WEM': '--', 'WAM': ':'}
cmap = plt.cm.YlGnBu  # A cool to warm color gradient
norm = plt.Normalize(vmin=0, vmax=n_bars-1)  # Normalize for hierarchy range

spacing = bar_width * len(scenarios) * 0.5
index = np.arange(n_groups) * (len(scenarios) + 1) * spacing  # Adjusted spacing between groups
offset = np.linspace(-bar_width * len(scenarios) / 2, bar_width * len(scenarios) / 2, num=len(scenarios))

fig, ax = plt.subplots(figsize=(14, 12))

for i, scenario in enumerate(scenarios):
  df = depassement_an_scenarios[scenario]
  bottom = np.zeros(n_groups)
  for j, col in enumerate(Proportion):
    bar_color = cmap(norm(j))  
    bars = ax.bar(index + offset[i], df[col], bar_width, bottom=bottom, alpha=opacity,
                  label=hierarchies[j] if i == 0 else "", color=bar_color,
                  edgecolor=edge_colors[scenario], linewidth=2,  
                  linestyle=edge_styles[scenario])
    bottom += df[col].values
    if scenario == "WEM" : 
        label_y_position = bottom[-1] +0.03  
    elif scenario == "WAM" : 
        label_y_position = bottom[-1] +0.03 
    else : # Buffer above the highest bar
        label_y_position = label_y_position = bottom[-1] +0.03  

  ax.text(index[0] + offset[i], label_y_position, scenario, ha='center', va='bottom', fontsize=16,
           color=edge_colors[scenario], rotation = 45)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

ax.set_xlabel('Year', fontsize=19)
ax.set_ylabel('Proportion of total street distance exceeding 20 µg/m³  NO₂ by Road Type(%)', fontsize=19)
ax.set_xticks(index + offset[len(scenarios) // 2])
ax.set_xticklabels(depassement_an_scenarios[scenarios[0]].index, fontsize=17)

ax.legend(handles=ax.patches, title='Scenarios', loc='upper right', fontsize=16, title_fontsize=19)

ax.legend(
    title='Road hierarchies',
    loc='upper center',
    bbox_to_anchor=(0.5, 1.1),  
    fancybox=True,
    fontsize=16, 
    title_fontsize=17,
    ncol=6,
    facecolor='white',
    borderaxespad=0.
)
ax.set_ylim([0, max(bottom) * 1.1])
file_name = 'StreetSProportionOver195.svg'
folder_path = '/Users/meganepourtois/Desktop/MapResults'
file_path = os.path.join(folder_path, file_name)
plt.savefig(file_path, format='svg')
ax.tick_params(axis='y', labelsize=19)
plt.show()
#%% Percent
import matplotlib.pyplot as plt


scenarios = ['BASE', 'WEM', 'WAM']  # The names of your scenarios
percents = ['PercentA0', 'PercentA1', 'PercentA2', 'PercentA3', 'PercentA4', 'PercentA5']  # List of percent categories
n_groups = len(depassement_an_scenarios[scenarios[0]])  # Number of groups (years?)

plt.figure(figsize=(12, 10))

cmap = plt.cm.tab10  

for i, percent in enumerate(percents):
  ax = plt.subplot(2, 3, i + 1)

  bar_width = 0.35  
  x = range(n_groups)  

  for j, scenario in enumerate(scenarios):
    df = depassement_an_scenarios[scenario]

    data = df[percent]

    bars = ax.bar(x, data, bar_width, label=scenario, alpha=0.7, color=cmap(j))

  ax.set_xlabel('Year', fontsize=12)
  ax.set_ylabel('Number of street segments', fontsize=12)
  ax.set_title(f'{percent}', fontsize=14)
  ax.set_xticks(x)  # Set x-axis ticks for year positions
  ax.set_xticklabels(depassement_an_scenarios[scenarios[0]].index, rotation=45, ha='right', fontsize=10)  # Rotate and adjust x-axis labels
  ax.legend(title='Scenarios', loc='upper left', fontsize=12)

plt.tight_layout()  

plt.show()


#%%   
for scenario, df in depassement_an_scenarios.items():
    plt.figure(figsize=(10, 6))
    plt.title(f'Share of street segment depassement between road hierarchies in {scenario} scenario')
    plt.xlabel('Year')
    plt.ylabel('Values')
    
    bottom = None
    
    for col in Proportion:
        plt.bar(df.index, df[col], bottom=bottom, label=col)
        if bottom is None:
            bottom = df[col]
        else:
            bottom += df[col]
    
    plt.xticks(df.index, rotation=45) 
    plt.legend(title='Columns')
    plt.grid(False)
    plt.show()  
#%%
for scenario, df in depassement_an_scenarios.items():
    plt.figure(figsize=(10, 6))
    plt.title(f'Share of street segment depassement between road hierarchies in {scenario} scenario')
    plt.xlabel('Year')
    plt.ylabel('Values')
    
    bottom = None
    
    for col in Percent:
        plt.bar(df.index, df[col], bottom=bottom, label=col)
        if bottom is None:
            bottom = df[col]
        else:
            bottom += df[col]
    
    plt.xticks(df.index, rotation=45) 
    plt.legend(title='Columns')
    plt.grid(False)
    plt.show() 
#%%
hierarchies = ['A0', 'A1', 'A2', 'A3', 'A4', 'A5'] 
for scenario, df in mean_hierarchy_scenarios.items():
    plt.figure(figsize=(10, 6))
    plt.title(f'Mean Hierarchy in {scenario} Scenario')
    plt.xlabel('Year')
    plt.ylabel('ug/m3')

    for col in hierarchies:
        plt.plot(df.index, df[col], marker='o', linestyle='-', label=col)

    plt.xticks(df.index, rotation=45) 
    plt.legend(title='Columns')
    plt.grid(True)  
    plt.show()
     


