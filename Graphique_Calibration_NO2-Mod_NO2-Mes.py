# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:34:26 2024

@author: cleme
"""
#%%
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import numpy as np


from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import LinearRegression

#%%

df1 = pd.read_csv('C:/SiraneDocument/Recepteur/2022/Recept_41R012.dat', delimiter='\s+')
df2 = pd.read_csv('C:/SiraneDocument/Recepteur/2022/Recept_41B011.dat', delimiter='\s+')
df3 = pd.read_csv('C:/SiraneDocument/Recepteur/2022/Recept_41MEU1.dat', delimiter='\s+')
df4 = pd.read_csv('C:/SiraneDocument/Recepteur/2022/Recept_41B006.dat', delimiter='\s+')
df5 = pd.read_csv('C:/SiraneDocument/Recepteur/2022/Recept_41N043.dat', delimiter='\s+')
df6 = pd.read_csv('C:/SiraneDocument/Recepteur/2022/Recept_41R001.dat', delimiter='\s+')
df7 = pd.read_csv('C:/SiraneDocument/Recepteur/2022/Recept_41B004.dat', delimiter='\s+')
df8 = pd.read_csv('C:/SiraneDocument/Recepteur/2022/Recept_41R002.dat', delimiter='\s+')
df9 = pd.read_csv('C:/SiraneDocument/Recepteur/2022/Recept_41CHA1.dat', delimiter='\s+')
df10 = pd.read_csv('C:/SiraneDocument/Recepteur/2022/Recept_41B008.dat', delimiter='\s+')
df11 = pd.read_csv('C:/SiraneDocument/Recepteur/2022/Recept_41B001.dat', delimiter='\s+')
df12 = pd.read_csv('C:/SiraneDocument/Recepteur/2022/Recept_41REG1.dat', delimiter='\s+')


date_format = '%d/%m/%Y %H:%M'  
dataframes = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10,df11, df12]
for df in dataframes:
    df['NO2_Mod'] = pd.to_numeric(df['NO2_Mod'], errors='coerce')
    df['NO2_Mes'] = pd.to_numeric(df['NO2_Mes'], errors='coerce')
    
    if 'Date' in df.columns:
        df.index = df.index.astype(str) + ' ' + df['Date']
        df.drop('Date', axis=1, inplace=True)
        
        df.index = pd.to_datetime(df.index, format=date_format)
        df['Date'] = df.index
        
df = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12], ignore_index=False)


df.dropna(inplace=True)
X = df['NO2_Mod'].values.reshape(-1, 1)
y = df['NO2_Mes'].values.reshape(-1, 1)

model = LinearRegression(fit_intercept=False).fit(X, y)
a = model.coef_[0][0]
slope = model.coef_[0][0]

print(f"The coefficient (slope) is: {slope}")
X_reshape = df['NO2_Mod'].values.reshape(-1, 1)
y_pred = model.predict(X_reshape)

plt.figure(figsize=(10, 6))
plt.scatter(df['NO2_Mod'], df['NO2_Mes'], color='blue', label='Data points')
plt.plot(df['NO2_Mod'], y_pred, color='red', label=f'Linear Regression\nSlope = {slope:.2f}')
plt.xlabel('NO2_Mod (Model)')
plt.ylabel('NO2_Mes (Measured)')
plt.title('Linear Regression Analysis Without Intercept')
plt.legend()
plt.grid(False)
plt.show()


#%% Régression linéaire entre NO2_Mod et NO2_Mes + coefficient Emis_rues


Emis_rues = pd.read_csv("C:/SiraneDocument/Emis_rues/Emis_rues.dat", delimiter=';')
# Afficher les premières lignes pour vérifier
print(df.head())

Mod_Temp_NO2 = pd.read_csv("C:/SiraneDocument/Emis_rues/Mod_Temp_NO2.dat", delimiter='\s+')
Mod_Temp_NO2['Date2'] = Mod_Temp_NO2.index
Mod_Temp_NO2['Index'] = Mod_Temp_NO2['Date2'].astype(str)+ ' '+Mod_Temp_NO2['Date'].astype(str)

Mod_Temp_NO2.set_index('Index', inplace=True)
Mod_Temp_NO2 = Mod_Temp_NO2.drop(['Date', 'Date2'], axis=1)
Mod_Temp_NO2.index.name = 'Date'


df.dropna(inplace=True)
X = df['NO2_Mod'].values.reshape(-1, 1)
y = df['NO2_Mes'].values.reshape(-1, 1)

model = LinearRegression(fit_intercept=False).fit(X, y)
a = model.coef_[0][0]
slope = model.coef_[0][0]

print(f"The coefficient (slope) is: {slope}")
X_reshape = df['NO2_Mod'].values.reshape(-1, 1)
y_pred = model.predict(X_reshape)

plt.figure(figsize=(10, 6))
plt.scatter(df['NO2_Mod'], df['NO2_Mes'], color='blue', label='Data points')
plt.plot(df['NO2_Mod'], y_pred, color='red', label=f'Linear Regression\nSlope = {slope:.2f}')
plt.xlabel('NO2_Mod (Model)')
plt.ylabel('NO2_Mes (Measured)')
plt.title('Linear Regression Analysis Without Intercept')
plt.legend()
plt.grid(False)
plt.show()

#%%
Emis_rues_new = Emis_rues.copy()
Emis_rues_new['NO'] = Emis_rues['NO'] * 0.75

Emis_rues_new['NO2'] = Emis_rues['NO2']
Emis_rues_new.set_index('Id',inplace=True)
Emis_rues_new.to_csv(path_or_buf='C:/BiblioSIRANE/Emis_rues.dat', sep='\t')

#%% Profil hebdomadaire moyen sur l'année des différentes stations


stations_dataframes = {
    "Uccle (41R012)": df1,
    "Berchem-Ste-Agathe (41B011)": df2,
    "Park Neder-Over-Heembeek (41MEU1)": df3,
    "Parliament UE (41B006)": df4,
    "Avant-Port (41N043)": df5,
    "Molenbeek (41R001)": df6,
    "Ste-Catherine (41B004)": df7,
    "Ixelles (41R002)": df8,
    "Ganshoren (41CHA1)": df9,
    "Belliard (41B008)": df10,
    "Arts-Loi (41B001)": df11,
    "Regent (41REG1)": df12
}

n_stations = len(stations_dataframes)
n_cols = 3
n_rows = int(np.ceil(n_stations / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 2 * n_rows), sharex=True, sharey=True)
axes = axes.flatten()

for i, (station_code, df) in enumerate(stations_dataframes.items()):
    df['Date'] = pd.to_datetime(df['Date'])
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Hour'] = df['Date'].dt.hour
    df['WeekHour'] = df['DayOfWeek'] * 24 + df['Hour']
    
    weekly_profile = df.groupby('WeekHour').mean()
    
    axes[i].plot(weekly_profile.index, weekly_profile['NO2_Mod'], label='Modèle', color='blue')
    axes[i].plot(weekly_profile.index, weekly_profile['NO2_Mes'], label='Mesure', color='orange')
    
    axes[i].set_title(station_code)
    axes[i].grid(False)

    if i >= n_stations - (n_stations % n_cols if n_stations % n_cols != 0 else n_cols):
        axes[i].set_xticks(np.arange(0, 168, 24))
        axes[i].set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], rotation=45, ha="right")
    else:
        axes[i].set_xticks(np.arange(0, 168, 24))
        axes[i].set_xticklabels(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

fig.text(0.5, 0.08, 'Day of the Week', ha='center', fontsize=18)
fig.text(0.08, 0.5, 'NO$_2$ concentration [µg/m³]', va='center', rotation='vertical', fontsize=18)

for idx in range(n_stations, n_rows * n_cols):
    fig.delaxes(axes[idx])


fig.tight_layout(rect=[0.09, 0.1, 0.95, 0.95])

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2, fontsize=15)

plt.savefig('C:/SiraneDocument/Presentationgraphseminar/profilhebdomadaire.png', format='png')

plt.show()




#%% Graphique pour une station spécifique


station_to_plot = "Regent (41REG1)"

stations_dataframes = {
    "Uccle (41R012)": df1,
    "Berchem-Ste-Agathe (41B011)": df2,
    "Park Neder-Over-Heembeek (41MEU1)": df3,
    "Parliament UE (41B006)": df4,
    "Avant-Port (41N043)": df5,
    "Molenbeek (41R001)": df6,
    "Ste-Catherine (41B004)": df7,
    "Ixelles (41R002)": df8,
    "Ganshoren (41CHA1)": df9,
    "Belliard (41B008)": df10,
    "Arts-Loi (41B001)": df11,
    "Regent (41REG1)": df12
}

df = stations_dataframes.get(station_to_plot)

df['Date'] = pd.to_datetime(df['Date'])
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Hour'] = df['Date'].dt.hour
df['WeekHour'] = df['DayOfWeek'] * 24 + df['Hour']

weekly_profile = df.groupby('WeekHour').mean()

plt.figure(figsize=(10, 6))
plt.plot(weekly_profile.index, weekly_profile['NO2_Mod'], label='Modeled', color='blue')
plt.plot(weekly_profile.index, weekly_profile['NO2_Mes'], label='Measure', color='orange')

plt.title(station_to_plot)
plt.xlabel('Day of the Week')
plt.ylabel('NO$_2$ concentration [µg/m³]')
plt.xticks(np.arange(0, 168, 24), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

plt.grid(False)
plt.legend()
plt.tight_layout()
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2, fontsize=10)


plt.savefig('C:/SiraneDocument/Presentationgraphseminar/Ixelles (41R002).png', format='png')

#%% Comparaison moyenne des stations par rapport à une moyenne parfaite


n_rows = 4
n_cols = 3
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15))  # Adjust the size as needed
axes = axes.flatten()  

for i, (station_name, df) in enumerate(stations_dataframes.items()):
    num_bins = 18
    bin_edges = np.linspace(df['NO2_Mes'].min(), df['NO2_Mes'].max(), num_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    df['Binned'] = pd.cut(df['NO2_Mes'], bins=bin_edges, labels=bin_centers, include_lowest=True)
    grouped = df.groupby('Binned')
    mean_modeled_NO2 = grouped['NO2_Mod'].mean().interpolate()
    
    axes[i].plot(grouped['NO2_Mes'].mean(), mean_modeled_NO2, color='blue', linestyle='-')
    
    max_val = max(df['NO2_Mes'].max(), df['NO2_Mod'].max())
    axes[i].plot([0, max_val], [0, max_val], 'r--')
    
    axes[i].set_title(station_name)

fig.tight_layout(rect=[0.09, 0.1, 0.95, 0.95])
fig.text(0.5, 0.08, 'Measured NO$_2$ concentrations [µg/m³]', ha='center', fontsize=16)
fig.text(0.08, 0.5, 'Modelled NO$_2$ concentrations [µg/m³]', va='center', rotation='vertical', fontsize=16)


plt.savefig('C:/SiraneDocument/Presentationgraphseminar/perfectligne.png', format='png')
plt.show()

#%% Histogramme de densité entre NO2_Mod et NO2_Mes


stations_dataframes = {
    "Uccle (41R012)": df1,
    "Berchem-Ste-Agathe (41B011)": df2,
    "Park Neder-Over-Heembeek (41MEU1)": df3,
    "Parliament UE (41B006)": df4,
    "Avant-Port (41N043)": df5,
    "Molenbeek (41R001)": df6,
    "Ste-Catherine (41B004)": df7,
    "Ixelles (41R002)": df8,
    "Ganshoren (41CHA1)": df9,
    "Belliard (41B008)": df10,
    "Arts-Loi (41B001)": df11,
    "Regent (41REG1)": df12
}
n_rows = 3
n_cols = 4
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15))  # Adjust the figure size as needed

for i, (station, df) in enumerate(stations_dataframes.items()):
    ax = axes[i // n_cols, i % n_cols]
    sns.histplot(df['NO2_Mod'], stat='density', bins=30, color='blue', label='Model', kde=False, element="step", linewidth=2, ax=ax,alpha=0)

    sns.histplot(df['NO2_Mes'], stat='density', bins=30, color='orange', label='Measure', kde=False, element="step", linewidth=2, ax=ax,alpha=0)

    ax.set_title(station, fontsize = 15)
    ax.set_xlabel('')  
    ax.set_ylabel('')
    ax.grid(False)

for j in range(i + 1, n_rows * n_cols):
    fig.delaxes(axes.flatten()[j])

fig.text(0.5, 0.04, 'NO$_2$ concentration [µg/m³]', ha='center', va='center', fontsize=20)
fig.text(0.04, 0.5, 'Density [-]', ha='center', va='center', rotation='vertical', fontsize=20)
fig.tight_layout()
fig.subplots_adjust(top=0.95, left=0.1, right=0.9, bottom=0.1)
fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=17)

plt.savefig('C:/SiraneDocument/Presentationgraphseminar/density.png', format='png')
plt.show()

#%% FB et NAD stastistics


stats_df = pd.DataFrame(columns=['DataFrame', 'FB', 'NAD'])

epsilon = 1e-8
def calculate_stats(df, df_name):
    df['NO2_Mes'] = pd.to_numeric(df['NO2_Mes'], errors='coerce')
    df['NO2_Mod'] = pd.to_numeric(df['NO2_Mod'], errors='coerce')
    
    valid_rows = df.dropna(subset=['NO2_Mes', 'NO2_Mod'])
    mean_observed = valid_rows['NO2_Mes'].mean()
    mean_predicted = valid_rows['NO2_Mod'].mean()
    
    FB = (mean_observed - mean_predicted) / (0.5 * (mean_observed + mean_predicted))
    NAD = valid_rows['NO2_Mes'].sub(valid_rows['NO2_Mod']).abs().div(valid_rows['NO2_Mes'].add(valid_rows['NO2_Mod']).add(epsilon)).mean()
    
    return {'DataFrame': df_name, 'FB': FB, 'NAD': NAD}

for i in range(1, 13):
    df_name = f'df{i}'
    df = globals()[df_name]  # Assuming dataframes df1 through df12 are already defined
    stats = calculate_stats(df, df_name)
    stats_df = stats_df.append(stats, ignore_index=True)

print(stats_df)

stats_df.to_excel('C:/SiraneDocument/stats_df.xlsx')


#%% Graphique création régression linéaire sur chaque station



fig, axs = plt.subplots(4, 3, figsize=(18, 24))  
axs = axs.flatten()  

for i, (station_name, df) in enumerate(stations_dataframes.items()):
    df['NO2_Mod'] = pd.to_numeric(df['NO2_Mod'], errors='coerce')
    df['NO2_Mes'] = pd.to_numeric(df['NO2_Mes'], errors='coerce')
    df.dropna(inplace=True)
    
    X = df['NO2_Mod'].values.reshape(-1, 1)
    y = df['NO2_Mes'].values.reshape(-1, 1)

    model = LinearRegression(fit_intercept=True).fit(X, y)
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]
    y_pred = model.predict(X)
    
    r_squared = model.score(X, y)

    axs[i].scatter(X, y, facecolors='none', edgecolors='black', label='Data points')
    axs[i].plot(X, y_pred, color='black', label=f'Linear Regression\ny = {slope:.2f}x + {intercept:.2f}\n$R^2$ = {r_squared:.2f}')
    axs[i].set_title(station_name,fontsize = 15)  # Use the actual station name
    data_min = min(X.min(), y.min())
    data_max = max(X.max(), y.max())
    axs[i].set_xlim(data_min, data_max)
    axs[i].set_ylim(data_min, data_max)
    axs[i].set_aspect('equal', 'box')
    axs[i].legend()
    axs[i].grid(False)

fig.text(0.5, 0.04, 'NO$_2$ modelled [µg/m³]', ha='center', va='center', fontsize=20)
fig.text(0.04, 0.5, 'NO$_2$ measured [µg/m³]', ha='center', va='center', rotation='vertical', fontsize=20)
fig.subplots_adjust(top=0.95, left=0.07, right=0.95, bottom=0.07)
plt.savefig('C:/SiraneDocument/Presentationgraphseminar/regression.png', format='png')


plt.show()

#%% Graphique comparaison sur un mois (janvier ou juin généralement) entre NO2_Mod et NO2_Mes

stations_dataframes = {
    "Uccle (41R012)": df1,
    "Berchem-Ste-Agathe (41B011)": df2,
    "Park Neder-Over-Heembeek (41MEU1)": df3,
    "Parliament UE (41B006)": df4,
    "Avant-Port (41N043)": df5,
    "Molenbeek (41R001)": df6,
    "Ste-Catherine (41B004)": df7,
    "Ixelles (41R002)": df8,
    "Ganshoren (41CHA1)": df9,
    "Belliard (41B008)": df10,
    "Arts-Loi (41B001)": df11,
    "Regent (41REG1)": df12
}
date_format = mdates.DateFormatter('%d-%b')  
week_locator = mdates.WeekdayLocator()  

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 10))
axes = axes.flatten()

for i, (station_name, df) in enumerate(stations_dataframes.items()):
    df['Date'] = pd.to_datetime(df['Date'])
    df_january = df[df['Date'].dt.month == 1]

    axes[i].plot(df_january['Date'], df_january['NO2_Mod'], label='NO2 Modelled')
    axes[i].plot(df_january['Date'], df_january['NO2_Mes'], label='NO2 Measured')
    axes[i].set_title(station_name)
    axes[i].set_xlabel('Date')
    axes[i].set_ylabel('NO2 Concentration')
    axes[i].legend()
    axes[i].xaxis.set_major_locator(week_locator)
    axes[i].xaxis.set_major_formatter(date_format)

plt.tight_layout()
plt.show()