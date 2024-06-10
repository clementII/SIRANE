# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 11:16:35 2024

@author: cleme
"""

import pandas as pd
import os
#%%
input_dir = 'C:/SiraneDocument/Source_Ponctuelle'
output_dir = 'C:/SiraneDocument/Source_Ponctuelle_resultat'

os.makedirs(output_dir, exist_ok=True)

def find_delimiter(file_path):
    with open(file_path, 'r') as file:
        line = file.readline()
        if '\t' in line:
            return '\t'
        else:
            return ' '  

for filename in os.listdir(input_dir):
    if filename.endswith('.dat'):
        file_path = os.path.join(input_dir, filename)
        
        
        delimiter = find_delimiter(file_path)
        
        df = pd.read_csv(file_path, sep=delimiter, engine='python')
        
        if 'NO' in df.columns:
            df['NO'] = df['NO'] * 0.75
            df['NO2']=df['NO2']
        
        output_file_path = os.path.join(output_dir, filename)
        
        df.to_csv(output_file_path, sep=delimiter, index=False)

print("All .dat files have been processed and saved to the output folder.")