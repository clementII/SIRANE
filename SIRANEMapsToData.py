#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 17:33:08 2024

@author: meganepourtois
"""

import geopandas as gpd
import rasterio
from rasterio.plot import show
from rasterstats import zonal_stats
import matplotlib.pyplot as plt
import pandas as pd


def load_and_buffer_vector(vector_path, buffer_distance):
    vector_data = gpd.read_file(vector_path)
    buffered_vector = vector_data.buffer(buffer_distance)
    return buffered_vector

def load_raster(raster_path):
    raster = rasterio.open(raster_path)
    return raster

def calculate_zonal_statistics(vector, raster):
    affine = raster.transform
    stats = zonal_stats(vector, raster.read(1), affine=affine, stats=['mean', 'max', 'min', 'median', 'std'])
    return stats

vector_path = '/Users/meganepourtois/Desktop/SIRANE2_segmenté.shp'
raster_path = '/Users/meganepourtois/Desktop/MapResults'

buffered_vector = load_and_buffer_vector(vector_path, 2)
raster = load_raster(raster_path)


statistics = calculate_zonal_statistics(buffered_vector, raster)
#%%
stat_df = pd.DataFrame(statistics)
stat_df.reset_index(inplace=True)  

excel_path = '/Users/meganepourtois/Desktop/Mémoire/Sirane2_segmenté.xlsx'
excel_df = pd.read_excel(excel_path)

excel_df = excel_df[['FID', 'Hierarchie']]

merged_df = pd.merge(stat_df, excel_df, left_on='index', right_on='FID')

merged_df.drop(['index', 'FID'], axis=1, inplace=True)

output_path = '/Users/meganepourtois/Desktop/MapData/merged_output.xlsx'
merged_df.to_excel(output_path, index=False)





