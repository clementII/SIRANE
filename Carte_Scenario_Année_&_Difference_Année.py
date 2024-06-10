# -*- coding: utf-8 -*-
"""
Created on Thu May  2 17:55:36 2024

@author: cleme
"""

import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import json  
import numpy as np
import pandas as pd

from rasterio.enums import Resampling
from rasterio.plot import show
from rasterstats import zonal_stats
from rasterio.mask import mask
#%%
shapefile_path = 'C:/SiraneDocument/RasterScenario/UrbAdm_REGION.shp'

vector_path = 'C:/SiraneDocument/RasterScenario/SIRANE2_segmenté.shp'
raster_path = 'C:/SiraneDocument/RasterScenario/Conc_NO2_Moy_2028BASE.grd'
def get_features(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

def mask_raster_with_shapefile(raster_path, shapefile_path):
    gdf = gpd.read_file(shapefile_path)
    
    raster = rasterio.open(raster_path)
    
    geom = get_features(gdf)
    
    out_image, out_transform = mask(raster, geom, crop=True)
    
    out_meta = raster.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })
    
    with rasterio.open("masked_raster.tif", "w", **out_meta) as dest:
        dest.write(out_image)
    
    return out_image

masked_raster = mask_raster_with_shapefile(raster_path, shapefile_path)

masked_raster_path = 'masked_raster.tif'  
masked_raster = rasterio.open(masked_raster_path)

data = masked_raster.read(1)  

print(data.max())
print(data.min())
data_min=data.min()
data_max = data.max()



#%%
shapefile_path = 'C:/SiraneDocument/RasterScenario/UrbAdm_REGION.shp'

shapefile = gpd.read_file(shapefile_path)

with rasterio.open(raster_path) as src:
    data = src.read(1)  
    data_min = data.min()  
    data_max = data.max()  
    rounded_min = np.floor(data_min)
    rounded_max = np.ceil(data_max)
    
    bounds = [13]+[14,16, 18,20, 30, 56]
    #bounds = [-6,-5,-4,-3,-2,-1,0]
    cmap = plt.get_cmap('magma_r', len(bounds) - 1)  
    norm = colors.BoundaryNorm(bounds, cmap.N)  

    clipped_data, transform = mask(src, shapefile.geometry, crop=True)
    meta = src.meta.copy()
    meta.update({"driver": "GTiff",
                 "height": clipped_data.shape[1],
                 "width": clipped_data.shape[2],
                 "transform": transform})

print(data_min)
print(data_max)

clipped_data = np.ma.masked_invalid(clipped_data)

valid_data = clipped_data.compressed()

interval_counts = []
total_pixels = valid_data.size

for i in range(len(bounds) - 1):
    interval_mask = (valid_data >= bounds[i]) & (valid_data < bounds[i + 1])
    interval_count = np.sum(interval_mask)
    interval_counts.append(interval_count)
    raw_percentage = (interval_count / total_pixels) * 100
    print(f"Interval {bounds[i]} to {bounds[i+1]}: {raw_percentage:.2f}% (raw)")

raw_total_percentage = sum(interval_counts)
normalized_percentages = [(count / raw_total_percentage) * 100 for count in interval_counts]

for i in range(len(bounds) - 1):
    print(f"Interval {bounds[i]} to {bounds[i+1]}: {normalized_percentages[i]:.2f}% (normalized)")

normalized_total_percentage = sum(normalized_percentages)
print(f"Normalized total percentage: {normalized_total_percentage:.2f}%")

plt.figure(figsize=(10, 10))
masked_data = np.ma.masked_where(clipped_data == clipped_data[0, 0], clipped_data)
plt.imshow(masked_data[0], cmap=cmap, norm=norm)
cbar = plt.colorbar(label='NO2 concentration difference [%]', ticks=bounds, fraction=0.04, pad=0.04)

cbar.set_label('NO$_2$ concentration  [µg/m³]', size=16)  # Ajustez 'size' selon vos besoins

cbar.ax.tick_params(labelsize=12)

plt.xlabel('')
plt.ylabel('')

plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.axis('off')

plt.savefig('C:/SiraneDocument/GraphiqueScenario/DIFFERENCE_entre_année/DIFFERENCE2028WAMWEM.png', format='png', bbox_inches='tight')
plt.show()

#%% CODE soustraction entre année



from matplotlib.colors import BoundaryNorm

def get_features(gdf):
    """Function to parse features from GeoDataFrame."""
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

def mask_raster(raster_path, shapefile_path,nodata_value=np.nan):
    """Load and mask a raster using a shapefile."""
    gdf = gpd.read_file(shapefile_path)
    geom = get_features(gdf)

    with rasterio.open(raster_path) as src:
        out_image, out_transform = mask(src, geom, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
    
    return out_image, out_meta

def subtract_and_save_rasters(base_raster_path,wem_raster_path, wam_raster_path, shapefile_path, output_path):
    """Mask both rasters, subtract them, and save the result."""

    wem_data, wem_meta = mask_raster(wem_raster_path, shapefile_path)
    wam_data, wam_meta = mask_raster(wam_raster_path, shapefile_path)
    base_data, base_meta = mask_raster(base_raster_path, shapefile_path)

    wem_data_concentrationfond= wem_data
    wam_data_concentrationfond = wam_data 
    base_data_concentrationfond = base_data-10.7989
    result_data = -(((wem_data_concentrationfond- wam_data_concentrationfond)/base_data_concentrationfond))*100


    with rasterio.open(output_path, "w", **wem_meta) as dest:
        dest.write(result_data)

wem_raster_path = 'C:/SiraneDocument/RasterScenario/Conc_NO2_Moy_2025WEM.grd'
wam_raster_path = 'C:/SiraneDocument/RasterScenario/Conc_NO2_Moy_2025WAM.grd'
base_raster_path ='C:/SiraneDocument/RasterScenario/Conc_NO2_Moy_2028BASE.grd'
shapefile_path = 'C:/SiraneDocument/RasterScenario/UrbAdm_REGION.shp'
output_path = 'Difference_2022BASE_WAM.tif'

subtract_and_save_rasters(base_raster_path,wem_raster_path, wam_raster_path, shapefile_path, output_path)



shapefile = gpd.read_file(shapefile_path)

with rasterio.open(output_path) as src:
    data = src.read(1)  
    data_min = data.min()  
    data_max = data.max()  
    rounded_min = np.floor(data_min)
    rounded_max = np.ceil(data_max)
    
    bounds = [-7.5,-5,-4,-3,-2,-1,-0.5,0]
    cmap = plt.get_cmap('coolwarm_r', len(bounds) - 1)  # Get the magma colormap with the number of intervals needed
    norm = colors.BoundaryNorm(bounds, cmap.N)  # Normalize data to fit the number of colors

    clipped_data, transform = mask(src, shapefile.geometry, crop=True)
    meta = src.meta.copy()
    meta.update({"driver": "GTiff",
                 "height": clipped_data.shape[1],
                 "width": clipped_data.shape[2],
                 "transform": transform})

print(data_min)
print(data_max)

clipped_data = np.ma.masked_invalid(clipped_data)

valid_data = clipped_data.compressed()

interval_counts = []
total_pixels = valid_data.size

for i in range(len(bounds) - 1):
    interval_mask = (valid_data >= bounds[i]) & (valid_data < bounds[i + 1])
    interval_count = np.sum(interval_mask)
    interval_counts.append(interval_count)
    raw_percentage = (interval_count / total_pixels) * 100
    print(f"Interval {bounds[i]} to {bounds[i+1]}: {raw_percentage:.2f}% (raw)")

raw_total_percentage = sum(interval_counts)
normalized_percentages = [(count / raw_total_percentage) * 100 for count in interval_counts]

for i in range(len(bounds) - 1):
    print(f"Interval {bounds[i]} to {bounds[i+1]}: {normalized_percentages[i]:.2f}% (normalized)")

normalized_total_percentage = sum(normalized_percentages)
print(f"Normalized total percentage: {normalized_total_percentage:.2f}%")

plt.figure(figsize=(10, 10))
masked_data = np.ma.masked_where(clipped_data == clipped_data[0, 0], clipped_data)
plt.imshow(masked_data[0], cmap=cmap, norm=norm)
cbar = plt.colorbar(label='NO2 relative concentration difference [%]', ticks=bounds, fraction=0.04, pad=0.04)

cbar.set_label('NO$_2$ relative concentration difference [%]', size=16)  # Ajustez 'size' selon vos besoins

cbar.ax.tick_params(labelsize=12)

plt.xlabel('')
plt.ylabel('')

plt.xticks([])
plt.yticks([])
plt.grid(False)
plt.axis('off')

plt.savefig('C:/SiraneDocument/GraphiqueScenario/DIFFERENCE_entre_année/DIFFERENCE2028WAMWEM.png', format='png', bbox_inches='tight')
plt.show()

