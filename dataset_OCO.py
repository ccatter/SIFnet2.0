# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4
import sys
sys.path.insert(1, '../')
import helpers
from scipy.stats import pearsonr

from sklearn.metrics import mean_squared_error, r2_score

#general packages
import os
from os import listdir
from os.path import isfile, join
import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from PIL import Image
import netCDF4 
import random
import sys
import datetime as dt

#PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
import torch.optim as optim
from torch.utils.data.dataset import Subset
from torch.nn import functional as F
#import torchvision

#other ML packags
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
#import optuna
from sklearn.decomposition import PCA

#import pytorch_ssim

#some settings
plt.rcParams.update({'font.size':14})

#own scripts
import sys
#sys.path.insert(1, '../')
import helpers


def get_500m_dataset(xLims=np.array([10,15]), yLims=np.array([45,50]), xLims_test = np.array([10, 15]), yLims_test = np.array([45, 50]), d_start=0, d_end=46, USE_MODEL_OUT=True,SCALE_FACTOR=10,DATA_OPTION=1):

    
            
    pathPref = '/mnt/mnt_folder/data/' 
    pathLoadLULC = pathPref + 'Gridded_LULC_E-10_50_N25_70_11bands.nc'
    pathLoadMeanStdInd = pathPref + 'Mean_Std_indices_16day.csv'
    pathLoadLandMask = pathPref + 'LandMask_0005_E-10_50_N25_70.nc'
    pathLoadTopography = pathPref + 'Topography_0005US_USGS_GMTED2010.nc'

    #16day
    pathLoadTropomi = pathPref + 'TROPOMI-SIF_04-2018--03-2021_16day_005deg.nc'
    pathLoadModis = pathPref + 'MODIS_04-2018-03-2021_CONUS_0005deg_16day_allBands_N40-56.8_E-10-30_interp.nc'
    pathLoadSZA = pathPref + 'cosSZA_2018-2021_16day_0005.csv'
    pathLoadSoMo = pathPref + 'SoMo_04-2018-03-2021_005deg_16day.nc'
    pathLoadERA5 = pathPref + 'ERA5_04-2018-03-2021_005deg_16day.nc'

    
    #OCO
    df_oco2 = helpers.read_OCO(pathPref, yLims, xLims, yLims_test, xLims_test)
    '''
    year = 2018
    X = 2
    df_oco2 = pd.read_csv(pathPref + 'OCO'+str(X) +'_'+ str(year) + '_3_Germany.csv')
    df_oco2 = df_oco2.drop('Unnamed: 0',axis=1)

    th = 0.01
    df = df_oco2[df_oco2.Latitude < (np.max(yLims) + th)]
    df = df[df_oco2.Latitude > (np.min(yLims) + th)]
    df = df[df_oco2.Longitude < (np.max(xLims) + th)]
    df = df[df_oco2.Longitude > (np.min(xLims) + th)]
    df = df.reset_index(drop=True)
    df_oco2 = df.copy()
    print('Number of footprints: ',df_oco2.shape)
    '''

    #%%
    # get grids
    Tropo, Modis, Sen2 = helpers.GetGrids(xLims, yLims)
    ddX_Modis,ddY_Modis,lon_Modis,lat_Modis = Modis['ddX'],Modis['ddY'],Modis['lon'],Modis['lat']
    ddX_Tropo,ddY_Tropo,lon_Tropo,lat_Tropo = Tropo['ddX'],Tropo['ddY'],Tropo['lon'],Tropo['lat']
    lonM, latM = np.meshgrid(lon_Modis, lat_Modis[::-1])


    # %%
    nc_tropomi = netCDF4.Dataset(pathLoadTropomi)
    lon, lat = nc_tropomi.variables['lon'][:].data, nc_tropomi.variables['lat'][:].data

    ti_days_since = nc_tropomi.variables['time'][d_start:d_end].data
    dates = helpers.GetDatesFromDaysSince(ti_days_since)

    #%%
    # get indices of reduced regions
    ix1,ix2 = np.argmin(abs(lon-np.min(lon_Tropo))),np.argmin(abs(lon-np.max(lon_Tropo))) + 1
    ix_min,ix_max = int(np.min([ix1, ix2])),int(np.max([ix1, ix2]))
    iy1,iy2 = np.argmin(abs(lat-np.min(lat_Tropo))) + 1,np.argmin(abs(lat-np.max(lat_Tropo)))
    iy_min,iy_max = int(np.min([iy1, iy2])),int(np.max([iy1, iy2]))
  
    #%%
    lon_red,lat_red = lon[ix_min:ix_max],lat[iy_min:iy_max]
    # %%
    #SIF
    sif = nc_tropomi.variables['sif'][iy_min:iy_max,ix_min:ix_max,d_start:d_end].data
    sif = np.moveaxis(sif,-1,0)


    sif_dc_feat = nc_tropomi.variables['sif_dc'][iy_min:iy_max,ix_min:ix_max,d_start:d_end].data
    sif_dc_feat[np.where(nc_tropomi.variables['sif_dc'][iy_min:iy_max,ix_min:ix_max,d_start:d_end].mask)] = np.nan
    sif_dc_feat = np.moveaxis(sif_dc_feat,-1,0)

    dcCorr_feat = nc_tropomi.variables['dcCorr'][iy_min:iy_max,ix_min:ix_max,d_start:d_end].data
    dcCorr_feat[np.where(nc_tropomi.variables['dcCorr'][iy_min:iy_max,ix_min:ix_max,d_start:d_end].mask)] = np.nan
    dcCorr_feat = np.moveaxis(dcCorr_feat,-1,0)

    


    #resample SIF from 0.05° to 0.005°
    sif = np.repeat(sif, SCALE_FACTOR, axis=1)
    sif = np.repeat(sif, SCALE_FACTOR, axis=2)

    sif_dc_feat = np.repeat(sif_dc_feat, SCALE_FACTOR, axis=1)
    sif_dc_feat = np.repeat(sif_dc_feat, SCALE_FACTOR, axis=2)

    dcCorr_feat = np.repeat(dcCorr_feat, SCALE_FACTOR, axis=1)
    dcCorr_feat = np.repeat(dcCorr_feat, SCALE_FACTOR, axis=2)    
    


    #%%
    nc_modis = netCDF4.Dataset(pathLoadModis)
    lat_modis_0005 = nc_modis.variables['lat'][:].data
    lon_modis_0005 = nc_modis.variables['lon'][:].data
    ix1,ix2 = np.argmin(abs(lon_modis_0005-np.min(lon_Modis))),np.argmin(abs(lon_modis_0005-np.max(lon_Modis))) + 1
    ix_min_Modis,ix_max_Modis = int(np.min([ix1, ix2])),int(np.max([ix1, ix2]))
    iy1,iy2 = np.argmin(abs(lat_modis_0005-np.min(lat_Modis))) + 1,np.argmin(abs(lat_modis_0005-np.max(lat_Modis)))
    iy_min_Modis,iy_max_Modis = int(np.min([iy1, iy2])),int(np.max([iy1, iy2]))

    lat_modis_0005 = lat_modis_0005[iy_min_Modis:iy_max_Modis]
    lon_modis_0005 = lon_modis_0005[ix_min_Modis:ix_max_Modis]
    
    nir = nc_modis.variables['nir'][iy_min_Modis:iy_max_Modis,ix_min_Modis:ix_max_Modis,d_start:d_end].data
    red = nc_modis.variables['red'][iy_min_Modis:iy_max_Modis,ix_min_Modis:ix_max_Modis,d_start:d_end].data
    blue = nc_modis.variables['blue'][iy_min_Modis:iy_max_Modis,ix_min_Modis:ix_max_Modis,d_start:d_end].data
    green = nc_modis.variables['green'][iy_min_Modis:iy_max_Modis,ix_min_Modis:ix_max_Modis,d_start:d_end].data
    swir1 = nc_modis.variables['swir1'][iy_min_Modis:iy_max_Modis,ix_min_Modis:ix_max_Modis,d_start:d_end].data
    swir2 = nc_modis.variables['swir2'][iy_min_Modis:iy_max_Modis,ix_min_Modis:ix_max_Modis,d_start:d_end].data
    swir3 = nc_modis.variables['swir3'][iy_min_Modis:iy_max_Modis,ix_min_Modis:ix_max_Modis,d_start:d_end].data
    nir[nir==0] = 0.0000001
    red[red==0] = 0.0000001
    nir = np.moveaxis(nir,-1,0)
    red = np.moveaxis(red,-1,0)
    blue = np.moveaxis(blue,-1,0)
    green = np.moveaxis(green,-1,0)
    swir1 = np.moveaxis(swir1,-1,0)
    swir2 = np.moveaxis(swir2,-1,0)
    swir3 = np.moveaxis(swir3,-1,0)


    
    #VIs
    ndvi = (nir-red)/(nir+red)
    nirv = nir * ndvi
    kNDVI = np.tanh(ndvi**2)
    L,C1,C2,G = 1,6,7.5,2.5
    evi = G * ((nir - red)/(nir + C1*red - C2*blue + L))
    evi[np.where(evi<-20)] = np.nan
    evi[np.where(evi>20)] = np.nan


    #%%
    #SSM & SUSM
    nc_SoMo = netCDF4.Dataset(pathLoadSoMo)
    ssm, susm = nc_SoMo.variables['ssm'][d_start:d_end,iy_min:iy_max,ix_min:ix_max].data, nc_SoMo.variables['susm'][d_start:d_end,iy_min:iy_max,ix_min:ix_max].data
    #change resolution to 0.005° from 0.05°
    ssm = np.repeat(ssm, SCALE_FACTOR, axis=1)
    ssm = np.repeat(ssm, SCALE_FACTOR, axis=2)
    susm = np.repeat(susm, SCALE_FACTOR, axis=1)
    susm = np.repeat(susm, SCALE_FACTOR, axis=2)

    # %%
    nc_ERA5 = netCDF4.Dataset(pathLoadERA5, 'r')
    precipitation, temperature = nc_ERA5.variables['precipitation'][d_start:d_end,iy_min:iy_max,ix_min:ix_max].data, nc_ERA5.variables['meantemp_2m'][d_start:d_end,iy_min:iy_max,ix_min:ix_max].data
    precipitation = np.repeat(precipitation, SCALE_FACTOR, axis=1)
    precipitation = np.repeat(precipitation, SCALE_FACTOR, axis=2)
    temperature = np.repeat(temperature, SCALE_FACTOR, axis=1)
    temperature = np.repeat(temperature, SCALE_FACTOR, axis=2)

    try:
        precipitation_del, temperature_del = nc_ERA5.variables['precipitation'][d_start-1:d_end-1,iy_min:iy_max,ix_min:ix_max].data, nc_ERA5.variables['meantemp_2m'][d_start-1:d_end-1,iy_min:iy_max,ix_min:ix_max].data 
        precipitation_del = np.repeat(precipitation_del, SCALE_FACTOR, axis=1)
        precipitation_del = np.repeat(precipitation_del, SCALE_FACTOR, axis=2)
        temperature_del = np.repeat(temperature_del, SCALE_FACTOR, axis=1)
        temperature_del = np.repeat(temperature_del, SCALE_FACTOR, axis=2)
    except:
        precipitation_del, temperature_del = precipitation, temperature
        print('Precipitation and temperature delay did not work!')
        pass

    nc_ERA5.close()

    #land mask
    nc_lm = netCDF4.Dataset(pathLoadLandMask)
    
    lat_lm_0005 = nc_lm.variables['lat'][:].data
    lon_lm_0005 = nc_lm.variables['lon'][:].data
    print('lat_lm_0005,lon_lm_0005',lat_lm_0005,lon_lm_0005)
    print(len(lat_lm_0005),len(lon_lm_0005))
    
    ix1,ix2 = np.argmin(abs(lon_lm_0005-np.min(lon_Modis))),np.argmin(abs(lon_lm_0005-np.max(lon_Modis))) + 1
    ix_min_lm,ix_max_lm = int(np.min([ix1, ix2])),int(np.max([ix1, ix2]))
    iy1,iy2 = np.argmin(abs(lat_lm_0005-np.min(lat_Modis))) + 1,np.argmin(abs(lat_lm_0005-np.max(lat_Modis)))
    iy_min_lm,iy_max_lm = int(np.min([iy1, iy2])),int(np.max([iy1, iy2]))

    lat_lm_0005 = lat_lm_0005[iy_min_lm:iy_max_lm]
    lon_lm_0005 = lon_lm_0005[ix_min_lm:ix_max_lm]
    
    
    land_mask = nc_lm.variables['land_mask'][iy_min_lm:iy_max_lm,ix_min_lm:ix_max_lm].data #
    ind_lm = np.where(np.isnan(land_mask))
    land_mask[ind_lm] = 0

    #Topography
    nc_to = netCDF4.Dataset(pathLoadTopography)
    topography = nc_to.variables['topography'][iy_min_Modis:iy_max_Modis,ix_min_Modis:ix_max_Modis].data
    topography[np.where(np.isnan(topography))] = 0


    # %%
    nc_lulc = netCDF4.Dataset(pathLoadLULC, 'r')
    keys = []
    for k in nc_lulc.variables.keys():
        keys.append(k)
    keys.remove('lon')
    keys.remove('lat')
    lon_lulc = nc_lulc.variables['lon'][:].data
    lat_lulc = nc_lulc.variables['lat'][:].data

    lulc_keys = keys

    ix1,ix2 = np.argmin(abs(lon_lulc-np.min(lon_Modis))),np.argmin(abs(lon_lulc-np.max(lon_Modis))) + 1
    ix_min_lulc,ix_max_lulc = int(np.min([ix1, ix2])),int(np.max([ix1, ix2]))
    iy1,iy2 = np.argmin(abs(lat_lulc-np.min(lat_Modis))) + 1,np.argmin(abs(lat_lulc-np.max(lat_Modis)))
    iy_min_lulc,iy_max_lulc = int(np.min([iy1, iy2]))-1,int(np.max([iy1, iy2]))+1

    
    lat_lulc = lat_lulc[iy_min_lulc:iy_max_lulc][::-1]
    lon_lulc = lon_lulc[ix_min_lulc:ix_max_lulc]

    lulc = np.zeros((len(lat_lulc), len(lon_lulc),len(keys)))

    for k,i in zip(keys,range(len(keys))):
        lulc[:,:,i] = np.flipud(nc_lulc.variables[k][iy_min_lulc:iy_max_lulc,ix_min_lulc:ix_max_lulc].data)
    
    # %%
    cosSZA = helpers.GetSZAMap(pathLoadSZA, xLims, yLims, dates, ddX_Modis)
    cosSZA = np.moveaxis(cosSZA,-1,0)


    # %%
    # get mean and standard deviation that are precalculated by a helpers function
    df_m_std = helpers.GetMeanAndStd(path=pathLoadMeanStdInd)

    sif = (sif-df_m_std.loc['mean', 'sif'])/df_m_std.loc['std', 'sif']
    sif_dc_feat = (sif_dc_feat-0.13955191)/0.22630583
    dcCorr_feat = (dcCorr_feat-0.38616973)/0.11590124
    nirv = (nirv - df_m_std.loc['mean', 'nirv'])/df_m_std.loc['std', 'nirv']
    kNDVI = (kNDVI - df_m_std.loc['mean', 'kNDVI'])/df_m_std.loc['std', 'kNDVI']
    evi = (evi - df_m_std.loc['mean', 'evi'])/df_m_std.loc['std', 'evi']
    ndvi = (ndvi - df_m_std.loc['mean', 'ndvi'])/df_m_std.loc['std', 'ndvi']
    nir = (nir - df_m_std.loc['mean', 'nir'])/df_m_std.loc['std', 'nir']
    red = (red - df_m_std.loc['mean', 'red'])/df_m_std.loc['std', 'red']
    blue = (blue - df_m_std.loc['mean', 'blue'])/df_m_std.loc['std', 'blue']
    green = (green - df_m_std.loc['mean', 'green'])/df_m_std.loc['std', 'green']
    swir1 = (swir1 - df_m_std.loc['mean', 'swir1'])/df_m_std.loc['std', 'swir1']  
    swir2 = (swir2 - df_m_std.loc['mean', 'swir2'])/df_m_std.loc['std', 'swir2']  
    swir3 = (swir3 - df_m_std.loc['mean', 'swir3'])/df_m_std.loc['std', 'swir3']  
    cosSZA = (cosSZA - df_m_std.loc['mean', 'cosSZA'])/df_m_std.loc['std', 'cosSZA']
    temperature = (temperature-df_m_std.loc['mean', 'temperature'])/df_m_std.loc['std', 'temperature']
    precipitation = (precipitation-df_m_std.loc['mean', 'precipitation'])/df_m_std.loc['std', 'precipitation']
    temperature_del = (temperature_del-df_m_std.loc['mean', 'temperature'])/df_m_std.loc['std', 'temperature']
    precipitation_del = (precipitation_del-df_m_std.loc['mean', 'precipitation'])/df_m_std.loc['std', 'precipitation']
    ssm = (ssm-df_m_std.loc['mean', 'ssm'])/df_m_std.loc['std', 'ssm']
    susm = (susm-df_m_std.loc['mean', 'susm'])/df_m_std.loc['std', 'susm']
    lulc = (lulc-df_m_std.loc['mean', 'lulc'])/df_m_std.loc['std', 'lulc']
    land_mask = (land_mask-df_m_std.loc['mean', 'land_mask'])/df_m_std.loc['std', 'land_mask']
    topography = (topography-df_m_std.loc['mean', 'topography'])/df_m_std.loc['std', 'topography']


    if DATA_OPTION == 1:
      
        keys = ['LR_SIF', 'NIRv','kNDVI','EVI','NDVI', 'NIR', 'RED', 'Blue', 'Green', 'SWIR1', 'SWIR2','SWIR3', 'cosSZA', 'temperature', 'precipitation', 'temperature_delay', 'precipitation_delay', 'ssm', 'susm','land_mask', 'topography', 'LULC_Others', 'LULC_ENF', 'LULC_EBF', 'LULC_DNF', 'LULC_DBF', 'LULC_Mixed Forest', 'LULC_Unknown Forest', 'LULC_Shrubland', 'LULC_Grassland', 'LULC_Cropland', 'LULC_Wetland']
        y_size, x_size = 5, 4
        inp = np.zeros((len(df_oco2), y_size, x_size, len(keys)))
        ocos = np.zeros((len(df_oco2), y_size, x_size))
        inp[:] = np.nan
        #ocos[:] = np.nan
        ys, xs = [], []
        #create training samples
        for N_oco in range(len(df_oco2)):
            #create mask for sif
            step = helpers.nearest_date(df_oco2['date_UTC'][N_oco],dates)
            r = sif.shape
            #print(r)
            #print('t:',r[0],'x:',r[2],'y:',r[1])    
            x_min, x_max, y_min, y_max = helpers.MaskGenerator(lon_modis_0005, lat_modis_0005, df=df_oco2.iloc[N_oco,:])
            Masked_sif = sif[step, y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_sif_dc_feat = sif_dc_feat[step, y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_dcCorr_feat = dcCorr_feat[step, y_min:y_min+y_size,x_min:x_min+x_size]
            #print('shape of masked_sif: ',Masked_sif.shape)
            #print('x_min, x_max, y_min, y_max',x_min, x_max, y_min, y_max)
            
            #oco target
            oco = Masked_sif.copy()
            oco.fill(df_oco2['Daily_SIF_740nm'][N_oco]) #ask
            #create mask for modis
            #x_min, x_max, y_min, y_max = helpers.MaskGenerator(xLims=np.array([0, 20]), yLims=np.array([40, 50]), Nx=ix_max_Modis-ix_min_Modis, Ny=iy_max_Modis-iy_min_Modis, df=df_oco2.iloc[N_oco,:])

            Masked_nir = nir[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_red = red[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_blue = blue[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_green = green[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_swir1 = swir1[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_swir2 = swir2[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_swir3 = swir3[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_nirv = nirv[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_kNDVI = kNDVI[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_evi = evi[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_ndvi = ndvi[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_cosSZA = cosSZA[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_temperature = temperature[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_precipitation = precipitation[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_temperature_del = temperature_del[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_precipitation_del = precipitation_del[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_ssm = ssm[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_susm = susm[step,y_min:y_min+y_size,x_min:x_min+x_size]
            

            
            y, x = Masked_sif.shape[0], Masked_sif.shape[1]
            ys.append(y)
            xs.append(x)
            #print(N_oco,y, x)

            inp[N_oco,0:y,0:x,0] = Masked_sif
            inp[N_oco,0:y,0:x,1] = Masked_nirv
            inp[N_oco,0:y,0:x,2] = Masked_kNDVI
            inp[N_oco,0:y,0:x,3] = Masked_evi
            inp[N_oco,0:y,0:x,4] = Masked_ndvi
            inp[N_oco,0:y,0:x,5] = Masked_nir
            inp[N_oco,0:y,0:x,6] = Masked_red
            inp[N_oco,0:y,0:x,7] = Masked_blue
            inp[N_oco,0:y,0:x,8] = Masked_green
            inp[N_oco,0:y,0:x,9] = Masked_swir1
            inp[N_oco,0:y,0:x,10] = Masked_swir2
            inp[N_oco,0:y,0:x,11] = Masked_swir3
            inp[N_oco,0:y,0:x,12] = Masked_cosSZA
            inp[N_oco,0:y,0:x,13] = Masked_temperature
            inp[N_oco,0:y,0:x,14] = Masked_precipitation
            inp[N_oco,0:y,0:x,15] = Masked_temperature_del
            inp[N_oco,0:y,0:x,16] = Masked_precipitation_del
            inp[N_oco,0:y,0:x,17] = Masked_ssm
            inp[N_oco,0:y,0:x,18] = Masked_susm            
            #
            inp[N_oco,0:y,0:x,19] = land_mask[y_min:y_min+y_size,x_min:x_min+x_size]
            inp[N_oco,0:y,0:x,20] = topography[y_min:y_min+y_size,x_min:x_min+x_size]            

            #for l in lulc_keys: #ask
            #    keys.append('LULC_'+l)

            inp[N_oco,0:y,0:x,21:32] = lulc[y_min:y_min+y_size,x_min:x_min+x_size,:] #shape: (lat,lon,11)
            
            
            ocos[N_oco,::] = oco.copy()
        Ys, Xs = np.array([ys]), np.array([xs])
        
        
        
        #create testing samples

    elif DATA_OPTION == 2: #no ssm ssum
      
        keys = ['LR_SIF', 'NIRv','kNDVI','EVI','NDVI', 'NIR', 'RED', 'Blue', 'Green', 'SWIR1', 'SWIR2','SWIR3', 'cosSZA', 'temperature', 'precipitation', 'temperature_delay', 'precipitation_delay','land_mask', 'topography', 'LULC_Others', 'LULC_ENF', 'LULC_EBF', 'LULC_DNF', 'LULC_DBF', 'LULC_Mixed Forest', 'LULC_Unknown Forest', 'LULC_Shrubland', 'LULC_Grassland', 'LULC_Cropland', 'LULC_Wetland']
        y_size, x_size = 5, 4
        inp = np.zeros((len(df_oco2), y_size, x_size, len(keys)))
        ocos = np.zeros((len(df_oco2), y_size, x_size))
        inp[:] = np.nan
        #ocos[:] = np.nan
        ys, xs = [], []
        #create training samples
        for N_oco in range(len(df_oco2)):
            #create mask for sif
            step = helpers.nearest_date(df_oco2['date_UTC'][N_oco],dates)
            r = sif.shape
            #print(r)
            #print('t:',r[0],'x:',r[2],'y:',r[1])    
            x_min, x_max, y_min, y_max = helpers.MaskGenerator(lon_modis_0005, lat_modis_0005, df=df_oco2.iloc[N_oco,:])
            Masked_sif = sif[step, y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_sif_dc_feat = sif_dc_feat[step, y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_dcCorr_feat = dcCorr_feat[step, y_min:y_min+y_size,x_min:x_min+x_size]
            #print('shape of masked_sif: ',Masked_sif.shape)
            #print('x_min, x_max, y_min, y_max',x_min, x_max, y_min, y_max)
            
            #oco target
            oco = Masked_sif.copy()
            oco.fill(df_oco2.iloc[N_oco,:][1])
            #create mask for modis
            #x_min, x_max, y_min, y_max = helpers.MaskGenerator(xLims=np.array([0, 20]), yLims=np.array([40, 50]), Nx=ix_max_Modis-ix_min_Modis, Ny=iy_max_Modis-iy_min_Modis, df=df_oco2.iloc[N_oco,:])

            Masked_nir = nir[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_red = red[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_blue = blue[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_green = green[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_swir1 = swir1[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_swir2 = swir2[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_swir3 = swir3[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_nirv = nirv[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_kNDVI = kNDVI[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_evi = evi[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_ndvi = ndvi[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_cosSZA = cosSZA[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_temperature = temperature[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_precipitation = precipitation[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_temperature_del = temperature_del[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_precipitation_del = precipitation_del[step,y_min:y_min+y_size,x_min:x_min+x_size]
            #Masked_ssm = ssm[step,y_min:y_min+y_size,x_min:x_min+x_size]
            #Masked_susm = susm[step,y_min:y_min+y_size,x_min:x_min+x_size]
            

            
            y, x = Masked_sif.shape[0], Masked_sif.shape[1]
            ys.append(y)
            xs.append(x)
            #print(N_oco,y, x)

            inp[N_oco,0:y,0:x,0] = Masked_sif
            inp[N_oco,0:y,0:x,1] = Masked_nirv
            inp[N_oco,0:y,0:x,2] = Masked_kNDVI
            inp[N_oco,0:y,0:x,3] = Masked_evi
            inp[N_oco,0:y,0:x,4] = Masked_ndvi
            inp[N_oco,0:y,0:x,5] = Masked_nir
            inp[N_oco,0:y,0:x,6] = Masked_red
            inp[N_oco,0:y,0:x,7] = Masked_blue
            inp[N_oco,0:y,0:x,8] = Masked_green
            inp[N_oco,0:y,0:x,9] = Masked_swir1
            inp[N_oco,0:y,0:x,10] = Masked_swir2
            inp[N_oco,0:y,0:x,11] = Masked_swir3
            inp[N_oco,0:y,0:x,12] = Masked_cosSZA
            inp[N_oco,0:y,0:x,13] = Masked_temperature
            inp[N_oco,0:y,0:x,14] = Masked_precipitation
            inp[N_oco,0:y,0:x,15] = Masked_temperature_del
            inp[N_oco,0:y,0:x,16] = Masked_precipitation_del
            #inp[N_oco,0:y,0:x,17] = Masked_ssm
            #inp[N_oco,0:y,0:x,18] = Masked_susm            
            #
            inp[N_oco,0:y,0:x,17] = land_mask[y_min:y_min+y_size,x_min:x_min+x_size]
            inp[N_oco,0:y,0:x,18] = topography[y_min:y_min+y_size,x_min:x_min+x_size]            

            #for l in lulc_keys: #ask
            #    keys.append('LULC_'+l)

            inp[N_oco,0:y,0:x,19:30] = lulc[y_min:y_min+y_size,x_min:x_min+x_size,:] #shape: (lat,lon,11)
            
            
            ocos[N_oco,0:y,0:x] = oco
        Ys, Xs = np.array([ys]), np.array([xs])
        


    elif DATA_OPTION == 3: #no ssm ssum, 'topography'
      
        keys = ['LR_SIF', 'NIRv','kNDVI','EVI','NDVI', 'NIR', 'RED', 'Blue', 'Green', 'SWIR1', 'SWIR2','SWIR3', 'cosSZA', 'temperature', 'precipitation', 'temperature_delay', 'precipitation_delay','land_mask', 'LULC_Others', 'LULC_ENF', 'LULC_EBF', 'LULC_DNF', 'LULC_DBF', 'LULC_Mixed Forest', 'LULC_Unknown Forest', 'LULC_Shrubland', 'LULC_Grassland', 'LULC_Cropland', 'LULC_Wetland']
        y_size, x_size = 5, 4
        inp = np.zeros((len(df_oco2), y_size, x_size, len(keys)))
        ocos = np.zeros((len(df_oco2), y_size, x_size))
        inp[:] = np.nan
        #ocos[:] = np.nan
        ys, xs = [], []
        #create training samples
        for N_oco in range(len(df_oco2)):
            #create mask for sif
            step = helpers.nearest_date(df_oco2.iloc[N_oco,:]['date_UTC'],dates)
            r = sif.shape
            #print(r)
            #print('t:',r[0],'x:',r[2],'y:',r[1])    
            x_min, x_max, y_min, y_max = helpers.MaskGenerator(lon_modis_0005, lat_modis_0005, df=df_oco2.iloc[N_oco,:])
            Masked_sif = sif[step, y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_sif_dc_feat = sif_dc_feat[step, y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_dcCorr_feat = dcCorr_feat[step, y_min:y_min+y_size,x_min:x_min+x_size]
            #print('shape of masked_sif: ',Masked_sif.shape)
            #print('x_min, x_max, y_min, y_max',x_min, x_max, y_min, y_max)
            
            #oco target
            oco = Masked_sif.copy()
            oco.fill(df_oco2.iloc[N_oco,:][1])
            #create mask for modis
            #x_min, x_max, y_min, y_max = helpers.MaskGenerator(xLims=np.array([0, 20]), yLims=np.array([40, 50]), Nx=ix_max_Modis-ix_min_Modis, Ny=iy_max_Modis-iy_min_Modis, df=df_oco2.iloc[N_oco,:])

            Masked_nir = nir[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_red = red[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_blue = blue[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_green = green[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_swir1 = swir1[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_swir2 = swir2[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_swir3 = swir3[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_nirv = nirv[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_kNDVI = kNDVI[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_evi = evi[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_ndvi = ndvi[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_cosSZA = cosSZA[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_temperature = temperature[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_precipitation = precipitation[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_temperature_del = temperature_del[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_precipitation_del = precipitation_del[step,y_min:y_min+y_size,x_min:x_min+x_size]
            #Masked_ssm = ssm[step,y_min:y_min+y_size,x_min:x_min+x_size]
            #Masked_susm = susm[step,y_min:y_min+y_size,x_min:x_min+x_size]
            

            
            y, x = Masked_sif.shape[0], Masked_sif.shape[1]
            ys.append(y)
            xs.append(x)
            #print(N_oco,y, x)

            inp[N_oco,0:y,0:x,0] = Masked_sif
            inp[N_oco,0:y,0:x,1] = Masked_nirv
            inp[N_oco,0:y,0:x,2] = Masked_kNDVI
            inp[N_oco,0:y,0:x,3] = Masked_evi
            inp[N_oco,0:y,0:x,4] = Masked_ndvi
            inp[N_oco,0:y,0:x,5] = Masked_nir
            inp[N_oco,0:y,0:x,6] = Masked_red
            inp[N_oco,0:y,0:x,7] = Masked_blue
            inp[N_oco,0:y,0:x,8] = Masked_green
            inp[N_oco,0:y,0:x,9] = Masked_swir1
            inp[N_oco,0:y,0:x,10] = Masked_swir2
            inp[N_oco,0:y,0:x,11] = Masked_swir3
            inp[N_oco,0:y,0:x,12] = Masked_cosSZA
            inp[N_oco,0:y,0:x,13] = Masked_temperature
            inp[N_oco,0:y,0:x,14] = Masked_precipitation
            inp[N_oco,0:y,0:x,15] = Masked_temperature_del
            inp[N_oco,0:y,0:x,16] = Masked_precipitation_del
            #inp[N_oco,0:y,0:x,17] = Masked_ssm
            #inp[N_oco,0:y,0:x,18] = Masked_susm            
            #
            inp[N_oco,0:y,0:x,17] = land_mask[y_min:y_min+y_size,x_min:x_min+x_size]
            #inp[N_oco,0:y,0:x,18] = topography[y_min:y_min+y_size,x_min:x_min+x_size]            

            #for l in lulc_keys: #ask
            #    keys.append('LULC_'+l)

            inp[N_oco,0:y,0:x,18:29] = lulc[y_min:y_min+y_size,x_min:x_min+x_size,:] #shape: (lat,lon,11)
            
            
            ocos[N_oco,0:y,0:x] = oco
        Ys, Xs = np.array([ys]), np.array([xs])
        

    
    elif DATA_OPTION == 4: #no ssm ssum, 'topography', 'temperature', 'precipitation', 'temperature_delay', 'precipitation_delay',
      
        keys = ['LR_SIF', 'NIRv','kNDVI','EVI','NDVI', 'NIR', 'RED', 'Blue', 'Green', 'SWIR1', 'SWIR2','SWIR3', 'cosSZA', 'land_mask', 'LULC_Others', 'LULC_ENF', 'LULC_EBF', 'LULC_DNF', 'LULC_DBF', 'LULC_Mixed Forest', 'LULC_Unknown Forest', 'LULC_Shrubland', 'LULC_Grassland', 'LULC_Cropland', 'LULC_Wetland']
        y_size, x_size = 5, 4
        inp = np.zeros((len(df_oco2), y_size, x_size, len(keys)))
        ocos = np.zeros((len(df_oco2), y_size, x_size))
        inp[:] = np.nan
        #ocos[:] = np.nan
        ys, xs = [], []
        #create training samples
        for N_oco in range(len(df_oco2)):
            #create mask for sif
            step = helpers.nearest_date(df_oco2.iloc[N_oco,:]['date_UTC'],dates)
            r = sif.shape
            #print(r)
            #print('t:',r[0],'x:',r[2],'y:',r[1])    
            x_min, x_max, y_min, y_max = helpers.MaskGenerator(lon_modis_0005, lat_modis_0005, df=df_oco2.iloc[N_oco,:])
            Masked_sif = sif[step, y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_sif_dc_feat = sif_dc_feat[step, y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_dcCorr_feat = dcCorr_feat[step, y_min:y_min+y_size,x_min:x_min+x_size]
            #print('shape of masked_sif: ',Masked_sif.shape)
            #print('x_min, x_max, y_min, y_max',x_min, x_max, y_min, y_max)
            
            #oco target
            oco = Masked_sif.copy()
            oco.fill(df_oco2.iloc[N_oco,:][1])
            #create mask for modis
            #x_min, x_max, y_min, y_max = helpers.MaskGenerator(xLims=np.array([0, 20]), yLims=np.array([40, 50]), Nx=ix_max_Modis-ix_min_Modis, Ny=iy_max_Modis-iy_min_Modis, df=df_oco2.iloc[N_oco,:])

            Masked_nir = nir[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_red = red[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_blue = blue[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_green = green[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_swir1 = swir1[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_swir2 = swir2[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_swir3 = swir3[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_nirv = nirv[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_kNDVI = kNDVI[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_evi = evi[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_ndvi = ndvi[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_cosSZA = cosSZA[step,y_min:y_min+y_size,x_min:x_min+x_size]
            #Masked_temperature = temperature[step,y_min:y_min+y_size,x_min:x_min+x_size]
            #Masked_precipitation = precipitation[step,y_min:y_min+y_size,x_min:x_min+x_size]
            #Masked_temperature_del = temperature_del[step,y_min:y_min+y_size,x_min:x_min+x_size]
            #Masked_precipitation_del = precipitation_del[step,y_min:y_min+y_size,x_min:x_min+x_size]
            #Masked_ssm = ssm[step,y_min:y_min+y_size,x_min:x_min+x_size]
            #Masked_susm = susm[step,y_min:y_min+y_size,x_min:x_min+x_size]
            

            
            y, x = Masked_sif.shape[0], Masked_sif.shape[1]
            ys.append(y)
            xs.append(x)
            #print(N_oco,y, x)

            inp[N_oco,0:y,0:x,0] = Masked_sif
            inp[N_oco,0:y,0:x,1] = Masked_nirv
            inp[N_oco,0:y,0:x,2] = Masked_kNDVI
            inp[N_oco,0:y,0:x,3] = Masked_evi
            inp[N_oco,0:y,0:x,4] = Masked_ndvi
            inp[N_oco,0:y,0:x,5] = Masked_nir
            inp[N_oco,0:y,0:x,6] = Masked_red
            inp[N_oco,0:y,0:x,7] = Masked_blue
            inp[N_oco,0:y,0:x,8] = Masked_green
            inp[N_oco,0:y,0:x,9] = Masked_swir1
            inp[N_oco,0:y,0:x,10] = Masked_swir2
            inp[N_oco,0:y,0:x,11] = Masked_swir3
            inp[N_oco,0:y,0:x,12] = Masked_cosSZA
            #inp[N_oco,0:y,0:x,13] = Masked_temperature
            #inp[N_oco,0:y,0:x,14] = Masked_precipitation
            #inp[N_oco,0:y,0:x,15] = Masked_temperature_del
            #inp[N_oco,0:y,0:x,16] = Masked_precipitation_del
            #inp[N_oco,0:y,0:x,17] = Masked_ssm
            #inp[N_oco,0:y,0:x,18] = Masked_susm            
            #
            inp[N_oco,0:y,0:x,13] = land_mask[y_min:y_min+y_size,x_min:x_min+x_size]
            #inp[N_oco,0:y,0:x,18] = topography[y_min:y_min+y_size,x_min:x_min+x_size]            

            #for l in lulc_keys: #ask
            #    keys.append('LULC_'+l)

            inp[N_oco,0:y,0:x,14:25] = lulc[y_min:y_min+y_size,x_min:x_min+x_size,:] #shape: (lat,lon,11)
            
            
            ocos[N_oco,0:y,0:x] = oco
        Ys, Xs = np.array([ys]), np.array([xs])
        


    elif DATA_OPTION == 5: #no ssm ssum, 'topography', 'temperature', 'precipitation', 'temperature_delay', 'precipitation_delay','kNDVI','EVI','NDVI',
      
        keys = ['LR_SIF', 'NIRv', 'NIR', 'RED', 'Blue', 'Green', 'SWIR1', 'SWIR2','SWIR3', 'cosSZA', 'land_mask', 'LULC_Others', 'LULC_ENF', 'LULC_EBF', 'LULC_DNF', 'LULC_DBF', 'LULC_Mixed Forest', 'LULC_Unknown Forest', 'LULC_Shrubland', 'LULC_Grassland', 'LULC_Cropland', 'LULC_Wetland']
        y_size, x_size = 5, 4
        inp = np.zeros((len(df_oco2), y_size, x_size, len(keys)))
        ocos = np.zeros((len(df_oco2), y_size, x_size))
        inp[:] = np.nan
        #ocos[:] = np.nan
        ys, xs = [], []
        #create training samples
        for N_oco in range(len(df_oco2)):
            #create mask for sif
            step = helpers.nearest_date(df_oco2.iloc[N_oco,:]['date_UTC'],dates)
            r = sif.shape
            #print(r)
            #print('t:',r[0],'x:',r[2],'y:',r[1])    
            x_min, x_max, y_min, y_max = helpers.MaskGenerator(lon_modis_0005, lat_modis_0005, df=df_oco2.iloc[N_oco,:])
            Masked_sif = sif[step, y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_sif_dc_feat = sif_dc_feat[step, y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_dcCorr_feat = dcCorr_feat[step, y_min:y_min+y_size,x_min:x_min+x_size]
            #print('shape of masked_sif: ',Masked_sif.shape)
            #print('x_min, x_max, y_min, y_max',x_min, x_max, y_min, y_max)
            
            #oco target
            oco = Masked_sif.copy()
            oco.fill(df_oco2.iloc[N_oco,:][1])
            #create mask for modis
            #x_min, x_max, y_min, y_max = helpers.MaskGenerator(xLims=np.array([0, 20]), yLims=np.array([40, 50]), Nx=ix_max_Modis-ix_min_Modis, Ny=iy_max_Modis-iy_min_Modis, df=df_oco2.iloc[N_oco,:])

            Masked_nir = nir[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_red = red[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_blue = blue[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_green = green[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_swir1 = swir1[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_swir2 = swir2[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_swir3 = swir3[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_nirv = nirv[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_kNDVI = kNDVI[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_evi = evi[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_ndvi = ndvi[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_cosSZA = cosSZA[step,y_min:y_min+y_size,x_min:x_min+x_size]
            #Masked_temperature = temperature[step,y_min:y_min+y_size,x_min:x_min+x_size]
            #Masked_precipitation = precipitation[step,y_min:y_min+y_size,x_min:x_min+x_size]
            #Masked_temperature_del = temperature_del[step,y_min:y_min+y_size,x_min:x_min+x_size]
            #Masked_precipitation_del = precipitation_del[step,y_min:y_min+y_size,x_min:x_min+x_size]
            #Masked_ssm = ssm[step,y_min:y_min+y_size,x_min:x_min+x_size]
            #Masked_susm = susm[step,y_min:y_min+y_size,x_min:x_min+x_size]
            

            
            y, x = Masked_sif.shape[0], Masked_sif.shape[1]
            ys.append(y)
            xs.append(x)
            #print(N_oco,y, x)

            inp[N_oco,0:y,0:x,0] = Masked_sif
            inp[N_oco,0:y,0:x,1] = Masked_nirv
            #inp[N_oco,0:y,0:x,2] = Masked_kNDVI
            #inp[N_oco,0:y,0:x,3] = Masked_evi
            #inp[N_oco,0:y,0:x,4] = Masked_ndvi
            inp[N_oco,0:y,0:x,2] = Masked_nir
            inp[N_oco,0:y,0:x,3] = Masked_red
            inp[N_oco,0:y,0:x,4] = Masked_blue
            inp[N_oco,0:y,0:x,5] = Masked_green
            inp[N_oco,0:y,0:x,6] = Masked_swir1
            inp[N_oco,0:y,0:x,7] = Masked_swir2
            inp[N_oco,0:y,0:x,8] = Masked_swir3
            inp[N_oco,0:y,0:x,9] = Masked_cosSZA
            #inp[N_oco,0:y,0:x,13] = Masked_temperature
            #inp[N_oco,0:y,0:x,14] = Masked_precipitation
            #inp[N_oco,0:y,0:x,15] = Masked_temperature_del
            #inp[N_oco,0:y,0:x,16] = Masked_precipitation_del
            #inp[N_oco,0:y,0:x,17] = Masked_ssm
            #inp[N_oco,0:y,0:x,18] = Masked_susm            
            #
            inp[N_oco,0:y,0:x,10] = land_mask[y_min:y_min+y_size,x_min:x_min+x_size]
            #inp[N_oco,0:y,0:x,18] = topography[y_min:y_min+y_size,x_min:x_min+x_size]            

            #for l in lulc_keys: #ask
            #    keys.append('LULC_'+l)

            inp[N_oco,0:y,0:x,11:22] = lulc[y_min:y_min+y_size,x_min:x_min+x_size,:] #shape: (lat,lon,11)
            
            
            ocos[N_oco,0:y,0:x] = oco
        Ys, Xs = np.array([ys]), np.array([xs])
        


    elif DATA_OPTION == 6: #no ssm ssum, 'topography', 'temperature', 'precipitation', 'temperature_delay', 'precipitation_delay','kNDVI','EVI','NDVI',, 'LULC_Others', 'LULC_ENF', 'LULC_EBF', 'LULC_DNF', 'LULC_DBF', 'LULC_Mixed Forest', 'LULC_Unknown Forest', 'LULC_Shrubland', 'LULC_Grassland', 'LULC_Cropland', 'LULC_Wetland'
      
        keys = ['LR_SIF', 'NIRv', 'NIR', 'RED', 'Blue', 'Green', 'SWIR1', 'SWIR2','SWIR3', 'cosSZA', 'land_mask']
        y_size, x_size = 5, 4
        inp = np.zeros((len(df_oco2), y_size, x_size, len(keys)))
        ocos = np.zeros((len(df_oco2), y_size, x_size))
        inp[:] = np.nan
        #ocos[:] = np.nan
        ys, xs = [], []
        #create training samples
        for N_oco in range(len(df_oco2)):
            #create mask for sif
            step = helpers.nearest_date(df_oco2.iloc[N_oco,:]['date_UTC'],dates)
            r = sif.shape
            #print(r)
            #print('t:',r[0],'x:',r[2],'y:',r[1])    
            x_min, x_max, y_min, y_max = helpers.MaskGenerator(lon_modis_0005, lat_modis_0005, df=df_oco2.iloc[N_oco,:])
            Masked_sif = sif[step, y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_sif_dc_feat = sif_dc_feat[step, y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_dcCorr_feat = dcCorr_feat[step, y_min:y_min+y_size,x_min:x_min+x_size]
            #print('shape of masked_sif: ',Masked_sif.shape)
            #print('x_min, x_max, y_min, y_max',x_min, x_max, y_min, y_max)
            
            #oco target
            oco = Masked_sif.copy()
            oco.fill(df_oco2.iloc[N_oco,:][1])
            #create mask for modis
            #x_min, x_max, y_min, y_max = helpers.MaskGenerator(xLims=np.array([0, 20]), yLims=np.array([40, 50]), Nx=ix_max_Modis-ix_min_Modis, Ny=iy_max_Modis-iy_min_Modis, df=df_oco2.iloc[N_oco,:])

            Masked_nir = nir[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_red = red[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_blue = blue[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_green = green[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_swir1 = swir1[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_swir2 = swir2[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_swir3 = swir3[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_nirv = nirv[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_kNDVI = kNDVI[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_evi = evi[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_ndvi = ndvi[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_cosSZA = cosSZA[step,y_min:y_min+y_size,x_min:x_min+x_size]
            #Masked_temperature = temperature[step,y_min:y_min+y_size,x_min:x_min+x_size]
            #Masked_precipitation = precipitation[step,y_min:y_min+y_size,x_min:x_min+x_size]
            #Masked_temperature_del = temperature_del[step,y_min:y_min+y_size,x_min:x_min+x_size]
            #Masked_precipitation_del = precipitation_del[step,y_min:y_min+y_size,x_min:x_min+x_size]
            #Masked_ssm = ssm[step,y_min:y_min+y_size,x_min:x_min+x_size]
            #Masked_susm = susm[step,y_min:y_min+y_size,x_min:x_min+x_size]
            

            
            y, x = Masked_sif.shape[0], Masked_sif.shape[1]
            ys.append(y)
            xs.append(x)
            #print(N_oco,y, x)

            inp[N_oco,0:y,0:x,0] = Masked_sif
            inp[N_oco,0:y,0:x,1] = Masked_nirv
            #inp[N_oco,0:y,0:x,2] = Masked_kNDVI
            #inp[N_oco,0:y,0:x,3] = Masked_evi
            #inp[N_oco,0:y,0:x,4] = Masked_ndvi
            inp[N_oco,0:y,0:x,2] = Masked_nir
            inp[N_oco,0:y,0:x,3] = Masked_red
            inp[N_oco,0:y,0:x,4] = Masked_blue
            inp[N_oco,0:y,0:x,5] = Masked_green
            inp[N_oco,0:y,0:x,6] = Masked_swir1
            inp[N_oco,0:y,0:x,7] = Masked_swir2
            inp[N_oco,0:y,0:x,8] = Masked_swir3
            inp[N_oco,0:y,0:x,9] = Masked_cosSZA
            #inp[N_oco,0:y,0:x,13] = Masked_temperature
            #inp[N_oco,0:y,0:x,14] = Masked_precipitation
            #inp[N_oco,0:y,0:x,15] = Masked_temperature_del
            #inp[N_oco,0:y,0:x,16] = Masked_precipitation_del
            #inp[N_oco,0:y,0:x,17] = Masked_ssm
            #inp[N_oco,0:y,0:x,18] = Masked_susm            
            #
            inp[N_oco,0:y,0:x,10] = land_mask[y_min:y_min+y_size,x_min:x_min+x_size]
            #inp[N_oco,0:y,0:x,18] = topography[y_min:y_min+y_size,x_min:x_min+x_size]            

            #for l in lulc_keys: #ask
            #    keys.append('LULC_'+l)

            #inp[N_oco,0:y,0:x,11:22] = lulc[y_min:y_min+y_size,x_min:x_min+x_size,:] #shape: (lat,lon,11)
            
            
            ocos[N_oco,0:y,0:x] = oco
        Ys, Xs = np.array([ys]), np.array([xs])
        


    elif DATA_OPTION == 7: #no ssm ssum, 'topography', 'temperature', 'precipitation', 'temperature_delay', 'precipitation_delay','kNDVI','EVI','NDVI',, 'LULC_Others', 'LULC_ENF', 'LULC_EBF', 'LULC_DNF', 'LULC_DBF', 'LULC_Mixed Forest', 'LULC_Unknown Forest', 'LULC_Shrubland', 'LULC_Grassland', 'LULC_Cropland', 'LULC_Wetland''NIR', 'RED', 'Blue', 'Green', 'SWIR1', 'SWIR2','SWIR3',
      
        keys = ['LR_SIF', 'NIRv',  'cosSZA', 'land_mask']
        y_size, x_size = 5, 4
        inp = np.zeros((len(df_oco2), y_size, x_size, len(keys)))
        ocos = np.zeros((len(df_oco2), y_size, x_size))
        inp[:] = np.nan
        #ocos[:] = np.nan
        ys, xs = [], []
        #create training samples
        for N_oco in range(len(df_oco2)):
            #create mask for sif
            step = helpers.nearest_date(df_oco2.iloc[N_oco,:]['date_UTC'],dates)
            r = sif.shape
            #print(r)
            #print('t:',r[0],'x:',r[2],'y:',r[1])    
            x_min, x_max, y_min, y_max = helpers.MaskGenerator(lon_modis_0005, lat_modis_0005, df=df_oco2.iloc[N_oco,:])
            Masked_sif = sif[step, y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_sif_dc_feat = sif_dc_feat[step, y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_dcCorr_feat = dcCorr_feat[step, y_min:y_min+y_size,x_min:x_min+x_size]
            #print('shape of masked_sif: ',Masked_sif.shape)
            #print('x_min, x_max, y_min, y_max',x_min, x_max, y_min, y_max)
            
            #oco target
            oco = Masked_sif.copy()
            oco.fill(df_oco2.iloc[N_oco,:][1])
            #create mask for modis
            #x_min, x_max, y_min, y_max = helpers.MaskGenerator(xLims=np.array([0, 20]), yLims=np.array([40, 50]), Nx=ix_max_Modis-ix_min_Modis, Ny=iy_max_Modis-iy_min_Modis, df=df_oco2.iloc[N_oco,:])

            Masked_nir = nir[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_red = red[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_blue = blue[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_green = green[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_swir1 = swir1[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_swir2 = swir2[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_swir3 = swir3[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_nirv = nirv[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_kNDVI = kNDVI[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_evi = evi[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_ndvi = ndvi[step,y_min:y_min+y_size,x_min:x_min+x_size]
            Masked_cosSZA = cosSZA[step,y_min:y_min+y_size,x_min:x_min+x_size]
            #Masked_temperature = temperature[step,y_min:y_min+y_size,x_min:x_min+x_size]
            #Masked_precipitation = precipitation[step,y_min:y_min+y_size,x_min:x_min+x_size]
            #Masked_temperature_del = temperature_del[step,y_min:y_min+y_size,x_min:x_min+x_size]
            #Masked_precipitation_del = precipitation_del[step,y_min:y_min+y_size,x_min:x_min+x_size]
            #Masked_ssm = ssm[step,y_min:y_min+y_size,x_min:x_min+x_size]
            #Masked_susm = susm[step,y_min:y_min+y_size,x_min:x_min+x_size]
            

            
            y, x = Masked_sif.shape[0], Masked_sif.shape[1]
            ys.append(y)
            xs.append(x)
            #print(N_oco,y, x)

            inp[N_oco,0:y,0:x,0] = Masked_sif
            inp[N_oco,0:y,0:x,1] = Masked_nirv
            #inp[N_oco,0:y,0:x,2] = Masked_kNDVI
            #inp[N_oco,0:y,0:x,3] = Masked_evi
            #inp[N_oco,0:y,0:x,4] = Masked_ndvi
            #inp[N_oco,0:y,0:x,2] = Masked_nir
            #inp[N_oco,0:y,0:x,3] = Masked_red
            #inp[N_oco,0:y,0:x,4] = Masked_blue
            #inp[N_oco,0:y,0:x,5] = Masked_green
            #inp[N_oco,0:y,0:x,6] = Masked_swir1
            #inp[N_oco,0:y,0:x,7] = Masked_swir2
            #inp[N_oco,0:y,0:x,8] = Masked_swir3
            inp[N_oco,0:y,0:x,2] = Masked_cosSZA
            #inp[N_oco,0:y,0:x,13] = Masked_temperature
            #inp[N_oco,0:y,0:x,14] = Masked_precipitation
            #inp[N_oco,0:y,0:x,15] = Masked_temperature_del
            #inp[N_oco,0:y,0:x,16] = Masked_precipitation_del
            #inp[N_oco,0:y,0:x,17] = Masked_ssm
            #inp[N_oco,0:y,0:x,18] = Masked_susm            
            #
            inp[N_oco,0:y,0:x,3] = land_mask[y_min:y_min+y_size,x_min:x_min+x_size]
            #inp[N_oco,0:y,0:x,18] = topography[y_min:y_min+y_size,x_min:x_min+x_size]            

            #for l in lulc_keys: #ask
            #    keys.append('LULC_'+l)

            #inp[N_oco,0:y,0:x,11:22] = lulc[y_min:y_min+y_size,x_min:x_min+x_size,:] #shape: (lat,lon,11)
            
            
            ocos[N_oco,0:y,0:x] = oco.copy()
        Ys, Xs = np.array([ys]), np.array([xs])
        

    
    else:
        0 == 0

    print('Number of nan in inp',np.count_nonzero(np.isnan(inp)))
    print('Number of nan in ocos',np.count_nonzero(np.isnan(ocos)))
        
    inp[np.where(np.isnan(inp))] = 0
    ocos[np.where(np.isnan(ocos))] = 0
    #ocos = (ocos-df_m_std.loc['mean', 'sif'])/df_m_std.loc['std', 'sif'] #standardize targets
    print('Number of nan in inp after process',np.count_nonzero(np.isnan(inp)))
    print('Number of nan in ocos after process',np.count_nonzero(np.isnan(ocos)))    
    ocos_all = np.count_nonzero(~np.isnan(ocos))
    
    print('shape of inp: ', inp.shape)
    print('shape of ocos: ', ocos.shape)
    print('np.mean(ocos): ', np.mean(ocos))
    print('np.std(ocos): ', np.std(ocos))
    ocos_0 = np.count_nonzero(ocos < 0) / ocos_all
    ocos_05 = np.count_nonzero(ocos < 0.5) / ocos_all
    ocos_10 = np.count_nonzero(ocos < 1.0) / ocos_all
    
    print('< 0: ', ocos_0)
    print('0 to 0.5: ', ocos_05 - ocos_0)
    print('0.5 to 1.0: ', ocos_10 - ocos_05)
    print('> 1.0: ', 1 - ocos_10)

    return inp, keys, ocos, dates, ti_days_since
    
    

def get_calc_500m_dataset(xLims=np.array([10,15]), yLims=np.array([45,50]), d_start=0, d_end=46, USE_MODEL_OUT=True,SCALE_FACTOR=10,DATA_OPTION=1):

    
            
    pathPref = '/mnt/mnt_folder/data/' 
    pathLoadLULC = pathPref + 'Gridded_LULC_E-10_50_N25_70_11bands.nc'
    pathLoadMeanStdInd = pathPref + 'Mean_Std_indices_16day.csv'
    pathLoadLandMask = pathPref + 'LandMask_0005_E-10_50_N25_70.nc'
    pathLoadTopography = pathPref + 'Topography_0005US_USGS_GMTED2010.nc'

    #16day
    pathLoadTropomi = pathPref + 'TROPOMI-SIF_04-2018--03-2021_16day_005deg.nc'
    pathLoadModis = pathPref + 'MODIS_04-2018-03-2021_CONUS_0005deg_16day_allBands_N40-56.8_E-10-30_interp.nc'
    pathLoadSZA = pathPref + 'cosSZA_2018-2021_16day_0005.csv'
    pathLoadSoMo = pathPref + 'SoMo_04-2018-03-2021_005deg_16day.nc'
    pathLoadERA5 = pathPref + 'ERA5_04-2018-03-2021_005deg_16day.nc'
    
    #oco footprints
    #df_oco2 = helpers.read_OCO(pathPref,yLims_test, xLims_test, yLims_=np.array([200]), xLims_=np.array([200]))
    

    #%%
    # get grids
    Tropo, Modis, Sen2 = helpers.GetGrids(xLims, yLims)
    ddX_Modis,ddY_Modis,lon_Modis,lat_Modis = Modis['ddX'],Modis['ddY'],Modis['lon'],Modis['lat']
    ddX_Tropo,ddY_Tropo,lon_Tropo,lat_Tropo = Tropo['ddX'],Tropo['ddY'],Tropo['lon'],Tropo['lat']
    lonM, latM = np.meshgrid(lon_Modis, lat_Modis[::-1])

    gridded = np.zeros((d_end-d_start,lonM.shape[0],lonM.shape[1]))

    print('gridded.shape: ', gridded.shape)
    #revised
    y_size, x_size = 1, 1000
    xLims_arr = np.arange(xLims.min(),xLims.max(),x_size*0.005)  #4*0.005=0.020
    yLims_arr = np.arange(yLims.min(),yLims.max(),y_size*0.005) #5*0.005=0.025

    
    xlims_gridded = np.arange(0,gridded.shape[2]+1,x_size) #1/.020=50
    ylims_gridded = np.arange(0,gridded.shape[1]+1,y_size)[::-1] #1/0.025=40

    print('xlims_gridded: ', xlims_gridded)
    print('ylims_gridded: ', ylims_gridded)
    print('len(xLims_arr)-1: ', len(xLims_arr)-1)
    print('len(yLims_arr)-1: ', len(yLims_arr)-1)
    # %%
    nc_tropomi = netCDF4.Dataset(pathLoadTropomi)
    lon, lat = nc_tropomi.variables['lon'][:].data, nc_tropomi.variables['lat'][:].data

    ti_days_since = nc_tropomi.variables['time'][d_start:d_end].data
    dates = helpers.GetDatesFromDaysSince(ti_days_since)

    #%%
    # get indices of reduced regions
    ix1,ix2 = np.argmin(abs(lon-np.min(lon_Tropo))),np.argmin(abs(lon-np.max(lon_Tropo))) + 1
    ix_min,ix_max = int(np.min([ix1, ix2])),int(np.max([ix1, ix2]))
    iy1,iy2 = np.argmin(abs(lat-np.min(lat_Tropo))) + 1,np.argmin(abs(lat-np.max(lat_Tropo)))
    iy_min,iy_max = int(np.min([iy1, iy2])),int(np.max([iy1, iy2]))
  
    #%%
    lon_red,lat_red = lon[ix_min:ix_max],lat[iy_min:iy_max]
    # %%
    #SIF
    sif = nc_tropomi.variables['sif'][iy_min:iy_max,ix_min:ix_max,d_start:d_end].data
    sif = np.moveaxis(sif,-1,0)

    sif_dc_feat = nc_tropomi.variables['sif_dc'][iy_min:iy_max,ix_min:ix_max,d_start:d_end].data
    sif_dc_feat[np.where(nc_tropomi.variables['sif_dc'][iy_min:iy_max,ix_min:ix_max,d_start:d_end].mask)] = np.nan
    sif_dc_feat = np.moveaxis(sif_dc_feat,-1,0)

    dcCorr_feat = nc_tropomi.variables['dcCorr'][iy_min:iy_max,ix_min:ix_max,d_start:d_end].data
    dcCorr_feat[np.where(nc_tropomi.variables['dcCorr'][iy_min:iy_max,ix_min:ix_max,d_start:d_end].mask)] = np.nan
    dcCorr_feat = np.moveaxis(dcCorr_feat,-1,0)

    


    #resample SIF from 0.05° to 0.005°
    sif = np.repeat(sif, SCALE_FACTOR, axis=1)
    sif = np.repeat(sif, SCALE_FACTOR, axis=2)

    
    sif_dc_feat = np.repeat(sif_dc_feat, SCALE_FACTOR, axis=1)
    sif_dc_feat = np.repeat(sif_dc_feat, SCALE_FACTOR, axis=2)

    dcCorr_feat = np.repeat(dcCorr_feat, SCALE_FACTOR, axis=1)
    dcCorr_feat = np.repeat(dcCorr_feat, SCALE_FACTOR, axis=2)    
    


    #%%
    nc_modis = netCDF4.Dataset(pathLoadModis)
    lat_modis_0005 = nc_modis.variables['lat'][:].data
    lon_modis_0005 = nc_modis.variables['lon'][:].data
    ix1,ix2 = np.argmin(abs(lon_modis_0005-np.min(lon_Modis))),np.argmin(abs(lon_modis_0005-np.max(lon_Modis))) + 1
    ix_min_Modis,ix_max_Modis = int(np.min([ix1, ix2])),int(np.max([ix1, ix2]))
    iy1,iy2 = np.argmin(abs(lat_modis_0005-np.min(lat_Modis))) + 1,np.argmin(abs(lat_modis_0005-np.max(lat_Modis)))
    iy_min_Modis,iy_max_Modis = int(np.min([iy1, iy2])),int(np.max([iy1, iy2]))

    lat_modis_0005 = lat_modis_0005[iy_min_Modis:iy_max_Modis]
    lon_modis_0005 = lon_modis_0005[ix_min_Modis:ix_max_Modis]
    
    nir = nc_modis.variables['nir'][iy_min_Modis:iy_max_Modis,ix_min_Modis:ix_max_Modis,d_start:d_end].data
    red = nc_modis.variables['red'][iy_min_Modis:iy_max_Modis,ix_min_Modis:ix_max_Modis,d_start:d_end].data
    blue = nc_modis.variables['blue'][iy_min_Modis:iy_max_Modis,ix_min_Modis:ix_max_Modis,d_start:d_end].data
    green = nc_modis.variables['green'][iy_min_Modis:iy_max_Modis,ix_min_Modis:ix_max_Modis,d_start:d_end].data
    swir1 = nc_modis.variables['swir1'][iy_min_Modis:iy_max_Modis,ix_min_Modis:ix_max_Modis,d_start:d_end].data
    swir2 = nc_modis.variables['swir2'][iy_min_Modis:iy_max_Modis,ix_min_Modis:ix_max_Modis,d_start:d_end].data
    swir3 = nc_modis.variables['swir3'][iy_min_Modis:iy_max_Modis,ix_min_Modis:ix_max_Modis,d_start:d_end].data
    nir = np.moveaxis(nir,-1,0)
    red = np.moveaxis(red,-1,0)
    blue = np.moveaxis(blue,-1,0)
    green = np.moveaxis(green,-1,0)
    swir1 = np.moveaxis(swir1,-1,0)
    swir2 = np.moveaxis(swir2,-1,0)
    swir3 = np.moveaxis(swir3,-1,0)


    
    #VIs
    ndvi = (nir-red)/(nir+red)
    nirv = nir * ndvi
    kNDVI = np.tanh(ndvi**2)
    L,C1,C2,G = 1,6,7.5,2.5
    evi = G * ((nir - red)/(nir + C1*red - C2*blue + L))
    evi[np.where(evi<-20)] = np.nan
    evi[np.where(evi>20)] = np.nan


    #%%
    #SSM & SUSM
    nc_SoMo = netCDF4.Dataset(pathLoadSoMo)
    ssm, susm = nc_SoMo.variables['ssm'][d_start:d_end,iy_min:iy_max,ix_min:ix_max].data, nc_SoMo.variables['susm'][d_start:d_end,iy_min:iy_max,ix_min:ix_max].data
    #change resolution to 0.005° from 0.05°
    ssm = np.repeat(ssm, SCALE_FACTOR, axis=1)
    ssm = np.repeat(ssm, SCALE_FACTOR, axis=2)
    susm = np.repeat(susm, SCALE_FACTOR, axis=1)
    susm = np.repeat(susm, SCALE_FACTOR, axis=2)

    # %%
    nc_ERA5 = netCDF4.Dataset(pathLoadERA5, 'r')
    precipitation, temperature = nc_ERA5.variables['precipitation'][d_start:d_end,iy_min:iy_max,ix_min:ix_max].data, nc_ERA5.variables['meantemp_2m'][d_start:d_end,iy_min:iy_max,ix_min:ix_max].data
    precipitation = np.repeat(precipitation, SCALE_FACTOR, axis=1)
    precipitation = np.repeat(precipitation, SCALE_FACTOR, axis=2)
    temperature = np.repeat(temperature, SCALE_FACTOR, axis=1)
    temperature = np.repeat(temperature, SCALE_FACTOR, axis=2)

    try:
        precipitation_del, temperature_del = nc_ERA5.variables['precipitation'][d_start-1:d_end-1,iy_min:iy_max,ix_min:ix_max].data, nc_ERA5.variables['meantemp_2m'][d_start-1:d_end-1,iy_min:iy_max,ix_min:ix_max].data 
        precipitation_del = np.repeat(precipitation_del, SCALE_FACTOR, axis=1)
        precipitation_del = np.repeat(precipitation_del, SCALE_FACTOR, axis=2)
        temperature_del = np.repeat(temperature_del, SCALE_FACTOR, axis=1)
        temperature_del = np.repeat(temperature_del, SCALE_FACTOR, axis=2)
    except:
        precipitation_del, temperature_del = precipitation, temperature
        print('Precipitation and temperature delay did not work!')
        pass

    nc_ERA5.close()

    #land mask
    nc_lm = netCDF4.Dataset(pathLoadLandMask)
    
    lat_lm_0005 = nc_lm.variables['lat'][:].data
    lon_lm_0005 = nc_lm.variables['lon'][:].data
    print('lat_lm_0005,lon_lm_0005',lat_lm_0005,lon_lm_0005)
    print(len(lat_lm_0005),len(lon_lm_0005))
    
    ix1,ix2 = np.argmin(abs(lon_lm_0005-np.min(lon_Modis))),np.argmin(abs(lon_lm_0005-np.max(lon_Modis))) + 1
    ix_min_lm,ix_max_lm = int(np.min([ix1, ix2])),int(np.max([ix1, ix2]))
    iy1,iy2 = np.argmin(abs(lat_lm_0005-np.min(lat_Modis))) + 1,np.argmin(abs(lat_lm_0005-np.max(lat_Modis)))
    iy_min_lm,iy_max_lm = int(np.min([iy1, iy2])),int(np.max([iy1, iy2]))

    lat_lm_0005 = lat_lm_0005[iy_min_lm:iy_max_lm]
    lon_lm_0005 = lon_lm_0005[ix_min_lm:ix_max_lm]
    
    
    land_mask = nc_lm.variables['land_mask'][iy_min_lm:iy_max_lm,ix_min_lm:ix_max_lm].data #
    ind_lm = np.where(np.isnan(land_mask))
    land_mask[ind_lm] = 0

    #Topography
    nc_to = netCDF4.Dataset(pathLoadTopography)
    topography = nc_to.variables['topography'][iy_min_Modis:iy_max_Modis,ix_min_Modis:ix_max_Modis].data
    topography[np.where(np.isnan(topography))] = 0


    # %%
    nc_lulc = netCDF4.Dataset(pathLoadLULC, 'r')
    keys = []
    for k in nc_lulc.variables.keys():
        keys.append(k)
    keys.remove('lon')
    keys.remove('lat')
    lon_lulc = nc_lulc.variables['lon'][:].data
    lat_lulc = nc_lulc.variables['lat'][:].data

    lulc_keys = keys

    ix1,ix2 = np.argmin(abs(lon_lulc-np.min(lon_Modis))),np.argmin(abs(lon_lulc-np.max(lon_Modis))) + 1
    ix_min_lulc,ix_max_lulc = int(np.min([ix1, ix2])),int(np.max([ix1, ix2]))
    iy1,iy2 = np.argmin(abs(lat_lulc-np.min(lat_Modis))) + 1,np.argmin(abs(lat_lulc-np.max(lat_Modis)))
    iy_min_lulc,iy_max_lulc = int(np.min([iy1, iy2]))-1,int(np.max([iy1, iy2]))+1

    
    lat_lulc = lat_lulc[iy_min_lulc:iy_max_lulc][::-1]
    lon_lulc = lon_lulc[ix_min_lulc:ix_max_lulc]

    lulc = np.zeros((len(lat_lulc), len(lon_lulc),len(keys)))

    for k,i in zip(keys,range(len(keys))):
        lulc[:,:,i] = np.flipud(nc_lulc.variables[k][iy_min_lulc:iy_max_lulc,ix_min_lulc:ix_max_lulc].data)
    
    # %%
    cosSZA = helpers.GetSZAMap(pathLoadSZA, xLims, yLims, dates, ddX_Modis)
    cosSZA = np.moveaxis(cosSZA,-1,0)


    # %%
    # get mean and standard deviation that are precalculated by a helpers function
    df_m_std = helpers.GetMeanAndStd(path=pathLoadMeanStdInd)

    sif = (sif-df_m_std.loc['mean', 'sif'])/df_m_std.loc['std', 'sif']
    sif_dc_feat = (sif_dc_feat-0.13955191)/0.22630583
    dcCorr_feat = (dcCorr_feat-0.38616973)/0.11590124
    nirv = (nirv - df_m_std.loc['mean', 'nirv'])/df_m_std.loc['std', 'nirv']
    kNDVI = (kNDVI - df_m_std.loc['mean', 'kNDVI'])/df_m_std.loc['std', 'kNDVI']
    evi = (evi - df_m_std.loc['mean', 'evi'])/df_m_std.loc['std', 'evi']
    ndvi = (ndvi - df_m_std.loc['mean', 'ndvi'])/df_m_std.loc['std', 'ndvi']
    nir = (nir - df_m_std.loc['mean', 'nir'])/df_m_std.loc['std', 'nir']
    red = (red - df_m_std.loc['mean', 'red'])/df_m_std.loc['std', 'red']
    blue = (blue - df_m_std.loc['mean', 'blue'])/df_m_std.loc['std', 'blue']
    green = (green - df_m_std.loc['mean', 'green'])/df_m_std.loc['std', 'green']
    swir1 = (swir1 - df_m_std.loc['mean', 'swir1'])/df_m_std.loc['std', 'swir1']  
    swir2 = (swir2 - df_m_std.loc['mean', 'swir2'])/df_m_std.loc['std', 'swir2']  
    swir3 = (swir3 - df_m_std.loc['mean', 'swir3'])/df_m_std.loc['std', 'swir3']  
    cosSZA = (cosSZA - df_m_std.loc['mean', 'cosSZA'])/df_m_std.loc['std', 'cosSZA']
    temperature = (temperature-df_m_std.loc['mean', 'temperature'])/df_m_std.loc['std', 'temperature']
    precipitation = (precipitation-df_m_std.loc['mean', 'precipitation'])/df_m_std.loc['std', 'precipitation']
    temperature_del = (temperature_del-df_m_std.loc['mean', 'temperature'])/df_m_std.loc['std', 'temperature']
    precipitation_del = (precipitation_del-df_m_std.loc['mean', 'precipitation'])/df_m_std.loc['std', 'precipitation']
    ssm = (ssm-df_m_std.loc['mean', 'ssm'])/df_m_std.loc['std', 'ssm']
    susm = (susm-df_m_std.loc['mean', 'susm'])/df_m_std.loc['std', 'susm']
    lulc = (lulc-df_m_std.loc['mean', 'lulc'])/df_m_std.loc['std', 'lulc']
    land_mask = (land_mask-df_m_std.loc['mean', 'land_mask'])/df_m_std.loc['std', 'land_mask']
    topography = (topography-df_m_std.loc['mean', 'topography'])/df_m_std.loc['std', 'topography']
    
    #df_oco2.iloc[:,1] = (df_oco2.iloc[:,1]-df_m_std.loc['mean', 'sif'])/df_m_std.loc['std', 'sif']


    if DATA_OPTION == 1:
      
        keys = ['LR_SIF', 'NIRv','kNDVI','EVI','NDVI', 'NIR', 'RED', 'Blue', 'Green', 'SWIR1', 'SWIR2','SWIR3', 'cosSZA', 'temperature', 'precipitation', 'temperature_delay', 'precipitation_delay', 'ssm', 'susm','land_mask', 'topography', 'LULC_Others', 'LULC_ENF', 'LULC_EBF', 'LULC_DNF', 'LULC_DBF', 'LULC_Mixed Forest', 'LULC_Unknown Forest', 'LULC_Shrubland', 'LULC_Grassland', 'LULC_Cropland', 'LULC_Wetland']

        inp_size = len(xlims_gridded-1)*len(ylims_gridded-1)*len(dates)
        inp = np.zeros((inp_size, y_size, x_size, len(keys)))
        ocos = np.zeros((inp_size, y_size, x_size)) #np.zeros((len(df_oco2), y_size, x_size))
        inp[:] = np.nan
        ocos[:] = np.nan
        ys, xs = [], []
        #create mask
        

        i = 0
        for y_grid in ylims_gridded[:-1]:
          #print('y_grid: ',y_grid)
          for x_grid in xlims_gridded[:-1]:
            #print('    x_grid: ',x_grid)
            for d in range(len(dates)):
              Masked_sif = sif[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_sif_dc_feat = sif_dc_feat[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_dcCorr_feat = dcCorr_feat[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]

              Masked_nir = nir[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_red = red[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_blue = blue[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_green = green[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_swir1 = swir1[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_swir2 = swir2[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_swir3 = swir3[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_nirv = nirv[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_kNDVI = kNDVI[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_evi = evi[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_ndvi = ndvi[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_cosSZA = cosSZA[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_temperature = temperature[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_precipitation = precipitation[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_temperature_del = temperature_del[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_precipitation_del = precipitation_del[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_ssm = ssm[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_susm = susm[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
            

              inp[i,:,:,0] = Masked_sif
              inp[i,:,:,1] = Masked_nirv
              inp[i,:,:,2] = Masked_kNDVI
              inp[i,:,:,3] = Masked_evi
              inp[i,:,:,4] = Masked_ndvi
              inp[i,:,:,5] = Masked_nir
              inp[i,:,:,6] = Masked_red
              inp[i,:,:,7] = Masked_blue
              inp[i,:,:,8] = Masked_green
              inp[i,:,:,9] = Masked_swir1
              inp[i,:,:,10] = Masked_swir2
              inp[i,:,:,11] = Masked_swir3
              inp[i,:,:,12] = Masked_cosSZA
              inp[i,:,:,13] = Masked_temperature
              inp[i,:,:,14] = Masked_precipitation
              inp[i,:,:,15] = Masked_temperature_del
              inp[i,:,:,16] = Masked_precipitation_del
              inp[i,:,:,17] = Masked_ssm
              inp[i,:,:,18] = Masked_susm            
              #
              inp[i,:,:,19] = land_mask[y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              inp[i,:,:,20] = topography[y_grid-y_size:y_grid,x_grid:x_grid+x_size]            
              inp[i,:,:,21:32] = lulc[y_grid-y_size:y_grid,x_grid:x_grid+x_size,:] #shape: (lat,lon,11)
              '''
              df = df_oco2[df_oco2.Latitude < yLims_arr[ylims_gridded = y_grid]]
              df = df[df.Latitude > yLims_arr[ylims_gridded = y_grid-y_size]]
              df = df[df.Longitude < xLims_arr[xlims_gridded = x_grid+x_size]]
              df = df[df.Longitude > xLims_arr[xlims_gridded = x_grid]]
              
              for j in len(df):
                step = helpers.nearest_date(dates[d],df['date_UTC'])
              
              oco = Masked_sif.copy()
              '''
              ocos[i,::] = Masked_sif.copy() #troposif as target

              i = i + 1    
          

    elif DATA_OPTION == 2:
      
        keys = ['LR_SIF', 'NIRv','kNDVI','EVI','NDVI', 'NIR', 'RED', 'Blue', 'Green', 'SWIR1', 'SWIR2','SWIR3', 'cosSZA', 'temperature', 'precipitation', 'temperature_delay', 'precipitation_delay','land_mask', 'topography', 'LULC_Others', 'LULC_ENF', 'LULC_EBF', 'LULC_DNF', 'LULC_DBF', 'LULC_Mixed Forest', 'LULC_Unknown Forest', 'LULC_Shrubland', 'LULC_Grassland', 'LULC_Cropland', 'LULC_Wetland']
        #y_size, x_size = 5, 4
        inp_size = len(xlims_gridded)*len(ylims_gridded)*len(dates)
        inp = np.zeros((inp_size, y_size, x_size, len(keys)))
        ocos = np.zeros((inp_size, 1, 1)) #np.zeros((len(df_oco2), y_size, x_size))
        inp[:] = np.nan
        #ocos[:] = np.nan
        ys, xs = [], []
        #create mask
        #for N_oco in range(len(df_oco2)):
        #create mask for sif
        #step = helpers.nearest_date(df_oco2.iloc[N_oco,:]['date_UTC'],dates)
        #r = sif.shape
        #print(r)
        #print('t:',r[0],'x:',r[2],'y:',r[1])    
        #x_min, x_max, y_min, y_max = helpers.MaskGenerator(lon_modis_0005, lat_modis_0005, Nx=r[2], Ny=r[1], df=df_oco2.iloc[N_oco,:])
        i = 0
        for y_grid in ylims_gridded[:-1]:
          for x_grid in xlims_gridded[:-1]:
            for d in range(len(dates)):
              Masked_sif = sif[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_sif_dc_feat = sif_dc_feat[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_dcCorr_feat = dcCorr_feat[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]

              Masked_nir = nir[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_red = red[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_blue = blue[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_green = green[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_swir1 = swir1[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_swir2 = swir2[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_swir3 = swir3[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_nirv = nirv[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_kNDVI = kNDVI[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_evi = evi[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_ndvi = ndvi[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_cosSZA = cosSZA[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_temperature = temperature[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_precipitation = precipitation[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_temperature_del = temperature_del[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_precipitation_del = precipitation_del[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              #Masked_ssm = ssm[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              #Masked_susm = susm[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
            

              inp[i,:,:,0] = Masked_sif
              inp[i,:,:,1] = Masked_nirv
              inp[i,:,:,2] = Masked_kNDVI
              inp[i,:,:,3] = Masked_evi
              inp[i,:,:,4] = Masked_ndvi
              inp[i,:,:,5] = Masked_nir
              inp[i,:,:,6] = Masked_red
              inp[i,:,:,7] = Masked_blue
              inp[i,:,:,8] = Masked_green
              inp[i,:,:,9] = Masked_swir1
              inp[i,:,:,10] = Masked_swir2
              inp[i,:,:,11] = Masked_swir3
              inp[i,:,:,12] = Masked_cosSZA
              inp[i,:,:,13] = Masked_temperature
              inp[i,:,:,14] = Masked_precipitation
              inp[i,:,:,15] = Masked_temperature_del
              inp[i,:,:,16] = Masked_precipitation_del
              #inp[i,:,:,17] = Masked_ssm
              #inp[i,:,:,18] = Masked_susm            
              #
              inp[i,:,:,17] = land_mask[y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              inp[i,:,:,18] = topography[y_grid-y_size:y_grid,x_grid:x_grid+x_size]            

              inp[i,:,:,19:30] = lulc[y_grid-y_size:y_grid,x_grid:x_grid+x_size,:] #shape: (lat,lon,11)

              i = i + 1    
         
        
    elif DATA_OPTION == 3:
      
        keys = ['LR_SIF', 'NIRv','kNDVI','EVI','NDVI', 'NIR', 'RED', 'Blue', 'Green', 'SWIR1', 'SWIR2','SWIR3', 'cosSZA', 'temperature', 'precipitation', 'temperature_delay', 'precipitation_delay','land_mask', 'LULC_Others', 'LULC_ENF', 'LULC_EBF', 'LULC_DNF', 'LULC_DBF', 'LULC_Mixed Forest', 'LULC_Unknown Forest', 'LULC_Shrubland', 'LULC_Grassland', 'LULC_Cropland', 'LULC_Wetland']
        #y_size, x_size = 5, 4
        inp_size = len(xlims_gridded)*len(ylims_gridded)*len(dates)
        inp = np.zeros((inp_size, y_size, x_size, len(keys)))
        ocos = np.zeros((inp_size, 1, 1)) #np.zeros((len(df_oco2), y_size, x_size))
        inp[:] = np.nan
        #ocos[:] = np.nan
        ys, xs = [], []
        #create mask
        #for N_oco in range(len(df_oco2)):
        #create mask for sif
        #step = helpers.nearest_date(df_oco2.iloc[N_oco,:]['date_UTC'],dates)
        #r = sif.shape
        #print(r)
        #print('t:',r[0],'x:',r[2],'y:',r[1])    
        #x_min, x_max, y_min, y_max = helpers.MaskGenerator(lon_modis_0005, lat_modis_0005, Nx=r[2], Ny=r[1], df=df_oco2.iloc[N_oco,:])
        i = 0
        for y_grid in ylims_gridded[:-1]:
          for x_grid in xlims_gridded[:-1]:
            for d in range(len(dates)):
              Masked_sif = sif[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_sif_dc_feat = sif_dc_feat[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_dcCorr_feat = dcCorr_feat[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]

              Masked_nir = nir[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_red = red[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_blue = blue[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_green = green[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_swir1 = swir1[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_swir2 = swir2[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_swir3 = swir3[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_nirv = nirv[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_kNDVI = kNDVI[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_evi = evi[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_ndvi = ndvi[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_cosSZA = cosSZA[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_temperature = temperature[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_precipitation = precipitation[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_temperature_del = temperature_del[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_precipitation_del = precipitation_del[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              #Masked_ssm = ssm[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              #Masked_susm = susm[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
            

              inp[i,:,:,0] = Masked_sif
              inp[i,:,:,1] = Masked_nirv
              inp[i,:,:,2] = Masked_kNDVI
              inp[i,:,:,3] = Masked_evi
              inp[i,:,:,4] = Masked_ndvi
              inp[i,:,:,5] = Masked_nir
              inp[i,:,:,6] = Masked_red
              inp[i,:,:,7] = Masked_blue
              inp[i,:,:,8] = Masked_green
              inp[i,:,:,9] = Masked_swir1
              inp[i,:,:,10] = Masked_swir2
              inp[i,:,:,11] = Masked_swir3
              inp[i,:,:,12] = Masked_cosSZA
              inp[i,:,:,13] = Masked_temperature
              inp[i,:,:,14] = Masked_precipitation
              inp[i,:,:,15] = Masked_temperature_del
              inp[i,:,:,16] = Masked_precipitation_del
              #inp[i,:,:,17] = Masked_ssm
              #inp[i,:,:,18] = Masked_susm            
              #
              inp[i,:,:,17] = land_mask[y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              #inp[i,:,:,18] = topography[y_grid:y_grid+y_size,x_grid:x_grid+x_size]            

              inp[i,:,:,18:29] = lulc[y_grid-y_size:y_grid,x_grid:x_grid+x_size,:] #shape: (lat,lon,11)

              i = i + 1    
        

    elif DATA_OPTION == 4: #no ssm ssum, 'topography', 'temperature', 'precipitation', 'temperature_delay', 'precipitation_delay',
      
        keys = ['LR_SIF', 'NIRv','kNDVI','EVI','NDVI', 'NIR', 'RED', 'Blue', 'Green', 'SWIR1', 'SWIR2','SWIR3', 'cosSZA','land_mask', 'LULC_Others', 'LULC_ENF', 'LULC_EBF', 'LULC_DNF', 'LULC_DBF', 'LULC_Mixed Forest', 'LULC_Unknown Forest', 'LULC_Shrubland', 'LULC_Grassland', 'LULC_Cropland', 'LULC_Wetland']
        #y_size, x_size = 5, 4
        inp_size = len(xlims_gridded)*len(ylims_gridded)*len(dates)
        inp = np.zeros((inp_size, y_size, x_size, len(keys)))
        ocos = np.zeros((inp_size, 1, 1)) #np.zeros((len(df_oco2), y_size, x_size))
        inp[:] = np.nan
        #ocos[:] = np.nan
        ys, xs = [], []
        #create mask
        #for N_oco in range(len(df_oco2)):
        #create mask for sif
        #step = helpers.nearest_date(df_oco2.iloc[N_oco,:]['date_UTC'],dates)
        #r = sif.shape
        #print(r)
        #print('t:',r[0],'x:',r[2],'y:',r[1])    
        #x_min, x_max, y_min, y_max = helpers.MaskGenerator(lon_modis_0005, lat_modis_0005, Nx=r[2], Ny=r[1], df=df_oco2.iloc[N_oco,:])
        i = 0
        for y_grid in ylims_gridded[:-1]:
          for x_grid in xlims_gridded[:-1]:
            for d in range(len(dates)):
              Masked_sif = sif[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_sif_dc_feat = sif_dc_feat[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_dcCorr_feat = dcCorr_feat[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]

              Masked_nir = nir[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_red = red[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_blue = blue[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_green = green[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_swir1 = swir1[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_swir2 = swir2[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_swir3 = swir3[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_nirv = nirv[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_kNDVI = kNDVI[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_evi = evi[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_ndvi = ndvi[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_cosSZA = cosSZA[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              #Masked_temperature = temperature[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              #Masked_precipitation = precipitation[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              #Masked_temperature_del = temperature_del[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              #Masked_precipitation_del = precipitation_del[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              #Masked_ssm = ssm[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              #Masked_susm = susm[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
            

              inp[i,:,:,0] = Masked_sif
              inp[i,:,:,1] = Masked_nirv
              inp[i,:,:,2] = Masked_kNDVI
              inp[i,:,:,3] = Masked_evi
              inp[i,:,:,4] = Masked_ndvi
              inp[i,:,:,5] = Masked_nir
              inp[i,:,:,6] = Masked_red
              inp[i,:,:,7] = Masked_blue
              inp[i,:,:,8] = Masked_green
              inp[i,:,:,9] = Masked_swir1
              inp[i,:,:,10] = Masked_swir2
              inp[i,:,:,11] = Masked_swir3
              inp[i,:,:,12] = Masked_cosSZA
              #inp[i,:,:,13] = Masked_temperature
              #inp[i,:,:,14] = Masked_precipitation
              #inp[i,:,:,15] = Masked_temperature_del
              #inp[i,:,:,16] = Masked_precipitation_del
              #inp[i,:,:,17] = Masked_ssm
              #inp[i,:,:,18] = Masked_susm            
              #
              inp[i,:,:,13] = land_mask[y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              #inp[i,:,:,18] = topography[y_grid:y_grid+y_size,x_grid:x_grid+x_size]            

              inp[i,:,:,14:25] = lulc[y_grid-y_size:y_grid,x_grid:x_grid+x_size,:] #shape: (lat,lon,11)

              i = i + 1    
        

    elif DATA_OPTION == 5: #no ssm ssum, 'topography', 'temperature', 'precipitation', 'temperature_delay', 'precipitation_delay',
      
        keys = ['LR_SIF', 'NIRv','kNDVI','EVI','NDVI', 'NIR', 'RED', 'Blue', 'Green', 'SWIR1', 'SWIR2','SWIR3', 'cosSZA','land_mask', 'LULC_Others', 'LULC_ENF', 'LULC_EBF', 'LULC_DNF', 'LULC_DBF', 'LULC_Mixed Forest', 'LULC_Unknown Forest', 'LULC_Shrubland', 'LULC_Grassland', 'LULC_Cropland', 'LULC_Wetland']
        #y_size, x_size = 5, 4
        inp_size = len(xlims_gridded)*len(ylims_gridded)*len(dates)
        inp = np.zeros((inp_size, y_size, x_size, len(keys)))
        ocos = np.zeros((inp_size, 1, 1)) #np.zeros((len(df_oco2), y_size, x_size))
        inp[:] = np.nan
        #ocos[:] = np.nan
        ys, xs = [], []
        #create mask
        #for N_oco in range(len(df_oco2)):
        #create mask for sif
        #step = helpers.nearest_date(df_oco2.iloc[N_oco,:]['date_UTC'],dates)
        #r = sif.shape
        #print(r)
        #print('t:',r[0],'x:',r[2],'y:',r[1])    
        #x_min, x_max, y_min, y_max = helpers.MaskGenerator(lon_modis_0005, lat_modis_0005, Nx=r[2], Ny=r[1], df=df_oco2.iloc[N_oco,:])
        i = 0
        for y_grid in ylims_gridded[:-1]:
          for x_grid in xlims_gridded[:-1]:
            for d in range(len(dates)):
              Masked_sif = sif[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_sif_dc_feat = sif_dc_feat[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_dcCorr_feat = dcCorr_feat[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]

              Masked_nir = nir[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_red = red[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_blue = blue[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_green = green[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_swir1 = swir1[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_swir2 = swir2[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_swir3 = swir3[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_nirv = nirv[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_kNDVI = kNDVI[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_evi = evi[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_ndvi = ndvi[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              Masked_cosSZA = cosSZA[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              #Masked_temperature = temperature[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              #Masked_precipitation = precipitation[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              #Masked_temperature_del = temperature_del[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              #Masked_precipitation_del = precipitation_del[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              #Masked_ssm = ssm[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              #Masked_susm = susm[d, y_grid-y_size:y_grid,x_grid:x_grid+x_size]
            

              inp[i,:,:,0] = Masked_sif
              inp[i,:,:,1] = Masked_nirv
              #inp[i,:,:,2] = Masked_kNDVI
              #inp[i,:,:,3] = Masked_evi
              #inp[i,:,:,4] = Masked_ndvi
              inp[i,:,:,2] = Masked_nir
              inp[i,:,:,3] = Masked_red
              inp[i,:,:,4] = Masked_blue
              inp[i,:,:,5] = Masked_green
              inp[i,:,:,6] = Masked_swir1
              inp[i,:,:,7] = Masked_swir2
              inp[i,:,:,8] = Masked_swir3
              inp[i,:,:,9] = Masked_cosSZA
              #inp[i,:,:,13] = Masked_temperature
              #inp[i,:,:,14] = Masked_precipitation
              #inp[i,:,:,15] = Masked_temperature_del
              #inp[i,:,:,16] = Masked_precipitation_del
              #inp[i,:,:,17] = Masked_ssm
              #inp[i,:,:,18] = Masked_susm            
              #
              inp[i,:,:,10] = land_mask[y_grid-y_size:y_grid,x_grid:x_grid+x_size]
              #inp[i,:,:,18] = topography[y_grid:y_grid+y_size,x_grid:x_grid+x_size]            

              inp[i,:,:,11:22] = lulc[y_grid-y_size:y_grid,x_grid:x_grid+x_size,:] #shape: (lat,lon,11)

              i = i + 1    
        
    else:
        0 == 0
    
    print('Number of nan in inp',np.count_nonzero(np.isnan(inp)))
    print('Number of nan in ocos',np.count_nonzero(np.isnan(ocos)))
        
    inp[np.where(np.isnan(inp))] = 0
    ocos[np.where(np.isnan(ocos))] = 0
    #ocos = (ocos-df_m_std.loc['mean', 'sif'])/df_m_std.loc['std', 'sif'] #standardize targets
    print('Number of nan in inp after process',np.count_nonzero(np.isnan(inp)))
    print('Number of nan in ocos after process',np.count_nonzero(np.isnan(ocos)))    
    ocos_all = np.count_nonzero(~np.isnan(ocos))
        
    print('shape of inp: ', inp.shape)
    print('shape of ocos: ', ocos.shape)
    print('np.mean(ocos): ', np.mean(ocos))
    print('np.std(ocos): ', np.std(ocos))
    ocos_0 = np.count_nonzero(ocos < 0) / ocos_all
    ocos_05 = np.count_nonzero(ocos < 0.5) / ocos_all
    ocos_10 = np.count_nonzero(ocos < 1.0) / ocos_all
    
    print('< 0: ', ocos_0)
    print('0 to 0.5: ', ocos_05 - ocos_0)
    print('0.5 to 1.0: ', ocos_10 - ocos_05)
    print('> 1.0: ', 1 - ocos_10)
    
    return inp, keys, ocos, ylims_gridded, xlims_gridded, ti_days_since






class DataSet(Dataset):
    def __init__(self, features, oco):
        self.features = features
        self.oco = oco


    def __len__(self):
        return self.oco.shape[0]

    def __getitem__(self, idx):
        sample = dict()
        sample['features'], sample['sif'] =self.features[idx], self.oco[idx]
        return  sample
'''
inp, keys, oco = get_500m_dataset(     ...)
dataset = DataSet(features = inp, OCO = ocos)
'''