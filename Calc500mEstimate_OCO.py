# %%
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
import optuna

#own scripts
import sys
#sys.path.insert(1, '../')
import helpers
from dataset_OCO import get_calc_500m_dataset
from dataset_OCO import DataSet
from ModelClass import SR_SIF

#%%
#some settings
plt.rcParams.update({'font.size':14})

# %%
#specify your device ('cuda' or 'cpu')
device = 'cuda'

"""
#settings for 50 to 5 km trained model
SCALE_FACTOR = 10
PRESCALE_FACTOR = 0
USE_MODEL_OUT = True
PRESCALE = False

folder_name_save = 'results'
model_name = 'OptmizedModel_scaling10_fullCV_optLRWD_fullRun_noTestSet.pth'
model_out_name = ''
results_data_name_save = 'ResultsCONUS_500m.nc'

# %%
model = load_checkpoint('/mnt/mnt_folder/MT_Vegetation/code/superresolution/'+model_name)
if torch.cuda.device_count() > 1:
  model = nn.DataParallel(model)
"""
def get_500m_dataset_OCO(xLims, yLims, pathPref, folder_name_save):
  Tropo, Modis, Sen2 = helpers.GetGrids(xLims, yLims)
  ddX_Modis,ddY_Modis,lon_Modis,lat_Modis = Modis['ddX'],Modis['ddY'],Modis['lon'],Modis['lat']
  lonM, latM = np.meshgrid(lon_Modis, lat_Modis[::-1])
            
  pathPrefpred = pathPref+folder_name_save+'/Results_500m.nc' 
    
  nc_pred = netCDF4.Dataset(pathPrefpred)
  lat_pred = nc_pred.variables['lat'][:].data
  lon_pred = nc_pred.variables['lon'][:].data
  ti_days_since = nc_pred.variables['time'][:].data
  dates = helpers.GetDatesFromDaysSince(ti_days_since)
  sif_pred = nc_pred.variables['sif_pred']
  #OCO
  pathPref_OCO = '/mnt/mnt_folder/data/'
  df_oco2 = helpers.read_OCO(pathPref_OCO, yLims, xLims, yLims_test=np.array([200]), xLims_test=np.array([200]))
  df_oco2_smp = df_oco2.sample(n = int(df_oco2.shape[0]/10))

      
  y_size, x_size = 5, 4
  ocos_target = np.zeros((len(df_oco2_smp), y_size, x_size))
  ocos_pred = np.zeros((len(df_oco2_smp), y_size, x_size))
  oco = np.zeros((y_size, x_size))

  #create training samples
  for N_oco in range(len(df_oco2_smp)):
    #create mask for sif
    step = helpers.nearest_date(df_oco2['date_UTC'][N_oco],dates)
  
    x_min, x_max, y_min, y_max = helpers.MaskGenerator(lon_pred, lat_pred, df=df_oco2_smp.iloc[N_oco,:])
    oco.fill(df_oco2['Daily_SIF_740nm'][N_oco])
    ocos_target[N_oco,::] = oco.copy()    
    ocos_pred[N_oco,::] = sif_pred[step,y_min:y_min+y_size,x_min:x_min+x_size]
    
  '''
  print('Number of nan in ocos',np.count_nonzero(np.isnan(ocos)))

  ocos[np.where(np.isnan(ocos))] = 0
  #ocos = (ocos-df_m_std.loc['mean', 'sif'])/df_m_std.loc['std', 'sif'] #standardize targets
  print('Number of nan in ocos after process',np.count_nonzero(np.isnan(ocos)))
  '''
  #print('shape of ocos: ', ocos_pred.shape)
  print('==prediction==')
  ocos_all = np.count_nonzero(~np.isnan(ocos_pred))
        
  print('shape of ocos: ', ocos_pred.shape)
  print('np.mean(ocos): ', np.mean(ocos_pred))
  print('np.std(ocos): ', np.std(ocos_pred))
  ocos_0 = np.count_nonzero(ocos_pred < 0) / ocos_all
  ocos_05 = np.count_nonzero(ocos_pred < 0.5) / ocos_all
  ocos_10 = np.count_nonzero(ocos_pred < 1.0) / ocos_all
    
  print('< 0: ', ocos_0)
  print('0 to 0.5: ', ocos_05 - ocos_0)
  print('0.5 to 1.0: ', ocos_10 - ocos_05)
  print('> 1.0: ', 1 - ocos_10)
  print('\n')
  
  print('==target==')
  ocos_all = np.count_nonzero(~np.isnan(ocos_target))
        
  print('shape of ocos: ', ocos_target.shape)
  print('np.mean(ocos): ', np.mean(ocos_target))
  print('np.std(ocos): ', np.std(ocos_target))
  ocos_0 = np.count_nonzero(ocos_target < 0) / ocos_all
  ocos_05 = np.count_nonzero(ocos_target < 0.5) / ocos_all
  ocos_10 = np.count_nonzero(ocos_target < 1.0) / ocos_all
    
  print('< 0: ', ocos_0)
  print('0 to 0.5: ', ocos_05 - ocos_0)
  print('0.5 to 1.0: ', ocos_10 - ocos_05)
  print('> 1.0: ', 1 - ocos_10)
  MSE = mean_squared_error(y_true = ocos_target.flatten(), y_pred = ocos_pred.flatten())

  return MSE
# %%
def RunSet(model, dataloader):
  sample = next(iter(dataloader))['sif'].detach().numpy()
  pred = np.zeros((len(dataloader), sample.shape[1], sample.shape[2]))
  targ = np.zeros((len(dataloader), sample.shape[1], sample.shape[2]))

  i = 0
  sample = next(iter(dataloader))
  model.eval()
  for sample in dataloader:
    with torch.no_grad():
        inp = sample['features'].to(device, dtype=torch.float)
        target = sample['sif'].to(device)
        pr = model(inp)
        
        pred[i,::] = pr.cpu().detach().numpy()[0,0,::] #revised
        targ[i,::] = target.cpu().detach().numpy()[0,0,::]
        i+=1

  return targ, pred

# load a model checkpoint
# Code referred from: https://discuss.pytorch.org/t/saving-customized-model-architecture/21512/2
def load_checkpoint(filepath):
    if torch.cuda.is_available() == False or device == 'cpu':
        checkpoint = torch.load(filepath, map_location=torch.device('cpu')) 
    else:
        checkpoint = torch.load(filepath)
    if torch.cuda.device_count() > 1:
      model = nn.DataParallel(checkpoint['model'])
    else:
      model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    return model



#revised
def Calc500m(model, DATA_OPTION, pathPref, SCALE_FACTOR,folder_name_save,d_start,d_end,USE_MODEL_OUT=False,xLims=np.array([10,15]),yLims=np.array([45,50])):
  
  feat_500m, keys, OCO, ylims_gridded, xlims_gridded, ti_days_since, lon, lat, lonM, latM, lon_Modis, lat_Modis = get_calc_500m_dataset(xLims=xLims, yLims=yLims, d_start=d_start, d_end=d_end, USE_MODEL_OUT=True,DATA_OPTION=DATA_OPTION)

  dataset_test = DataSet(feat_500m, OCO) #
  dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
      
  sif_lr, pred_test = RunSet(model, dataloader_test)
  print('pred_test.shape: ',pred_test.shape)
  print('pred_test[0,::]: ',pred_test[0,0,0])
  print('pred_test[-1,::]: ',pred_test[-1,0,0])
  
  gridded = np.zeros((d_end-d_start,lonM.shape[0],lonM.shape[1]))
  print('gridded.shape: ', gridded.shape)
  
  i = 0
  y_size, x_size = 100, 100

  for y_grid in ylims_gridded[:-1]:
    for x_grid in xlims_gridded[:-1]:
      for d in range(len(ti_days_since)):
        iy_min = y_grid-y_size
        iy_max = y_grid
        ix_min = x_grid
        ix_max = x_grid+x_size
          
        gridded[d,iy_min:iy_max,ix_min:ix_max] = pred_test[i,:,:]
        i = i + 1

  fig, ax = plt.subplots(figsize = (10,5))
  # Histogram:
  # Bin it
  n, bin_edges = np.histogram(gridded.flatten(), 10)
  # Normalize it, so that every bins value gives the probability of that bin
  bin_probability = n/float(n.sum())
  # Get the mid points of every bin
  bin_middles = (bin_edges[1:]+bin_edges[:-1])/2.
  # Compute the bin-width
  bin_width = bin_edges[1]-bin_edges[0]
  # Plot the histogram as a bar plot
  plt.bar(bin_middles, bin_probability, width=bin_width)
  plt.xlabel('sif')
  plt.ylabel('Probability')
  plt.title('Prediction')

  plt.grid(True)
  plt.xlim(-0.2,2)
  plt.show()
  plt.savefig(pathPref+folder_name_save+'Profile_of_prediction_target.pdf', format='pdf', bbox_inches = 'tight', pad_inches = 0)

  '''
  pred_mean = gridded.mean
  pred_std = gridded.std
  gridded = (gridded - pred_mean) / pred_std * sif_mean + sif_std
  '''
  """
  #direct run
  feat_500m,lab,keys,ti_days_since = get_500m_dataset(xLims=xLims, yLims=yLims, d_start=d_start, d_end=d_end, USE_MODEL_OUT=True)
  dataset_test = DataSet(feat_500m,lab, TrainFlag=False, SCALE_FACTOR=SCALE_FACTOR, USE_MODEL_OUT=USE_MODEL_OUT) #
  dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
  sif_lr, gridded = RunSet(model, dataloader_test)

  print('gridded: ', gridded)
  print('gridded.shape: ', gridded.shape)
  """

  OutputFile = netCDF4.Dataset(pathPref+folder_name_save+'/Results_500m.nc', "w", format="NETCDF4")

  len1 = OutputFile.createDimension("lon", gridded.shape[2])
  len2 = OutputFile.createDimension("lat", gridded.shape[1])
  len3 = OutputFile.createDimension("time", gridded.shape[0])

  var_lon = OutputFile.createVariable("lon", "f4", ("lon",))
  var_lat = OutputFile.createVariable("lat", "f4", ("lat",))
  var_dates = OutputFile.createVariable("time", "f4", ("time",))
  var_sif_pred = OutputFile.createVariable("sif_pred", "f4", ("time","lat","lon"))

  var_dates.units = 'days since 1970-01-01'
  var_lon[:] = lon_Modis
  var_lat[:] = lat_Modis
  var_sif_pred[:] = gridded
  var_dates[:] = ti_days_since
  
  #compute MSE between prediction and OCOs footprints
  #MSE = get_500m_dataset_OCO(xLims = np.array([10, 15]), yLims = np.array([45, 50]), pathPref=pathPref, folder_name_save=folder_name_save)
  #print('MSE: ', MSE)

  OutputFile.close()

  cmap_SIF = helpers.CreateSIFCbar()
  my_dpi = 96
  plt.figure(figsize=(gridded.shape[2]/my_dpi, gridded.shape[1]/my_dpi), dpi=my_dpi)
  im = plt.imshow(np.nanmean(gridded,axis=0),cmap=cmap_SIF,vmin=0,vmax=1.2,extent=[lon_Modis[0],lon_Modis[-1],lat_Modis[0],lat_Modis[-1]])
  plt.savefig(pathPref+folder_name_save+'/ResultsCONUS_500m.pdf', format='pdf', bbox_inches = 'tight', pad_inches = 0)

#revised
  #Nuernberg
  xLims_SF = np.array([11, 12])
  yLims_SF = np.array([48, 50])

  ix1,ix2 = np.argmin(abs(lon_Modis-np.min(xLims_SF))),np.argmin(abs(lon_Modis-np.max(xLims_SF))) + 1
  ix_min,ix_max = np.min([ix1, ix2]),np.max([ix1, ix2])
  iy1,iy2 = np.argmin(abs(lat_Modis-np.min(yLims_SF))) + 1,np.argmin(abs(lat_Modis-np.max(yLims_SF)))
  iy_min,iy_max = np.min([iy1, iy2]),np.max([iy1, iy2])

  sif_SF = np.nanmean(gridded,axis=0)
  sif_SF = sif_SF[iy_min:iy_max,ix_min:ix_max]
  plt.figure(figsize=(gridded.shape[2]/my_dpi, gridded.shape[1]/my_dpi), dpi=my_dpi)
  im = plt.imshow(sif_SF,cmap=cmap_SIF,vmin=0,vmax=1.2)
  plt.savefig(pathPref+folder_name_save+'/ResultsSF_500m.pdf', format='pdf', bbox_inches = 'tight', pad_inches = 0)

#revised
  #Munich
  xLims_SF = np.array([11, 12])
  yLims_SF = np.array([45.5, 46.5])

  ix1,ix2 = np.argmin(abs(lon_Modis-np.min(xLims_SF))),np.argmin(abs(lon_Modis-np.max(xLims_SF))) + 1
  ix_min,ix_max = np.min([ix1, ix2]),np.max([ix1, ix2])
  iy1,iy2 = np.argmin(abs(lat_Modis-np.min(yLims_SF))) + 1,np.argmin(abs(lat_Modis-np.max(yLims_SF)))
  iy_min,iy_max = np.min([iy1, iy2]),np.max([iy1, iy2])

  sif_SF = np.nanmean(gridded,axis=0)
  sif_SF = sif_SF[iy_min:iy_max,ix_min:ix_max]
  plt.figure(figsize=(gridded.shape[2]/my_dpi, gridded.shape[1]/my_dpi), dpi=my_dpi)
  im = plt.imshow(sif_SF,cmap=cmap_SIF,vmin=0,vmax=1.2)
  plt.savefig(pathPref+folder_name_save+'/ResultsLA_500m.pdf', format='pdf', bbox_inches = 'tight', pad_inches = 0)



  #plt.figure(figsize=(gridded.shape[0]/48, gridded.shape[1]/48))
  #plt.imshow(gridded[0,::])
  #plt.savefig('/mnt/mnt_folder/test.pdf', bbox_inches = 'tight', pad_inches = 0)
