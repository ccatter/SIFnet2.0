
import numpy as np
import pandas as pd
import os
from os.path import isfile, join
from numba import jit
from scipy.signal import convolve2d
import datetime as dt
import netCDF4
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib.colors import ListedColormap
import matplotlib

### Grid function
# Typical call:
"""
xLims = [-180,180]
yLims = [-90,90]
Tropo, Modis, Sen2 = helpers.GetGrids(xLims, yLims)
ddX_Tropo = Tropo['ddX']
ddY_Tropo = Tropo['ddY']
lon_Tropo = Tropo['lon']
lat_Tropo = Tropo['lat']
lonT, latT = np.meshgrid(lon_Tropo, lat_Tropo[::-1])
"""
def GetGrids(xLims, yLims):

    xLims = np.array(xLims)
    yLims = np.array(yLims)

    #Tropomi SIF
    ddX_Tropo = 0.05
    ddY_Tropo = 0.05

    #Modis
    ddX_Modis = 0.0045 #results in 495m
    ddY_Modis = 0.0045
    ddX_Modis = 0.005 #results in 555m
    ddY_Modis = 0.005


    #Sentinel 2 - 10m
    ddX_Sen2 = 0.00009 #results in 9.99m
    ddY_Sen2 = 0.00009
    ddX_Sen2 = 0.0001 #results in 11.1m
    ddY_Sen2 = 0.0001

    #Tropomi SIF grid
    xLims_Tropo =  xLims + ddX_Tropo/2
    yLims_Tropo = yLims + ddY_Tropo/2

    lon_Tropo = np.arange(xLims_Tropo[0], xLims_Tropo[1] - ddX_Tropo/5, ddX_Tropo)
    lat_Tropo = np.arange(yLims_Tropo[0], yLims_Tropo[1] - ddY_Tropo/5, ddY_Tropo)

    Tropo = {'lon': lon_Tropo, 'lat': lat_Tropo, 'ddX': ddX_Tropo, 'ddY': ddY_Tropo}

    #Modis grid
    xLims_Modis =  xLims + ddX_Modis/2
    yLims_Modis = yLims + ddY_Modis/2

    lon_Modis = np.arange(xLims_Modis[0], xLims_Modis[1], ddX_Modis)
    lat_Modis = np.arange(yLims_Modis[0], yLims_Modis[1], ddY_Modis)

    Modis = {'lon': lon_Modis, 'lat': lat_Modis, 'ddX': ddX_Modis, 'ddY': ddY_Modis}

    #Sentinel 2 grid
    xLims_Sen2 =  xLims + ddX_Sen2/2
    yLims_Sen2 = yLims + ddY_Sen2/2

    lon_Sen2 = np.arange(xLims_Sen2[0], xLims_Sen2[1], ddX_Sen2)
    lat_Sen2 = np.arange(yLims_Sen2[0], yLims_Sen2[1], ddY_Sen2)

    Sen2 = {'lon': lon_Sen2, 'lat': lat_Sen2, 'ddX': ddX_Sen2, 'ddY': ddY_Sen2}

    return Tropo, Modis, Sen2

def GetSZAMap(path, xLims, yLims, dates, ddXY):


    xLims = xLims.astype('float32')
    yLims = yLims.astype('float32')

    df = pd.read_csv(path, index_col ='latitude')
    #reduce dataframe to dates of interest
    df = df[dates]

    #reduce dataframe to latitude borders
    yLims_ = yLims.copy()
    yLims_[0] = yLims_[0] + ddXY/2
    yLims_[1] = yLims_[1] - ddXY/2
    ind = df.index.to_numpy()
    i1 = np.argmin(abs(ind - yLims_[0]))
    i2 = np.argmin(abs(ind - yLims_[1]))
    mi = np.min([i1,i2])
    ma = np.max([i1,i2])

    df = df.loc[ind[mi]:ind[ma]]

    # generate maps of cos SZA
    # there are some issues when using np.arange with a non-integer step size. 
    # I'll just substract a little bit so the end point is not included.
    X = np.arange(xLims[0]+ddXY/2,xLims[1]+ddXY/2 - ddXY/5,ddXY)
    Y = np.arange(yLims[0]+ddXY/2,yLims[1]+ddXY/2 - ddXY/5,ddXY)
    cosSZA = np.zeros((len(Y), len(X), len(dates)))

    l = cosSZA.shape[1]
    for i in range(len(dates)):
        cosSZA[:,:,i] = np.repeat(np.reshape(df[dates[i]].to_numpy(),(-1,1)), l, axis=1)

    return cosSZA


def GetDCMap(path, xLims, yLims, dates, ddXY):


    xLims = xLims.astype('float32')
    yLims = yLims.astype('float32')

    df = pd.read_csv(path, index_col ='latitude')
    #reduce dataframe to dates of interest
    df = df[dates]

    

    #reduce dataframe to latitude borders
    yLims_ = yLims.copy()
    yLims_[0] = yLims_[0] + ddXY/2
    yLims_[1] = yLims_[1] - ddXY/2
    ind = df.index.to_numpy()
    i1 = np.argmin(abs(ind - yLims_[0]))
    i2 = np.argmin(abs(ind - yLims_[1]))
    mi = np.min([i1,i2])
    ma = np.max([i1,i2])+1

    df = df.loc[ind[mi]:ind[ma]]

    print('DC df lat: ', df.index)
    

    # generate maps of cos SZA
    # there are some issues when using np.arange with a non-integer step size. 
    # I'll just substract a little bit so the end point is not included.
    X = np.arange(xLims[0]+ddXY/2,xLims[1]+ddXY/2 - ddXY/5,ddXY)
    Y = np.arange(yLims[0]+ddXY/2,yLims[1]+ddXY/2 - ddXY/5,ddXY)
    DC = np.zeros((len(Y)+1, len(X)+1, len(dates)))

    l = DC.shape[1]
    for i in range(len(dates)):
        DC[:,:,i] = np.repeat(np.reshape(df[dates[i]].to_numpy(),(-1,1)), l, axis=1)

    print('DC.shape: ', DC.shape)
    return DC
    
def GetLULCFrac(pathLoadLULC,iy_min=0,iy_max=0,ix_min=0,ix_max=0):
    nc = netCDF4.Dataset(pathLoadLULC, 'r')
    keys = []
    for l in nc.variables.keys():
        keys.append(l)
    keys.remove('lon')
    keys.remove('lat')

    iy_max_ = 3600 - iy_min
    iy_min_ = 3600 - iy_max
    iy_max,iy_min = iy_max_,iy_min_
    if np.mean([iy_min,iy_max,ix_min,ix_max]) == 0:
        sh = nc.variables[keys[0]][:].data.shape
    else:
        sh = nc.variables[keys[0]][iy_min:iy_max,ix_min:ix_max].data.shape
    lulc = np.zeros((len(keys), sh[0], sh[1]))

    i = 0
    for k in keys:
        if np.mean([iy_min,iy_max,ix_min,ix_max]) == 0:
            lulc[i,:,:] = np.flipud(nc.variables[k][:].data)        
        else:
            lulc[i,:,:] = np.flipud(nc.variables[k][iy_min:iy_max,ix_min:ix_max].data)
        i +=1

    nc.close()

    return lulc, keys

def GetDatesFromDaysSince(ti, start=dt.date(1970,1,1)):
    ti = ti.astype('int16')
    dates = []
    for t in ti:
        days = int(t)                             
        delta = dt.timedelta(days)    
        offset = start + delta      
        dates.append(offset.strftime('%Y-%m-%d'))
        
    return dates

# List of files in directory with specific ending
def GetFilesInDir(path, ending):
    return [f for f in os.listdir(path) if f.endswith(ending) and
              isfile(join(path, f)) and not f.startswith('.')]

#s=10
# arr2 = np.ones((s,s))/s**2
#lonM_T = strideConv(lon, arr2,s=s)
def strideConv(arr, arr2, s):
    return convolve2d(arr, arr2[::-1, ::-1], mode='valid')[::s, ::s]



# fast linear interpolation of NaN values over the 3rd axis in a 3d array.
# function is adapted from an answer in:
# https://stackoverflow.com/questions/30910944/fast-1d-linear-np-nan-interpolation-over-large-3d-array
@jit(nopython=True)
def interp_3d(arr_3d):
    result=arr_3d.copy()
    for i in range(arr_3d.shape[0]):
        for j in range(arr_3d.shape[1]):
            arr=arr_3d[i,j,:]
            # If all elements are nan then cannot conduct linear interpolation.
            if np.sum(np.isnan(arr))==arr.shape[0]:
                pass
            else:
                # If the element is in the middle and its value is nan, do linear interpolation using neighbor values.
                for k in range(arr.shape[0]):
                    if np.isnan(arr[k]):
                        x=k
                        x1=x-1
                        x2=x+1
                        y1 = np.nan
                        y2 = np.nan
                        # Find left neighbor whose value is not nan.
                        while x1>=0:
                            if np.isnan(arr[x1]):
                                x1=x1-1
                            else:
                                y1=arr[x1]
                                break
                        # Find right neighbor whose value is not nan.
                        while x2<arr.shape[0]:
                            if np.isnan(arr[x2]):
                                x2=x2+1
                            else:
                                y2=arr[x2]
                                break
                        # Calculate the slope and intercept determined by the left and right neighbors.
                        slope=(y2-y1)/(x2-x1)
                        intercept=y1-slope*x1
                        # Linear interpolation and assignment.
                        y=slope*x+intercept
                        arr[x]=y
                result[i,j,:]=arr
    return result


def GetMeanAndStd(path='/Users/johannesgensheimer/data/Mean_Std_indices.csv'):
    return pd.read_csv(path, index_col='index')

def CalcMeanAndStd():
    PathLoadTropomi = '/Users/johannesgensheimer/data/Tropomi/TROPO_03-2018-06-2019_005deg_8day_interp.nc'
    PathLoadModis = '/Users/johannesgensheimer/data/Modis/MODIS_03-2018-06-2019_005deg_8day_interp.nc'
    pathLoadSZA = '/Users/johannesgensheimer/data/cosSZA_2018_2019.csv'
    pathLoadLULC = '/Users/johannesgensheimer/data/CopernicusLULC/Gridded_LULC_005.nc'
    pathLoadSoMo = '/Users/johannesgensheimer/data/SoilMoisture/SoMo_03-2018-06-2019_005deg_8day.nc'
    pathLoadERA5 = '/Users/johannesgensheimer/data/ERA5/ERA5_03-2018-06-2019_005deg_8day.nc'

    nc_modis = netCDF4.Dataset(PathLoadModis)
    nc_tropomi = netCDF4.Dataset(PathLoadTropomi)

    index = ['mean', 'std']
    var = ['sif','nir', 'red', 'nirv', 'kNDVI', 'lulc', 'cosSZA', 'temperature', 'precipitation', 'ssm', 'susm']

    nc_ERA5 = netCDF4.Dataset(pathLoadERA5)
    prec = nc_ERA5.variables['precipitation'][:].data
    prec_m, prec_std = np.nanmean(prec), np.nanstd(prec)
    del prec
    temp = nc_ERA5.variables['meantemp_2m'][:].data
    temp_m, temp_std = np.nanmean(temp), np.nanstd(temp)
    del temp

    nc_SoMo = netCDF4.Dataset(pathLoadSoMo)
    ssm = nc_SoMo.variables['ssm'][:].data
    ssm_m, ssm_std = np.nanmean(ssm), np.nanstd(ssm)
    del ssm
    susm = nc_SoMo.variables['susm'][:].data
    susm_m, susm_std = np.nanmean(susm), np.nanstd(susm)
    del susm

    # cosSZA
    start=dt.date(2018,1,1)
    d = start
    dates = [start.strftime('%Y-%m-%d')]
    days_delta = 16
    for i in range(365//days_delta):
        d = d + dt.timedelta(days=days_delta)
        dates.append(d.strftime('%Y-%m-%d'))

    cosSZA = GetSZAMap(pathLoadSZA, xLims=np.array([-180,180]), yLims=np.array([-90,90]), dates=dates, ddXY=0.05)
    cosSZA = cosSZA[:,0,:]
    cosSZA_m = np.nanmean(cosSZA)
    cosSZA_std = np.nanstd(cosSZA)

    # MODIS NIR
    nir = nc_modis.variables['nir'][:].data
    nir_m = np.nanmean(nir)
    nir_std = np.nanstd(nir)
    red = nc_modis.variables['red'][:].data
    red_m = np.nanmean(red)
    red_std = np.nanstd(red)

    nirv = nir*(nir-red)/(nir+red)
    nirv_m = np.nanmean(nirv)
    nirv_std = np.nanstd(nirv)
    del nirv
    kNDVI = np.tanh(((nir-red)/(nir+red))**2)
    kNDVI_m = np.nanmean(kNDVI)
    kNDVI_std = np.nanstd(kNDVI)
    del kNDVI, nir, red

    sif = nc_tropomi.variables['sif'][:].data
    sif_m = np.nanmean(sif)
    sif_std = np.nanstd(sif)
    del sif


    # LULC
    gridded = {}
    nc = netCDF4.Dataset(pathLoadLULC, 'r')
    keys = []
    for l in nc.variables.keys():
        keys.append(l)
    keys.remove('lon')
    keys.remove('lat')

    sh = nc.variables[keys[0]][:].data.shape

    lulc = np.zeros((len(keys), sh[0], sh[1]))

    i = 0
    for k in keys:
        lulc[i,:,:] = np.flipud(nc.variables[k][:].data)
        i +=1

    nc.close()

    lulc_m = np.nanmean(lulc)
    lulc_std = np.nanstd(lulc)

    # create dataframe

    df = pd.DataFrame({}, index=index)
    df['sif'] = [sif_m, sif_std]
    df['nir'] = [nir_m, nir_std]
    df['red'] = [red_m, red_std]
    df['nirv'] = [nirv_m, nirv_std]
    df['kNDVI'] = [kNDVI_m, kNDVI_std]
    df['lulc'] = [lulc_m, lulc_std]
    df['cosSZA'] = [cosSZA_m, cosSZA_std]
    df['precipitation'] = [prec_m, prec_std]
    df['temperature'] = [temp_m, temp_std]
    df['ssm'] = [ssm_m, ssm_std]
    df['susm'] = [susm_m, susm_std]
    df.index.name = 'index'

    df.to_csv('/Users/johannesgensheimer/data/Mean_Std_indices.csv')


def PlotRegion(xLims, yLims, PathLoadTropomi='/Users/johannesgensheimer/data/Tropomi/TROPO_03-2018-06-2019_005deg_8day_interp.nc', pathSave='/Users/johannesgensheimer/Desktop/region.pdf', title = '', save=True):

    nc_tropomi = netCDF4.Dataset(PathLoadTropomi)

    lon = nc_tropomi.variables['lon'][:].data
    lat = nc_tropomi.variables['lat'][:].data

    lulc, lulc_keys = GetLULCFrac('/Users/johannesgensheimer/data/CopernicusLULC/Gridded_LULC_005.nc',0,3600,0,7200)
    lulc = np.argmax(lulc,axis=0)

    Tropo, Modis, Sen2 = GetGrids(xLims, yLims)
    ddX_Tropo = Tropo['ddX']
    ddY_Tropo = Tropo['ddY']
    lon_Tropo = Tropo['lon']
    lat_Tropo = Tropo['lat'][::-1]
    lonT, latT = np.meshgrid(lon_Tropo, lat_Tropo[::-1])

    ix1 = np.argmin(abs(lon-np.min(lon_Tropo)))
    ix2 = np.argmin(abs(lon-np.max(lon_Tropo))) + 1
    iy1 = np.argmin(abs(lat-np.min(lat_Tropo))) + 1
    iy2 = np.argmin(abs(lat-np.max(lat_Tropo)))

    ix_min = np.min([ix1, ix2])
    ix_max = np.max([ix1, ix2])
    iy_min = np.min([iy1, iy2])
    iy_max = np.max([iy1, iy2])

    my_dpi = 96
    plt.figure(figsize=(7200/my_dpi, 3600/my_dpi), dpi=my_dpi)
    plt.imshow(lulc, cmap='Greens') #, extent=[lon_Tropo[0], lon_Tropo[-1],lat_Tropo[0], lat_Tropo[-1]]
    plt.plot([ix_min, ix_max], [iy_min,iy_min], linewidth=10, color='blue')
    plt.plot([ix_min, ix_max], [iy_max,iy_max], linewidth=10, color='blue')
    plt.plot([ix_min, ix_min], [iy_min,iy_max], linewidth=10, color='blue')
    plt.plot([ix_max, ix_max], [iy_min,iy_max], linewidth=10, color='blue')
    plt.title(title, fontsize=70)
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([]);
    frame1.axes.yaxis.set_ticklabels([]);
    plt.plot([0,7200],[1800,1800], '--', c='black',alpha=0.5,linewidth=2)
    plt.ylim([3000,400])
    plt.xlim([0,7200])
    if save: 
        plt.savefig(pathSave, bbox_inches = 'tight', pad_inches = 0)



def Plot2dHistogram(X,Y,bins=(500,500),x_label_='',y_label_='',title_='',pathSave='', xlim_=[-1.5, 3.5], ylim_=[-1.5, 3.5],vmin_=0,vmax_=10):
    X = X.flatten()
    Y = Y.flatten()

    #drop NaNs
    if np.sum(np.isnan(X)) != 0:
        print('Dropping NaNs')
        df = pd.DataFrame({'X': X, 'Y': Y})
        df = df.dropna(axis=0)
        X,Y = df.X.to_numpy(), df.Y.to_numpy()

    # calculate metrics
    r2 = np.round(r2_score(X, Y), 2)
    rmse = np.round(np.sqrt(mean_squared_error(X, Y)), 2)

    bound_min, bound_max = np.min([X,Y]),np.max([X,Y])

    #histogram
    #h,x,y,j = plt.hist2d(Y, X, bins=bins, cmap='jet')
    #plt.close()
    h,x,y = np.histogram2d(X,Y, bins=bins)
    plt.figure(figsize=(8,8))
    plt.grid()
    min_x,max_x,min_y,max_y = np.min(x),np.max(x),np.min(y),np.max(y)
    #im = plt.imshow(np.log(np.flipud(h)), extent=[min_x, max_x, min_y, max_y])
    im = plt.imshow(np.log(h.T), extent=[x[0], x[-1], y[0], y[-1]], origin='lower', vmin=vmin_, vmax=vmax_) #, aspect='auto'
    #im = plt.imshow(np.log(h), extent=[min_y, max_y, min_x, max_x])
    plt.plot([bound_min, bound_max],[bound_min, bound_max], '--', alpha=0.8)
    cbar = plt.colorbar(im)
    cbar.set_label('log$_{10}$(# points in bin)')
    plt.title(title_ + ', r$^2$=' + str(r2) + ' RMSE=' + str(rmse))
    #plt.ylim([y[0], y[-1]])
    #plt.xlim([x[0], x[-1]])
    plt.ylim(ylim_)
    plt.xlim(xlim_)
    plt.xlabel(x_label_)
    plt.ylabel(y_label_)

    print('##### Results')
    print('r2 = ', r2)
    print('RMSE = ', rmse)
    if pathSave != '':
        plt.savefig(pathSave, format='pdf', bbox_inches = 'tight', pad_inches = 0)
    return min_x,max_x,min_y,max_y

def inpolygon(xq, yq, xv, yv):
    shape = xq.shape
    xq = xq.reshape(-1)
    yq = yq.reshape(-1)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)
    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
    p = matplotlib.path.Path([(xv[i], yv[i]) for i in range(xv.shape[0])])
    return p.contains_points(q).reshape(shape)

def CreateSIFCbar(st=100,rbg_water=(0.8314, 0.9451, 0.9765)): #light blue for water

    TN = (255/255,232/255,179/255) #brown
    FG = (33/255, 75/255,  2/255) #green
    
    R = np.linspace(TN[0], FG[0], st)
    G = np.linspace(TN[1], FG[1], st)
    B = np.linspace(TN[2], FG[2], st)


    cmap_sif = []
    for i in range(st):
        cmap_sif.append( (R[i], G[i], B[i], 1) )
    
    cmap_sif = ListedColormap(cmap_sif)
     
    #rbg_water = (89/255, 77/255, 91/255) #dark gray for water
    cmap_sif.set_bad(rbg_water,1.)
    return cmap_sif
    
    #new functions
def nearest_date(items, pivot): 
    #items=df_oco2.iloc[50,:]['date_UTC'], pivot=dates
    cloz_dict = { 
    abs(dt.datetime.strptime(items, '%Y-%m-%d').timestamp() - dt.datetime.strptime(date, '%Y-%m-%d').timestamp()) : date 
    for date in pivot}
  
    # extracting minimum key using min()
    keys = list(cloz_dict.keys())
    min_key = min(keys)
    res = keys.index(min_key)

    return res
    
def MaskGenerator(lon_modis_0005, lat_modis_0005, df):
    #xLims=np.array([-10,50]), yLims=np.array([25,70]), shape=sif_dc_feat.shape, df=df_oco2.iloc[2,:]
    Xs = np.array([df['Longitude_Corners1'], df['Longitude_Corners2'], df['Longitude_Corners3'], df['Longitude_Corners4']])
    Ys = np.array([df['Latitude_Corners1'], df['Latitude_Corners2'], df['Latitude_Corners3'], df['Latitude_Corners4']])
    
    x_min = np.argmin(abs(lon_modis_0005-np.min(Xs)))
    x_max = np.argmin(abs(lon_modis_0005-np.max(Xs)))+1
    y_min = np.argmin(abs(lat_modis_0005-np.max(Ys)))+1
    y_max = np.argmin(abs(lat_modis_0005-np.min(Ys)))
    #print(x_min, x_max, y_min, y_max)

    #create a mask matrix with 1 values in the certain area
    #mask = np.zeros((shape[1], shape[2]))
    #mask[x_min:x_max,y_min:y_max] = 1
    #mask[mask == 0] = 'nan'
    return x_min, x_max, y_min, y_max
    
def read_OCO(pathPref, yLims, xLims, yLims_test, xLims_test):
    pathPref = pathPref
    yLims = yLims
    xLims = xLims
    xLims_test = xLims_test
    yLims_test = yLims_test
    year = range(2019,2022)
    X = range(2,4)
    th = 0.05
    #get footprints in 2018
    df_oco_all = pd.read_csv(pathPref + 'OCO2_SIF_2018_4_Germany.csv')
    df_oco_all = df_oco_all.drop('Unnamed: 0',axis=1)
    #select footprints in yLims, xLims
    df = df_oco_all[df_oco_all.Latitude < (np.max(yLims) - th)]
    df = df[df.Latitude > (np.min(yLims) + th)]
    df = df[df.Longitude < (np.max(xLims) - th)]
    df = df[df.Longitude > (np.min(xLims) + th)]
    df_oco_all = df.copy()

    if yLims_test[0]==200 and xLims_test[0]==200: #yLims, xLims are testing area
        for y in year:
            for x in X:
                if y==2021:
                    df_oco2 = pd.read_csv(pathPref + 'OCO'+str(x) +'_SIF_'+ str(y) + '_3_Germany.csv')
                    #csv files with '_Germany' ranged in xLims = [0,20], yLims = [40,55]
                    #2018 footprints start from 2018-04-01
                    #2021 footprints stop from 2021-04-02
                else:
                    df_oco2 = pd.read_csv(pathPref + 'OCO'+str(x) +'_SIF_'+ str(y) + '_Germany.csv')
                    #csv files with '_Germany' ranged in xLims = [0,20], yLims = [40,55]
                
                df_oco2 = df_oco2.drop('Unnamed: 0',axis=1)
                #select footprints in yLims, xLims
                df = df_oco2[df_oco2.Latitude < (np.max(yLims) - th)]
                df = df[df.Latitude > (np.min(yLims) + th)]
                df = df[df.Longitude < (np.max(xLims) - th)]
                df = df[df.Longitude > (np.min(xLims) + th)]
                df = df.reset_index(drop=True)
                df_oco2 = df.copy()
        
                df_oco_all = df_oco_all.append(df_oco2, ignore_index=True)
                #print('Number of footprints: ',df_oco_all.shape)
        #print('Testing area only:')
        #print('shape of footprints: ',df_oco_all.shape)

    else:
        #get footprints in xLims_test, yLims_test    
        df_drop = df_oco_all[df_oco_all.Latitude < (np.max(yLims_test) - th)]  
        df_drop = df_drop[df_drop.Latitude > (np.min(yLims_test) + th)]
        df_drop = df_drop[df_drop.Longitude > (np.min(xLims_test) - th)]
        df_drop = df_drop[df_drop.Longitude < (np.max(xLims_test) + th)]
        df_drop = df_drop.reset_index(drop=True)
        df_drop_all = df_drop.copy()
     
        for y in year:
            for x in X:
                if y==2021:
                    df_oco2 = pd.read_csv(pathPref + 'OCO'+str(x) +'_SIF_'+ str(y) + '_3_Germany.csv')
                    #csv files with '_Germany' ranged in xLims = [0,20], yLims = [40,55]
                    #2018 footprints start from 2018-04-01
                    #2021 footprints stop from 2021-04-02
                else:
                    df_oco2 = pd.read_csv(pathPref + 'OCO'+str(x) +'_SIF_'+ str(y) + '_Germany.csv')
                    #csv files with '_Germany' ranged in xLims = [0,20], yLims = [40,55]
                
                df_oco2 = df_oco2.drop('Unnamed: 0',axis=1)
                #select footprints in yLims, xLims
                df = df_oco2[df_oco2.Latitude < (np.max(yLims) - th)]
                df = df[df.Latitude > (np.min(yLims) + th)]
                df = df[df.Longitude < (np.max(xLims) - th)]
                df = df[df.Longitude > (np.min(xLims) + th)]
                df = df.reset_index(drop=True)
                df_oco2 = df.copy()
                #footprints in xLims_test, yLims_test    
                df_drop = df_oco2[df_oco2.Latitude < (np.max(yLims_test) - th)]  
                df_drop = df_drop[df_drop.Latitude > (np.min(yLims_test) + th)]
                df_drop = df_drop[df_drop.Longitude > (np.min(xLims_test) - th)]
                df_drop = df_drop[df_drop.Longitude < (np.max(xLims_test) + th)]
                df_drop = df_drop.reset_index(drop=True)
        
                df_oco_all = df_oco_all.append(df_oco2, ignore_index=True)
                df_drop_all = df_drop_all.append(df_drop, ignore_index=True)
                #print('Number of footprints: ',df_oco_all.shape)
        #print('Without test area')
        #print('shape of footprints in yLims, xLims: ',df_oco_all.shape)
        #print('shape of footprints in xLims_test, yLims_test: ',df_drop_all.shape)
        #drop footprints in testing area
        df_oco_all = df_oco_all[~((df_oco_all.Longitude.isin(df_drop_all['Longitude']))&(df_oco_all.DateTime_UTC.isin(df_drop_all['DateTime_UTC']))&(df_oco_all.Latitude.isin(df_drop_all['Latitude']))&(df_oco_all.SIF_740nm.isin(df_drop_all['SIF_740nm'])))]
        df_oco_all = df_oco_all.reset_index(drop=True)
        #print('shape of footprints for training: ',df_oco_all.shape)
    #df_oco_all = df_oco_all[df_oco_all['Daily_SIF_740nm'] = 0]
    df = df_oco_all[df_oco_all.Latitude < (np.max(yLims) - th)]
    df = df[df.Latitude > (np.min(yLims) + th)]
    df = df[df.Longitude < (np.max(xLims) - th)]
    df = df[df.Longitude > (np.min(xLims) + th)]
    df_oco_all = df.copy()
    
    df_oco_all = df_oco_all[df_oco_all['Latitude_Corners1'] > -200]
    df_oco_all = df_oco_all.reset_index(drop=True)
    df_oco_all = df_oco_all[df_oco_all['Longitude_Corners1'] > -200]
    df_oco_all = df_oco_all.reset_index(drop=True)
    
    print('quality flag = 0',len(df_oco_all[df_oco_all['Quality_Flag'] == 0]))
    print('quality flag = 1',len(df_oco_all[df_oco_all['Quality_Flag'] == 1]))
    print('quality flag = 2',len(df_oco_all[df_oco_all['Quality_Flag'] == 2]))
    df_oco_all = df_oco_all[df_oco_all['Quality_Flag'] == 0]
    df_oco_all = df_oco_all.reset_index(drop=True)
    
    sif_up = 3
    print('number of daily SIF>'+str(sif_up)+':',len(df_oco_all[df_oco_all['Daily_SIF_740nm']>=sif_up]))
    df_oco_all = df_oco_all.drop(df_oco_all[df_oco_all['Daily_SIF_740nm']>=sif_up].index) 
    df_oco_all = df_oco_all.reset_index(drop=True)
    
    sif_low = 0 
    print('number of daily SIF<'+str(sif_low)+':',len(df_oco_all[df_oco_all['Daily_SIF_740nm']<=sif_low]))
    df_oco_all = df_oco_all.drop(df_oco_all[df_oco_all['Daily_SIF_740nm']<=sif_low].index)
    df_oco_all = df_oco_all.reset_index(drop=True)
    
    sif_std = df_oco_all['SIF_740nm'].std()
    print('SIF_740nm_std',sif_std)
    print('SIF_740nm_mean',df_oco_all['SIF_740nm'].mean())
    #
    print('SIF + 2-s Uncertainty<0: ',len(df_oco_all[df_oco_all['SIF_740nm']<(-2+sif_std * df_oco_all['SIF_Uncertainty_740nm'])]))
    df_oco_all = df_oco_all.drop(df_oco_all[df_oco_all['SIF_740nm']<(-2+sif_std * df_oco_all['SIF_Uncertainty_740nm'])].index) 
    #) 
    df_oco_all = df_oco_all.reset_index(drop=True)

    print('shape of footprints after process: ',df_oco_all.shape)
    if df_oco_all.shape[0] > 2000000:
        df_oco_all = df_oco_all.sample(n=2000000)
        df_oco_all = df_oco_all.reset_index(drop=True)
    else:
        0==0
    return df_oco_all
    
def OCO_filter(df_oco_all, sif_low, sif_up):
    df_oco_all = df_oco_all[df_oco_all.Daily_SIF_740nm > sif_low]
    df_oco_all = df_oco_all.reset_index(drop=True)
    df_oco_all = df_oco_all[df_oco_all.Daily_SIF_740nm < sif_up]
    df_oco_all = df_oco_all.reset_index(drop=True)    
    return df_oco_all