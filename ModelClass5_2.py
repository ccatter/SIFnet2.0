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
from skimage.metrics import structural_similarity as ssim
import optuna

import pytorch_ssim

# Module for Google Drive
#from google.colab import drive
#mount drive
#from google.colab import drive
#drive.mount('/content/drive')

#some settings
plt.rcParams.update({'font.size':14})

#own scripts
import sys
#sys.path.insert(1, '../')
import helpers

#%%

"""
#MOD1: benchmark model
class SR_SIF(nn.Module):
    def __init__(self, INPUT_CHANNELS):
        super(SR_SIF, self).__init__()
        
        #initial convolutional layer
        self.conv0 = nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=64, kernel_size=1, stride=1, padding=0)
        
        #residual block 1
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=3//2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=3//2)
        
        
        #a relu layer between residual blocks
        self.relu2 = nn.ReLU()

        #residual block 2
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=3//2)
        self.relu3 = nn.ReLU()  
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=3//2)
              
        self.relu3 = nn.ReLU()

        #residual block 3
        self.conv51 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=3//2)
        self.relu51 = nn.ReLU()  
        self.conv52 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=3//2)
        
        #two final convolutional layers with relu
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.relu4 = nn.ReLU() 

    def forward(self, img):
        #print('Sample in Model!')
        #initial convolutional layer
        res1 = self.conv0(img)
        
        #residual block 1
        out = self.conv1(res1)
        out = self.relu1(out)
        out = self.conv2(out)
        out = out + res1
        
        #a relu layer between residual blocks
        out = self.relu2(out)

        #residual block 2
        res2 = out
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = out + res2

        #ReLu layer
        out = self.relu3(out)

        #block 3
        out = self.conv51(out)
        out = self.relu51(out)
        out = self.conv52(out)

        #two final convolutional layers with relu
        out = self.conv5(out)
        out = self.relu4(out)
        out = self.conv6(out)
        
        #return output
        return out

"""
"""
#MOD2: more channels 
class SR_SIF(nn.Module):
    def __init__(self, INPUT_CHANNELS):
        super(SR_SIF, self).__init__()
        
        #initial convolutional layer
        self.conv0 = nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=128, kernel_size=1, stride=1, padding=0)
        
        #residual block 1
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=1, padding=3//2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=3//2)
        
        
        #a relu layer between residual blocks
        self.relu2 = nn.ReLU()

        #residual block 2
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=1, padding=3//2)
        self.relu3 = nn.ReLU()  
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=3//2)
              
        self.relu3 = nn.ReLU()

        #residual block 3
        self.conv51 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=3//2)
        self.relu51 = nn.ReLU()  
        self.conv52 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=3//2)
        
        #two final convolutional layers with relu
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.relu4 = nn.ReLU() 

    def forward(self, img):
        #print('Sample in Model!')
        #initial convolutional layer
        res1 = self.conv0(img)
        
        #residual block 1
        out = self.conv1(res1)
        out = self.relu1(out)
        out = self.conv2(out)
        out = out + res1
        
        #a relu layer between residual blocks
        out = self.relu2(out)

        #residual block 2
        res2 = out
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = out + res2

        #ReLu layer
        out = self.relu3(out)

        #block 3
        out = self.conv51(out)
        out = self.relu51(out)
        out = self.conv52(out)

        #two final convolutional layers with relu
        out = self.conv5(out)
        out = self.relu4(out)
        out = self.conv6(out)
        
        #return output
        return out
"""
"""
#MOD03: Reduced Model: one res block
class SR_SIF(nn.Module):
    def __init__(self, INPUT_CHANNELS):
        super(SR_SIF, self).__init__()
        
        #initial convolutional layer
        self.conv0 = nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=128, kernel_size=1, stride=1, padding=0)
        
        #residual block 1
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=1, padding=3//2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=3//2)
        
        #a relu layer between residual blocks
        self.relu2 = nn.ReLU()

        #conv layer
        self.conv51 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=3//2)
        
        #two final convolutional layers with relu
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, img):
        #print('Sample in Model!')
        #initial convolutional layer
        res1 = self.conv0(img)
        
        #residual block 1
        out = self.conv1(res1)
        out = self.relu1(out)
        out = self.conv2(out)
        out = out + res1
        
        #a relu layer between residual blocks
        out = self.relu2(out)

        #block 3
        out = self.conv51(out)

        #two final convolutional layers with relu
        out = self.conv5(out)
        out = self.conv6(out)
        
        #return output
        return out
"""
"""
#MOD04: only one res block with more channels
class SR_SIF(nn.Module):
    def __init__(self, INPUT_CHANNELS):
        super(SR_SIF, self).__init__()
        
        #initial convolutional layer
        self.conv0 = nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=128, kernel_size=1, stride=1, padding=0)
        
        #residual block 1
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=1, padding=3//2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=3//2)
        
        
        #a relu layer between residual blocks
        self.relu2 = nn.ReLU()

        #residual block 2
        #self.conv3 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=1, padding=3//2)
        #self.relu3 = nn.ReLU()  
        #self.conv4 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=3//2)
              
        #self.relu3 = nn.ReLU()

        #residual block 3
        self.conv51 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=3//2)
        self.relu51 = nn.ReLU()  
        self.conv52 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=3//2)
        
        #two final convolutional layers with relu
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.relu4 = nn.ReLU() 

    def forward(self, img):
        #print('Sample in Model!')
        #initial convolutional layer
        res1 = self.conv0(img)
        
        #residual block 1
        out = self.conv1(res1)
        out = self.relu1(out)
        out = self.conv2(out)
        out = out + res1
        
        #a relu layer between residual blocks
        out = self.relu2(out)


        #block 3
        out = self.conv51(out)
        out = self.relu51(out)
        out = self.conv52(out)

        #two final convolutional layers with relu
        out = self.conv5(out)
        out = self.relu4(out)
        out = self.conv6(out)
        
        #return output
        return out



# MOD 05: feature reduction model
class SR_SIF(nn.Module):
    def __init__(self, INPUT_CHANNELS):
        super(SR_SIF, self).__init__()
        
        #initial convolutional layer
        self.conv0 = nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=16, kernel_size=1, stride=1, padding=0)
        
        #residual block 1
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=3//2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=3//2)
        
        #a relu layer between residual blocks
        self.relu2 = nn.ReLU()

        #conv layer
        self.conv51 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=3//2)
        
        #two final convolutional layers with relu
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, img):
        #print('Sample in Model!')
        #initial convolutional layer
        res1 = self.conv0(img)
        
        #residual block 1
        out = self.conv1(res1)
        out = self.relu1(out)
        out = self.conv2(out)
        out = out + res1
        
        #a relu layer between residual blocks
        out = self.relu2(out)

        #block 3
        out = self.conv51(out)

        #two final convolutional layers with relu
        out = self.conv5(out)
        out = self.conv6(out)
        
        #return output
        return out


# MOD 05.1: feature reduction model
class SR_SIF(nn.Module):
    def __init__(self, INPUT_CHANNELS):
        super(SR_SIF, self).__init__()
        
        #initial convolutional layer
        self.conv0 = nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=8, kernel_size=1, stride=1, padding=0)
        
        #residual block 1
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=2, kernel_size=3, stride=1, padding=3//2)
        self.relu1 = nn.ReLU()
        #self.conv2 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=3//2)
        
        #a relu layer between residual blocks
        #self.relu2 = nn.ReLU()

        #conv layer
        #self.conv51 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=3//2)
        
        #two final convolutional layers with relu
        #self.conv5 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, img):
        #print('Sample in Model!')
        #initial convolutional layer
        res1 = self.conv0(img)
        
        #residual block 1
        out = self.conv1(res1)
        out = self.relu1(out)
        #out = self.conv2(out)
        #out = out + res1
        
        #a relu layer between residual blocks
        #out = self.relu2(out)

        #block 3
        #out = self.conv51(out)

        #two final convolutional layers with relu
        #out = self.conv5(out)
        out = self.conv6(out)
        
        #return output
        return out

# MOD 05.2: feature reduction model
class SR_SIF(nn.Module):
    def __init__(self, INPUT_CHANNELS):
        super(SR_SIF, self).__init__()
        
        #initial convolutional layer
        self.conv0 = nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=8, kernel_size=3, stride=1, padding=3//2)
        
        #residual block 1
        #self.conv1 = nn.Conv2d(in_channels=8, out_channels=2, kernel_size=3, stride=1, padding=3//2)
        self.relu1 = nn.ReLU()
        #self.conv2 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=3//2)
        
        #a relu layer between residual blocks
        #self.relu2 = nn.ReLU()

        #conv layer
        #self.conv51 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=3//2)
        
        #two final convolutional layers with relu
        #self.conv5 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, img):
        #print('Sample in Model!')
        #initial convolutional layer
        res1 = self.conv0(img)
        
        #residual block 1
        #out = self.conv1(res1)
        out = self.relu1(res1)
        #out = self.conv2(out)
        #out = out + res1
        
        #a relu layer between residual blocks
        #out = self.relu2(out)

        #block 3
        #out = self.conv51(out)

        #two final convolutional layers with relu
        #out = self.conv5(out)
        out = self.conv6(out)
        
        #return output
        return out

"""
# MOD 05.3: feature reduction model
class SR_SIF(nn.Module):
    def __init__(self, INPUT_CHANNELS):
        super(SR_SIF, self).__init__()
        
        #initial convolutional layer
        self.conv0 = nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=16, kernel_size=1, stride=1, padding=0)
        #residual block 1
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=3//2)
        self.conv12 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1, padding=0)
        
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=3//2)
        
        #a relu layer between residual blocks
        self.relu2 = nn.ReLU()

        #conv layer
        self.conv51 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=3//2)
        self.conv52 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, stride=1, padding=0)
        
        #two final convolutional layers with relu
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, img):
        #print('Sample in Model!')
        #initial convolutional layer
        
        res1 = self.conv0(img)
        
        #residual block 1
        out = self.conv1(res1)
        out = self.conv12(out)
        out = self.conv12(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = out + res1
        
        #a relu layer between residual blocks
        out = self.relu2(out)

        #block 3
        out = self.conv51(out)
        out = self.conv52(out)
        out = self.conv52(out)
        #two final convolutional layers with relu
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.relu2(out)
        #return output
        return out
"""
# MOD 05.4: feature reduction model testloss
class SR_SIF(nn.Module):
    def __init__(self, INPUT_CHANNELS):
        super(SR_SIF, self).__init__()
        
        #initial convolutional layer
        self.conv0 = nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=64, kernel_size=1, stride=1, padding=0)
        
        #residual block 1
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=8, kernel_size=3, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        #self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=3//2)
        
        #a relu layer between residual blocks
        #self.relu2 = nn.ReLU()

        #conv layer
        #self.conv51 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=3//2)
        
        #two final convolutional layers with relu
        #self.conv5 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, img):
        #print('Sample in Model!')
        #initial convolutional layer
        res1 = self.conv0(img)
        
        #residual block 1
        out = self.conv1(res1)
        out = self.relu1(out)
        #out = self.conv2(out)
        #out = out + res1
        
        #a relu layer between residual blocks
        #out = self.relu2(out)

        #block 3
        #out = self.conv51(out)

        #two final convolutional layers with relu
        #out = self.conv5(out)
        out = self.conv6(out)
        
        #return output
        return out
  

# MOD 05.5: feature reduction model
class SR_SIF(nn.Module):
    def __init__(self, INPUT_CHANNELS):
        super(SR_SIF, self).__init__()
        
        #initial convolutional layer
        self.conv0 = nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=16, kernel_size=(4,4), stride=1, padding=0)
        
        #residual block 1
        #self.conv1 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=3//2)
        self.relu1 = nn.ReLU()
        #self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=3//2)
        
        #a relu layer between residual blocks
        #self.relu2 = nn.ReLU()

        #conv layer
        #self.conv51 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=3//2)
        
        #two final convolutional layers with relu
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        #self.conv6 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(5,4), stride=1, padding=0)

    def forward(self, img):
        #print('Sample in Model!')
        #initial convolutional layer
        res1 = self.conv0(img)
        
        #residual block 1
        #out = self.conv1(res1)
        out = self.relu1(res1)
        #out = self.conv2(out)
        #out = out + res1
        
        #a relu layer between residual blocks
        #out = self.relu2(out)

        #block 3
        #out = self.conv51(out)

        #two final convolutional layers with relu
        out = self.conv5(out)
        #out = self.conv6(out)
        
        #return output
        return out
      
# MOD 06: feature reduction model without residual block
class SR_SIF(nn.Module):
    def __init__(self, INPUT_CHANNELS):
        super(SR_SIF, self).__init__()
        
        #initial convolutional layer
        self.conv0 = nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=16, kernel_size=1, stride=1, padding=0)
        
        #residual block 1
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=3//2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=3//2)
        
        #a relu layer between residual blocks
        self.relu2 = nn.ReLU()

        #conv layer
        self.conv51 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=3//2)
        
        #two final convolutional layers with relu
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, img):
        #print('Sample in Model!')
        #initial convolutional layer
        res1 = self.conv0(img)
        
        #residual block 1
        out = self.conv1(res1)
        out = self.relu1(out)
        out = self.conv2(out)
        
        #a relu layer between residual blocks
        out = self.relu2(out)

        #block 3
        out = self.conv51(out)

        #two final convolutional layers with relu
        out = self.conv5(out)
        out = self.conv6(out)
        
        #return output
        return out

"""

"""
# MOD 07: minor model with 4 conv layers
class SR_SIF(nn.Module):
    def __init__(self, INPUT_CHANNELS):
        super(SR_SIF, self).__init__()
        
        #initial convolutional layer
        self.conv0 = nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=16, kernel_size=1, stride=1, padding=0)

        #conv layer
        self.conv51 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=3//2)
        
        #two final convolutional layers with relu
        self.conv5 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, img):
        #print('Sample in Model!')
        #initial convolutional layer
        out = self.conv0(img)
        

        #block 3
        out = self.conv51(out)

        #two final convolutional layers with relu
        out = self.conv5(out)
        out = self.conv6(out)
        
        #return output
        return out

"""
"""
# MOD 08
class SR_SIF(nn.Module):
    def __init__(self, INPUT_CHANNELS):
        super(SR_SIF, self).__init__()
        
        #initial convolutional layer
        self.conv0 = nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=8, kernel_size=1, stride=1, padding=0)
        
        #residual block 1
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=3//2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=3//2)
        
        #a relu layer between residual blocks
        self.relu2 = nn.ReLU()

        #conv layer
        self.conv51 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=3//2)
        
        #two final convolutional layers with relu
        self.conv5 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, img):
        #print('Sample in Model!')
        #initial convolutional layer
        res1 = self.conv0(img)
        
        #residual block 1
        out = self.conv1(res1)
        out = self.relu1(out)
        out = self.conv2(out)
        out = out + res1
        
        #a relu layer between residual blocks
        out = self.relu2(out)

        #block 3
        out = self.conv51(out)

        #two final convolutional layers with relu
        out = self.conv5(out)
        out = self.conv6(out)
        
        #return output
        return out
"""