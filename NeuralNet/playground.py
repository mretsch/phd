import numpy as np
import xarray as xr
from keras.layers import Dense
from keras.models import Sequential

ds_predictors = xr.open_dataset('/Volumes/GoogleDrive/My Drive/Data/LargeScale/CPOL_large-scale_forcing.nc')
predictor = ds_predictors.T
target = xr.open_dataarray('/Volumes/GoogleDrive/My Drive/Data_Analysis/rom.nc')


c = target[2:4]
c[0:2] = [3., 5.]
cre = c.resample(time='T0min').interpolate('linear')
print(cre.time)
print(c.time)

# is it a bug? I want 10min frequency but have to say 10-1 = 9
a = predictor[:2]
b = a.resample(time='T9min').interpolate('linear')


# https://datascience.stackexchange.com/questions/27506/back-propagation-in-cnn
from scipy import signal
o = np.array([(0.51, 0.9, 0.88, 0.84, 0.05),
              (0.4, 0.62, 0.22, 0.59, 0.1),
              (0.11, 0.2, 0.74, 0.33, 0.14),
              (0.47, 0.01, 0.85, 0.7, 0.09),
              (0.76, 0.19, 0.72, 0.17, 0.57)])
d = np.array([(0, 0, 0.0686, 0),
              (0, 0.0364, 0, 0),
              (0, 0.0467, 0, 0),
              (0, 0, 0, -0.0681)])

gradient = signal.convolve(np.rot90(np.rot90(d)), o, 'valid')
I = np.array([[3,0], [1,0]])
K = np.array([[0,2], [0,3]])
signal.convolve(I, K)