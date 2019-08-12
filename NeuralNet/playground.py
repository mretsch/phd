import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import keras.models as kmodels
import keras.layers as klayers
import keras.utils as kutils

# ds_predictors = xr.open_dataset('/Volumes/GoogleDrive/My Drive/Data/LargeScale/CPOL_large-scale_forcing.nc')
# predictor = ds_predictors.T
# target = xr.open_dataarray('/Volumes/GoogleDrive/My Drive/Data_Analysis/rom.nc')
#
#
# c = target[2:4]
# c[0:2] = [3., 5.]
# cre = c.resample(time='T0min').interpolate('linear')
# print(cre.time)
# print(c.time)
#
# # is it a bug? I want 10min frequency but have to say 10-1 = 9
# a = predictor[:2]
# b = a.resample(time='T9min').interpolate('linear')
#
#
# # https://datascience.stackexchange.com/questions/27506/back-propagation-in-cnn
# from scipy import signal
# o = np.array([(0.51, 0.9, 0.88, 0.84, 0.05),
#               (0.4, 0.62, 0.22, 0.59, 0.1),
#               (0.11, 0.2, 0.74, 0.33, 0.14),
#               (0.47, 0.01, 0.85, 0.7, 0.09),
#               (0.76, 0.19, 0.72, 0.17, 0.57)])
# d = np.array([(0, 0, 0.0686, 0),
#               (0, 0.0364, 0, 0),
#               (0, 0.0467, 0, 0),
#               (0, 0, 0, -0.0681)])
#
# gradient = signal.convolve(np.rot90(np.rot90(d)), o, 'valid')
# I = np.array([[3,0], [1,0]])
# K = np.array([[0,2], [0,3]])
# signal.convolve(I, K)


# x = np.random.randint(1, 50, size=(50))
# y = np.square(x)
# model = kmodels.Sequential()
# model.add(klayers.Dense(2, activation='linear', input_shape=(1,)))
# model.add(klayers.Dense(2, activation='linear'))
# model.add(klayers.Dense(2, activation='linear'))
# model.add(klayers.Dense(2, activation='linear'))
# model.add(klayers.Dense(1, activation='linear'))
# model.compile(optimizer='adam', loss='mean_absolute_error')
# model.fit(x, y, batch_size=1, epochs=2000)
# model.predict([10])
# array([[46.559216]], dtype=float32)
# model.predict([20])
# array([[531.67847]], dtype=float32)
# model.predict([40])
# array([[1501.9172]], dtype=float32)
# model.predict([500])
# array([[23817.406]], dtype=float32)

# kutils.plot_model(model, to_file='a.png')
# data = plt.imread('a.png')
# plt.imshow(data)
# plt.show()

x = np.random.randint(1, 50, size=(50))
y = np.sqrt(x)
model = kmodels.Sequential()
model.add(klayers.Dense(150, activation='linear', input_shape=(1,)))
model.add(klayers.Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x, y, batch_size=1, epochs=2000)
# model.predict([9])
# array([[2.8867426]], dtype=float32)
# model.predict([25])
# array([[4.751915]], dtype=float32)