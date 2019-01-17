import xarray as xr
import numpy as np


radar = xr.open_dataarray('/Users/matthiasretsch/Google Drive File Stream/My Drive/Data/CPOL_STEINER_ECHO_CLASSIFICATION_threedays.nc')

rand = np.random.random(radar.shape)

field = xr.where(radar.isnull(), radar, rand)