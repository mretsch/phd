import netCDF4 as nc
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

ds = xr.open_dataset('~/Desktop/CPOL_RADAR_ESTIMATED_RAIN_RATE_threedays.nc')

# artificial field the same size as the radar data
# art[_,58,58] is the centre.
#art = xr.DataArray(np.zeros(shape=(10, len(ds.lat), len(ds.lon))))  # 'time', 'y', 'x'
art = xr.zeros_like(ds.radar_estimated_rain_rate[:10, :, :])

# large and small, together
art[0, 53:64, 46:57] = 1
art[0, 42:75, 60:93] = 1

# large and small, apart
art[1, 53:64, 16: 27] = 1
art[1, 42:75, 80:113] = 1

# small and small, together
art[2, 53:64, 47:58] = 1
art[2, 53:64, 59:70] = 1

# large and large, together
art[3, 42:75, 24:57] = 1
art[3, 42:75, 60:93] = 1

