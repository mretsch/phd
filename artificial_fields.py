import netCDF4 as nc
import xarray as xr
import numpy as np

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

# uniform cross
art[4,  6: 39, 42: 75] = 1
art[4, 42: 75,  6: 39] = 1
art[4, 42: 75, 42: 75] = 1
art[4, 42: 75, 78:111] = 1
art[4, 78:111, 42: 75] = 1

# contracted cross
art[5, 28: 39, 53: 64] = 1
art[5, 53: 64, 28: 39] = 1
art[5, 42: 75, 42: 75] = 1
art[5, 53: 64, 78: 89] = 1
art[5, 78: 89, 53: 64] = 1

