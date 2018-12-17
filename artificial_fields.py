import netCDF4 as nc
import xarray as xr
import numpy as np

try:
    ds = xr.open_dataset('~/Desktop/CPOL_RADAR_ESTIMATED_RAIN_RATE_threedays.nc')
except FileNotFoundError:
    ds = xr.open_dataset('~/Data/RainRate/CPOL_RADAR_ESTIMATED_RAIN_RATE_threedays.nc')

# artificial field the same size as the radar data
# art[_,58,58] is the centre.
#art = xr.DataArray(np.zeros(shape=(10, len(ds.lat), len(ds.lon))))  # 'time', 'y', 'x'
art = xr.zeros_like(ds.radar_estimated_rain_rate[:8, :, :])

# the centre: art[0, 58: 58, 58: 58] = 1

# =========================

# large and small, apart
art[0, 53:64, 16: 27] = 2
art[0, 42:75, 80:113] = 2

# large and small, together
art[1, 53:64, 46:57] = 2
art[1, 42:75, 60:93] = 2

# =========================

# small row
art[2, 57: 60, 49: 52] = 2
art[2, 54: 63, 54: 63] = 2
art[2, 57: 60, 65: 68] = 2

# large row
art[3, 54: 63, 30: 39] = 2
art[3, 45: 72, 45: 72] = 2
art[3, 54: 63, 78: 87] = 2

# =========================

# contracted cross
art[4, 28: 39, 53: 64] = 2
art[4, 53: 64, 28: 39] = 2
art[4, 42: 75, 42: 75] = 2
art[4, 53: 64, 78: 89] = 2
art[4, 78: 89, 53: 64] = 2

# uniform cross
art[5,  6: 39, 42: 75] = 2
art[5, 42: 75,  6: 39] = 2
art[5, 42: 75, 42: 75] = 2
art[5, 42: 75, 78:111] = 2
art[5, 78:111, 42: 75] = 2

# =========================

# stick and square
art[6, 29: 88, 30: 39] = 2
art[6, 54: 63, 78: 87] = 2

# tilted stick and square
art[7, 54: 63,  5: 64] = 2
art[7, 54: 63, 78: 87] = 2

# =========================

# # small and small, together
# art[2, 53:64, 47:58] = 2
# art[2, 53:64, 59:70] = 2
#
# # large and large, together
# art[3, 42:75, 24:57] = 2
# art[3, 42:75, 60:93] = 2

# =========================

