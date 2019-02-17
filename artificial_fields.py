import netCDF4 as nc
import xarray as xr
import numpy as np

try:
    ds = xr.open_dataset('~/Data/Steiner/CPOL_STEINER_ECHO_CLASSIFICATION_threedays.nc')
except FileNotFoundError:
    ds = xr.open_dataset('~/Google Drive File Stream/My Drive/Data/CPOL_STEINER_ECHO_CLASSIFICATION_threedays.nc')

# artificial field the same size as the radar data
# art[_,58,58] is the centre.
#art = xr.DataArray(np.zeros(shape=(10, len(ds.lat), len(ds.lon))))  # 'time', 'y', 'x'
art = xr.zeros_like(ds.steiner_echo_classification[:14, :, :])

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

# large square surrounded by small ones
art[6, 42: 75, 42: 75] = 2
art[6, 78: 81, 57: 60] = 2
art[6, 78: 81, 36: 39] = 2
art[6, 57: 60, 36: 39] = 2
art[6, 36: 39, 36: 39] = 2
art[6, 36: 39, 57: 60] = 2
art[6, 36: 39, 78: 81] = 2
art[6, 57: 60, 78: 81] = 2
art[6, 78: 81, 78: 81] = 2

# no cross, only small and large square
art[7, 42: 75, 42: 75] = 2
art[7, 78: 81, 57: 60] = 2

# =========================

# vertical stick and square
art[8, 29: 88, 30: 39] = 2
art[8, 54: 63, 78: 87] = 2

# horizontal stick and square
art[9, 54: 63,  5: 64] = 2
art[9, 54: 63, 78: 87] = 2

# =========================

# large and small square
art[10, 42: 75, 31: 64] = 2
art[10, 54: 63, 78: 87] = 2

# same sized stick and small square
art[11,  9:108, 53: 64] = 2
art[11, 54: 63, 78: 87] = 2

# =========================

# small stick and small square
art[12, 35: 82, 49: 58] = 2
art[12, 54: 63, 78: 87] = 2
# art[12, 37: 80, 53: 64] = 2
# art[12, 54: 63, 78: 87] = 2

# large and small square
art[13, 35: 82, 11: 58] = 2
art[13, 54: 63, 78: 87] = 2
# art[13, 37: 80, 21: 64] = 2
# art[13, 54: 63, 78: 87] = 2

# =========================

