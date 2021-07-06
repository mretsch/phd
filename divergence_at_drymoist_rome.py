from os.path import expanduser
home = expanduser("~")
import timeit
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
import cartopy.io as cio
from Plotscripts.colors_solarized import sol



start = timeit.default_timer()
plt.rc('font'  , size=20)

rome_highfreq = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/rome_10mmhour.nc')
rome_3h = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/rome_10mmhour_3hmaxavg.nc')
rh_3h = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/rh500_3hmaxavg_by_rome10mm.nc')
div = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/Divergence900/div900_avg.nc')

rome = rome_3h.sel(time=div['time'].values)
rh   = rh_3h.  sel(time=div['time'].values)

rome_thresh = np.nanpercentile(rome_highfreq, 90)
dry_thresh = 40
moist_thresh = 70

rome_rh_mask = (rome > rome_thresh) & ((rh < dry_thresh) | (moist_thresh < rh))
# rome_rh_mask = (rome > rome_thresh) &  (rh < dry_thresh)

rome_and_rh    = xr.where(rome_rh_mask, rome, np.nan)
div_at_rome    = xr.where(rome_rh_mask, div , np.nan)
rome_rh_div    = rome_and_rh.where(div_at_rome.notnull())
rh_div         = xr.where(div_at_rome.notnull(), rh, np.nan)

div_flat  = xr.DataArray(np.ravel(div_at_rome))
rome_flat = xr.DataArray(np.ravel(rome_rh_div))
rh_flat   = xr.DataArray(np.ravel(rh_div))

div_vector  = div_flat[div_flat.notnull()]
rome_vector = rome_flat[rome_flat.notnull()]
rh_vector   = rh_flat[rh_flat.notnull()]

stop = timeit.default_timer()
print(f'Time used: {stop - start}')