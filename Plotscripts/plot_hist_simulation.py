from os.path import expanduser
home = expanduser("~")
import timeit
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
import cartopy.io as cio
from Plotscripts.colors_solarized import sol
from Plotscripts.plot_hist import histogram_2d

start = timeit.default_timer()
plt.rc('font'  , size=20)

rome = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/rome_14mmhour.nc')
rh500 = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/rh500.nc')
w500 = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/omega500.nc')

# rh_where_conv = xr.where(rome.notnull(), rh500, np.nan)
rh_np = np.array(rh500)
rh_where_conv = xr.DataArray(rh_np[rome.notnull()])
rh =rh_where_conv.rename({'dim_0': 'time'})

w_np = np.array(w500)
w_where_conv = xr.DataArray(w_np[rome.notnull()])
w =w_where_conv.rename({'dim_0': 'time'})

rome_np = np.array(rome)
rome_where_conv = xr.DataArray(rome_np[rome.notnull()])
r = rome_where_conv.rename({'dim_0': 'time'})

# make a 'scatter'-plot via a 2d-histogram
fig, ax = plt.subplots(nrows=1, ncols=1)#, figsize=(5, 5))
ax, h_2d = histogram_2d(w,
                        rh,
                        nbins=9,
                        l_cut_off=True,
                        ax=ax,
                        y_label='rh',
                        x_label='omega',
                        cbar_label='[%]')
fig.savefig(home + '/Desktop/hist.pdf', bbox_inches='tight')
h_2d.to_netcdf(home+'/Desktop/hist.nc', mode='w')

stop = timeit.default_timer()
print(f'Time used: {stop - start}')