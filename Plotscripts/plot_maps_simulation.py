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

start = timeit.default_timer()
plt.rc('font'  , size=20)

rome = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/rome_14mmhour.nc')
rh = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/rh500.nc')

rome_avg = rome.sum(dim='time') / len(rome.time)
# rome_avg = rome.max(dim='time')
l_rome_p90 = (rome > np.nanpercentile(rome, q=90))
rh_p90 = rh.where(l_rome_p90, other=np.nan)
rhlow_p90 = (rh_p90 < 40)

rome_p90_count = (rome > np.nanpercentile(rome, q=90)).sum(dim='time')
# highest count of rome above 90th percentile
rome_southofindia = rome.sel(lat=-8.064, lon=80.18, method='nearest')

area = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/o_area_14mmhour.nc')
area_avg = area.sum(dim='time') / len(area.time)

number = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/o_number_14mmhour.nc')

rome_ni = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/rome_ni_14mmhour.nc')
delta_size = rome_ni - area
delta_prox = rome - rome_ni

land_sea = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/land_sea_avg.nc')
coast_mask = (0.2<land_sea) & (land_sea<0.8)
coast_mask['lat'] = rome['lat']
coast_mask['lon'] = rome['lon']
rome_coast = rome.where(coast_mask, other=np.nan)



def make_map(projection=ccrs.PlateCarree()):
    fig, ax = plt.subplots(figsize=(36,4),
                           subplot_kw=dict(projection=projection))
    #gl = ax.gridlines(draw_labels=True)
    #gl.xlabels_top = gl.ylabels_right = False
    #gl.xformatter = LONGITUDE_FORMATTER
    #gl.yformatter = LATITUDE_FORMATTER
    return fig, ax

# ratio = rome / area
# map_to_plot = (delta_prox.sum(dim='time') / delta_size.sum(dim='time')) # rome.mean(dim='time')# ratio.sum(dim='time') / len(ratio.time)
# map_to_plot = delta_prox.sum(dim='time') / len(rome.time)
# map_to_plot = rhlow_p90.sum(dim='time').sel(lat=slice(-20, -10), lon=slice(120, 150))
# map_to_plot = (rome > np.nanpercentile(rome, q=90)).sum(dim='time')#.sel(lat=slice(-15, 0), lon=slice(70, 90))
map_to_plot = rhlow_p90.sum(dim='time').where(coast_mask, other=np.nan).sel(lat=slice(-20, -10), lon=slice(120, 134))

fig, ax = make_map(projection=ccrs.PlateCarree())
ax.coastlines()
map_to_plot.plot(ax=ax, cmap='rainbow')# cmap='BrBG')#, vmin=0. , vmax=0.3)

# ax.collections[0].colorbar.set_label('RH_500 [1]')
ax.collections[0].colorbar.set_label('Count ROME_p90 below 40% RH [1]')
# ax.collections[0].colorbar.set_label('$\Delta_\mathrm{prox}$ [km$^2$]')
# ax.collections[0].colorbar.set_label('$\Delta_\mathrm{prox}$ / $\Delta_\mathrm{size}$ [1]')

# fig, ax = plt.subplots(figsize=(48, 3))
# rome_southofindia.plot()
# ax.axhline(y=np.nanpercentile(rome, q=90), color='r')
# ax.xaxis.set_major_locator(ticker.MultipleLocator(1))


plt.savefig(home+'/Desktop/map14.pdf', bbox_inches='tight', transparent=True)
# plt.show()

stop = timeit.default_timer()
print(f'Time used: {stop - start}')

