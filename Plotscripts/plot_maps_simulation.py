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


def make_map(projection=ccrs.PlateCarree()):
    fig, ax = plt.subplots(figsize=(36,4),
                           subplot_kw=dict(projection=projection))
    #gl = ax.gridlines(draw_labels=True)
    #gl.xlabels_top = gl.ylabels_right = False
    #gl.xformatter = LONGITUDE_FORMATTER
    #gl.yformatter = LATITUDE_FORMATTER
    return fig, ax


start = timeit.default_timer()
plt.rc('font'  , size=20)

rome = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/rome_14mmhour.nc')
rh = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/rh500.nc')
# precip = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/pr_avg_0201to301.nc') * 3600

# rome_avg = rome.sum(dim='time') / len(rome.time)
# rome_avg = rome.max(dim='time')
l_rome_p90 = (rome > np.nanpercentile(rome, q=90))
rh_p90 = rh.where(l_rome_p90, other=np.nan)
rhlow_p90 = (rh_p90 < 40)

# rome_p90_count = (rome > np.nanpercentile(rome, q=90)).sum(dim='time')
# highest count of rome above 90th percentile
# rome_southofindia = rome.sel(lat=-8.064, lon=80.18, method='nearest')

# area = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/o_area_14mmhour.nc')
# area_avg = area.sum(dim='time') / len(area.time)

# number = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/o_number_14mmhour.nc')

# rome_ni = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/rome_ni_14mmhour.nc')
# delta_size = rome_ni - area
# delta_prox = rome - rome_ni

l_pick_surface = False
if l_pick_surface:
    land_sea = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/land_sea_avg.nc')
    # coast_mask = (0.2 < land_sea) & (land_sea < 0.8) # coasts
    coast_mask =  land_sea < 0.2                     # ocean
    coast_mask['lat'] = rome['lat']
    coast_mask['lon'] = rome['lon']
    rome_coast = rome.where(coast_mask, other=np.nan)

# west Australia: .sel(lat=slice(-20, -10), lon=slice(120, 134))
# Amazonas Delta: .sel(lat=slice(-1, 5), lon=slice(-52, -44))
# North Pacific: .where(coast_mask & ((160 < rome['lon']) | (rome['lon'] < -90)) & (0 < rome['lat']), other=np.nan)
# map_to_plot = rhlow_p90.sum(dim='time').sel(lat=slice(-1, 5), lon=slice(-52, -44))
# map_to_plot = rhlow_p90.sum(dim='time').where(coast_mask & ((160 < rome['lon']) | (rome['lon'] < -90)) & (0 < rome['lat']), other=np.nan)
# map_to_plot = precip.squeeze()
map_to_plot = rhlow_p90.sum(dim='time')

# n_roll = 140
# center_lat = map_to_plot[:, -n_roll-1]['lon'].item() + 180
# map_to_plot = map_to_plot.roll(shifts={'lon': n_roll}, roll_coords=False)

# exploit that lons are now a view only of the lon-values, i.e. point into its memory
lons = map_to_plot['lon'].values
lons -= 160
lons = xr.where(lons < -180, lons + 360, lons)

fig, ax = make_map(projection=ccrs.PlateCarree(central_longitude=160))
ax.coastlines()
ax.axhline(y=0, color='r')
map_to_plot.plot(ax=ax, cmap='rainbow')#, vmin=0. , vmax=2.5)# cmap='BrBG')#, )
# ax.pcolormesh(map_to_plot)#, vmin=0. , vmax=2.5)# cmap='BrBG')#, )
ax.collections[0].colorbar.set_label('Count ROME_p90 below 40% RH [1]')
# ax.collections[0].colorbar.set_label('[mm/hour]')
# ax.collections[0].colorbar.set_label('')
# ax.set_title('Count ROME_p90 below 40% RH$_{500}$')
# ax.set_title('Average precip Feb. 2020, DYAMOND-Winter nwp 2.5km')
ax.set_title('')

plt.savefig(home+'/Desktop/map.pdf', bbox_inches='tight', transparent=True)
# plt.show()

# fig, ax = plt.subplots(figsize=(48, 3))
# # rome_southofindia.plot()
# rome_domain = rome_coast.where(coast_mask & ((160 < rome['lon']) | (rome['lon'] < -90)) & (0 < rome['lat']), other=np.nan)
# rome_domain_high = rome_domain.where(rome_domain > np.nanpercentile(rome_domain, q=90), other=np.nan)
# rome_avg = rome_domain_high.mean(dim='lat').mean(dim='lon')
# rome_avg.plot()
# ax.axhline(y=np.nanpercentile(rome_domain, q=90), color='r')
# plt.title(f'P90 avg. NW-Aussie, p90_domainpixel={round(np.nanpercentile(rome_domain, q=90))}')
# ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
# plt.savefig(home+'/Desktop/time.pdf', bbox_inches='tight', transparent=True)

stop = timeit.default_timer()
print(f'Time used: {stop - start}')

