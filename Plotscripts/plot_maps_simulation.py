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
# div = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/Divergence900/div900_timemean.nc')
# precip = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/pr_avg_0201to301.nc') * 3600
# precip = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/pr_20200210T0000_reggrid.nc').sel(time='20200210T19:45:00') * 3600
# precip = xr.open_dataset(home+'/Documents/Data/Simulation/r2b10/pr_observations.HDF5', group='Grid')['precipitation']
# precip = xr.open_dataset(home+'/Documents/Data/Simulation/r2b10/3B-MO.MS.MRG.3IMERG.20200201-S000000-E235959.02.V06B.HDF5',
#                          group='Grid')['precipitation'].sel(lat=slice(-20, 20)).transpose()


# pr_latavg = precip.coarsen(lat=4, boundary='trim').mean()
# pr_avg = pr_latavg.coarsen(lon=4, boundary='trim').mean()
rh_avg = rh.mean(dim='time')
rome_avg = rome.sum(dim='time') / len(rome.time)
rome_p90 = np.nanpercentile(rome, q=90)
# rome_avg = rome.max(dim='time')
l_rome_p90 = (rome > rome_p90)
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

l_pick_surface = True
if l_pick_surface:
    land_sea = xr.open_dataarray(home + '/Documents/Data/Simulation/r2b10/land_sea_avg.nc')

    ocean_mask = land_sea < 0.2  # ocean
    ocean_mask['lat'] = rome['lat']
    ocean_mask['lon'] = rome['lon']
    rome_ocean = rome.where(ocean_mask, other=np.nan)

    coast_mask = (0.2 < land_sea) & (land_sea < 0.8)  # coasts
    coast_mask['lat'] = rome['lat']
    coast_mask['lon'] = rome['lon']
    rome_coast = rome.where(coast_mask, other=np.nan)

l_plot_map = True
if l_plot_map:
    # west Australia: .sel(lat=slice(-20, -10), lon=slice(120, 134))
    # Amazonas Delta: .sel(lat=slice(-1, 5), lon=slice(-52, -44))
    # North Pacific: .where(coast_mask & ((160 < rome['lon']) | (rome['lon'] < -90)) & (0 < rome['lat']), other=np.nan)
    # South of India: .sel(lat=slice(-12, -6), lon=slice(77, 82))

    pac = rh_avg.where(ocean_mask &
                       ((160 < rome['lon']) | (rome['lon'] < -90)) &
                       (0 < rome['lat']), other=0)
    aus = rh_avg.where(coast_mask &
                       ((120 < rome['lon']) & (rome['lon'] < 134)) &
                       ((-20 < rome['lat']) & (rome['lat'] < -10)), other=0)
    ama = rh_avg.where(((-52 < rome['lon']) & (rome['lon'] < -44)) &
                       (( -1 < rome['lat']) & (rome['lat'] <   5)), other=0)
    ind = rh_avg.where((( 77 < rome['lon']) & (rome['lon'] <  82)) &
                       ((-12 < rome['lat']) & (rome['lat'] <  -6)), other=0)
    allregions = pac + aus + ama + ind
    allregions = xr.where(allregions != 0., 1, np.nan)

    # map_to_plot = rhlow_p90.sum(dim='time').sel(lat=slice(-1, 5), lon=slice(-52, -44))
    # map_to_plot = rhlow_p90.sum(dim='time').where(coast_mask & ((160 < rome['lon']) | (rome['lon'] < -90)) & (0 < rome['lat']), other=np.nan)
    # map_to_plot = rhlow_p90.sum(dim='time')
    map_to_plot = allregions
    # map_to_plot = pr_avg.squeeze()
    # map_to_plot = rome_avg.sel(lat=slice(-12, -6), lon=slice(77, 82))

    # n_roll = 140
    # center_lat = map_to_plot[:, -n_roll-1]['lon'].item() + 180
    # map_to_plot = map_to_plot.roll(shifts={'lon': n_roll}, roll_coords=False)

    # exploit that lons are now a view only of the lon-values, i.e. point into its memory
    longitude_offset = 180
    # lons = map_to_plot['lon'].values
    # lons -= longitude_offset
    map_to_plot['lon'] = map_to_plot['lon'] - longitude_offset
    lons = map_to_plot['lon']
    lons = xr.where(lons < -180, lons + 360, lons)

    fig, ax = make_map(projection=ccrs.PlateCarree(central_longitude=longitude_offset))
    ax.coastlines()
    # ax.axhline(y=0, color='r')

    map_to_plot.plot(ax=ax, cmap='cool')# cmap='OrRd') #cmap='gist_earth_r')# cmap='GnBu')#, vmin=0. , vmax=2.143688

    ax.collections[0].colorbar.set_label('Count ROME$_\mathrm{p90}$ & RH$_{500}$ < 40% [1]')
    # ax.collections[0].colorbar.set_label('ROME [km$^2$]')
    # ax.collections[0].colorbar.set_label('Precipitation [mm/hour]')
    # ax.collections[0].colorbar.set_label('Avg. 900 hPa divergence, [1/s]')
    # ax.set_title('Count ROME_p90 below 40% RH$_{500}$')
    # ax.set_title('Average precip Feb. 2020, DYAMOND-Winter nwp 2.5km')
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')

    ax.collections[0].colorbar.ax.set_anchor((-0.25, 0.))
    ax.set_xticks([30, 90, 150, 210, 270, 330], crs=ccrs.PlateCarree())
    ax.set_yticks([-20, -10, 0, 10, 20], crs=ccrs.PlateCarree())
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    # ax.grid()

    plt.savefig(home+'/Desktop/map.pdf', bbox_inches='tight', transparent=True)
    # plt.savefig(home+'/Desktop/map', dpi=200, bbox_inches='tight', transparent=True)

l_plot_time = False
if l_plot_time:
    fig, ax = plt.subplots(figsize=(48, 3))
    # rome_southofindia.plot()

    rome_pac = rome_ocean.where(((160 < rome['lon']) | (rome['lon'] < -90)) & (0 < rome['lat']), other=np.nan)
    rome_ama = rome.sel(lat=slice(-1, 5), lon=slice(-52, -44))
    rome_aus = rome_coast.sel(lat=slice(-20, -10), lon=slice(120, 134))
    rome_ind = rome.sel(lat=slice(-12, -6), lon=slice(77, 82))

    rome_domain = rome_aus
    legend_text = 'NW Australia'

    # rome_domain_high = rome_domain.where(rome_domain > np.nanpercentile(rome_domain, q=90), other=np.nan)
    rome_domain_high = rome_domain.where(rome_domain > rome_p90, other=np.nan)
    rome_avg = rome_domain_high.mean(dim='lat').mean(dim='lon')
    # ax.plot(rome_avg['time'], rome_avg, color='k', label=legend_text)
    rome_avg.plot(color='k', label=legend_text)
    # n_high_pixels = rome_domain_high.notnull().sum(dim=('lat', 'lon'))
    # total_domain = len(rome_domain['lat']) * len(rome_domain['lon'])
    # (n_high_pixels / total_domain).plot(label=legend_text)

    ax.axhline(y=rome_p90, color='r')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlim(pd.to_datetime('20200131'), pd.to_datetime('20200301'))
    plt.title(f'') #, p90_domainpixel={round(rome_p90)}')
    plt.ylabel('ROME [km$^2$]')
    plt.xlabel('Time')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.legend(loc='upper left')
    plt.savefig(home+'/Desktop/time.pdf', bbox_inches='tight', transparent=True)
    # plt.show()

stop = timeit.default_timer()
print(f'Time used: {stop - start}')

