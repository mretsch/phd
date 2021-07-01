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

rome = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/rome_10mmhour.nc')
rh = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/rh500.nc')
# div = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/Divergence900/div900_dailycycle.nc')\
#     .sel(time='2020-01-31T03:00:00')
# precip = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/pr_avg_0201to301.nc') * 3600
# precip = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/pr_20200210T0000_reggrid.nc')\
#              .sel(time='20200210T19:45:00') * 3600
# precip = xr.open_dataset(home+'/Documents/Data/Simulation/r2b10/pr_observations.HDF5', group='Grid')['precipitation']
# precip = xr.open_dataset(home+
#                          '/Documents/Data/Simulation/r2b10/3B-MO.MS.MRG.3IMERG.20200201-S000000-E235959.02.V06B.HDF5',
#                          group='Grid')['precipitation'].sel(lat=slice(-20, 20)).transpose()


# pr_latavg = precip.coarsen(lat=4, boundary='trim').mean()
# pr_avg = pr_latavg.coarsen(lon=4, boundary='trim').mean()
# rh_avg = rh.mean(dim='time')
rome_avg = rome.sum(dim='time') / len(rome.time)
rome_p90 = np.nanpercentile(rome, q=90)
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

    # west Australia: .sel(lat=slice(-20, -10), lon=slice(120, 134))
    # Amazonas Delta: .sel(lat=slice(-1, 5), lon=slice(-52, -44))
    # North Pacific: .where(coast_mask & ((160 < rome['lon']) | (rome['lon'] < -90)) & (0 < rome['lat']), other=np.nan)
    # South of India: .sel(lat=slice(-12, -6), lon=slice(77, 82))

    pac = rome_avg.where(ocean_mask &
                         ((160 < rome['lon']) | (rome['lon'] < -90)) &
                         (0 < rome['lat']), other=0)
    pa1 = rome_avg.where(ocean_mask &
                         ((170 < rome['lon']) | (rome['lon'] < -178)) &
                         ((6 < rome['lat']) & (rome['lat'] < 8)), other=0)
    pa2 = rome_avg.where(ocean_mask &
                         ((-145 < rome['lon']) & (rome['lon'] < -133)) &
                         ((6 < rome['lat']) & (rome['lat'] < 8)), other=0)
    pa3 = rome_avg.where(ocean_mask &
                         ((-145 < rome['lon']) & (rome['lon'] < -139)) &
                         ((14 < rome['lat']) & (rome['lat'] < 20)), other=0)
    aus = rome_avg.where(coast_mask &
                         ((120 < rome['lon']) & (rome['lon'] < 134)) &
                         ((-20 < rome['lat']) & (rome['lat'] < -10)), other=0)
    ama = rome_avg.where(((-52 < rome['lon']) & (rome['lon'] < -44)) &
                         ((-1 < rome['lat']) & (rome['lat'] < 5)), other=0)
    ind = rome_avg.where(((77 < rome['lon']) & (rome['lon'] < 82)) &
                         ((-12 < rome['lat']) & (rome['lat'] < -6)), other=0)
    allregions = pa1 + pa2 + pa3 + aus + ama + ind

    allregions = xr.where(allregions != 0., 1, np.nan)

    # allregions[:, :] = np.nan
    # allregions.loc[{'lon': rome_avg.sel(lon=126, method='nearest')['lon'],
    #                 'lat': rome_avg.sel(lat=-16, method='nearest')['lat']}] = 1

l_plot_map = False
if l_plot_map:

    # map_to_plot = rhlow_p90.sum(dim='time')
    map_to_plot = allregions

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

    map_to_plot.plot(ax=ax, cmap='cool')#, vmin=-0.0002, vmax=0.0002)
                     # cmap='PuOr' #cmap='OrRd' # cmap='PuRd' #cmap='gist_earth_r')# cmap='GnBu')#,
                     # vmin=0. , vmax=2.143688

    # ax.set_extent((120, 140, -10, -20), crs=ccrs.PlateCarree())

    # ax.collections[0].colorbar.set_label('Count ROME$_\mathrm{p90}$ & RH$_{500}$ < 40% [1]')
    # ax.collections[0].colorbar.set_label('Avg. ROME [km$^2$]')
    # ax.collections[0].colorbar.set_label('Precipitation [mm/hour]')
    # ax.collections[0].colorbar.set_label('900 hP div. [1/s]')
    # ax.collections[0].colorbar.ax.ticklabel_format(scilimits=(0, 0))

    # ax.set_title('Count ROME_p90 below 40% RH$_{500}$')
    # ax.set_title('Average precip Feb. 2020, DYAMOND-Winter nwp 2.5km')
    ax.set_title('') # '12:30$\,$pm')
    ax.set_xlabel('')
    ax.set_ylabel('')

    # ax.collections[0].colorbar.ax.set_anchor((-0.25, 0.))
    ax.set_xticks([30, 90, 150, 210, 270, 330], crs=ccrs.PlateCarree())
    ax.set_yticks([-20, -10, 0, 10, 20], crs=ccrs.PlateCarree())
    # ax.set_xticks([125, 135], crs=ccrs.PlateCarree())
    # ax.set_yticks([-10, -20], crs=ccrs.PlateCarree())
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    # ax.grid()

    plt.savefig(home+'/Desktop/map.pdf', bbox_inches='tight', transparent=True)
    # plt.savefig(home+'/Desktop/map', dpi=200, bbox_inches='tight', transparent=True)

l_plot_time = True
if l_plot_time:
    fig, ax = plt.subplots(figsize=(48, 3))

    rome_pac = rome_ocean.where(((160 < rome['lon']) | (rome['lon'] < -90)) & (0 < rome['lat']), other=np.nan)
    rome_pa1 = rome_ocean.where((( 170 < rome['lon']) | (rome['lon'] < -178)) & ((6 < rome['lat']) & (rome['lat'] < 8)), other=np.nan)
    rome_pa2 = rome_ocean.where(((-145 < rome['lon']) & (rome['lon'] < -133)) & ((6 < rome['lat']) & (rome['lat'] < 8)), other=np.nan)
    rome_pa3 = rome_ocean.where(((-145 < rome['lon']) & (rome['lon'] < -139)) & ((14 < rome['lat']) & (rome['lat'] < 20)), other=np.nan)
    rome_ama = rome.sel(lat=slice(-1, 5), lon=slice(-52, -44))
    rome_aus = rome_coast.sel(lat=slice(-20, -10), lon=slice(120, 134))
    rome_ind = rome.sel(lat=slice(-12, -6), lon=slice(77, 82))

    rome_domain = rome_pa1
    title_text = 'Pacific Region 1'

    rome_domain_high = rome_domain.where(rome_domain > rome_p90, other=np.nan)
    rome_avg = rome_domain_high.mean(dim='lat').mean(dim='lon')
    p0, = rome_avg.plot(color='k', label='ROME', lw=3)
    ax.axhline(y=rome_p90, color='grey', label='High ROME')

    # rh_cutout = rh.sel(lat=rome_domain['lat'], lon=rome_domain['lon'])
    # rh_domain = xr.where(rome_domain.notnull(), rh_cutout, np.nan)
    rh_domain = xr.where(rome_domain.notnull(), rh       , np.nan)
    # rh_domain_high = xr.where(rome_domain_high.notnull(), rh, np.nan)
    rh_avg = rh_domain.mean(dim='lat').mean(dim='lon')

    ax_1 = ax.twinx()
    p1, = ax_1.plot(rh_avg['time'], rh_avg, color=sol['violet'], label='RH$_{500}$', lw=3)
    ax_1.axhline(y=80, color='red')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_title('')
    ax.set_ylabel('ROME [km$^2$]')

    ax_1.set_ylim(-2, 102)
    ax_1.set_title(title_text) #, p90_domainpixel={round(rome_p90)}')
    ax_1.set_ylabel('RH$_{500}$ [1]', fontdict={'color': sol['violet']})

    plt.xlim(pd.to_datetime('20200131'), pd.to_datetime('20200301'))
    plt.xlabel('Time')
    plt.legend([p0, p1], [p.get_label() for p in [p0, p1]], loc='upper left')
    plt.savefig(home+'/Desktop/time.pdf', bbox_inches='tight', transparent=True)
    # plt.show()

stop = timeit.default_timer()
print(f'Time used: {stop - start}')

