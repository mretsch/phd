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
from divergence_at_drymoist_rome import smallregion_in_tropics


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

# rome = rome.where(rome.notnull(), other=0.)
# rome_time_avg = rome.median(dim='time')
rome_time_avg = rome.sum(dim='time') / len(rome['time'])
rome_p90 = np.nanpercentile(rome, q=90)
rome_p20 = np.nanpercentile(rome, q=20)
l_rome_p90 = (rome > rome_p90)
rh_p90 = rh.where(l_rome_p90, other=np.nan)
rhlow_p90 = (rh_p90 < 40)
rome_p90_count = l_rome_p90.sum(dim='time')

# highest count of rome above 90th percentile
# rome_southofindia = rome.sel(lat=-8.064, lon=80.18, method='nearest')

# area = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/o_area_14mmhour.nc')
# area_avg = area.sum(dim='time') / len(area.time)

# number = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/o_number_14mmhour.nc')

# rome_ni = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/rome_ni_14mmhour.nc')
# delta_size = rome_ni - area
# delta_prox = rome - rome_ni

l_pick_region = True
if l_pick_region:

    pa1 = smallregion_in_tropics(rome_time_avg, 'Pacific Region 1', 'ocean', fillvalue=0.)
    pa2 = smallregion_in_tropics(rome_time_avg, 'Pacific Region 2', 'ocean', fillvalue=0.)
    pa3 = smallregion_in_tropics(rome_time_avg, 'Pacific Region 3', 'ocean', fillvalue=0.)
    aus = smallregion_in_tropics(rome_time_avg, 'NW Australia'    , 'coast', fillvalue=0.)
    ama = smallregion_in_tropics(rome_time_avg, 'Amazon Delta'    , 'all'  , fillvalue=0.)
    ind = smallregion_in_tropics(rome_time_avg, 'South of India'  , 'ocean', fillvalue=0.)

    allregions = xr.zeros_like(rome_time_avg)

    for region in [pa1, pa2, pa3, aus, ama, ind]:
        allregions.loc[{'lat': region['lat'], 'lon': region['lon']}] = region

    allregions = xr.where(allregions != 0., 1., np.nan)

l_plot_map = True
if l_plot_map:

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

    map_to_plot.plot(ax=ax,cmap='cool')#, vmin=0. , vmax=rome.mean(dim='time').max())#, vmin=-0.0002, vmax=0.0002)
                     # cmap='PuOr' cmap='GnBu' # cmap='PuRd' # cmap='OrRd' #cmap='gist_earth_r')# cmap='GnBu')#,

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

    ax.collections[0].colorbar.ax.set_anchor((-0.25, 0.))
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

l_plot_time = False
if l_plot_time:
    fig, ax = plt.subplots(figsize=(48, 3))

    rome_pa1 = smallregion_in_tropics(rome, 'Pacific Region 1', 'ocean', fillvalue=np.nan)
    rome_pa2 = smallregion_in_tropics(rome, 'Pacific Region 2', 'ocean', fillvalue=np.nan)
    rome_pa3 = smallregion_in_tropics(rome, 'Pacific Region 3', 'ocean', fillvalue=np.nan)
    rome_aus = smallregion_in_tropics(rome, 'NW Australia'    , 'coast', fillvalue=np.nan)
    rome_ama = smallregion_in_tropics(rome, 'Amazon Delta'    , 'all'  , fillvalue=np.nan)
    rome_ind = smallregion_in_tropics(rome, 'South of India'  , 'ocean', fillvalue=np.nan)

    rome_domain = rome_aus
    title_text = 'NW Australia'

    rome_times = [
        # np.datetime64('2020-02-01T10:45'),  # Australia
        # np.datetime64('2020-02-03T10:45'),  # Australia
        # np.datetime64('2020-02-11T19:45'),  # Australia
        # np.datetime64('2020-02-22T14:30'),  # Australia
        # np.datetime64('2020-02-29T11:45'),  # Australia

        np.datetime64('2020-02-23T13:45'),  # Australia daily
        np.datetime64('2020-02-24T11:00'),  # Australia daily
        np.datetime64('2020-02-25T15:00'),  # Australia daily
        np.datetime64('2020-02-26T10:00'),  # Australia daily
        np.datetime64('2020-02-27T18:00'),  # Australia daily

        # np.datetime64('2020-02-06T19:30'),  # South of India
        # np.datetime64('2020-02-10T00:15'),  # South of India
        # np.datetime64('2020-02-10T22:45'),  # South of India
        # np.datetime64('2020-02-13T17:00'),  # South of India
        # np.datetime64('2020-02-15T04:00'),  # South of India

        # np.datetime64('2020-02-03T09:00'),  # Amazon Delta
        # np.datetime64('2020-02-09T09:45'),  # Amazon Delta
        # np.datetime64('2020-02-12T09:00'),  # Amazon Delta
        # np.datetime64('2020-02-20T21:45'),  # Amazon Delta
        # np.datetime64('2020-02-24T14:15'),  # Amazon Delta

        # np.datetime64('2020-02-09T12:45'),  # Pacific Region 1
        # np.datetime64('2020-02-12T18:30'),  # Pacific Region 1
        # np.datetime64('2020-02-16T11:30'),  # Pacific Region 1
        # np.datetime64('2020-02-19T06:00'),  # Pacific Region 1
        # np.datetime64('2020-02-20T17:15'),  # Pacific Region 1

        # np.datetime64('2020-02-08T05:45'),  # Pacific Region 2
        # np.datetime64('2020-02-13T09:00'),  # Pacific Region 2
        # np.datetime64('2020-02-15T16:00'),  # Pacific Region 2
        # np.datetime64('2020-02-17T08:15'),  # Pacific Region 2
        # np.datetime64('2020-02-18T11:15'),  # Pacific Region 2

        # np.datetime64('2020-02-06T03:45'),  # Pacific Region 3
        # np.datetime64('2020-02-07T04:45'),  # Pacific Region 3
        # np.datetime64('2020-02-08T07:00'),  # Pacific Region 3
        # np.datetime64('2020-02-10T03:30'),  # Pacific Region 3
        # np.datetime64('2020-02-19T23:15'),  # Pacific Region 3
    ]

    rome_domain_high = rome_domain.where(rome_domain > rome_p90, other=np.nan)
    rome_avg = rome_domain_high.stack({'z': ('lat', 'lon')}).mean(dim='z')
    p0, = rome_avg.plot(color='k', label='ROME', lw=3)
    ax.axhline(y=rome_p90, color='grey', label='High ROME')

    relhum_cutout = rh.sel(lat=rome_domain['lat'], lon=rome_domain['lon'])
    relhum_domain = xr.where(rome_domain.notnull(), relhum_cutout, np.nan)
    relhum_avg = relhum_domain.stack({'z': ('lat', 'lon')}).mean(dim='z')

    for t in rome_times:
        if relhum_avg.sel(time=t) < 40.:
            colour = sol['red']
        elif relhum_avg.sel(time=t) > 70.:
            colour = sol['blue']
        else:
            colour = sol['yellow']
        ax.plot(rome_avg.sel(time=t)['time'], rome_avg.sel(time=t),
                ls='', marker='o', ms=17, color=colour, alpha=0.5)

    ax_1 = ax.twinx()
    p1, = ax_1.plot(relhum_avg['time'], relhum_avg, color=sol['violet'], label='RH$_{500}$', lw=3)
    # ax_1.axhline(y=80, color='red')

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
    plt.show()

stop = timeit.default_timer()
print(f'Time used: {stop - start}')

