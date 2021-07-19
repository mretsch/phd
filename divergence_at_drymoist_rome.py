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


def smallregion_in_tropics(tropic_wide_field, region, surface_type, fillvalue):


    land_sea = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/land_sea_avg.nc')

    land_sea['lat'] = tropic_wide_field['lat']
    land_sea['lon'] = tropic_wide_field['lon']

    if surface_type == 'ocean':
        surface_mask = land_sea < 0.2
        field_ocean  = tropic_wide_field.where(surface_mask, other=fillvalue)
    if surface_type == 'coast':
        surface_mask = (0.2 < land_sea) & (land_sea < 0.8)
        field_coast  = tropic_wide_field.where(surface_mask, other=fillvalue)
    if surface_type == 'land':
        surface_mask = land_sea > 0.8
        field_land   = tropic_wide_field.where(surface_mask, other=fillvalue)

    field = tropic_wide_field

    if 'North Pacific' in region:
        pac = field_ocean.where(((160 < field_ocean['lon']) | (field_ocean['lon'] < -90)) &
                                (   0 < field_ocean['lat']),
                                other=fillvalue)
        small_region = pac
    if 'Pacific Region 1' in region:
        pa1 = field.where((( 170 < field['lon']) | (field['lon'] < -178)) &
                          ((   6 < field['lat']) & (field['lat'] <    8)),
                          other=fillvalue)
        small_region = pa1
    if 'Pacific Region 2' in region:
        pa2 = field.where(((-145 < field['lon']) & (field['lon'] < -133)) &
                          ((   6 < field['lat']) & (field['lat'] <    8)),
                          other=fillvalue)
        small_region = pa2
    if 'Pacific Region 3' in region:
        pa3 = field.where(((-145 < field['lon']) & (field['lon'] < -139)) &
                          ((  14 < field['lat']) & (field['lat'] <   20)),
                          other=fillvalue)
        small_region = pa3
    if 'Amazon delta' in region:
        ama = field.where((-52 < field['lon']) & (field['lon'] < -44) &
                          ( -1 < field['lat']) & (field['lat'] <   5),
                          other=fillvalue)
        small_region = ama
    if 'NW Australia' in region:
        aus = field_coast.where((120 < field_coast['lon']) & (field_coast['lon'] < 134) &
                                (-20 < field_coast['lat']) & (field_coast['lat'] < -10),
                                other=fillvalue)
        small_region = aus
    if 'South of India' in region:
        ind = field.where(( 77 < field['lon']) & (field['lon'] < 82) &
                          (-12 < field['lat']) & (field['lat'] < -6),
                          other=fillvalue)
        small_region = ind

    return small_region


def a_few_times_in_regions(region):
    if region == 'Pacific Region 1':
        times = [
            np.datetime64('2020-02-09T12:00'),
            np.datetime64('2020-02-12T18:00'),
            np.datetime64('2020-02-16T12:00'),
            np.datetime64('2020-02-19T06:00'),
            np.datetime64('2020-02-20T18:00'),
        ]

    if region == 'Pacific Region 2':
        times = [
            np.datetime64('2020-02-08T06:00'),
            np.datetime64('2020-02-13T09:00'),
            np.datetime64('2020-02-15T15:00'),
            np.datetime64('2020-02-17T09:00'),
            np.datetime64('2020-02-18T12:00'),
        ]

    if region == 'Pacific Region 3':
        times = [
            np.datetime64('2020-02-06T03:00'),
            np.datetime64('2020-02-07T06:00'),
            np.datetime64('2020-02-08T06:00'),
            np.datetime64('2020-02-10T03:00'),
            np.datetime64('2020-02-19T21:00'), # original ROME-time is 02-19T23:15 but no div available at 00:00
        ]

    if region == 'NW Australia':
        times = [
            np.datetime64('2020-02-01T12:00'),
            np.datetime64('2020-02-03T12:00'),
            np.datetime64('2020-02-11T21:00'),
            np.datetime64('2020-02-22T15:00'),
            np.datetime64('2020-02-29T12:00'),
        ]

    return times


def composite_based_on_timeshift(list_of_arrays, n_hours, operation):

    crunched_series = []

    time_step = 3
    for hour_shift in np.arange(-n_hours, n_hours + time_step, time_step):

        timedelta = np.timedelta64(hour_shift, 'h')

        raw_values = []
        for series in list_of_arrays:
            assert hasattr(series, 'coords')
            assert 'timeshift' in series.coords
            timeshift_as_dim = series.swap_dims({'time': 'timeshift'})

            try:
                raw_values.append(timeshift_as_dim.sel(timeshift=timedelta).values.ravel())
            except KeyError:
                continue

        if operation == 'avg':
            crunched_series.append( np.nanmean(np.concatenate(raw_values)) )

        if operation == 'std':
            crunched_series.append( np.nanstd(np.concatenate(raw_values)) )

    return crunched_series


start = timeit.default_timer()
plt.rc('font'  , size=20)

rome_highfreq = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/rome_10mmhour.nc')
rome_3h = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/rome_10mmhour_3hmaxavg.nc')
rh_3h = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/rh500_3hmaxavg_by_rome10mm.nc')
div = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/Divergence900/div900_avg.nc').\
    transpose('time', 'lon', 'lat')

rome = rome_3h.sel(time=div['time'].values)
rh   = rh_3h.  sel(time=div['time'].values)

rome_thresh = np.nanpercentile(rome_highfreq, 90)
dry_thresh = 40
moist_thresh = 70

# rome_rh_mask = (rome > rome_thresh) & ((rh < dry_thresh) | (moist_thresh < rh))
rome_dry_mask = (rome > rome_thresh) &  (rh < dry_thresh)
rome_moist_mask = (rome > rome_thresh) &  (moist_thresh < rh)

n_hours = 12
div_vector_list = []
i=-1
fig, ax = plt.subplots(1, 1)
for r_mask, region in zip([rome_dry_mask], ['Pacific Region 3']):

    i += 1

    l_whole_tropics = False
    if l_whole_tropics:
        short_rome_mask = r_mask.isel({'time': slice(4, -4)}).values

        rome_and_rh    = xr.where(r_mask, rome, np.nan)
        div_at_rome    = xr.where(r_mask, div , np.nan)
        rome_rh_div    = rome_and_rh.where(div_at_rome.notnull())
        rh_div         = xr.         where(div_at_rome.notnull(), rh, np.nan)

        l_concat_to_vectors = False
        if l_concat_to_vectors:
            div_flat  = xr.DataArray(np.ravel(div_at_rome))
            rome_flat = xr.DataArray(np.ravel(rome_rh_div))
            rh_flat   = xr.DataArray(np.ravel(rh_div))

            div_vector_list.append(div_flat[div_flat.notnull()])
            rome_vector = rome_flat[rome_flat.notnull()]
            rh_vector   = rh_flat[rh_flat.notnull()]

            ax.hist(div_vector_list[-1], bins=np.linspace(-5e-5, 2.5e-5, 40), density=True)

    else: # select by region
        region_mask = smallregion_in_tropics(tropic_wide_field=r_mask, region=region, surface_type='coast',
                                             fillvalue=bool(0))

    l_select_few_times = True and not l_whole_tropics
    if l_select_few_times:
        few_times = a_few_times_in_regions(region=region)

    single_timeslices, avg_7timesteps, std_7timesteps = [], [], []

    for a_time in few_times:

        divselect = div.sel(time=slice(a_time - np.timedelta64(n_hours, 'h'),
                                       a_time + np.timedelta64(n_hours, 'h')))

        divselect.coords['timeshift'] = 'time', (divselect['time'] - a_time)

        single_timeslices.append(divselect.where(region_mask))

    comp_avg = composite_based_on_timeshift(single_timeslices, n_hours=n_hours, operation='avg')
    comp_std = composite_based_on_timeshift(single_timeslices, n_hours=n_hours, operation='std')

    composite_avg = xr.DataArray(np.zeros(len(comp_avg)),
                                 coords={'timeshift': np.arange(-n_hours, n_hours + 3, 3)},
                                 dims='timeshift')
    composite_std = xr.zeros_like(composite_avg)

    composite_avg[:] = comp_avg
    composite_std[:] = comp_std




    if i==0:
        sol_col = sol['red']
    else:
        sol_col = sol['blue']
    plt.plot(composite_avg['timeshift'], composite_avg, label=region, lw=2, color=sol_col)
    plt.fill_between(x=composite_avg['timeshift'],
                     y1=composite_avg - composite_std,
                     y2=composite_avg + composite_std,
                     alpha=0.1,
                     color=sol_col)

    for series in single_timeslices:
        series.stack({'z': ('lat', 'lon')})
        series_avg.append()

plt.axvline(x=0, color='lightgrey', zorder=0)
plt.axhline(y=0, color='lightgrey', zorder=0)
plt.legend()
plt.ylim(-2e-5, 0.5e-5)
plt.ylabel('Divergence [1/s]')
# plt.xlim(0, len(avg_7timesteps)-1)
plt.xlabel('Time around high ROME [h]')
# plt.xticks(ticks=np.arange(len(avg_7timesteps)), labels=['-12', '-9', '-6', '-3', '0', '3', '6', '9', '12'])
plt.savefig(home+'/Desktop/div_composite.pdf', bbox_inches='tight')
plt.show()

stop = timeit.default_timer()
print(f'Time used: {stop - start}')