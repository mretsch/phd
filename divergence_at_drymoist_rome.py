from os.path import expanduser
home = expanduser("~")
import timeit
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from Plotscripts.colors_solarized import sol


def smallregion_in_tropics(tropic_wide_field, region, surface_type, fillvalue):

    land_sea = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/land_sea_avg.nc')

    land_sea['lat'] = tropic_wide_field['lat']
    land_sea['lon'] = tropic_wide_field['lon']

    if surface_type == 'ocean':
        surface_mask = land_sea < 0.2
        field = tropic_wide_field.where(surface_mask, other=fillvalue)
    elif surface_type == 'coast':
        surface_mask = (0.2 < land_sea) & (land_sea < 0.8)
        field = tropic_wide_field.where(surface_mask, other=fillvalue)
    elif surface_type == 'land':
        surface_mask = land_sea > 0.8
        field = tropic_wide_field.where(surface_mask, other=fillvalue)
    else:
        field = tropic_wide_field

    if 'Pacific Region 1' in region:
        selected_lons = field.where(((170 < field['lon']) | (field['lon'] < -178)), other=fillvalue)
        small_region = field.sel(lat=slice(6, 8), lon=selected_lons['lon'].values)

    if 'Pacific Region 2' in region:
        small_region = field.sel(lat=slice(6, 8), lon=slice(-145, -133))

    if 'Pacific Region 3' in region:
        small_region = field.sel(lat=slice(14, 20), lon=slice(-145, -139))

    if 'Amazon Delta' in region:
        small_region = field.sel(lat=slice(-1, 5), lon=slice(-52, -44))

    if 'NW Australia' in region:
        small_region = field.sel(lat=slice(-20, -10), lon=slice(120, 134))

    if 'South of India' in region:
        small_region = field.sel(lat=slice(-12, -6), lon=slice(77, 82))

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
            # np.datetime64('2020-02-01T12:00'),
            # np.datetime64('2020-02-03T12:00'),
            # np.datetime64('2020-02-11T21:00'),
            # np.datetime64('2020-02-22T15:00'),
            # np.datetime64('2020-02-29T12:00'),
            # daily cycle:
            np.datetime64('2020-02-23T15:00'),
            np.datetime64('2020-02-24T12:00'),
            np.datetime64('2020-02-25T15:00'),
            np.datetime64('2020-02-26T09:00'),
            np.datetime64('2020-02-27T18:00'),
        ]

    if region == 'Amazon Delta':
        times = [
            np.datetime64('2020-02-03T09:00'),
            np.datetime64('2020-02-09T09:00'),
            np.datetime64('2020-02-12T09:00'),
            np.datetime64('2020-02-21T21:00'),
            np.datetime64('2020-02-24T15:00'),
        ]

    if region == 'South of India':
        times = [
            np.datetime64('2020-02-06T18:00'),
            np.datetime64('2020-02-09T21:00'), # original ROME-time is 02-10T00:15 but no div available at 00:00
            np.datetime64('2020-02-10T21:00'), # original ROME-time is 02-10T22:45 but no div available at 00:00
            np.datetime64('2020-02-13T18:00'),
            np.datetime64('2020-02-15T03:00'),
        ]

    return times


def composite_based_on_timeshift(list_of_arrays, n_hours, step, operation):

    crunched_series = []

    for hour_shift in np.arange(-n_hours, n_hours + step, step):

        raw_values = []
        for series in list_of_arrays:
            assert hasattr(series, 'coords')
            assert 'timeshift' in series.coords
            timeshift_as_dim = series.swap_dims({'time': 'timeshift'})

            try:
                raw_values.append(timeshift_as_dim.sel(timeshift=hour_shift).values.ravel())
            except KeyError:
                continue

        if operation == 'avg':
            crunched_series.append( np.nanmean(np.concatenate(raw_values)) )

        if operation == 'std':
            crunched_series.append( np.nanstd(np.concatenate(raw_values)) )

    return crunched_series


def daily_maxrome_time(high_freq_field):
    days = np.unique(high_freq_field['time'].values.astype('datetime64[D]').astype('str'))

    maxtime = []
    for day in days:
        one_day = high_freq_field.sel(time=day)
        mtime = one_day[one_day == one_day.max()]['time'].values

        if len(mtime) == 1:
            maxtime.append(np.datetime64(mtime.astype('str').item()))

    return maxtime


def nearest_div_time(time_list, low_freq_field):

    lowfreq_time = []
    for t in time_list:
        lowfreq_time.append(low_freq_field.sel(time=t, method='nearest')['time'].values)

    return [np.datetime64(one_time.astype('str').item()) for one_time in lowfreq_time]


if __name__ == '__main__':

    start = timeit.default_timer()
    plt.rc('font'  , size=20)

    rome_highfreq = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/rome_10mmhour.nc')
    rome_3h = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/rome_10mmhour_3hmaxavg.nc')
    rh_highfreq = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/rh500.nc')
    rh_3h = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/rh500_3hmaxavg_by_rome10mm.nc')
    div = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/Divergence900/div900_avg.nc').\
        transpose('time', 'lon', 'lat')

    rome = rome_3h.sel(time=div['time'].values)
    rh   = rh_3h.  sel(time=div['time'].values)

    rome_thresh = np.nanpercentile(rome_highfreq, 90)
    dry_thresh = 40
    moist_thresh = 70

    # rome_rh_mask = (rome > rome_thresh) & ((rh < dry_thresh) | (moist_thresh < rh))
    rome_mask = rome > rome_thresh
    rome_dry_mask = (rome > rome_thresh) &  (rh < dry_thresh)
    rome_moist_mask = (rome > rome_thresh) &  (moist_thresh < rh)

    n_hours = 12
    hour_step = 3
    i = -1
    fig, ax = plt.subplots(1, 1)
    div_vector_list = []
    single_timeslices = []
    r_mask = rome_mask
    region = 'South of India'
    # region = 'Amazon Delta'
    # region = 'NW Australia'
    # region = 'Pacific Region 1'
    surfacetype = 'ocean'


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
        region_mask = smallregion_in_tropics(tropic_wide_field=r_mask, region=region, surface_type=surfacetype,
                                             fillvalue=bool(0))

    l_select_few_times = True and not l_whole_tropics
    if l_select_few_times:

        rome_domain = smallregion_in_tropics(rome_highfreq, region, surface_type=surfacetype, fillvalue=np.nan)
        rome_domain_high = rome_domain.where(rome_domain > np.nanpercentile(rome_highfreq, 90), other=np.nan)
        rome_avg = rome_domain_high.stack({'z': ('lat', 'lon')}).mean(dim='z')

        relhum_cutout = rh_highfreq.sel(lat=rome_domain['lat'], lon=rome_domain['lon'])
        relhum_domain = xr.where(rome_domain.notnull(), relhum_cutout, np.nan)
        relhum_avg = relhum_domain.stack({'z': ('lat', 'lon')}).mean(dim='z')

        daily_maxtimes = daily_maxrome_time(rome_avg)

        l_dry_times   = (relhum_avg.sel(time=daily_maxtimes) < dry_thresh).values
        l_moist_times = (relhum_avg.sel(time=daily_maxtimes) > moist_thresh).values
        dry_maxtimes   = np.array(daily_maxtimes)[l_dry_times]
        moist_maxtimes = np.array(daily_maxtimes)[l_moist_times]

        few_more_times  = nearest_div_time(daily_maxtimes, div)
        few_dry_times   = nearest_div_time(dry_maxtimes  , div)
        few_moist_times = nearest_div_time(moist_maxtimes, div)

        cmp_avg, cmp_std, slice_avg = [], [], []
        for few_times in [few_dry_times, few_moist_times, few_more_times]:

            if len(few_times) == 0:
                continue

            single_timeslices = []
            for a_time in few_times:

                divselect = div.sel(time=slice(a_time - np.timedelta64(n_hours, 'h'),
                                               a_time + np.timedelta64(n_hours, 'h')),
                                    lat=region_mask['lat'],
                                    lon=region_mask['lon']
                                    ).load()

                divselect.coords['timeshift'] = ('time', ((divselect['time'] - a_time) / 3600e9).values.astype(int))

                latlon_mask = region_mask.sel(time=a_time)
                if latlon_mask.sum() == 0:
                    continue

                for t in divselect['time']:

                    divselect.loc[{'time': t}] = divselect.sel(time=t).where(latlon_mask)

                single_timeslices.append(divselect)

            composite_avg = composite_based_on_timeshift(single_timeslices, n_hours=n_hours, step=hour_step, operation='avg')
            composite_std = composite_based_on_timeshift(single_timeslices, n_hours=n_hours, step=hour_step, operation='std')

            comp_avg = xr.DataArray(np.zeros(len(composite_avg)),
                                         coords={'timeshift': np.arange(-n_hours, n_hours + hour_step, hour_step)},
                                         dims='timeshift')
            comp_std = xr.zeros_like(comp_avg)
            comp_avg[:] = composite_avg
            comp_std[:] = composite_std

            cmp_avg.append(comp_avg)
            cmp_std.append(comp_std)

            sum_by_singleslices   = xr.zeros_like(cmp_avg[0])
            count_by_singleslices = xr.zeros_like(cmp_avg[0])
            for series in single_timeslices:
                series_avg = series.stack({'z': ('lat', 'lon')}).mean(dim='z')
                sum_by_singleslices.loc[{'timeshift': series['timeshift']}] += series_avg.values
                count_by_singleslices.loc[{'timeshift': series['timeshift']}] += 1
            slice_avg.append(sum_by_singleslices / count_by_singleslices)




    ##### PLOTS ######



    colours = pd.cut(x=relhum_avg.sel(time=daily_maxtimes), bins=[0, 40, 70, 100], labels=['red', 'yellow', 'blue'])
    alphas  = pd.cut(x=relhum_avg.sel(time=daily_maxtimes), bins=[0, 40, 70, 100], labels=[0.3  , 0.0     , 0.3   ],
                     ordered=False)

    # plt.plot(cmp_avg[0]['timeshift'], cmp_avg[0], label=region, lw=2.5, color=sol['red'])
    plt.plot(cmp_avg[1]['timeshift'], cmp_avg[1], label=region, lw=2.5, color=sol['blue'])

    # plt.plot(slice_avg[0]['timeshift'], slice_avg[0], label=region, lw=2.5, color=sol['red'])
    # plt.plot(slice_avg[1]['timeshift'], slice_avg[1], label=region, lw=2.5, color=sol['blue'])

    # plt.fill_between(x=composite_avg['timeshift'],
    #                  y1=composite_avg - composite_std,
    #                  y2=composite_avg + composite_std,
    #                  alpha=0.1,
    #                  color=sol_col)

    # sol_col = [sol['yellow'], sol['blue'], sol['red'], sol['red'], sol['yellow'], ]  # Australia
    # sol_col = [sol['red'], sol['red'], sol['yellow'], sol['red'], sol['yellow'], ]  # Australia daily
    # sol_col = [sol['blue'], sol['blue'], sol['blue'], sol['blue'], sol['blue'], ]  # South of India
    # sol_col = [sol['red'], sol['yellow'], sol['yellow'], sol['yellow'], sol['yellow'], ]  # Amazon
    # sol_col = [sol['yellow'], sol['red'], sol['red'], sol['red'], sol['red'], ]  # Pacific 1
    # sol_col = [sol['yellow'], sol['blue'], sol['red'], sol['blue'], sol['yellow'], ]  # Pacific 2
    # sol_col = [sol['blue'], sol['yellow'], sol['red'], sol['red'], sol['blue'], ]  # Pacific 3


    for i, series in enumerate(single_timeslices):
        series_avg = series.stack({'z': ('lat', 'lon')}).mean(dim='z')
        plt.plot(series_avg['timeshift'], series_avg, color=sol[colours[i]], lw=0.5, marker='o', alpha=alphas[i])

    plt.axvline(x=0, color='lightgrey', zorder=0)
    plt.axhline(y=0, color='lightgrey', zorder=0)
    # plt.legend()
    plt.title(region)
    plt.ylim(-6e-5, 1.5e-5)
    plt.ylabel('Divergence [1/s]')
    plt.xlabel('Time around high ROME [h]')
    plt.xticks(ticks=np.arange(-n_hours, n_hours + hour_step, hour_step))
    plt.savefig(home+'/Desktop/div_composite_ind.pdf', bbox_inches='tight')
    plt.show()

    stop = timeit.default_timer()
    print(f'Time used: {stop - start}')