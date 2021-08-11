from os.path import expanduser
home = expanduser("~")
import timeit
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.ticker as ticker
from Plotscripts.colors_solarized import sol


def smallregion_in_tropics(tropic_wide_field, region, surface_type, other_surface_fillvalue):

    land_sea = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/land_sea_avg.nc')

    land_sea['lat'] = tropic_wide_field['lat']
    land_sea['lon'] = tropic_wide_field['lon']

    if surface_type == 'ocean':
        surface_mask = land_sea < 0.2
        field = tropic_wide_field.where(surface_mask, other=other_surface_fillvalue)
    elif surface_type == 'coast':
        surface_mask = (0.2 < land_sea) & (land_sea < 0.8)
        field = tropic_wide_field.where(surface_mask, other=other_surface_fillvalue)
    elif surface_type == 'land':
        surface_mask = land_sea > 0.8
        field = tropic_wide_field.where(surface_mask, other=other_surface_fillvalue)
    else:
        field = tropic_wide_field

    if 'Pacific Region 1' in region:
        selected_lons = field['lon'][((170 < field['lon']) | (field['lon'] < -178))]
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

    if 'Tropic' in region:
        small_region = field

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

    composite = xr.DataArray(np.zeros(len(crunched_series)),
                             coords={'timeshift': np.arange(-n_hours, n_hours + hour_step, hour_step)},
                             dims='timeshift')
    composite[:] = crunched_series

    return composite


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

    # area = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/o_area_10mmhour.nc')
    # number = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/o_number_10mmhour.nc')
    # rome_highfreq = area * number

    rh_highfreq = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/rh500.nc')

    div = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/Divergence900/div900_avg.nc').\
        transpose('time', 'lon', 'lat')

    rome_thresh = np.nanpercentile(rome_highfreq, 90)
    dry_thresh = 40
    moist_thresh = 70
    n_hours = 12
    hour_step = 3
    fig, ax = plt.subplots(1, 1)
    relhum_at_maxtime = []
    single_timeslices = []
    # region = 'South of India'
    region = 'Amazon Delta'
    # region = 'NW Australia'
    # region = 'Pacific Region 1'
    # region = 'Tropical coast'
    surfacetype = 'all'

    rome_domain = smallregion_in_tropics(rome_highfreq, region, surface_type=surfacetype,
                                         other_surface_fillvalue=np.nan)
    rome_domain_high = rome_domain.where(rome_domain > rome_thresh, other=np.nan)

    relhum_cutout = rh_highfreq.sel(lat=rome_domain['lat'], lon=rome_domain['lon'])
    relhum_domain = xr.where(rome_domain.notnull(), relhum_cutout, np.nan)

    spatial_stack = rome_domain_high.stack({'z': ('lat', 'lon')})

    for latlon in spatial_stack['z']:

        domain_rome_series = spatial_stack.sel(z=latlon)
        domain_maxtimes = daily_maxrome_time(domain_rome_series)
        domain_even_maxtimes = nearest_div_time(domain_maxtimes, div)

        lat, lon = latlon.values.item()[0], latlon.values.item()[1]

        for a_time in domain_even_maxtimes:

            divselect = div.sel(time=slice(a_time - np.timedelta64(n_hours, 'h'),
                                           a_time + np.timedelta64(n_hours, 'h')),
                                lat=lat,
                                lon=lon
                                ).load()
            divselect.coords['timeshift'] = ('time', ((divselect['time'] - a_time) / 3600e9).values.astype(int))

            single_timeslices.append(divselect)
            relhum_at_maxtime.append( relhum_domain.sel(time=a_time, lat=lat, lon=lon) )

    assert len(relhum_at_maxtime) == len(single_timeslices)
    moist_indices = np.arange(len(single_timeslices))[(np.array(relhum_at_maxtime) > moist_thresh)]
    moist_timeslices = [single_timeslices[j] for j in moist_indices]
    dry_indices   = np.arange(len(single_timeslices))[(np.array(relhum_at_maxtime) < dry_thresh)]
    dry_timeslices   = [single_timeslices[k] for k in   dry_indices]

    print('Timeslices done.')

    # composite_avg_all = composite_based_on_timeshift(single_timeslices, n_hours=n_hours, step=hour_step,
    #                                                  operation='avg')
    if len(moist_indices) != 0:
        composite_avg_moist = composite_based_on_timeshift(moist_timeslices, n_hours=n_hours, step=hour_step,
                                                           operation='avg')
        composite_std_moist = composite_based_on_timeshift(moist_timeslices, n_hours=n_hours, step=hour_step,
                                                           operation='std')
    if len(dry_indices) != 0:
        composite_avg_dry = composite_based_on_timeshift(dry_timeslices, n_hours=n_hours, step=hour_step,
                                                         operation='avg')
        composite_std_dry = composite_based_on_timeshift(dry_timeslices, n_hours=n_hours, step=hour_step,
                                                           operation='std')

    if len(dry_timeslices) != 0 & len(moist_timeslices) != 0:
        pvalue_ttest = []
        for shift in composite_avg_dry['timeshift']:
            pvalue_ttest.append(stats.ttest_ind_from_stats(
                mean1=composite_avg_dry.sel(timeshift=shift),
                std1=composite_std_dry.sel(timeshift=shift),
                nobs1=len(dry_timeslices),
                mean2=composite_avg_moist.sel(timeshift=shift),
                std2=composite_std_moist.sel(timeshift=shift),
                nobs2=len(moist_timeslices),
                equal_var=False
            ))

    ##### PLOTS ######

    colours = pd.cut(x=relhum_at_maxtime, bins=[0, dry_thresh, moist_thresh, 100], labels=['red', 'yellow', 'blue'])
    # alphas  = pd.cut(x=relhum_avg.sel(time=daily_maxtimes), bins=[0, 40, 70, 100], labels=[0.3  , 0.0     , 0.3   ],
    #                  ordered=False)

    # plt.plot(composite_avg_all['timeshift'], composite_avg_all, label=region, lw=2.5, color=sol['yellow'])
    if len(moist_indices) != 0:
        plt.fill_between(x=composite_avg_moist['timeshift'],
                         y1=composite_avg_moist - composite_std_moist,
                         y2=composite_avg_moist + composite_std_moist,
                         alpha=0.1,
                         color=sol['blue'])
        plt.plot(composite_avg_moist['timeshift'], composite_avg_moist, label=region, lw=2.5, color=sol['blue'])
    if len(dry_indices) != 0:
        plt.fill_between(x=composite_avg_dry['timeshift'],
                         y1=composite_avg_dry - composite_std_dry,
                         y2=composite_avg_dry + composite_std_dry,
                         alpha=0.1,
                         color=sol['red'])
        plt.plot(composite_avg_dry['timeshift'], composite_avg_dry, label=region, lw=2.5, color=sol['red'])

    l_plot_significance = True
    if l_plot_significance:
        ps = np.array([testresult.pvalue for testresult in pvalue_ttest])
        l_p_below_005 = ps < 0.05
        plt.plot(composite_avg_dry['timeshift'][l_p_below_005], np.repeat(-3.9e-5, repeats=l_p_below_005.sum()),
                 ls='', marker='x', color='k')

    # for i, series in enumerate(moist_timeslices):
    #     plt.plot(series['timeshift'], series, color=sol['blue'], lw=0.5, marker='o', alpha=0.2)
    #
    # for i, series in enumerate(dry_timeslices):
    #     plt.plot(series['timeshift'], series, color=sol['red'], lw=0.5, marker='o', alpha=0.2)

    plt.axvline(x=0, color='lightgrey', zorder=0)
    plt.axhline(y=0, color='lightgrey', zorder=0)
    plt.legend([f'Moist ({len(moist_timeslices)} samples)', f'Dry ({len(dry_timeslices)} samples)'])
    plt.title(region)
    plt.ylim(-4e-5, 5.e-6)
    plt.ylabel('Divergence [1/s]')
    plt.xlabel('Time around high ROME [h]')
    plt.xticks(ticks=np.arange(-n_hours, n_hours + hour_step, hour_step))
    plt.savefig(home+'/Desktop/div_composite.pdf', bbox_inches='tight')
    plt.show()

    stop = timeit.default_timer()
    print(f'Time used: {stop - start}')