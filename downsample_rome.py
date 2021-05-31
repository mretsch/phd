from os.path import expanduser
import timeit
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

home = expanduser("~")


def maximum_time(x):

    # x.name = 'soso'
    # x = x.rename({'time', 'gonzo'})

    x_max = x[x == x.max()]
    if len(x_max) > 1:
        x_return = x_max[0]['time']
        # print(f'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA, {x_return}')
        # x_return.load()
    else:
        x_return = x_max['time']
        # print(f'HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH, {x_return}')
        # x_return.load()

    # x_return.name = 'soso'
    # x_return = x_return.rename({'time': 'gonzo'})

    # if len(x_return.dims) > 1:
    #     print('KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK')
    # if len(x_return) > 1:
    #     print('TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT')

    return x_return
    # return x_return['gonzo']
    # return x_return + np.timedelta64(1, 'm')
    # return x_return.values # no, the numpy array does not have the time dimension attached
    # return x[1]['time']
    # if the 19:30 value is responsible, because the earlier max-time was 19:15, then change it by 1 minute


def average_around_max(series, m_max, **rsmpl):

    series_rsmpl = series.resample(indexer={'time': rsmpl['sample_period']}, skipna=False,
                                      closed=rsmpl['closed_side'], label=rsmpl['closed_side'],
                                      base=rsmpl['sample_center'], loffset=rsmpl['center_time'])

    time_of_max = series_rsmpl.apply(maximum_time)

    assert m_max.notnull().sum() == time_of_max.notnull().sum()

    # time_20min_before = time_of_max - np.timedelta64(20, 'm')
    time_10min_before = time_of_max - np.timedelta64(15, 'm')
    time_10min_after = time_of_max + np.timedelta64(15, 'm')
    # time_20min_after = time_of_max + np.timedelta64(20, 'm')

    # l_time_available = time_20min_before.isin(metric.time) & \
    #                    time_10min_before.isin(metric.time) & \
    #                    time_10min_after .isin(metric.time) & \
    #                    time_20min_after .isin(metric.time)
    l_time_available = time_10min_before.isin(series.time) & \
                       time_10min_after.isin(series.time)

    time_not_available = time_of_max[np.logical_not(l_time_available)]

    # rome0 = metric.sel(time=time_20min_before[l_time_available])
    rome1 = series.sel(time=time_10min_before[l_time_available])
    rome2 = series.sel(time=time_of_max[l_time_available])
    rome3 = series.sel(time=time_10min_after[l_time_available])
    # rome4 = metric.sel(time=time_20min_after [l_time_available])
    # array_rome = np.array([rome0, rome1, rome2, rome3, rome4])
    array_rome = np.array([rome1, rome2, rome3])

    avg_maximum = np.nanmean(array_rome, axis=0)


    # For ROME from C-POL at Darwin
    m_max.loc[{'time': m_max.sel(time=time_not_available, method='nearest').time}] = np.nan
    m_max[m_max.notnull()] = avg_maximum

    # For ROME from simulation
    # m_maxavg = xr.full_like(m_max, fill_value=np.nan)
    # m_maxavg.loc[{'time': m_max.sel(time=time_of_max[l_time_available], method='backfill').time}] = avg_maximum
    # m_maxavg.loc[{'time': m_max.sel(time=time_of_max[l_time_available], method='nearest').time}] = avg_maximum

    m_max.coords['percentile'] = m_max.rank(dim='time', pct=True)

    return m_max


def downsample_timeseries(series, **kwargs):
                          # sample_period, closed_side, sample_center, center_time, n_sample):

    series_resample = series.resample(indexer={'time': kwargs['sample_period']}, skipna=False,
                                      closed=kwargs['closed_side'], label=kwargs['closed_side'],
                                      base=kwargs['sample_center'], loffset=kwargs['center_time'])

    # take means over 6 hours each, starting at 3, 9, 15, 21 h. The time labels are placed in the middle of
    # the averaging period. Thus the labels are aligned to the large scale data set.
    # For reasons unknown, averages crossing a day of no data, not even NaN, into a normal day have wrongly
    # calculated averages. Overwrite manually with correct values.
    # Take sum and divide by 36 time steps (for 10 min intervals in 6 hours), to avoid one single value in 6 hours
    # (with the rest NaNs) to have that value as the average. sum()/36. treats NaNs as Zeros basically.
    m_avg = series_resample.sum() / kwargs['n_sample']

    # .sum() applied to NaNs in xarray yields Zero, not NaN as with .mean().
    # So put NaNs where .mean() yields NaNs.
    m_mean = series_resample.mean()
    m_avg = m_avg.where(m_mean.notnull(), other=np.nan)

    manual_overwrite = False
    if manual_overwrite:
        m_avg.loc[
            [np.datetime64('2003-03-15T00:00'), np.datetime64('2003-03-17T00:00'), np.datetime64('2003-10-30T00:00'),
             np.datetime64('2003-11-25T00:00'), np.datetime64('2006-11-11T00:00')]] = \
            [series.sel(time=slice('2003-03-14T21', '2003-03-15T02:50')).sum() / 36.,
             series.sel(time=slice('2003-03-16T21', '2003-03-17T02:50')).sum() / 36.,
             series.sel(time=slice('2003-10-29T21', '2003-10-30T02:50')).sum() / 36.,
             series.sel(time=slice('2003-11-24T21', '2003-11-25T02:50')).sum() / 36.,
             series.sel(time=slice('2006-11-10T21', '2006-11-11T02:50')).sum() / 36.]

    m_avg.coords['percentile'] = m_avg.rank(dim='time', pct=True)

    m_max = series_resample.max()

    m_maxavg = average_around_max(series, m_max, **kwargs)

    return m_maxavg

if __name__=='__main__':

    start = timeit.default_timer()

    # metric = xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/rom_kilometres.nc')
    # metric = xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/o_area.nc') * 6.25
    # metric = xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/o_number.nc') \
    #        * xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/o_area.nc') * 6.25
    metric = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/rome_14mmhour.nc') \
        # .sel(lat=-16, lon=126, method='nearest') # land-domain at Kimberley coast

    kwa = {
        'sample_period' : '3H'     , # '3H'   ,  # '6H'
        'sample_center' : 1.5      , # 0      ,  # 3
        'center_time'   : '1H30min', # '0H'   ,  # '3H'
        'closed_side'   : 'left'   , # 'right',  # 'left'
        'n_sample'      : 12       , # 12     ,  # 36
    }


    # hand-pick 4 domains in each regions of interest

    # Kimberley coast & Darwin
    aus = [
        [125.33, -16.02],
        [125.33, -14.70],
        [126.66, -14.70],
        [131.97, -12.04],
    ]

    ind = [
        [81.52, -6.73],
        [80.20, -6.73],
        [80.20, -8.06],
        [81.52, -8.06],
    ]

    ama = [
        [-51.22, 1.22],
        [-49.89, 1.22],
        [-48.56, 1.22],
        [-45.91, 2.55],
    ]

    pac = [
        [-136.18,  6.53],
        [-142.81, 15.83],
        [-142.81,  6.53],
        [ 174.45,  6.53],
    ]

    all_coordinates = aus + ind + ama + pac

    rome_3h = []
    for lon, lat in all_coordinates:
        domain_series = metric.sel(lat=lat, lon=lon, method='nearest')
        rome_3h.append( downsample_timeseries(domain_series, **kwa) )


    #    # # get time difference of maximum ROME time and large-scale time
    #    # time_max_avail = time_of_max[l_time_available]
    #    # ls_time = m_max.sel(time=time_max_avail, method='nearest').time
    #    # time_diff_list = [(ls_time[i] - time_max_avail[i]).values.item() for i in range(len(ls_time))]
    #    # time_diff = xr.full_like(m_max, fill_value=np.nan)
    #    # time_diff[m_max.notnull()] = time_diff_list
    #    # time_diff /= 60e9 # timedelta in minutes instead of nanoseconds
    #    # time_diff.attrs['units'] = 'seconds'
    #    # time_diff.attrs['long_name'] = 'timedelta_LargeScale_maxROME'
    #    # del time_diff['percentile']
    #
    #
    #    l_split_and_redistribute = False
    #    if l_split_and_redistribute:
    #        # metric = xr.open_dataarray('/Volumes/GoogleDrive/My Drive/Data_Analysis/rom_kilometres_avg6h.nc')
    #        metric = m_avg
    #
    #        # ROME-value at given percentile
    #        threshold = metric[abs((metric.percentile - 0.85)).argmin()]
    #        n_above_thresh = (metric > threshold).sum().item()
    #        sample_ind = xr.DataArray(np.zeros(shape=2 * n_above_thresh))
    #        sample_ind[:] = -1
    #
    #        # find arguments (meaning indizes) for the highest ROME-values
    #        m_present = metric.where(metric.notnull(), drop=True)
    #        sort_ind = m_present.argsort()
    #        sample_ind[-n_above_thresh:] = sort_ind[-n_above_thresh:]
    #
    #        # stride through ROME-values (not the percentiles or sorted indizes) linearly
    #        # With this method some indizes might be taken twice as they have shortest distance to two ROME-values.
    #        # We sort that out below.
    #        check_values = np.linspace(6.25, threshold, n_above_thresh)
    #        for i, v in enumerate(check_values):
    #            ind = abs((m_present - v)).argmin()
    #            sample_ind[i] = ind
    #
    #        unique, indizes, inverse, count = np.unique(sample_ind, return_index=True, return_inverse=True, return_counts=True)
    #
    #        # take the samples in samples_ind which recreate the outcome of the unique function, which orders the unique values.
    #        # Here thats the order of indizes we use to get a sample from ROME-values. Hence they will be timely ordered. Okay.
    #        sample_ind_unique = sample_ind[indizes]
    #
    #        metric_sample = m_present[sample_ind_unique.astype(int)]
    #        sample = metric_sample.rename({'dim_0': 'time'})
    #        sample.to_netcdf(home + '/Desktop/rom_sample.nc')

    stop = timeit.default_timer()
    print('This script needed {} seconds.'.format(stop-start))
