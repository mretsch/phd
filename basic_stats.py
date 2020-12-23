from os.path import expanduser
from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np
# from dask.distributed import Client
import bottleneck as bn
import timeit
home = expanduser("~")


def into_pope_regimes(series, l_upsample=True, l_percentile=False, l_all=False):
    """Mask radar/metric time series according to the 5 possible Pope regimes."""
    # get the Pope regimes per day
    dfr_pope = pd.read_csv(home+'/Documents/Data/PopeRegimes/Pope_regimes.csv',
                           header=None, names=['timestring', 'regime'], index_col=0)
    dse = pd.Series(dfr_pope['regime'])

    da_pope = xr.DataArray(dse)
    pope_years = da_pope.sel({'timestring': slice('1998-12-06', '2017-03-31')})
    pope_years.coords['time'] = ('timestring', pd.to_datetime(pope_years.timestring.values))
    pope = pope_years.swap_dims({'timestring': 'time'})
    del pope['timestring']

    if l_percentile:
        try:
            var = series.percentile * 100
        except AttributeError:
            var =series
    else:
        var = series

    if not l_upsample:
        daily = var.resample(time='1D', skipna=True).mean()
        var = daily.sel({'time': pope.time})
    else:
        pope = pope.resample(time='6H').interpolate('zero')

    # filter each Pope regime
    pope = pope.where(var.notnull())
    var_p1 = var.where(pope == 1)
    var_p2 = var.where(pope == 2)
    var_p3 = var.where(pope == 3)
    var_p4 = var.where(pope == 4)
    var_p5 = var.where(pope == 5)
    if l_all:
        ds = xr.Dataset({'var_all': var.where(pope), 'var_p1': var_p1, 'var_p2': var_p2, 'var_p3': var_p3, 'var_p4': var_p4, 'var_p5': var_p5})
    else:
        ds = xr.Dataset({'var_p1': var_p1, 'var_p2': var_p2, 'var_p3': var_p3, 'var_p4': var_p4, 'var_p5': var_p5})
    return ds


def diurnal_cycle(series, group='time', frequency='10T', period=144, time_shift=9*6+3):
    """Compute grouped average of time series.
    Diurnal cycle for 10-minute data in Darwin time zone (UTC+9.5h) as default."""
    try:
        del series['percentile']
    except KeyError:
        pass
    # the second time of time.time groups by every unique timestamp per day --> daily cycle.
    # other possibilities to group datetime-objects are
    # https://xray.readthedocs.io/en/latest/api.html#datetimelike-properties
    day = series.groupby('time.'+group).mean()

    # create time objects for one (arbitrary) day
    dti = pd.date_range('2019-01-07T00:00:00', periods=period, freq=frequency)

    day.coords['new_time'] = ('time', dti)
    day = day.swap_dims({'time': 'new_time'})
    del day['time']
    day = day.rename({'new_time': 'time'})
    return day.roll(shifts={'time': time_shift}, roll_coords=False)


def root_mean_square_error(x, y):
    """RMSE for two data series, which are subtractable and have a .mean() method."""
    return ((x - y) ** 2).mean() ** 0.5


def covariance(x, y):
    """Covariance. From http://xarray.pydata.org/en/stable/dask.html#automatic-parallelization. """
    return ((x - x.mean(axis=-1)) * (y - y.mean(axis=-1))).mean(axis=-1)


def pearson_correlation(x, y):
    """Pearson's r, or correlation coefficient.
    From http://xarray.pydata.org/en/stable/dask.html#automatic-parallelization. """
    return covariance(x, y) / (x.std(axis=-1) * y.std(axis=-1))


def spearman_correlation(x, y):
    """Spearman's s, or rank correlation coefficient.
    From http://xarray.pydata.org/en/stable/dask.html#automatic-parallelization. """
    x_ranks = bn.rankdata(x, axis=-1)
    y_ranks = bn.rankdata(y, axis=-1)
    return pearson_correlation(x_ranks, y_ranks)


def interpolate_repeating_values(dataset, l_sort_it=False):
    """Takes a dataset (must have time dimension) and linearly interpolates all recurring values in its variables.
    Optionally sorts variable data first."""
    ds = dataset.copy(deep=True)
    for variable in ds:
        var = ds[variable]
        # get rid of nans
        v = var.where(var.notnull(), drop=True)
        if l_sort_it:
            # sort ascending
            v_sort = v.sortby(v)
        else:
            v_sort = v
        # xarray itself does not capture the small differences between elements. But numpy does.
        first  = np.array(v_sort[:-1])
        second = np.array(v_sort[1:])
        # the last n-1 elements (need to be larger than previous element)
        greater = xr.DataArray(second > first)
        # the first n-1 elements (need to be smaller than following element)
        smaller = xr.DataArray(first < second)
        # both conditions
        both = (smaller[:-1]) & (greater[1:])
        # put original data onto nans where no nans shall be
        inner    = np.zeros(shape=len(v_sort))
        inner[:] = np.nan
        np.putmask(inner[1:-1], both, v_sort[1:-1])
        inner[ 0] = v_sort[ 0]
        inner[-1] = v_sort[-1]
        # linearly interpolate remaining nans and put into original data
        v_sort[:] = xr.DataArray(inner).interpolate_na(dim='dim_0')
        # put original order back into place
        v = v_sort.sortby('time')
        # put data back into long series
        ds[variable].loc[v.time] = v
    return ds


def convective_rainrates():
    """Returns an 1D-array (without particular order inside) containing values of precipitation
    for all convective pixels."""

    # # convective rain rates in xarray_array
    # rain_conv_xr = rain.where(stein == 2)
    #
    # # make a numpy-copy of the rain-array
    # rain_conv_np = np.array(rain_conv_xr)
    #
    # # use xarray-version to apply .notnull() to, to mask the NaNs and return boolean array.
    # # np.nonzero() returns indices of True values (indices for each dimension in single array-> 3 arrays total)
    # idxs = np.asarray(rain_conv_xr.notnull()).nonzero()
    #
    # # use numpy-version to subselect on the indices, the 3 arrays in idxs are automatically applied correctly
    # rain_conv_1d = rain_conv_np[idxs]

    # we need a loop, the numpy-array containing the whole array of shape (378000, 117, 117) would be 40GB in memory.

    data_path = Path(home + '/Documents/Data/Steiner/')
    glob_pattern_2d = 'STEINER_ECHO_CLASSIFICATION_'
    file_names_stein = sorted([str(f) for f in data_path.rglob(f'*{glob_pattern_2d}*.nc')])

    data_path = Path(home + '/Documents/Data/RainRate/')
    glob_pattern_2d = 'RADAR_ESTIMATED_RAIN_RATE_'
    file_names_rain = sorted([str(f) for f in data_path.rglob(f'*{glob_pattern_2d}*.nc')])

    rain_rates = []
    for stein_file, rain_file in zip(file_names_stein, file_names_rain):
        print(stein_file[-11:-3])  # print date, to have tracker of loop and assess progress
        stein = xr.open_dataset(stein_file).steiner_echo_classification
        rain = xr.open_dataset(rain_file).radar_estimated_rain_rate

        assert rain.time.isin(stein.time).all()

        # numpy supports 3-dimensional boolean indexing, xarray not. Returns the values as 1d-array
        rain_np = np.asarray(rain)
        rain_rates.append(rain_np[stein == 2])

    return np.concatenate(rain_rates)


def notnull_area(array):
    """Pixels not NaN for the two fast dimensions of a 3D x-array."""
    i = 0
    while array[i, :, :].isnull().all():
        i += 1
    return int(array[i, :, :].notnull().sum())


def precip_stats(rain, stein, period='', group=''):
    """Calculates area and rate of stratiform and convective precipitation and area ratio to scan area."""
    grouped = (period == '')
    rain_conv = rain.where(stein == 2)
    rain_stra = rain.where(stein == 1)
    rain_conv = rain_conv.where(rain_conv != 0.)
    rain_stra = rain_stra.where(rain_stra != 0.)

    # The total area (number of pixels) of one radar scan. All cells have same area.
    area_scan = notnull_area(rain)

    # Total rain in one time slice. And the corresponding number of cells.
    if grouped:
        group = group if (group == '') else '.' + group
        conv_rain = rain_conv.groupby('time' + group).sum(skipna=True)
        stra_rain = rain_stra.groupby('time' + group).sum(skipna=True)
        conv_area = rain_conv.notnull().groupby('time' + group).sum()
        stra_area = rain_stra.notnull().groupby('time' + group).sum()
    else:
        conv_rain = rain_conv.resample(time=period).sum(skipna=True)
        stra_rain = rain_stra.resample(time=period).sum(skipna=True)
        conv_area = rain_conv.notnull().resample(time=period).sum()
        stra_area = rain_stra.notnull().resample(time=period).sum()

    if not conv_area._in_memory:
        conv_area.load()
        stra_area.load()
    if not conv_rain._in_memory:
        conv_rain.load()
        stra_rain.load()

    # conv_intensity gives mean precip over its area in a time interval.
    # Summation of it hence would be a sum of means. Beware, mean(a) + mean(b) != mean(a + b).
    # xarray automatically throws NaN if division by zero
    conv_intensity = conv_rain / conv_area
    stra_intensity = stra_rain / stra_area

    area_period = area_scan * (len(rain.time) / len(conv_rain))
    conv_mean = conv_rain / area_period
    stra_mean = stra_rain / area_period
    conv_area_ratio = conv_area / area_period * 100
    stra_area_ratio = stra_area / area_period * 100

    return conv_intensity, conv_mean, conv_area_ratio, stra_intensity, stra_mean, stra_area_ratio


if __name__ == '__main__':

    start = timeit.default_timer()

    files = "Steiner/CPOL_STEINER_ECHO_CLASSIFICATION_*.nc"
    ds_st = xr.open_mfdataset(home+"/Documents/Data/"+files, chunks={'time': 1000})
    files = "RainRate/CPOL_RADAR_ESTIMATED_RAIN_RATE_*.nc"
    ds_rr = xr.open_mfdataset(home+"/Documents/Data/"+files, chunks={'time': 1000})

    # run with 4 parallel threads on my local laptop
    # c = Client()

    # Rain rate units are mm/hour, dividing by 86400 yields mm/s == kg/m^2s. No, the factor is 3600, not 86400.
    # And 6 (=600/3600) for 10 minutes, the measurement interval.
    conv_intensity, conv_mean, conv_area,\
    stra_intensity, stra_mean, stra_area = precip_stats(rain=ds_rr.radar_estimated_rain_rate,  # / 6.,
                                                              stein=ds_st.steiner_echo_classification,
                                                              period='',
                                                              group='')

    # add some attributes for convenience to the stats
    conv_intensity.attrs['units'] = 'mm/hour'
    conv_mean.attrs['units'] = 'mm/hour'
    conv_area.attrs['units'] = '% of radar area'
    stra_intensity.attrs['units'] = 'mm/hour'
    stra_mean.attrs['units'] = 'mm/hour'
    stra_area.attrs['units'] = '% of radar area'

    # save as netcdf-files
    path = '/Users/mret0001/Desktop/'
    xr.save_mfdataset([xr.Dataset({'conv_intensity': conv_intensity}), xr.Dataset({'conv_rr_mean': conv_mean}),
                       xr.Dataset({'conv_area': conv_area}),
                       xr.Dataset({'stra_intensity': stra_intensity}), xr.Dataset({'stra_rr_mean': stra_mean}),
                       xr.Dataset({'stra_area': stra_area})],
                      [path+'conv_intensity.nc', path+'conv_rr_mean.nc',
                       path+'conv_area.nc',
                       path+'stra_intensity.nc', path+'stra_rr_mean.nc',
                       path+'stra_area.nc'])

    # sanity check
    check = False
    if check:
        r = ds_rr.radar_estimated_rain_rate / 6.
        cr = r.where(ds_st.steiner_echo_classification == 2)
        cr = cr.where(cr != 0.)
        cr_1h = cr[9774:9780, :, :].load()  # the most precip hour in the 09/10-season
        # cr_1h = cr[9774, :, :].load()  # '2010-02-25T21:00:00'
        npixels = cr_1h.notnull().sum()
        cr_intens_appro = cr_1h.sum() / npixels
        cr_intens_orig = conv_intensity.sel(time='2010-02-25T21:00:00')
        print('Simple 2.5 x 2.5 km square assumption approximated the most precipitating hour by {} %.'
              .format(str((cr_intens_appro/cr_intens_orig).values * 100)))

    stop = timeit.default_timer()
    print('Run Time: ', stop - start)
