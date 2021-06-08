from os.path import expanduser
from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np
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
