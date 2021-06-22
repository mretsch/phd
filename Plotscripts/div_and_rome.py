from os.path import expanduser
import timeit
import numpy as np
import numpy.testing as npt
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
from basic_stats import covariance, pearson_correlation
from downsample_rome import downsample_timeseries
from scipy.signal import correlate


home = expanduser("~")
start = timeit.default_timer()


def cross_corr(x, y):

    assert len(x) == len(y)

    cc = np.zeros_like(x)

    n_half = len(x) // 2
    cc[n_half - 1] = pearson_correlation(x, y)

    for i in range(1, n_half):
        # x preceding y <- 'early' values in x & 'late' values in y
        cc[n_half - i - 1] = pearson_correlation(x[ :-i], y[i:  ])
        # y preceding x <- 'late' values in x & 'early' values in y
        cc[n_half + i - 1] = pearson_correlation(x[i:  ], y[ :-i])

    if len(cc) % 2 == 0:
        # in even case, the last entry hasn't been set, hence [:-1]
        corr = xr.DataArray(cc[:-1],
                            {'lag': np.arange(-n_half + 1, n_half)},
                            dims='lag')
    else:
        corr = xr.DataArray(cc,
                            {'lag': np.arange(-n_half + 1, n_half + 2)},
                            dims='lag')

    return corr


# rome_3h = xr.open_dataarray(home+'/Desktop/rome_3h_kimberley.nc')
metric = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/rome_14mmhour.nc')
div = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/Divergence900/div900_avg.nc')

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

rome_3h, divrome_corr = [], []

lag_map = xr.full_like(metric.isel(time=0), fill_value=np.nan)

m_stack = metric.stack({'z': ('lon', 'lat')})

counter = 0

for lon_apprx, lat_apprx in list(m_stack['z'].values):
# for lon_apprx, lat_apprx in all_coordinates:

    rome_series = metric.sel(lat=lat_apprx, lon=lon_apprx, method='nearest')
    print(f"lat: {round(lat_apprx, 2)}, lon: {round(lon_apprx, 2)}")

    # do not take short series of ROME
    if rome_series.count() < 500:
        continue

    try:
        rome_3h.append(downsample_timeseries(rome_series, **kwa))
    except xr.core.variable.MissingDimensionsError:
        counter += 1
        continue

    div_domain = div.sel(lat=rome_series['lat'], lon=rome_series['lon'])
    div_domain = div_domain[div_domain.notnull()]

    # do not take short series of divergence
    if div_domain.count() < 100:
        continue

    div_both = div_domain.where(rome_3h[-1].notnull(), drop=True)
    rome_both = rome_3h[-1].where(div_both)

    divrome_corr.append( cross_corr(rome_both.values, div_both.values) )

    divrome_corr[-1].attrs['lat'] = rome_series['lat'].values
    divrome_corr[-1].attrs['lon'] = rome_series['lon'].values

    # check that at lag 0, there is the correct correlation
    npt.assert_almost_equal(np.corrcoef(rome_both.values, div_both.values)[0, 1],
                           divrome_corr[-1].sel(lag=0),
                           4)

    idx_anticorr = divrome_corr[-1].sel(lag=slice(-5, 5)).argmin()
    lag_anticorr = divrome_corr[-1].sel(lag=slice(-5, 5))[idx_anticorr]['lag'].values

    lag_map.loc[{'lat': rome_series['lat'], 'lon': rome_series['lon']}] = lag_anticorr


l_plot = False
if l_plot:
    fig, axes = plt.subplots(nrows=4, ncols=4, sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        ax.plot(
            divrome_corr[i].sel(lag=slice(-5, 5))['lag'],
            divrome_corr[i].sel(lag=slice(-5, 5))
        )
        ax.axhline(y=0, c='r')
        ax.axvline(x=0, c='grey')
        ax.set_title(f"{divrome_corr[i].attrs['lon'].round(2)}, {divrome_corr[i].attrs['lat'].round(2)}")

    plt.savefig(home+'/Desktop/corr_div_rome3h.pdf')

stop = timeit.default_timer()
print('This script needed {} seconds.'.format(stop - start))


#    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(24, 8))
#
#
#    ax.plot(rome_both['time'],#.sel(time=slice('2020-02-16', '2020-02-18'))['time'],
#            rome_both,#.sel(time=slice('2020-02-16', '2020-02-18')),
#            c='b', label='rome', ls='--', marker='o')
#    plt.grid(axis='x')
#    plt.legend(loc='lower right')
#
#    ax_1 = ax.twinx()
#    ax_1.plot(div_both['time'],#.sel(time=slice('2020-02-16', '2020-02-18'))['time'],
#              div_both,#.sel(time=slice('2020-02-16', '2020-02-18')),
#              c='r', label='div', ls='--', marker='o')
#    ax_1.axhline(y=0)
#    plt.legend()
#
#    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#    ax.tick_params(axis="x", rotation=50)
#
#    # plt.show()
#    plt.savefig(home+'/Desktop/a.pdf')