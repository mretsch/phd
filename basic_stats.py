import math as m
import xarray as xr
import bottleneck as bn
from dask.distributed import Client
import numpy as np
import matplotlib.pyplot as plt
import timeit


def total_area(array):
    """Calculate total and single cell area of an array based on regular lat and lon grid."""
    lat_dist, lon_dist = 0.5 * (array.lat[1] - array.lat[0]), array.lon[1] - array.lon[0]
    stacked = array.stack(z=('lat', 'lon'))
    valid = stacked.where(stacked.notnull(), drop=True)

    earth_r = 6378000  # in metre
    large_wedge = m.pi * earth_r**2 * np.sin(np.deg2rad(abs(valid.z.lat) + lat_dist)) * (lon_dist / 180.0)
    small_wedge = m.pi * earth_r**2 * np.sin(np.deg2rad(abs(valid.z.lat) - lat_dist)) * (lon_dist / 180.0)
    area = large_wedge - small_wedge
    areas = area.unstack('z')
    return area.sum(), areas


def nan_sum(array, axis):
    """Sum an xarray containing NaNs and which doesn't work with .sum(skipna=True)."""
    return bn.nansum(array, axis=axis)


def mask_sum(selector, background, period='', group=''):
    """Sum an array after applying a mask to it."""
    grouped = period == ''

    valid = background.where(selector.notnull()).fillna(0.)
    if grouped:
        # calling nan_sum on the groupby('time') object here
        # takes 150 seconds at 4 processes for 3 days of data (time=432,lat=117,lon=117). Too long -> .fillna(0.)
        valid_sum = valid.groupby('time' + group).sum()
    else:
        valid_sum = valid.resample(time=period).sum()
    return valid_sum


def precip_stats(rain, stein, period='', group=''):
    """Calculates area and rate of stratiform and convective precipitation and ratios between them."""

    grouped = period == ''
    rain_conv = rain.where(stein == 2)
    rain_stra = rain.where(stein == 1)

    i = 0
    while rain[i, :, :].isnull().all():
        i += 1
    else:
        area_scan, area_cell_single = total_area(rain[i, :, :])

    expanded = area_cell_single.expand_dims('time')
    expanded.time[0] = rain.time[0]
    bcasted, _ = xr.broadcast(expanded, rain)
    area_cell = bcasted.ffill('time')

    # Multiply rain-rate (unit is kg/sm^2), with the respective grid cell area before summation.
    if grouped:
        group = group if (group == '') else '.' + group
        conv_rain = (rain_conv * area_cell).groupby('time' + group).sum(skipna=True)
        stra_rain = (rain_stra * area_cell).groupby('time' + group).sum(skipna=True)
    else:
        conv_rain = (rain_conv * area_cell).resample(time=period).sum(skipna=True)
        stra_rain = (rain_stra * area_cell).resample(time=period).sum(skipna=True)

    # The ratio toward the total scan area is not of interest, only absolute values and their ratio (calculated later).
    conv_area = mask_sum(rain_conv, area_cell, period, group)  # / area_period
    stra_area = mask_sum(rain_stra, area_cell, period, group)  # / area_period

    if not conv_area._in_memory:
        conv_area.load()
        stra_area.load()
    if not conv_rain._in_memory:
        conv_rain.load()
        stra_rain.load()

    # xarray automatically throws NaN if division by zero
    conv_intensity = conv_rain / conv_area
    stra_intensity = stra_rain / stra_area

    area_period = area_scan * (len(rain.time) / len(conv_rain))
    conv_mean = conv_rain / area_period
    stra_mean = stra_rain / area_period
    conv_area_ratio = conv_area / area_period
    stra_area_ratio = stra_area / area_period

    return conv_intensity, conv_mean, conv_area_ratio, stra_intensity, stra_mean, stra_area_ratio, area_period


if __name__ == '__main__':

    start = timeit.default_timer()

    files = "Steiner/CPOL_STEINER_ECHO_CLASSIFICATION_season0910.nc"
    ds_st = xr.open_mfdataset("/Users/mret0001/Data/"+files, chunks={'time': 1000})
    files = "RainRate/CPOL_RADAR_ESTIMATED_RAIN_RATE_season0910.nc"
    ds_rr = xr.open_mfdataset("/Users/mret0001/Data/"+files, chunks={'time': 1000})

    # run with 4 parallel threads on my local laptop
    # c = Client()

    # Rain rate units are mm/hour, dividing by 86400 yields mm/s == kg/m^2s. No, the factor is 3600, not 86400.
    # And 6 (=600/3600) for 10 minutes, the measurement interval.
    conv_intensity, conv_mean, conv_area_ratio,\
    stra_intensity, stra_mean, stra_area_ratio,\
    area_period = precip_stats(rain=ds_rr.radar_estimated_rain_rate / 6.,
                               stein=ds_st.steiner_echo_classification,
                               period='1H',
                               group='hour')

    # sanity check 1
    # total_rain = conv_rain.sum() + ((1 - ratio_rain) * conv_rain).sum().load()
    # appro_rain = (ds_rr.radar_estimated_rain_rate * 2500 ** 2 / 86400.).sum().load()
    # print('Simple 2.5x2.5 km square assumption approximated real total rain sum by {} %.'
    #       .format(round(abs(1 - (appro_rain/total_rain - 1))*100)))
    # sanity check 2
    r = ds_rr.radar_estimated_rain_rate / 6.
    cr = r.where(ds_st.steiner_echo_classification == 2)
    cr_1h = cr[9768:9773, :, :].load()  # the most precip hour in the 09/10-season
    npixels = cr_1h.notnull().sum()
    cr_intens_appro = cr_1h.sum() / npixels
    cr_intens_orig = conv_intensity.sel(time='2010-02-25T21:00:00')
    print('Simple 2.5 x 2.5 km square assumption approximated the most precipitating hour by {} %.'
          .format(round(abs(1 - (cr_intens_appro/cr_intens_orig - 1))*100)))

    stop = timeit.default_timer()
    print('Run Time: ', stop - start)
