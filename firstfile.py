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

    earth_r = 6378000 # in metre
    large_wedge = m.pi * earth_r**2 * np.sin(np.deg2rad(abs(valid.z.lat) + lat_dist)) * (lon_dist / 180.0)
    small_wedge = m.pi * earth_r**2 * np.sin(np.deg2rad(abs(valid.z.lat) - lat_dist)) * (lon_dist / 180.0)
    area = large_wedge - small_wedge
    areas = area.unstack('z')
    return area.sum(), areas


def nan_sum(array, axis):
    """Sum an xarray containing NaNs and which doesn't work with .sum(skipna=True)."""
    return bn.nansum(array, axis=axis)


def mask_sum(selector, background, per_timestep=False):
    """Sum an array after applying a mask to it."""
    valid = background.where(selector.notnull()).fillna(0.)
    if per_timestep:
        # calling nan_sum on the groupby('time') object here
        # takes 150 seconds at 4 processes for 3 days of data (time=432,lat=117,lon=117). Too long -> .fillna(0.)
        valid_sum = valid.groupby('time').sum()
    else:
        valid_sum = bn.nansum(valid)
    return valid_sum


def precip_stats(rain, stein):
    """Calculates area and rate of stratiform and convective precipitation and ratios between them."""
    rain_conv = rain.where(stein == 2)
    rain_stra = rain.where(stein == 1)

    i = 0
    while rain[i,:,:].isnull().all():
        i += 1
    else:
        area_scan, area_cell = total_area(rain[i, :, :])

    expanded = area_cell.expand_dims('time')
    expanded.time[0] = rain.time[0]
    bcasted, _ = xr.broadcast(expanded, rain)
    area_cell_all = bcasted.ffill('time')

    conv_area = mask_sum(rain_conv, area_cell_all, per_timestep=True) / area_scan
    stra_area = mask_sum(rain_stra, area_cell_all, per_timestep=True) / area_scan
    if not conv_area._in_memory:
        conv_area.load()
        stra_area.load()

    conv_rain = rain_conv.groupby('time').sum(skipna=True)
    stra_rain = rain_stra.groupby('time').sum(skipna=True)
    if not conv_rain._in_memory:
        conv_rain.load()
        stra_rain.load()

    # xarray automatically throws NaN if division by zero
    ratio_area = conv_area / stra_area
    ratio_rain = conv_rain / stra_rain
    ratio_area[xr.ufuncs.logical_and(conv_area > 0., stra_area == 0.)] = -1
    ratio_rain[xr.ufuncs.logical_and(conv_rain > 0., stra_rain == 0.)] = -1

    return ratio_area, ratio_rain, conv_rain, stra_rain


if __name__ == '__main__':

    start = timeit.default_timer()

    files = "Steiner/CPOL_STEINER_ECHO_CLASSIFICATION_season0910.nc"
    ds_st = xr.open_mfdataset("/Users/mret0001/Data/"+files, chunks={'time': 1000})
    files = "RainRate/CPOL_RADAR_ESTIMATED_RAIN_RATE_season0910.nc"
    ds_rr = xr.open_mfdataset("/Users/mret0001/Data/"+files, chunks={'time': 1000})

    # run with 4 parallel threads on my local laptop
    # c = Client()

    # create an array which has the time-dim as 'height'. For ParaView.
    #   array = xr.DataArray(np.array(stein),coords=[('height',list(range(1,len(stein.time)+1))),('lat',stein.lat),('lon',stein.lon)])
    #   stein_ds = xr.Dataset({'SteinerClass':array})
    #   stein_ds.to_netcdf('height_steiner_data.nc')

    stats = precip_stats(rain=ds_rr.radar_estimated_rain_rate, stein=ds_st.steiner_echo_classification)

    stop = timeit.default_timer()
    print('Run Time: ', stop - start)
