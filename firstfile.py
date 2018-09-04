import math as m
import xarray as xr
import bottleneck as bn
from dask.distributed import Client
import numpy as np
import matplotlib.pyplot as plt
import timeit

start = timeit.default_timer()
import base_stats

files = "Steiner/CPOL_STEINER_ECHO_CLASSIFICATION_oneday.nc"
ds_st = xr.open_mfdataset("/Users/mret0001/Data/"+files,chunks={'time':40})
files = "RainRate/CPOL_RADAR_ESTIMATED_RAIN_RATE_oneday.nc"
ds_rr = xr.open_mfdataset("/Users/mret0001/Data/"+files, chunks={'time':40})

rain  = ds_rr.radar_estimated_rain_rate
stein = ds_st.steiner_echo_classification

# run with 4 parallel threads on my local laptop
c = Client()

# create an array which has the time-dim as 'height'. For ParaView.
#   array = xr.DataArray(np.array(stein),coords=[('height',list(range(1,len(stein.time)+1))),('lat',stein.lat),('lon',stein.lon)])
#   stein_ds = xr.Dataset({'SteinerClass':array})
#   stein_ds.to_netcdf('height_steiner_data.nc')

rain_conv = rain.where(stein == 2)
rain_stra = rain.where(stein == 1)


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


area_scan, area_cell = total_area(rain[40, :, :])

expanded = area_cell.expand_dims('time')
expanded.time[0] = rain.time[0]
bcasted, _ = xr.broadcast(expanded, rain)
area_cell_all = bcasted.ffill('time')


def nan_sum(array, axis):
    """Summarise an xarray containing NaNs and which doesn't work with .sum(skipna=True)."""
    return bn.nansum(array, axis=axis)


def mask_sum(selector, background, per_timestep=False):
    """Summarise an array after applying a mask to it."""
    valid = background.where(selector.notnull())
    if per_timestep:
        # calling nan_sum on the groupby('time') object
        # takes 150 seconds at 4 processes for 3 days of data (time=432,lat=117,lon=117). Too long.
        valid_sum = valid.groupby('time').reduce(nan_sum)
    else:
        valid_sum = bn.nansum(valid)
    return valid_sum


conv_area = mask_sum(rain_conv, area_cell_all, per_timestep=True) / area_scan
stra_area = mask_sum(rain_stra, area_cell_all, per_timestep=True) / area_scan

conv_rain = rain_conv.groupby('time').reduce(nan_sum)
stra_rain = rain_stra.groupby('time').reduce(nan_sum)

# xarray automatically throws NaN if division by zero
conv_stra_area = conv_area / stra_area
conv_stra_rain = conv_rain / stra_rain

conv_stra_area[xr.ufuncs.logical_and(conv_area > 0., stra_area == 0.)] = -1
conv_stra_rain[xr.ufuncs.logical_and(conv_rain > 0., stra_rain == 0.)] = -1

stop = timeit.default_timer()
print('Run Time: ', stop - start)