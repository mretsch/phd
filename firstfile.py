import math as m
import xarray as xr
from dask.distributed import Client
import numpy as np
import base_stats

files = "Steiner/CPOL_STEINER_ECHO_CLASSIFICATION_season0910.nc"
ds_st = xr.open_mfdataset("/Users/mret0001/Data/"+files,chunks={'time':500})
files = "RainRate/CPOL_RADAR_ESTIMATED_RAIN_RATE_season0910.nc"
ds_rr = xr.open_mfdataset("/Users/mret0001/Data/"+files, chunks={'time':500})

rain  = ds_rr.radar_estimated_rain_rate
stein = ds_st.steiner_echo_classification

# run with 4 parallel threads on my local laptop
# c = Client()

# create an array which has the time-dim as 'height'. For ParaView.
#   array = xr.DataArray(np.array(stein),coords=[('height',list(range(1,len(stein.time)+1))),('lat',stein.lat),('lon',stein.lon)])
#   stein_ds = xr.Dataset({'SteinerClass':array})
#   stein_ds.to_netcdf('height_steiner_data.nc')

rain_conv = rain.where(stein == 2)
rain_stra = rain.where(stein == 1)


# def total_area(lats, lons):
#     """Calculate total area based on latitudes and longitudes. Neglects earth's curvature per rectangle."""
#     earth_r = 6378000
#     dist = earth_r * m.pi / 180.0
#     lats_m = np.array(map(lambda lat: dist * lat, lats))
#     lons_m = np.array(               [dist * lon * m.cos(m.radians(lat))
#                                       for lat, lon in zip(lats, lons)])


def total_area(array):
    """Calculate total area of an array based on regular lat and lon grid."""
    lat_dist, lon_dist = 0.5 * (array.lat[1] - array.lat[0]), array.lon[1] - array.lon[0]
    stacked = array.stack(z=('lat', 'lon'))
    valid = stacked.where(xr.ufuncs.isnan(stacked) == False, drop=True)

    earth_r = 6378000 # in metre
    large_wedge = m.pi * earth_r**2 * np.sin(np.deg2rad(abs(valid.z.lat) + lat_dist)) * (lon_dist / 180.0)
    small_wedge = m.pi * earth_r**2 * np.sin(np.deg2rad(abs(valid.z.lat) - lat_dist)) * (lon_dist / 180.0)
    return (large_wedge - small_wedge).sum()

scan_area = total_area(rain[40,:,:])