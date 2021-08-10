from os.path import expanduser
import timeit
import numpy as np
import xarray as xr
from downsample_rome import downsample_timeseries

home = expanduser("~")
start = timeit.default_timer()

# metric = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/rome_10mmhour.nc')
area = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/o_area_10mmhour.nc')
number = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/o_number_10mmhour.nc')
metric = area * number
div = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/Divergence900/div900_avg.nc')
rh = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/rh500.nc')

kwa = {
    'sample_period' : '3H'     , # '3H'   ,  # '6H'
    'sample_center' : 1.5      , # 0      ,  # 3
    'center_time'   : '1H30min', # '0H'   ,  # '3H'
    'closed_side'   : 'left'   , # 'right',  # 'left'
    'n_sample'      : 12       , # 12     ,  # 36
    'rh_series'     : None,
}


m_stack = metric.stack({'z': ('lon', 'lat')})

counter, error_counter = 0, 0

for lon_apprx, lat_apprx in list(m_stack['z'].values):

    rome_series = metric.sel(lat=lat_apprx, lon=lon_apprx, method='nearest')
    # print(f"lat: {round(lat_apprx, 2)}, lon: {round(lon_apprx, 2)}")

    if 'rh_series' in kwa:
        kwa['rh_series'] = rh.sel(lat=lat_apprx, lon=lon_apprx, method='nearest')

    try:
        rome_3h = downsample_timeseries(rome_series, **kwa)
        counter += 1
    except xr.core.variable.MissingDimensionsError:
        error_counter += 1
        print(f"lat: {round(lat_apprx, 2)}, lon: {round(lon_apprx, 2)}")
        continue

    if counter==1:
        n_time = len(rome_3h['time'])
        stack_shorttime = xr.full_like(m_stack[:n_time], fill_value=np.nan)
        stack_shorttime['time'] = rome_3h['time'].values

    lat = float(rome_3h['lat'].values)
    lon = float(rome_3h['lon'].values)

    stack_shorttime.loc[{'z': (lon, lat)}] = rome_3h.values

metric_3h = stack_shorttime.unstack()

stop = timeit.default_timer()
print(f'Time used: {stop - start}')
