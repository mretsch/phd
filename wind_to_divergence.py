from os.path import expanduser
home = expanduser("~")
import timeit
from pathlib import Path
import xarray as xr
import numpy as np
import metpy.calc as mpcalc
import matplotlib.pyplot as plt

start = timeit.default_timer()

data_path = Path('/Users/mret0001/Desktop/')
glob_pattern_2d = 'v900_*.nc'
v_files = sorted([str(f) for f in data_path.rglob(f'{glob_pattern_2d}')])
glob_pattern_2d = 'u900_*.nc'
u_files = sorted([str(f) for f in data_path.rglob(f'{glob_pattern_2d}')])

for ufile, vfile in zip(u_files, v_files):

    date = ufile[-24:-11]
    u_wind = xr.open_dataarray(home+f'/Desktop/u900_{date}_reggrid.nc')
    v_wind = xr.open_dataarray(home+f'/Desktop/v900_{date}_reggrid.nc')

    div = xr.DataArray(mpcalc.divergence(u_wind, v_wind).squeeze())

    ds = div.to_dataset(name='div')
    ds['div'].attrs['height'] = "900 hPa"
    ds['div'].attrs['units'] = "1/s"
    ds['div'].attrs['long_name'] = "Divergence"

    ds.to_netcdf(home+f'/Desktop/div900_{date}_reggrid.nc')
    del ds, u_wind, v_wind

stop = timeit.default_timer()
print(f'This script needed {stop-start} seconds.')
