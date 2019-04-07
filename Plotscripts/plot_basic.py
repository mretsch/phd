from os.path import expanduser
home = expanduser("~")
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import timeit

start = timeit.default_timer()

plt.rc('font'  , size=12)
plt.rc('legend', fontsize=12)

start_date = '2017-03-30T14:50:00' # '2009-12-07T09:10:00'
end_date   = '2017-03-30T18:00:00'  # '2009-12-07T12:20:00'

path1 = '/Data/Analysis/No_Boundary/'
path2 = '/Google Drive File Stream/My Drive/'
try:
    var_1   = xr.open_dataarray(home+path1+'/rom.nc').sel({'time':slice(start_date, end_date)})
    var_2   = xr.open_dataarray(home+path1+'/low_rom_limit.nc').sel({'time':slice(start_date, end_date)})
    var_3   = xr.open_dataarray(home+path1+'/o_area.nc').sel({'time':slice(start_date, end_date)})
    var_4   = xr.open_dataarray(home+path1+'/cop.nc').sel({'time':slice(start_date, end_date)})
    ds_steiner = xr.open_mfdataset(home+'/Data/Steiner/*season*', chunks=40)
except FileNotFoundError:
    #var_1   = xr.open_dataarray(home+path2+'/Data_Analysis/rom.nc').percentile.sel({'time':slice(start_date, end_date)})
    #var_2   = xr.open_dataarray(home+path2+'/Data_Analysis/iorg.nc').percentile.sel({'time':slice(start_date, end_date)})
    var_3   = xr.open_dataarray(home+path2+'/Data_Analysis/cop.nc').percentile.sel({'time':slice(start_date, end_date)})
    #var_4   = xr.open_dataarray(home+path2+'/Data_Analysis/scai.nc').percentile.sel({'time':slice(start_date, end_date)})

    var_41  = xr.open_dataarray(home+path2+'/Data_Analysis/o_area.nc').sel({'time':slice(start_date, end_date)})
    var_42  = xr.open_dataarray(home+path2+'/Data_Analysis/o_number.nc').sel({'time':slice(start_date, end_date)})
    var_4 = 2 * var_41.where(var_42 != 1., np.nan)

    var_1   = xr.open_dataarray(home+'/Desktop/rom_30032017.nc').sel({'time':slice(start_date, end_date)})
    var_2   = xr.open_dataarray(home+'/Desktop/lrl_30032017.nc').sel({'time':slice(start_date, end_date)})

    #ds_steiner = xr.open_mfdataset(home+path2+'/Data/Steiner/*season*', chunks=40)

try:
    del var_1['percentile']
    #del var_2['percentile']
    #del var_3['percentile']
    del var_4['percentile']
except KeyError:
    pass

var_3 = 2*var_3.where(var_4.notnull(), np.nan)
var_4 = 2*var_1 - var_3

ds = xr.Dataset({'v1':var_1,'v2':var_2,'v3':var_3,'v4':abs(1-var_4)})

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))

for v in ds:
    var = ds[v]
    ax.plot(var.time, var)

ax.legend(['ROM', 'LowLimit', 'upper', 'ROME'])

plt.show()
