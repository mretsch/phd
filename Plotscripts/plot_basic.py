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

var_1 = xr.open_dataarray(home+'/Data/Analysis/No_Boundary/rom.nc').percentile.sel({'time':slice(start_date, end_date)})
var_2 = xr.open_dataarray(home+'/Data/Analysis/No_Boundary/iorg.nc').percentile.sel({'time':slice(start_date, end_date)})
var_3 = xr.open_dataarray(home+'/Data/Analysis/No_Boundary/cop.nc').percentile.sel({'time':slice(start_date, end_date)})
var_4 = xr.open_dataarray(home+'/Data/Analysis/No_Boundary/scai.nc').percentile.sel({'time':slice(start_date, end_date)})

del var_1['percentile']
del var_2['percentile']
del var_3['percentile']
del var_4['percentile']


ds = xr.Dataset({'v1':var_1,'v2':var_2,'v3':var_3,'v4':abs(1-var_4)})

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
for v in ds:
    var = ds[v]
    ax.plot(var)

plt.show()
