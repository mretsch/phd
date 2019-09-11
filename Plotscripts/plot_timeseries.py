from os.path import expanduser
home = expanduser("~")
import timeit
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import Plotscripts.colors_solarized as col

start = timeit.default_timer()

plt.rc('font'  , size=20)
plt.rc('legend', fontsize=20)

hourFmt = mdates.DateFormatter('%H:%M')

start_date = '2015-11-10T03:00:00' # '2009-12-07T09:10:00' # '2017-03-30T14:50:00' #
end_date   = '2015-11-10T06:10:00' # '2009-12-07T12:20:00' # '2017-03-30T18:00:00' #

path1 = '/Data/Analysis/No_Boundary/'
path2 = '/Google Drive File Stream/My Drive/'

var_1   = xr.open_dataarray(home+path1+'/rom.nc')['percentile'].sel({'time':slice(start_date, end_date)})
var_2   = xr.open_dataarray(home+path1+'/cop.nc')['percentile'].sel({'time':slice(start_date, end_date)})
var_3   = xr.open_dataarray(home+path1+'/iorg.nc')['percentile'].sel({'time':slice(start_date, end_date)})
var_4   = xr.open_dataarray(home+path1+'/scai.nc')['percentile'].sel({'time':slice(start_date, end_date)})
#ds_steiner = xr.open_mfdataset(home+'/Data/Steiner/*season*', chunks=40)

try:
    del var_1['percentile']
    del var_2['percentile']
    del var_3['percentile']
    del var_4['percentile']
except KeyError:
    pass

var_4 = 1 - var_4
ds = xr.Dataset({'v1':var_1,'v2':var_2,'v3':var_3,'v4':var_4})

darwin_time = np.timedelta64(570, 'm')  # UTC + 9.5 hours

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))

color = [col.sol['cyan'], col.sol['magenta'], col.sol['violet'], col.sol['yellow'], col.sol['magenta'], col.sol['cyan']]

for i, v in enumerate(ds):
    var = ds[v]
    var['time'] = var.time + darwin_time
    ax.plot(var.time, var*100, color=color[i], lw=2)

plot_dates = mdates.date2num(var.time)
k = 0
ax.set_xticks([plot_dates[k   ], plot_dates[k+ 3], plot_dates[k+ 6], plot_dates[k+ 9],
               plot_dates[k+12], plot_dates[k+15], plot_dates[k+18]])
ax.xaxis.set_major_formatter(hourFmt)

#xtl = ax.xaxis.get_majorticklabels()
#xtl[0].get_text()
#ax.set_xticklabels(['a','b','c'])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='x',direction='in')
ax.yaxis.set_ticks_position('none')

ax.set_xlabel('Darwin time [h]')
ax.set_ylabel('Percentile of organisation [%]')
lg = ax.legend(['ROME', 'COP', 'I$_\mathrm{org}$', 'SCAI'], framealpha=1.)
lg.get_frame().set_facecolor('none')

#plt.show()
plt.savefig(home+'/Desktop/metric_series.pdf', transparent=True, bbox_inches='tight')
