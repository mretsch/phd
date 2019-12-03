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

# var_1   = xr.open_dataarray(home+path1+'/rom.nc')['percentile'].sel({'time':slice(start_date, end_date)})
# var_2   = xr.open_dataarray(home+path1+'/cop.nc')['percentile'].sel({'time':slice(start_date, end_date)})
# var_3   = xr.open_dataarray(home+path1+'/iorg.nc')['percentile'].sel({'time':slice(start_date, end_date)})
# var_4   = xr.open_dataarray(home+path1+'/scai.nc')['percentile'].sel({'time':slice(start_date, end_date)})
# var_4 = 1 - var_4
#ds_steiner = xr.open_mfdataset(home+'/Data/Steiner/*season*', chunks=40)
var_1   = xr.open_dataarray(home+path1+'/rom_kilometres.nc') .sel({'time':slice(start_date, end_date)})
var_2   = xr.open_dataarray(home+path1+'/cop.nc') .sel({'time':slice(start_date, end_date)})
var_3   = xr.open_dataarray(home+path1+'/iorg.nc').sel({'time':slice(start_date, end_date)})
var_4   = xr.open_dataarray(home+path1+'/scai.nc').sel({'time':slice(start_date, end_date)})
var_4 = -1 * var_4

try:
    del var_1['percentile']
    del var_2['percentile']
    del var_3['percentile']
    del var_4['percentile']
except KeyError:
    pass

ds = xr.Dataset({'v1':var_1,'v2':var_2,'v3':var_3,'v4':var_4})

darwin_time = np.timedelta64(570, 'm')  # UTC + 9.5 hours

l_single_plot = False
if l_single_plot:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
else:
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 4), sharex=True)
    plt.subplots_adjust(wspace=0.1)

color = [col.sol['cyan'], col.sol['magenta'], col.sol['violet'], col.sol['yellow'], col.sol['magenta'], col.sol['cyan']]

for i, v in enumerate(ds):
    var = ds[v]
    var['time'] = var.time + darwin_time
    if l_single_plot:
        ax.plot(var.time, var, color=color[i], lw=2)
    else:
        axes.flatten()[i].plot(var.time, var, color=color[i], lw=2)

plot_dates = mdates.date2num(var.time)
k = 0

if l_single_plot:
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
else:
    legend_text = ['ROME', 'COP', 'I$_\mathrm{org}$', 'SCAI']
    ylabel_text = ['[km$^2$]', '[1]', '[1]', '[-1]']
    for i, p in enumerate(axes.flatten()):
        k = 0
        p.set_xticks([plot_dates[k+ 3],
                                      plot_dates[k+12]])
        p.set_xticks(ticks=[plot_dates[k], plot_dates[k + 6], plot_dates[k + 9],
                            plot_dates[k + 15], plot_dates[k + 18]], minor=True)
        p.xaxis.set_major_formatter(hourFmt)
        p.grid(axis='x', which='both')
        if i in (1, 3):
            p.yaxis.set_ticks_position('right')
            p.yaxis.set_label_position('right')
        p.set_ylabel(ylabel_text[i])
        if i == 1:
            lg = p.legend([legend_text[i]], framealpha=1., handlelength=0.5, handletextpad=0.5, loc=9)
        else:
            lg = p.legend([legend_text[i]], framealpha=1., handlelength=0.5, handletextpad=0.5, loc='best')
        lg.get_frame().set_facecolor('none')
        if i in (2, 3):
            p.set_xlabel('Darwin time [h]')


# plt.show()
plt.savefig(home+'/Desktop/metric_series.pdf', transparent=True, bbox_inches='tight')
