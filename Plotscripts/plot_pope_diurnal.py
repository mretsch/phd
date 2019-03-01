from os.path import expanduser
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
home = expanduser("~")

plt.rc('font'  , size=12)
plt.rc('legend', fontsize=12)

def into_pope_regimes(series, l_upsample=True, l_percentile=False):
    """Mask radar/metric time series according to the 5 possible Pope regimes."""

    # get the Pope regimes per day
    dfr_pope = pd.read_csv(home+'/Data/PopeRegimes/Pope_regimes.csv', header=None, names=['timestring', 'regime'], index_col=0)
    dse = pd.Series(dfr_pope['regime'])

    da_pope = xr.DataArray(dse)
    pope_years = da_pope.sel({'timestring': slice('2009-11-30', '2017-03-31')})
    pope_years.coords['time'] = ('timestring', pd.to_datetime(pope_years.timestring))
    pope = pope_years.swap_dims({'timestring': 'time'})
    del pope['timestring']

    if l_percentile:
        var = series.percentile * 100
    else:
        var = series

    if not l_upsample:
        daily = var.resample(time='1D', skipna=True).mean()
        var = daily.sel({'time': pope.time})
    else:
        pope = pope.resample(time='10T').interpolate('zero')

    # filter each Pope regime
    pope = pope.where(var.notnull())
    var_p1 = var.where(pope == 1)
    var_p2 = var.where(pope == 2)
    var_p3 = var.where(pope == 3)
    var_p4 = var.where(pope == 4)
    var_p5 = var.where(pope == 5)

    return xr.Dataset({'var_p1': var_p1, 'var_p2': var_p2, 'var_p3': var_p3, 'var_p4': var_p4, 'var_p5': var_p5})


# get the quantities
var1 = xr.open_dataarray(home+'/Data/Analysis/No_Boundary/rome.nc')
var2 = xr.open_dataarray(home+'/Data/Analysis/With_Boundary/conv_intensity.nc')

for i, var in enumerate([var1, var2]):
#for i, var in enumerate([var2]):

    ds_var_p = into_pope_regimes(var)
    var_actual = []
    for variable in ds_var_p:
        var_p = ds_var_p[variable]
        try:
            del var_p['percentile']
        except KeyError:
            pass
        var_day = var_p.groupby('time.time').mean()

        dti = pd.date_range('2019-01-07T00:00:00', periods=144, freq='10T')

        var_day.coords['new_time'] = ('time', dti)
        var_day = var_day.swap_dims({'time': 'new_time'})
        del var_day['time']
        var_day = var_day.rename({'new_time': 'time'})
        var_local = var_day.roll(shifts={'time': 9*6+3}, roll_coords=False)
        var_actual.append(var_local)

    # to pandas series, to have easy multi-axes plot
    if i == 0:
        ds_1 = var_actual[0].to_pandas()
        ds_2 = var_actual[1].to_pandas()
        ds_3 = var_actual[2].to_pandas()
        ds_4 = var_actual[3].to_pandas()
        ds_5 = var_actual[4].to_pandas()
    else:
        ds_conv_1 = var_actual[0].to_pandas()
        ds_conv_2 = var_actual[1].to_pandas()
        ds_conv_3 = var_actual[2].to_pandas()
        ds_conv_4 = var_actual[3].to_pandas()
        ds_conv_5 = var_actual[4].to_pandas()

# # from: https://matplotlib.org/gallery/text_labels_and_annotations/date.html?highlight=ticks
hourFmt = mdates.DateFormatter('%H')

# plot per Pope regime, or everything together
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))

#p1, =   ax.plot(var_actual[0].time, var_actual[0], lw=2, label="pr1")
#p2, =   ax.plot(var_actual[1].time, var_actual[1], lw=2, label="pr2")
#p3, =   ax.plot(var_actual[2].time, var_actual[2], lw=2, label="pr3")
#p4, =   ax.plot(var_actual[3].time, var_actual[3], lw=2, label="pr4")
#p5, =   ax.plot(var_actual[4].time, var_actual[4], lw=2, label="pr5")

ds_1.plot()
plt.ylabel('PR-1 ROM [1]')
ds_conv_1.plot(secondary_y=True)
plt.ylabel('PR-1 Conv. intensity [mm/hour]')
ax.legend(['ROM', 'Intensity'])

#plt.ylabel('Convective intensity [mm/hour]')
#plt.legend([p1.get_label() ,p2.get_label() ,p3.get_label() ,p4.get_label() ,p5.get_label() ])

ax.set_xlim('2019-01-07T00:00:00', '2019-01-07T23:50:00')
ax.grid()
plt.savefig(home+'/Desktop/a.pdf', transparent=True, bbox_inches='tight')
#plt.show()
