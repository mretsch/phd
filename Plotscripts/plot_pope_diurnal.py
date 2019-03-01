from os.path import expanduser
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
home = expanduser("~")

# get the Pope regimes per day
dfr_pope = pd.read_csv(home+'/Data/PopeRegimes/Pope_regimes.csv', header=None, names=['timestring', 'regime'], index_col=0)
dse = pd.Series(dfr_pope['regime'])

da_pope = xr.DataArray(dse)
pope_years = da_pope.sel({'timestring': slice('2009-11-30', '2017-03-31')})
pope_years.coords['time'] = ('timestring', pd.to_datetime(pope_years.timestring))
pope = pope_years.swap_dims({'timestring': 'time'})
del pope['timestring']

# get the metric
var1 = xr.open_dataarray(home+'/Data/Analysis/No_Boundary/rome.nc')
var2 = xr.open_dataarray(home+'/Data/Analysis/With_Boundary/conv_intensity.nc')

for i, var in enumerate([var1, var2]):
#for i, var in enumerate([var2]):
    try:
        var_perc = var.thisisnotanattribute# percentile * 100
    except AttributeError:
        var_perc = var

    downsample = False
    if downsample:
        perc_day_max = var_perc.resample(time='1D', skipna=True).mean()
        var_perc = perc_day_max.sel({'time': pope.time})

    upsample = not downsample
    if upsample:
        pope = pope.resample(time='10T').interpolate('zero')

    # filter each Pope regime
    pope = pope.where(var_perc.notnull())

    perc_pope_1 = var_perc.where(pope == 1)# , drop=True)
    perc_pope_2 = var_perc.where(pope == 2)# , drop=True)
    perc_pope_3 = var_perc.where(pope == 3)# , drop=True)
    perc_pope_4 = var_perc.where(pope == 4)# , drop=True)
    perc_pope_5 = var_perc.where(pope == 5)# , drop=True)

    vars = [perc_pope_1, perc_pope_2, perc_pope_3, perc_pope_4, perc_pope_5]
    var_actual = []
    for var in vars:
        try:
            del var['percentile']
        except KeyError:
            pass
        var_day = var.groupby('time.time').mean()

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
plt.savefig(home+'/Desktop/pr1_daily.pdf', transparent=True, bbox_inches='tight')
#plt.show()
