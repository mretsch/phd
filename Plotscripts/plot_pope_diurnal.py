from os.path import expanduser
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from basic_stats import into_pope_regimes, diurnal_cycle
home = expanduser("~")

plt.rc('font'  , size=12)
plt.rc('legend', fontsize=12)

# get the quantities
var1 = xr.open_dataarray(home+'/Google Drive File Stream/My Drive/Data_Analysis/rom.nc')
var2 = xr.open_dataarray(home+'/Google Drive File Stream/My Drive/Data_Analysis/conv_intensity.nc')

# seperate quantities into Pope regimes and compute diurnal cycle
for i, var in enumerate([var1, var2]):  # ([var2])

    ds_var_pope = into_pope_regimes(var, l_percentile=True)
    var_daily = []
    for variable in ds_var_pope:
        var_daily.append(diurnal_cycle(ds_var_pope[variable]))

    # to pandas series, to have easy multi-axes plot
    if i == 0:
        ds_1 = var_daily[0].to_pandas()
        ds_2 = var_daily[1].to_pandas()
        ds_3 = var_daily[2].to_pandas()
        ds_4 = var_daily[3].to_pandas()
        ds_5 = var_daily[4].to_pandas()
    else:
        ds_conv_1 = var_daily[0].to_pandas()
        ds_conv_2 = var_daily[1].to_pandas()
        ds_conv_3 = var_daily[2].to_pandas()
        ds_conv_4 = var_daily[3].to_pandas()
        ds_conv_5 = var_daily[4].to_pandas()

# # from: https://matplotlib.org/gallery/text_labels_and_annotations/date.html?highlight=ticks
hourFmt = mdates.DateFormatter('%H')

# plot per Pope regime, or everything together
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))

#p1, =   ax.plot(var_daily[0].time, var_daily[0], lw=2, label="pr1")
#p2, =   ax.plot(var_daily[1].time, var_daily[1], lw=2, label="pr2")
#p3, =   ax.plot(var_daily[2].time, var_daily[2], lw=2, label="pr3")
#p4, =   ax.plot(var_daily[3].time, var_daily[3], lw=2, label="pr4")
#p5, =   ax.plot(var_daily[4].time, var_daily[4], lw=2, label="pr5")

ds_5.plot()
plt.ylabel('PR-5 ROM percentile [%]')
ds_conv_5.plot(secondary_y=True)
plt.ylabel('PR-5 Conv. intensity [mm/hour]')
ax.legend(['ROM perc.', 'Intensity'])

#plt.ylabel('Convective intensity [mm/hour]')
#plt.legend([p1.get_label() ,p2.get_label() ,p3.get_label() ,p4.get_label() ,p5.get_label() ])

ax.set_xlim('2019-01-07T00:00:00', '2019-01-07T23:50:00')
ax.grid()
plt.savefig(home+'/Desktop/pr5.pdf', transparent=True, bbox_inches='tight')
#plt.show()
