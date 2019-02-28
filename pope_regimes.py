from os.path import expanduser
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
from Plotscripts.plot_hist import histogram_1d
home = expanduser("~")

a = date(2009, 10,  1)
b = date(2017,  3, 31)

# get the Pope regimes per day
dfr_pope = pd.read_csv(home+'/Google Drive File Stream/My Drive/Data/PopeRegimes/Pope_regimes.csv', header=None, names=['timestring', 'regime'], index_col=0)
dse = pd.Series(dfr_pope['regime'])

da_pope = xr.DataArray(dse)
pope_years = da_pope.sel({'timestring': slice('2009-11-30', '2017-03-31')})
pope_years.coords['time'] = ('timestring', pd.to_datetime(pope_years.timestring))
pope = pope_years.swap_dims({'timestring': 'time'})
del pope['timestring']

# get the metric
var = xr.open_dataarray(home+'/Google Drive File Stream/My Drive/Data_Analysis/rom.nc')
var_perc = var.percentile * 100

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

# create dataset and plot it
ds = xr.Dataset({'all': var_perc.where(pope),
                 'pp1': perc_pope_1, 'pp2': perc_pope_2, 'pp3': perc_pope_3, 'pp4': perc_pope_4, 'pp5': perc_pope_5})
bins = np.linspace(0., 100., num=25+1)
fig_h_1d = histogram_1d(ds, l_xlog=False, nbins=bins,
                        x_label='ROM Percentile $P$ [%]',
                        y_label='% / d$P$',
                        legend_label=['All', 'PR 1', 'PR 2', 'PR 3', 'PR 4', 'PR 5'],
                        l_color=True,
                        l_percentage=False,
                        l_rel_mode=False)
save = True
if save:
    plt.savefig(home+'/Desktop/pope.pdf', bbox_inches='tight', transparent=True)
else:
    plt.show()