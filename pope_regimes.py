import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import Plotscripts.plot_hist as histo

a = date(2009,11,30)
b = date(2017,3,31)


dfr_pope = pd.read_csv('/Users/mret0001/Desktop/Pope_regimes.csv', header=None, names=['timestring', 'regime'], index_col=0)

dse = pd.Series(dfr_pope['regime'])

da_pope = xr.DataArray(dse)
pope_years = da_pope.sel({'timestring': slice('2009-11-30','2017-03-31')})
pope_years.coords['time'] = ('timestring', pd.to_datetime(pope_years.timestring))
pope = pope_years.swap_dims({'timestring': 'time'})
del pope['timestring']


sic = xr.open_dataarray('/Users/mret0001/Data/Analysis/No_Boundary/sic.nc')
sic_perc = sic.percentile

#test = sic_perc.groupby('time.daz')
perc_day_max = sic_perc.resample(time='1D', skipna=True).mean()
perc_trunc = perc_day_max.sel({'time': pope.time})

fig, h = histo.histogram_2d(y_series=perc_trunc, x_series=pope, y_label='Daily avg. SIC percentile', x_label='Pope regime')
#plt.savefig('/Users/mret0001/Desktop/pope_sic_avg_hist.pdf')

pope_trunc = pope.where(perc_trunc.notnull())


perc_pope_1 = perc_trunc.where(pope_trunc == 1)# , drop=True)
perc_pope_2 = perc_trunc.where(pope_trunc == 2)# , drop=True)
perc_pope_3 = perc_trunc.where(pope_trunc == 3)# , drop=True)
perc_pope_4 = perc_trunc.where(pope_trunc == 4)# , drop=True)
perc_pope_5 = perc_trunc.where(pope_trunc == 5)# , drop=True)

ds = xr.Dataset({'pp1': perc_pope_1, 'pp2': perc_pope_2, 'pp3': perc_pope_3, 'pp4': perc_pope_4, 'pp5': perc_pope_5})

fig_h_1d = histo.histogram_1d(ds, l_xlog=False, nbins=10,
                        x_label='',
                        y_label='',
                        legend_label=['1', '2', '3', '4', '5'])