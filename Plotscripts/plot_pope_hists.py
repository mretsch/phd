from os.path import expanduser
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from Plotscripts.plot_hist import histogram_1d
from basic_stats import into_pope_regimes, interpolate_repeating_values
home = expanduser("~")

var = xr.open_dataarray(home+'/Data/Analysis/No_Boundary/rom.nc')

ds_pope_perc = into_pope_regimes(var, l_percentile=True, l_all=True)
# 'PR 1', 'PR 2', 'PR 3', 'PR 4', 'PR 5'
# 'DE'  , 'DW'  , 'E'   , 'SW'  , 'ME'

del ds_pope_perc['var_p1']
del ds_pope_perc['var_p3']

ds_inter = interpolate_repeating_values(dataset=ds_pope_perc, l_sort_it=True)

# This is cosmetics, because the percentiles are still not increasing with the same Delta between them. This is all
# due to the fact, that the data first gets ranked and then divided by data length to attain percentiles.
# But some ranks occur multiple times, getting assigned their average rank, causing non-equal bin counts.
ds_inter.var_all[0:155521] = np.linspace(0, 100., 155521.)

bins = np.linspace(0., 100., num=10+1)
fig_h_1d = histogram_1d(ds_inter, l_xlog=False, nbins=bins,
                        x_label='ROME percentile $P$  [%]',
                        y_label='d$\mathcal{P}$ / d$P$  [1 / %]',
                        legend_label=['All', 'DW', 'SW', 'ME'],
                        l_color=True,
                        l_percentage=False,
                        l_rel_mode=False)
save = True
if save:
    plt.savefig(home+'/Desktop/pope.pdf', bbox_inches='tight', transparent=True)
else:
    plt.show()