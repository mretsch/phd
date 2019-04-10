from os.path import expanduser
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from Plotscripts.plot_hist import histogram_1d
from basic_stats import into_pope_regimes, interpolate_repeating_values
home = expanduser("~")

var = xr.open_dataarray(home+'/Data/Analysis/No_Boundary/rome.nc')

ds_pope = into_pope_regimes(var, l_percentile=True, l_all=True)

ds_inter = interpolate_repeating_values(dataset=ds_pope, l_sort_it=True)

bins = np.linspace(0., 100., num=10+1)
fig_h_1d = histogram_1d(ds_inter, l_xlog=False, nbins=bins,
                        x_label='ROME percentile $P$  [%]',
                        y_label='d$\mathcal{P}$ / d$P$  [% $\cdot$ %$^{-1}$]',
                        legend_label=['All', 'PR 1', 'PR 2', 'PR 3', 'PR 4', 'PR 5'],
                        l_color=True,
                        l_percentage=False,
                        l_rel_mode=False)
save = True
if save:
    plt.savefig(home+'/Desktop/pope.pdf', bbox_inches='tight', transparent=True)
else:
    plt.show()