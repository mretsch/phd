from os.path import expanduser
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from Plotscripts.colors_solarized import sol
from basic_stats import into_pope_regimes
home = expanduser("~")

plt.rc('font', size=12  )

# rome = xr.open_dataarray(home + '/Documents/Data/Analysis/No_Boundary/AllSeasons/rom_km_max6h_avg_pm20minutes.nc')
rome = xr.open_dataarray(home + '/Documents/Data/Analysis/No_Boundary/AllSeasons/totalarea_km_avg6h.nc')

ds_pope = into_pope_regimes(rome, l_upsample=True, l_all=True)

p_regime = xr.full_like(ds_pope.var_all, np.nan)
p_regime[:] = xr.where(ds_pope.var_p1.notnull(), 1, p_regime)
p_regime[:] = xr.where(ds_pope.var_p2.notnull(), 2, p_regime)
p_regime[:] = xr.where(ds_pope.var_p3.notnull(), 3, p_regime)
p_regime[:] = xr.where(ds_pope.var_p4.notnull(), 4, p_regime)
p_regime[:] = xr.where(ds_pope.var_p5.notnull(), 5, p_regime)

# What percentiles?
percentiles = rome.percentile

bins = []
# should be 2 at least
n_bins = 10
# the percentile-separating numbers
p_edges = np.linspace(0., 1., n_bins + 1)

# taking rome-values into the bins is okay, sometimes we use the time information only, sometimes the values itself.
# The binning itself is still done based on 'percentile' as assigned above.
bins.append(rome.where(percentiles < p_edges[1], drop=True))
for i in range(1, n_bins - 1):
    bins.append(rome.where((p_edges[i] <= percentiles) & (percentiles < p_edges[i + 1]), drop=True))
bins.append(rome.where(p_edges[-2] <= percentiles, drop=True))

colours = ['yellow', 'orange', 'red', 'magenta', 'violet', 'blue', 'cyan', 'green', 'base01', 'base03']
color_list = [sol[c] for c in colours]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 3))
ax.hist([
    # p_regime.where(rome.notnull(), drop=True).values,
    p_regime.where(bins[0]).values,
    p_regime.where(bins[1]).values,
    p_regime.where(bins[2]).values,
    p_regime.where(bins[3]).values,
    p_regime.where(bins[4]).values,
    p_regime.where(bins[5]).values,
    p_regime.where(bins[6]).values,
    p_regime.where(bins[7]).values,
    p_regime.where(bins[8]).values,
    p_regime.where(bins[9]).values
],
    bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
    density=True,
    color=color_list,
    label=['1. Decile',
           '2. Decile',
           '3. Decile',
           '4. Decile',
           '5. Decile',
           '6. Decile',
           '7. Decile',
           '8. Decile',
           '9. Decile',
           '10. Decile',
           ])
ax.legend(fontsize=6.5)
ax.set_xticklabels(('xxx', 'DE', 'DW', 'E', 'SW', 'ME'))
ax.set_title('TCA')
ax.set_ylabel('Histogram [1]')
ax.set_ylim((0.0, 0.4877659574468085))
ax.set_xlabel('Pope regimes')

save = True
if save:
    plt.savefig(home+'/Desktop/pope.pdf', bbox_inches='tight', transparent=True)
else:
    plt.show()