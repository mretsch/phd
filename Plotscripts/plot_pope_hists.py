from os.path import expanduser
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from Plotscripts.colors_solarized import sol
from Plotscripts.plot_hist import histogram_1d
from basic_stats import into_pope_regimes, interpolate_repeating_values
home = expanduser("~")

l_hists_on_top = False
if l_hists_on_top:
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
                            x_label='ROME percentile $X$  [%]',
                            y_label='d$\mathcal{P}$ / d$X$  [%$^{-1}$]',
                            legend_label=['All', 'DW', 'SW', 'ME'],
                            l_color=True,
                            l_percentage=False,
                            l_rel_mode=False,
                            l_pope=True)

l_hists_side_by = True
if l_hists_side_by:
    rome = xr.open_dataarray(home + '/Documents/Data/Analysis/No_Boundary/AllSeasons/rom_km_max6h_avg_pm20minutes.nc')

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
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
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
    ax.legend(fontsize=10)
    ax.set_xticklabels(('xxx', '(DE)', 'DW', '(E)', 'SW', 'ME'))
    ax.set_title('Max. ROME $\pm$ 20 minutes')

save = True
if save:
    plt.savefig(home+'/Desktop/pope.pdf', bbox_inches='tight', transparent=True)
else:
    plt.show()