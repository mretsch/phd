from os.path import expanduser
home = expanduser("~")
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import seaborn as sns
import scipy as sp
import bottleneck as bn
import Plotscripts.plot_hist as h
import basic_stats as stats

if __name__ == '__main__':

    var_in_1 = xr.open_dataarray(home+'/Data/Analysis/No_Boundary/rom.nc') #.sel({'time': slice('2009-10-01', '2010-03-31')})
    var_in_2 = xr.open_dataarray(home+'/Data/Analysis/No_Boundary/rome.nc')

    if len(var_in_1) != len(var_in_2):
        var1 = var_in_1[var_in_1.notnull()]
        var2 = var_in_2[var_in_1.notnull()]
    else:
        var1 = var_in_1
        var2 = var_in_2

    # a, b = sp.stats.pearsonr (ds.sic[ds.sic.notnull()], ds.eso[ds.eso.notnull()])
    # c, d = sp.stats.spearmanr(ds.sic[ds.sic.notnull()], ds.eso[ds.eso.notnull()])
    # sns.jointplot(ds.sic, ds.eso)
    # plt.show()
    # plt.close()

    r   = stats.pearson_correlation (var1[var1.notnull()], var2[var2.notnull()])
    rho = stats.spearman_correlation(var1[var1.notnull()], var2[var2.notnull()])

    percentiles = True
    if percentiles:
        fig, h_2d = h.histogram_2d(var1.percentile.fillna(-1.) * 100., var2.percentile.fillna(-1) * 100., nbins=100,
                                   x_label='ROM percentiles [%]',
                                   y_label='ROME percentiles [%]',
                                   cbar_label='[%]')
        # plot identity
        plt.plot(var1.percentile.sortby(var1.percentile) * 100., var1.percentile.sortby(var1.percentile) * 100.,
                 color='w', linewidth=0.5)

    else:
        var1_rank = xr.DataArray(bn.nanrankdata(var1))
        var2_rank = xr.DataArray(bn.nanrankdata(var2))

        fig, h_2d = h.histogram_2d(var1_rank.fillna(-1.), var2_rank.fillna(-1.), nbins=100,
                                   x_label='SIC rank',
                                   y_label='ROM rank',
                                   cbar_label='[%]')
        # plot identity
        plt.plot(var1_rank.sortby(var1_rank), var1_rank.sortby(var1_rank),
                 color='w', linewidth=0.5)

    save = True
    if save:
        plt.savefig('/Users/mret0001/Desktop/corr_hist.pdf')

