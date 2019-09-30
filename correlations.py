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
from LargeScale.cape_matthias import mixratio_to_spechum, temp_to_virtual

if __name__ == '__main__':

    metric = xr.open_dataarray(home+'/Data/Analysis/No_Boundary/rom_kilometres.nc') #.sel({'time': slice('2009-10-01', '2010-03-31')})
    ls = xr.open_dataset(home+'/Data/LargeScaleState/CPOL_large-scale_forcing_cape_cin_rh.nc')

    # get air density from Large Scale state variables
    r = ls.r[:, 1:]
    p = xr.zeros_like(r)
    p[:, :]  = ls.lev[1:]
    t = ls.T[:, 1:]
    q = mixratio_to_spechum(r, p)
    temp_v = temp_to_virtual(t, q)
    density = p*100 / (287.1 * temp_v)

    var = ls.T[:, 1:]
    # arithmetic mean
    ls_var2 = var.mean(dim='lev')
    # density weighted mean
    ls_var = ((density * var).sum(dim='lev') / density.sum(dim='lev'))
    ls_var._copy_attrs_from(var)

    m_where_ls = metric.where(ls_var[ls_var.notnull()])
    var_in_1 = m_where_ls.where(m_where_ls.notnull(), drop=True)
    var_in_2 = ls_var.where(var_in_1)

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

    percentiles = False
    if percentiles:
        fig, h_2d = h.histogram_2d(var1.percentile.fillna(-1.) * 100., var2.percentile.fillna(-1) * 100., nbins=100,
                                   x_label='ROM percentiles [%]',
                                   y_label='ROME percentiles [%]',
                                   cbar_label='[%]')
        # plot identity
        plt.plot(var1.percentile.sortby(var1.percentile) * 100., var1.percentile.sortby(var1.percentile) * 100.,
                 color='w', linewidth=0.5)

    else:
        # var1_rank = xr.DataArray(bn.nanrankdata(var1))
        # var2_rank = xr.DataArray(bn.nanrankdata(var2))

        fig, h_2d = h.histogram_2d(var1, var2, nbins=100,
                                   x_label='ROME at VA time steps [km$^2$]',
                                   y_label=ls_var.long_name+', ['+ls_var.units+']',
                                   cbar_label='[%]')
        # plot identity
        # plt.plot(var1_rank.sortby(var1_rank), var1_rank.sortby(var1_rank),
        #          color='w', linewidth=0.5)

    save = False
    if save:
        plt.savefig('/Users/mret0001/Desktop/corr_hist.pdf')
    else:
        plt.show()
