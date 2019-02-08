import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
import scipy as sp
import bottleneck as bn
import Plotscripts.plot_hist as h
import basic_stats as stats

if __name__ == '__main__':

    sic = xr.open_dataarray(
        '/Users/mret0001/Data/Analysis/No_Boundary/sic.nc').sel({'time': slice('2009-10-01', '2010-03-31')})
    ior = xr.open_dataarray('/Users/mret0001/Data/Analysis/No_Boundary/iorg_season0910.nc')

    # a, b = sp.stats.pearsonr (ds.sic[ds.sic.notnull()], ds.eso[ds.eso.notnull()])
    # c, d = sp.stats.spearmanr(ds.sic[ds.sic.notnull()], ds.eso[ds.eso.notnull()])
    # sns.jointplot(ds.sic, ds.eso)
    # plt.show()
    # plt.close()

    r   = stats.pearson_correlation (sic[sic.notnull()], ior[ior.notnull()])
    rho = stats.spearman_correlation(sic[sic.notnull()], ior[ior.notnull()])

    percentiles = False
    if percentiles:
        fig, h_2d = h.histogram_2d(sic.percentile.fillna(-1.) * 100., eso.percentile.fillna(-1)  * 100., nbins=100,
                                   x_label='SIC percentiles [%]',
                                   y_label='ESO percentiles [%]',
                                   cbar_label='[%]')
        plt.plot(sic.percentile.sortby(sic.percentile)*100., sic.percentile.sortby(sic.percentile)*100.,
                 color='w', linewidth=0.5)

    else:
        sic_rank = xr.DataArray(bn.nanrankdata(sic))
        ior_rank = xr.DataArray(bn.nanrankdata(ior))

        fig, h_2d = h.histogram_2d(sic_rank.fillna(-1.), ior_rank.fillna(-1.), nbins=100,
                                   x_label='SIC rank',
                                   y_label='I$_{org}$ rank',
                                   cbar_label='[%]')
        plt.plot(sic_rank.sortby(sic_rank), sic_rank.sortby(sic_rank),
                 color='w', linewidth=0.5)

    save = True
    if save:
        plt.savefig('/Users/mret0001/Desktop/corr_hist.pdf')

