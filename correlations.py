import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
import scipy as sp
import bottleneck as bn
import Plotscripts.plot_hist as h
import basic_stats as stats

if __name__ == '__main__':

    ds = xr.open_mfdataset(['/Users/mret0001/Data/Analysis/No_Boundary/sic.nc',
                            '/Users/mret0001/Data/Analysis/No_Boundary/eso.nc',
                            '/Users/mret0001/Data/Analysis/No_Boundary/cop.nc',
                            ])  # .sel({'time': slice('2009-10-01', '2010-03-01')})


    # a, b = sp.stats.pearsonr (ds.sic[ds.sic.notnull()], ds.eso[ds.eso.notnull()])
    # c, d = sp.stats.spearmanr(ds.sic[ds.sic.notnull()], ds.eso[ds.eso.notnull()])
    # sns.jointplot(ds.sic, ds.eso)
    # plt.show()
    # plt.close()

    sic_rank = xr.DataArray(bn.nanrankdata(ds.sic))
    eso_rank = xr.DataArray(bn.nanrankdata(ds.eso))
    sic_rank_clean = sic_rank[sic_rank.notnull()]
    eso_rank_clean = eso_rank[eso_rank.notnull()]

    rho = stats.spearman_correlation(ds.sic[ds.sic.notnull()], ds.eso[ds.eso.notnull()])

    fig, _ = h.histogram_2d(sic_rank_clean, eso_rank_clean, nbins=500)
    plt.plot(sic_rank_clean, sic_rank_clean)
    plt.xlabel('SIC rank')
    plt.ylabel('ESO rank')
    plt.show()

