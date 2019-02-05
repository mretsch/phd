import matplotlib.pyplot as plt
import math as m
import numpy as np
import xarray as xr
import seaborn as sns
import scipy as sp
import bottleneck as bn
import Plotscripts.plot_2Dhist as h


# Code taken from http://xarray.pydata.org/en/stable/dask.html#automatic-parallelization

def covariance(x, y):
    return ( (x - x.mean(axis=-1))
            *(y - y.mean(axis=-1))).mean(axis=-1)


def pearson_correlation(x, y):
    return covariance(x, y) / (x.std(axis=-1) * y.std(axis=-1))


def spearman_correlation(x, y):
    x_ranks = bn.rankdata(x, axis=-1)
    y_ranks = bn.rankdata(y, axis=-1)
    return pearson_correlation(x_ranks, y_ranks)


# TODO split script into a part for plotting (to 2D-hists) and put functions above to basic_stats.


ds = xr.open_mfdataset(['/Users/mret0001/Data/Analysis/No_Boundary/sic.nc',
                        '/Users/mret0001/Data/Analysis/No_Boundary/eso.nc',
                        '/Users/mret0001/Data/Analysis/No_Boundary/iorg.nc',
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

eso_rank_clean[-1] = 7000.

rho = spearman_correlation(ds.sic[ds.sic.notnull()], ds.eso[ds.eso.notnull()])

fig, _ = h.histogram_2d(sic_rank_clean, eso_rank_clean, nbins=500)
plt.plot(sic_rank_clean, sic_rank_clean)
plt.show()

