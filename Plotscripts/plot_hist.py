from os.path import expanduser
import timeit
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import sub as FORTRAN
home = expanduser("~")

plt.rc('font'  , size=18)
plt.rc('legend', fontsize=18)


def histogram_2d(x_series, y_series, nbins=10,
                 ax=None,
                 x_label='', y_label='', cbar_label='',
                 l_same_axis_length=False, l_cut_off=False):
    """Computes and plots a 2D histogram."""
    start_h = timeit.default_timer()

    x_series_min, x_series_max = x_series.min(), x_series.max()
    y_series_min, y_series_max = y_series.min(), y_series.max()
    # Assign metric to plot and get rid of NaNs.
    # x_series = x_series.fillna(-10000.)
    # y_series = y_series.fillna(-10000.)

    if l_same_axis_length:
        bin_edges = [np.linspace(start=0., stop=max(x_series_max, y_series_max), num=nbins+1),
                     np.linspace(start=0., stop=max(x_series_max, y_series_max), num=nbins+1)]
    else:
        xlow, xupp = np.nanpercentile(x_series, q=1.), np.nanpercentile(x_series, q=99)
        ylow, yupp = np.nanpercentile(y_series, q=1.), np.nanpercentile(y_series, q=99)
        bin_edges = [np.linspace(start=np.round(xlow, decimals=1), stop=np.round(xupp, decimals=1), num=nbins + 1),
                     np.linspace(start=np.round(ylow, decimals=1), stop=np.round(yupp, decimals=1), num=nbins + 1)]

    x_edges = bin_edges[0]
    y_edges = bin_edges[1]

    l_fortran = True
    # takes seconds
    if l_fortran:
        H, xbinseries, ybinseries = FORTRAN.histogram_2d(xseries=x_series, yseries=y_series,
                                                         xedges=x_edges, yedges=y_edges,
                                                         l_density=False,
                                                         l_cut_off=l_cut_off, cut_off=55)
        xbinseries[xbinseries == -1.] = np.nan
        ybinseries[ybinseries == -1.] = np.nan
        # the cut-away part
        # H = np.ma.masked_greater(H, 50)
        # percentages
        Hsum = H.sum()
        H = H * 100.   / Hsum
    # takes minutes
    else:
        # range option gets rid of the original NaNs
        H, x_edges, y_edges = np.histogram2d(x_series, y_series, bins=bin_edges,
                                             range=[[0, x_series.max()], [0, y_series.max()]],
                                             density=True)
        # percentages
        # Hsum = H.sum()
        # H = H * 100. / Hsum
        # to have "density=True", don't multiply by 100 and divide by dx*dy (bin-area)
        # H needs to transposed for correct plot
        H = H.T # * 100.

    # Mask zeros, hence they do not show in plot
    Hmasked = np.ma.masked_where(H == 0, H)

    # create xarray dataset from 2D histogram
    abscissa = x_edges[:-1] + 0.5 * (x_edges[1:] - x_edges[:-1])
    ordinate = y_edges[:-1] + 0.5 * (y_edges[1:] - y_edges[:-1])

    x_bin_series = xr.DataArray(xbinseries)
    y_bin_series = xr.DataArray(ybinseries)

    samplesize = min(x_bin_series.notnull().sum(), y_bin_series.notnull().sum())

    ds_out = xr.Dataset(data_vars={'hist_2D': (['y', 'x'], Hmasked, {'units': '%'}),
                                   'x_series_bins': (['time'], x_bin_series, {'units': 'bin by value'}),
                                   'y_series_bins': (['time'], y_bin_series, {'units': 'bin by value'})},
                        coords={'x': (['x'], abscissa),
                                'y': (['y'], ordinate),
                                'time': (['time'], x_series[x_series.dims[-1]])},
                        attrs={'Sample size': '{:g}'.format(samplesize.values)})

    # Plot 2D histogram
    if ax==None:
        ax = plt.figure() # actually a figure shouldnt be called ax, very ambiguous
    else:
        plt.sca(ax)
        # ax.yaxis.set_label_position("right")
        # ax.tick_params(left=False, labelleft=False)
        # ax.tick_params(right=True, labelright=True)

    plot = plt.pcolormesh(x_edges, y_edges, Hmasked, cmap='rainbow')#'gnuplot2')#'gist_ncar')#  # , cmap='tab20c')

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # try:
    #     cb = plt.colorbar(plot, ax=[ax], location='left')
    # except AttributeError:
    #     cb = plt.colorbar()
    cb = plt.colorbar()
    cb.ax.set_ylabel(cbar_label+', Sample size: {:g}'.format(samplesize.values))

    stop_h = timeit.default_timer()
    print('Histogram Run Time: ', stop_h - start_h)
    return ax, ds_out


if __name__ == '__main__':
    start = timeit.default_timer()

    ls = xr.open_dataset(home + '/Documents/Data/LargeScaleState/' +
                         'CPOL_large-scale_forcing_cape990hPa_cin990hPa_rh_shear_dcape_NoDailyCycle.nc')#.nc')#

    ls_day = xr.open_dataset(home + '/Documents/Data/LargeScaleState/' +
                         'CPOL_large-scale_forcing_cape990hPa_cin990hPa_rh_shear_dcape.nc')#

    # remove false data in precipitable water
    ls_day['PW'].loc[{'time': slice(None, '2002-02-27T12')}] = np.nan
    ls_day['PW'].loc[{'time': slice('2003-02-07T00', '2003-10-19T00')}] = np.nan
    ls_day['PW'].loc[{'time': slice('2005-11-14T06', '2005-12-09T12')}] = np.nan
    ls_day['PW'].loc[{'time': slice('2006-02-25T00', '2006-04-07T00')}] = np.nan
    ls_day['PW'].loc[{'time': slice('2011-11-09T18', '2011-12-01T06')}] = np.nan
    ls_day['PW'].loc[{'time': slice('2015-01-05T00', None)}] = np.nan
    ls_day['LWP'].loc[{'time': slice(None, '2002-02-27T12')}] = np.nan
    ls_day['LWP'].loc[{'time': slice('2003-02-07T00', '2003-10-19T00')}] = np.nan
    ls_day['LWP'].loc[{'time': slice('2005-11-14T06', '2005-12-09T12')}] = np.nan
    ls_day['LWP'].loc[{'time': slice('2006-02-25T00', '2006-04-07T00')}] = np.nan
    ls_day['LWP'].loc[{'time': slice('2011-11-09T18', '2011-12-01T06')}] = np.nan
    ls_day['LWP'].loc[{'time': slice('2015-01-05T00', None)}] = np.nan

    xr.set_options(keep_attrs=True)

    var1 = ls['omega'].sel(lev=515) + ls_day['omega'].sel(lev=515).mean(dim='time')
    var2 = ls['RH']   .sel(lev=515) + ls_day['RH']   .sel(lev=515).mean(dim='time')

    xr.set_options(keep_attrs=False)

    var1 = var1.where(var2)
    var2 = var2.where(var1)
    both_valid = var1.notnull() & var2.notnull()
    var1 = var1[both_valid]
    var2 = var2[both_valid]

    fig_h_2d, h_2d = histogram_2d(var1, var2,  nbins=9,
                                  x_label=var1.long_name+' ['+var1.units+']', #'Total conv. area [km$^2$]', #
                                  y_label=var2.long_name+' ['+var2.units+']',
                                  cbar_label='%',
                                  l_cut_off=True)
    fig_h_2d.show()

    fig_h_2d.savefig(home+'/Desktop/hist.pdf', transparent=True, bbox_inches='tight')
    h_2d.to_netcdf(home+'/Desktop/hist.nc', mode='w')

    stop = timeit.default_timer()
    print('Run Time: ', stop - start)

