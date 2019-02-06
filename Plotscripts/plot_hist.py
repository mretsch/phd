import timeit
import matplotlib.pyplot as plt
import math as m
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import sub as FORTRAN

plt.rc('font'  , size=12)
plt.rc('legend', fontsize=12)
#font = {'fontname': 'Helvetica'}
#plt.rc('font',**{'family':'serif','serif':['Times']})
# plt.rc('text.latex' , preamble=r'\usepackage{cmbright}')


def histogram_1d(dataset, nbins=None, l_xlog=False, x_label='', y_label='', legend_label=[]):
    """Probability distributions for multiple variables in a xarray-dataset."""

    fig, ax = plt.subplots()
    linestyle = ['dashed', 'solid', 'dotted']

    for i, metric in enumerate([dataset.eso, dataset.sic, dataset.cop]):

        if type(nbins) == int:
            bins = np.linspace(start=0., stop=metric.max(), num=nbins+1)  # 50
        else:
            bins = np.linspace(start=m.sqrt(metric.min()), stop=m.sqrt(metric.max()), num=18)**2

        # sns.distplot(metric[metric.notnull()], bins=bins, kde=False, norm_hist=True)  # hist_kws={'log': True})

        total = metric.notnull().sum().values
        metric_clean = metric.fillna(-1)
        h, edges = np.histogram(metric_clean, bins=bins)  # , density=True)

        bin_centre = 0.5* (edges[1:] + edges[:-1])
        dx         =       edges[1:] - edges[:-1]
        dlogx      = dx / (bin_centre * m.log(10))

        if l_xlog:
            h_normed = h / dlogx / total * 100  # equals density=True in percent
        else:
            h_normed = h / dx / total  # equals density=True

        plt.plot(bin_centre, h_normed, color='k', linewidth=2., linestyle=linestyle[i])

    if l_xlog:
        plt.xscale('log')

    plt.ylabel(y_label)
    plt.xlabel(x_label)#, **font)
    plt.legend(legend_label)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    #ax.yaxis.set_ticks_position('none')  # 'left', 'right'
    ax.tick_params(axis='y', direction='in')
    ax.tick_params(axis='x', length=5)

    return fig


def histogram_2d(x_series, y_series, nbins=None, x_label='', y_label=''):
    """Computes and plots a 2D histogram."""
    start_h = timeit.default_timer()

    # Assign metric to plot and get rid of NaNs.
    x_series = x_series.fillna(-1.)
    y_series = y_series.fillna(-1.)

    if type(nbins) == int:
        bin_edges = [np.linspace(start=0., stop=x_series.max(), num=nbins+1),
                     np.linspace(start=0., stop=y_series.max(), num=nbins+1)]
    else:
        # bin_edges = [np.linspace(start=0., stop=m.sqrt(x_series.max()), num=18)**2,
        #              np.linspace(start=0., stop=       y_series.max(), num=40+1)]
        bin_edges = [np.linspace(start=0., stop=m.sqrt(250), num=18)**2,
                     np.linspace(start=0., stop=        80 , num=40+1)]
    x_edges = bin_edges[0]
    y_edges = bin_edges[1]

    l_fortran = True
    # takes seconds
    if l_fortran:
        H = FORTRAN.histogram_2d(xseries=x_series, yseries=y_series,
                                 xedges=x_edges, yedges=y_edges,
                                 l_cut_off=False, cut_off=50, l_density=False)
                                 # l_cut_off=True, cut_off=50)
        # the cut-away part
        # H = np.ma.masked_greater(H, 50)
        # percentages
        Hsum = H.sum()
        # H = H * 100.  # / Hsum
    # takes minutes
    else:
        # range option gets rid of the original NaNs
        H, x_edges, y_edges = np.histogram2d(x_series, y_series, bins=bin_edges,
                                             range=[[0, x_series.max()], [0, y_series.max()]],
                                             density=True)
        # percentages
        # Hsum = H.sum()
        # H = H * 100. / Hsum
        # to have "density=True", don't multiply by 100 and divide by dx*dy (bin-area),
        # which in case of COP vs. M1 with 40 bins is:
        # H = H / Hsum / (6.795294 * 0.013159)
        # H needs to transposed for correct plot
        H = H.T # * 100.

    # Mask zeros, hence they do not show in plot
    Hmasked = np.ma.masked_where(H == 0, H)

    # create xarray dataset from 2D histogram
    abscissa = x_edges[:-1] + 0.5 * (x_edges[1:] - x_edges[:-1])
    ordinate = y_edges[:-1] + 0.5 * (y_edges[1:] - y_edges[:-1])

    x_bin_series = xr.DataArray(pd.cut(np.array(x_series), x_edges,
                                       labels=abscissa,  # np.linspace(1, len(x_edges)-1, len(x_edges)-1),
                                       right=False).get_values())
    y_bin_series = xr.DataArray(pd.cut(np.array(y_series), y_edges,
                                       labels=ordinate,  # np.linspace(1, len(y_edges)-1, len(y_edges)-1),
                                       right=False).get_values())

    ds_out = xr.Dataset(data_vars={'hist_2D': (['y', 'x'], Hmasked, {'units': '%'}),
                                   'x_series_bins': (['time'], x_bin_series, {'units': 'bin by value'}),
                                   'y_series_bins': (['time'], y_bin_series, {'units': 'bin by value'})},
                        coords={'x': (['x'], abscissa),
                                'y': (['y'], ordinate),
                                'time': (['time'], x_series[x_series.dims[-1]])},
                        attrs={'Sample size': '{:g}'.format(x_bin_series.notnull().sum().values)})

    # Plot 2D histogram
    fig = plt.figure()
    plt.pcolormesh(x_edges, y_edges, Hmasked)  # , cmap='tab20c')
    # plt.grid()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('[% dx$^{{-1}}$ dy$^{{-1}}$], Sample size: {:g}'.format(x_bin_series.notnull().sum().values))

    stop_h = timeit.default_timer()
    print('Histogram Run Time: ', stop_h - start_h)
    return fig, ds_out


if __name__ == '__main__':
    start = timeit.default_timer()

    hist_2d = False
    if hist_2d:
        ds = xr.open_mfdataset(["/Users/mret0001/Data/Analysis/No_Boundary/o_area.nc",
                                "/Users/mret0001/Data/Analysis/No_Boundary/o_number.nc",
                                ])

        # don't take scenes where convection is 1 pixel large only
        # area_max = ds.o_area_max.where(ds.o_area_max != 1)
        # h_2d = histogram_2d(area_max, ds.o_number, bins=60, x_label='Max object area', y_label='Number of objects')

        fig_h_2d, h_2d = histogram_2d(ds.o_area, ds.o_number,  # nbins=40,
                                      x_label='Avg. no boundary object area', y_label='Number of no boundary objects')
        fig_h_2d.show()

        h_2d.to_netcdf('/Users/mret0001/Desktop/hist.nc', mode='w')

    hist_1d = True
    if hist_1d:
        ds = xr.open_mfdataset(['/Users/mret0001/Data/Analysis/No_Boundary/sic.nc',
                                '/Users/mret0001/Data/Analysis/No_Boundary/eso.nc',
                                '/Users/mret0001/Data/Analysis/No_Boundary/cop.nc',
                                ])

        fig_h_1d = histogram_1d(ds, l_xlog=True,
                                x_label='Metric $\mathcal{M}$  [1]',
                                y_label='d$\mathcal{P}$ / dlog($\mathcal{M}$)  [% $\cdot 1^{-1}$]',
                                legend_label=['ESO', 'SIC', 'COP'])

        fig_h_1d.show()




    stop = timeit.default_timer()
    print('Run Time: ', stop - start)
