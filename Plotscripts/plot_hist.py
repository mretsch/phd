from os.path import expanduser
import timeit
import matplotlib.pyplot as plt
import math as m
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import matplotlib.ticker as ticker
import Plotscripts.colors_solarized as col
import sub as FORTRAN
home = expanduser("~")

plt.rc('font'  , size=12)
plt.rc('legend', fontsize=12)
#sns.set()

def histogram_1d(dataset, nbins=None, l_adjust_bins=False, l_xlog=False, x_label='', y_label='', legend_label=[],
                 l_color=True, l_percentage=True, l_rel_mode=False):
    """Probability distributions for multiple variables in a xarray-dataset."""

    fig, ax = plt.subplots(figsize=(7*0.8, 5*0.8))
    linestyle = ['solid', 'dashed', 'dotted', (0, (1,1)), (0, (3,5,1,5)), (0, (3,1,1,1,1,1))]
    color = [col.sol['blue'], col.sol['red'], col.sol['green'], col.sol['cyan'], col.sol['magenta'], col.sol['yellow']]
    lw = [1., 2., 2., 2., 2., 2.]

    for i, variable in enumerate(dataset):

        var = dataset[variable]
        if type(nbins) == int:
            bins = np.linspace(start=var.min(), stop=var.max(), num=nbins+1)  # 50
        else:
            if l_adjust_bins:
                bins = np.linspace(start=m.sqrt(var.min()), stop=m.sqrt(var.max()), num=18)**2
            else:
                bins = nbins

        # sns.distplot(var[var.notnull()], bins=bins, kde=False, norm_hist=True)  # hist_kws={'log': True})

        total = var.notnull().sum().values
        metric_clean = var.fillna(-1)
        h, edges = np.histogram(metric_clean, bins=bins)  # , density=True)

        if l_rel_mode:
            total = h.max()

        bin_centre = 0.5* (edges[1:] + edges[:-1])
        dx         =       edges[1:] - edges[:-1]
        dlogx      = dx / (bin_centre * m.log(10, m.e))

        if l_xlog:
            h_normed = h / dlogx / total * 100  # equals density=True in percent
        else:
            if l_percentage:
                h_normed = h / total * 100
            else:
                h_normed = h / dx / total * 100  # equals density=True in percent

        if l_color:
            plt.plot(bin_centre, h_normed, color=color[i], linewidth=lw[i])
        else:
            plt.plot(bin_centre, h_normed, color='k', linewidth=2., linestyle=linestyle[i])

    if l_xlog:
        plt.xscale('log')

    plt.ylabel(y_label)
    plt.xlabel(x_label)#, **font)
    lg = plt.legend(legend_label)
    # this sets only the legend background color to transparent (not the surrounding box)
    lg.get_frame().set_facecolor('none')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    #ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    #ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    #ax.tick_params(axis='x', length=5)
    #ax.spines['bottom'].set_position('zero')
    ax.tick_params(axis='x', direction='in')
    ax.xaxis.set_ticks_position('none')  # 'left', 'right'
    ax.set_xlim(0,100)

    ax.tick_params(axis='y', direction='in')
    ax.yaxis.set_ticks_position('none')  # 'left', 'right'

    return fig


def histogram_2d(x_series, y_series, nbins=None, x_label='', y_label='', cbar_label=''):
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
        #bin_edges = [np.linspace(start=0., stop=m.sqrt(250), num=18)**2,
        #             np.linspace(start=0., stop=        80 , num=40+1)]
        bin_edges = [np.linspace(start=0.5, stop= 5.5 , num=5+1),
                     np.linspace(start=0., stop=y_series.max(), num=100+1)]
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

    samplesize = min(x_bin_series.notnull().sum(), y_bin_series.notnull().sum())

    ds_out = xr.Dataset(data_vars={'hist_2D': (['y', 'x'], Hmasked, {'units': '%'}),
                                   'x_series_bins': (['time'], x_bin_series, {'units': 'bin by value'}),
                                   'y_series_bins': (['time'], y_bin_series, {'units': 'bin by value'})},
                        coords={'x': (['x'], abscissa),
                                'y': (['y'], ordinate),
                                'time': (['time'], x_series[x_series.dims[-1]])},
                        attrs={'Sample size': '{:g}'.format(samplesize.values)})

    # Plot 2D histogram
    fig = plt.figure()
    plt.pcolormesh(x_edges, y_edges, Hmasked)  # , cmap='tab20c')
    # plt.grid()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel(cbar_label+', Sample size: {:g}'.format(samplesize.values))

    stop_h = timeit.default_timer()
    print('Histogram Run Time: ', stop_h - start_h)
    return fig, ds_out


if __name__ == '__main__':
    start = timeit.default_timer()

    hist_2d = True
    if hist_2d:
        var1 = xr.open_dataarray(home+'/Data/Analysis/No_Boundary/rom_old_shapelyarea.nc')#.sel({'time': slice('2009-10-01', '2012-03-31')})
        var2 = xr.open_dataarray(home+'/Data/Analysis/No_Boundary/o_area.nc')
        #ds2 = xr.open_mfdataset(home+'/Data/Analysis/No_Boundary/iorg*.nc')
        #var2 = ds2.iorg

        # don't take scenes where convection is 1 pixel large only
        # area_max = ds.o_area_max.where(ds.o_area_max != 1)
        # h_2d = histogram_2d(area_max, ds.o_number, bins=60, x_label='Max object area', y_label='Number of objects')

        fig_h_2d, h_2d = histogram_2d(var1, var2,  nbins=100,
                                      x_label='ROM [pixel]',
                                      y_label='Object mean area [pixel]',
                                      cbar_label='%')  # '[% dx$^{{-1}}$ dy$^{{-1}}$]')
        fig_h_2d.show()

        h_2d.to_netcdf(home+'/Desktop/hist.nc', mode='w')

    hist_1d = False
    if hist_1d:
        var1 = xr.open_dataarray(home+'/Data/Analysis/No_Boundary/sic.nc')
        var2 = xr.open_dataarray(home+'/Data/Analysis/No_Boundary/rom.nc')
        var3 = xr.open_dataarray(home+'/Data/Analysis/No_Boundary/cop.nc')
        del var1['percentile']
        del var2['percentile']
        ds = xr.Dataset({'sic': var1, 'rom': var2, 'cop': var3})
        #ds = xr.Dataset({'rom': var2})#, 'cop': var3})

        fig_h_1d = histogram_1d(ds, l_xlog=True, l_adjust_bins=True,
                                x_label='Metric $\mathcal{M}$  [1]',
                                y_label='d$\mathcal{P}$ / dlog($\mathcal{M}$)  [% $\cdot$ 1$^{-1}$]',
                                legend_label=['SIC', 'ROM', 'COP'],
                                l_color=False)

        fig_h_1d.show()




    stop = timeit.default_timer()
    print('Run Time: ', stop - start)
