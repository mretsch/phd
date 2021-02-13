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

plt.rc('font'  , size=18)
plt.rc('legend', fontsize=18)
#sns.set()

def histogram_1d(data, nbins=None, l_adjust_bins=False, l_xlog=False, x_label='', y_label='', legend_label=[],
                 l_color=True, l_percentage=True, l_rel_mode=False, l_pope=False):
    """Probability distributions for multiple variables in a xarray-dataset."""

    fig, ax = plt.subplots(figsize=(7*0.8, 5*0.8))
    linestyle = ['solid', 'dashed', 'dotted', (0, (1,1)), (0, (3,5,1,5)), (0, (3,1,1,1,1,1))]
    color = [col.sol['blue'], col.sol['red'], col.sol['green'], col.sol['yellow'], col.sol['magenta'], col.sol['cyan']]
    lw = [1., 2., 2., 2., 2., 2.]

    for i, var in enumerate(data):

        # var = dataset[variable]
        if type(nbins) == int:
            bins = np.linspace(start=var.min(), stop=var.max(), num=nbins+1)  # 50
        else:
            if l_adjust_bins:
                bins = np.linspace(start=m.sqrt(var.min()), stop=m.sqrt(var.max()), num=18)**2
            else:
                bins = nbins[i]

        # sns.distplot(var[var.notnull()], bins=bins, kde=False, norm_hist=True)  # hist_kws={'log': True})

        total = var.notnull().sum().values
        # metric_clean = var.fillna(-1)  # works if var is positive and thus bins as well
        metric_clean = var.where(var.notnull(), drop=True)  # works if var is positive and thus bins as well
        h, edges = np.histogram(metric_clean, bins=bins)  # , density=True)

        if l_rel_mode:
            total = h.max()

        bin_centre = 0.5* (edges[1:] + edges[:-1])
        dx         =       edges[1:] - edges[:-1]
        dlogx      = dx / (bin_centre * m.log(10, m.e))

        if l_xlog:
            h_normed = h / dlogx / total # equals density=True
        else:
            if l_percentage:
                h_normed = h / total * 100
            else:
                h_normed = h / dx / total # equals density=True

        if l_color:
            plt.plot(bin_centre, h_normed,
                     linestyle='-',
                     marker='o',
                     color=color[i],
                     linewidth=lw[i])
        else:
            h_normed_ext = np.zeros(shape=len(h_normed)+1)
            h_normed_ext[ 0] = h_normed[0]
            h_normed_ext[1:] = h_normed
            # plot a step function instead of a continuous line
            h_to_plot = h_normed_ext
            # h_to_plot = np.cumsum(h_normed_ext)
            plt.step(edges, h_to_plot, color=color[i], linewidth=2.)#, linestyle=linestyle[i])

    if l_xlog:
        plt.xscale('log')

        x_major = ticker.LogLocator(base=10.0, numticks=10)
        ax.xaxis.set_major_locator(x_major)
        x_minor = ticker.LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=20)
        ax.xaxis.set_minor_locator(x_minor)
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    plt.ylabel(y_label)
    plt.xlabel(x_label)#, **font)
    if l_pope:
        lg = plt.legend(legend_label, fontsize=14)
        # this sets only the legend background color to transparent (not the surrounding box)
        lg.get_frame().set_facecolor('none')

    if l_pope:
        ax.set_xticks([10, 30, 50, 70, 90], minor=True)
        ax.grid(b=True, which='both', axis='x')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if not l_pope:
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        ax.tick_params(axis='x', length=8)
        ax.spines['bottom'].set_position('zero')
        ax.tick_params(axis='x', direction='out')
        # ax.xaxis.set_ticks_position('none')  # 'left', 'right'
        # ax.set_xlim(6)

    ax.tick_params(axis='y', direction='out')
    ax.yaxis.set_ticks_position('none')  # 'left', 'right'

    return fig


def histogram_2d(x_series, y_series, nbins=None, x_label='', y_label='', cbar_label='', l_same_axis_length=False):
    """Computes and plots a 2D histogram."""
    start_h = timeit.default_timer()

    x_series_min, x_series_max = x_series.min(), x_series.max()
    y_series_min, y_series_max = y_series.min(), y_series.max()
    # Assign metric to plot and get rid of NaNs.
    x_series = x_series.fillna(-10000.)
    y_series = y_series.fillna(-10000.)

    if type(nbins) == int:
        if l_same_axis_length:
            bin_edges = [np.linspace(start=0., stop=max(x_series_max, y_series_max), num=nbins+1),
                         np.linspace(start=0., stop=max(x_series_max, y_series_max), num=nbins+1)]
        else:
            bin_edges = [np.linspace(start=x_series_min, stop=x_series_max, num=nbins+1),
                         np.linspace(start=y_series_min, stop=y_series_max, num=nbins+1)]
        bin_edges = [np.linspace(start=0., stop=3000, num=nbins+1),
                     np.linspace(start=0., stop=150, num=nbins+1)]
    else:
        # bin_edges = [np.linspace(start=0., stop=m.sqrt(x_series.max()), num=18)**2,
        #              np.linspace(start=0., stop=       y_series.max(), num=40+1)]
        #bin_edges = [np.linspace(start=0., stop=m.sqrt(250), num=18)**2,
        #             np.linspace(start=0., stop=        80 , num=40+1)]
        # bin_edges = [np.linspace(start=0.5, stop= 5.5 , num=5+1),
        #              np.linspace(start=0., stop=y_series.max(), num=100+1)]
        bin_edges = [np.linspace(start=-40, stop=10, num=16),
                     np.linspace(start=y_series_min, stop=y_series_max, num=16)]
    x_edges = bin_edges[0]
    y_edges = bin_edges[1]

    l_fortran = True
    # takes seconds
    if l_fortran:
        H, xbinseries, ybinseries = FORTRAN.histogram_2d(xseries=x_series, yseries=y_series,
                                                         xedges=x_edges, yedges=y_edges,
                                                         l_density=False,
                                                         l_cut_off=True, cut_off=40)#75)#2600)
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

    # x_bin_series = xr.DataArray(pd.cut(np.array(x_series), x_edges,
    #                                    labels=abscissa,  # np.linspace(1, len(x_edges)-1, len(x_edges)-1),
    #                                    right=False).get_values())
    # y_bin_series = xr.DataArray(pd.cut(np.array(y_series), y_edges,
    #                                    labels=ordinate,  # np.linspace(1, len(y_edges)-1, len(y_edges)-1),
    #                                    right=False).get_values())
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
    fig = plt.figure()
    plt.pcolormesh(x_edges, y_edges, Hmasked)#, cmap='gist_ncar')#  # , cmap='tab20c')
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
        ls = xr.open_dataset(home + '/Documents/Data/LargeScaleState/' +
                             'CPOL_large-scale_forcing_cape990hPa_cin990hPa_rh_shear_dcape.nc')#_NoDailyCycle.nc')

        subselect = False
        if subselect:
            # subselect specific times during a day
            ls.coords['hour'] = ls.indexes['time'].hour
            ls_sub = ls.where(ls.hour.isin([6]), drop=True)
            ls = ls_sub

        var1 = ls.cape#.resample(time='10min').interpolate('linear')
        var2 = ls.cin#.resample(time='10min').interpolate('linear') #PW#lw_net_toa##h2o_adv_col
        # var1 = xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/totalarea_km_avg6h.nc')
        # var2 = xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/o_number.nc')
        # var1 = xr.open_dataarray(home + '/Documents/Data/Analysis/No_Boundary/AllSeasons/rom_km_avg6h_nanzero.nc')
        # model_path = '/Documents/Data/NN_Models/ROME_Models/Kitchen_NoDiurnal/'
        # var2 = xr.open_dataarray(home + model_path + 'predicted.nc')
        # var1 = var1.where(var2)
        # var2 *= -1.

        l_no_singlepixel = True
        if l_no_singlepixel:
            # don't take scenes where convection is 1 pixel large only
            # var1 = var1[var1 != 6.25]
            # var1 = var1[var1 < 800.]
            # var1 = var1[(-10 < var1) & (var1 < 5)]
            # Zoom in via subsetting data
            # var2 = var2[var2 <= 250.]

            # # remove false data in precipitable water
            # var2.loc[{'time': slice(None, '2002-02-27T12')}] = np.nan
            # var2.loc[{'time': slice('2003-02-07T00', '2003-10-19T00')}] = np.nan
            # var2.loc[{'time': slice('2005-11-14T06', '2005-12-09T12')}] = np.nan
            # var2.loc[{'time': slice('2006-02-25T00', '2006-04-07T00')}] = np.nan
            # var2.loc[{'time': slice('2011-11-09T18', '2011-12-01T06')}] = np.nan
            # var2.loc[{'time': slice('2015-01-05T00', None)}] = np.nan

            var1 = var1.where(var2)
            var2 = var2.where(var1)
            both_valid = var1.notnull() & var2.notnull()
            var1 = var1[both_valid]
            var2 = var2[both_valid]

        fig_h_2d, h_2d = histogram_2d(var1, var2*-1,  nbins=9,# 37,#23,
                                      x_label=var1.long_name+' ['+var1.units+']', #'Total conv. area [km$^2$]', #
                                      y_label=var2.long_name+' ['+var2.units+']', # 'Number of objects [1]', #
                                      cbar_label='%', # '[% dx$^{{-1}}$ dy$^{{-1}}$]')
                                      l_same_axis_length=False)
        fig_h_2d.show()

        fig_h_2d.savefig(home+'/Desktop/hist.pdf', transparent=True, bbox_inches='tight')
        h_2d.to_netcdf(home+'/Desktop/hist.nc', mode='w')

    hist_1d = False
    if hist_1d:
        #var1 = xr.open_dataarray(home+'/Data/Analysis/No_Boundary/sic.nc')
        # var2 = xr.open_dataarray(home+'/Data/Analysis/No_Boundary/rom_kilometres.nc')
        #var3 = xr.open_dataarray(home+'/Data/Analysis/No_Boundary/cop.nc')
        #del var1['percentile']
        #del var2['percentile']
        # ds = xr.Dataset({'rom': var2})
        #ds = xr.Dataset({'sic': var1, 'rom': var2, 'cop': var3})
        #ds = xr.Dataset({'rom': var2})#, 'cop': var3})

        ds_sim = xr.open_dataset(home+'/Documents/Data/Analysis/With_Boundary/r2b10/conv_rain_values_3mmhour.nc')
        sim_in_3mm = ds_sim['pr']*3600
        del ds_sim
        ds_sim = xr.open_dataset(home+'/Documents/Data/Analysis/With_Boundary/r2b10/conv_rain_values_7mmhour.nc')
        sim_in_7mm = ds_sim['pr']*3600
        del ds_sim
        ds_sim = xr.open_dataset(home+'/Documents/Data/Analysis/With_Boundary/r2b10/conv_rain_values_14mmhour.nc')
        sim_in_14mm = ds_sim['pr']*3600
        del ds_sim
        # ds = sim_in_mm.to_dataset(name='rain_sim')
        var = xr.open_dataarray(home+'/Documents/Data/Analysis/With_Boundary/conv_rain_radar.nc')
        var_gt0 = var.where(var != 0., drop=True)


        fig_h_1d = histogram_1d([sim_in_3mm, sim_in_7mm, sim_in_14mm, var_gt0], l_xlog=True, l_adjust_bins=False,
                                # nbins=int(np.ceil(ds['pr'].max())), #555, #
                                nbins=[np.arange(int(np.floor(sim_in_3mm.min()))+0.00003,
                                                 int(np.ceil(sim_in_3mm.max()))+0.00003, 1),
                                       np.arange(int(np.floor(sim_in_7mm.min())) + 0.00003,
                                                 int(np.ceil(sim_in_7mm.max())) + 0.00003, 1),
                                       np.arange(int(np.floor(sim_in_14mm.min())) + 0.00003,
                                                 int(np.ceil(sim_in_14mm.max())) + 0.00003, 1),
                                       np.arange(var_gt0.min(), int(np.ceil(var_gt0.max()))+var_gt0.min(), 1)],
                                l_percentage=False,
                                x_label='$P_\mathrm{conv}$ [mm/hour]',
                                # y_label='d$\mathcal{P}$ / d($P_\mathrm{conv}$) [hour/mm]',
                                y_label='d$\mathcal{P}$ / dlog($P_\mathrm{conv}$) [hour/mm]',
                                # legend_label=[''],
                                l_color=False)

        fig_h_1d.show()
        fig_h_1d.savefig(home + '/Desktop/1d_hist.pdf', transparent=True, bbox_inches='tight')




    stop = timeit.default_timer()
    print('Run Time: ', stop - start)

