import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import timeit
import sub as FORTRAN


def histogram_2d(x_series, y_series, nbins=None, x_label='', y_label=''):
    """Computes and plots a 2D histogram."""
    start_h = timeit.default_timer()

    # Assign metric to plot and get rid of NaNs.
    x_series = x_series.fillna(-1.)
    y_series = y_series.fillna(-1.)

    if type(nbins) == int:
        bin_edges = [np.linspace(start=0., stop=x_series.max(), num=nbins),
                     np.linspace(start=0., stop=y_series.max(), num=nbins)]
    else:
        bin_edges = [np.linspace(start=0., stop=0.8, num=18)**2, np.linspace(start=0., stop=17, num=18)**2]
    x_edges = bin_edges[0]
    y_edges = bin_edges[1]

    l_fortran = False
    # takes seconds
    if l_fortran:
        H = FORTRAN.histogram_2d(xseries=x_series, yseries=y_series,
                                 xedges=x_edges, yedges=y_edges,
                                 # xbound=[0, x_series.max()], ybound=[0, y_series.max()],
                                 l_cut_off=False, cut_off=50)
                                 # xbound=[0, 200.], ybound=[0, 80],
                                 # l_cut_off=True, cut_off=50)
        # the cut-away part
        # H = np.ma.masked_greater(H, 50)
        # percentages
        Hsum = H.sum()
        # H = H / Hsum * 100.
    # takes minutes
    else:
        # range option gets rid of the original NaNs
        H, x_edges, y_edges = np.histogram2d(x_series, y_series, bins=bin_edges,
                                             range=[[0, x_series.max()], [0, y_series.max()]],
                                             density=False
                                             )
        # percentages
        Hsum = H.sum()
        H = H / Hsum * 100.
        # to have "density=True", don't multiply by 100 and divide by dx*dy (bin-area),
        # which in case of COP vs. M1 with 40 bins is:
        # H = H / Hsum / (6.795294 * 0.013159)
        # H needs to transposed for correct plot
        H = H.T # * 100.

    # Mask zeros, hence they do not show in plot
    Hmasked = np.ma.masked_where(H == 0, H)

    # create xarray dataset from 2D histogram
    x_bin_series = pd.cut(np.array(x_series), x_edges, labels=np.linspace(1, len(x_edges)-1, len(x_edges)-1),
                          right=False).get_values()
    y_bin_series = pd.cut(np.array(y_series), y_edges, labels=np.linspace(1, len(y_edges)-1, len(y_edges)-1),
                          right=False).get_values()
    abscissa = x_edges[:-1] + 0.5 * (x_edges[1:] - x_edges[:-1])
    ordinate = y_edges[:-1] + 0.5 * (y_edges[1:] - y_edges[:-1])
    ds_out = xr.Dataset(data_vars={'hist_2D': (['y', 'x'], Hmasked, {'units': '%'}),
                                   'x_series_bins': (['time'], x_bin_series, {'units': 'bin number'}),
                                   'y_series_bins': (['time'], y_bin_series, {'units': 'bin_number'})},
                        coords={'x': (['x'], abscissa),
                                'y': (['y'], ordinate),
                                'time': (['time'], x_series.time)},
                        attrs={'Sample size': '{:g}'.format(Hsum)})

    # Plot 2D histogram
    fig = plt.figure()
    plt.pcolormesh(x_edges, y_edges, Hmasked)  # , cmap='tab20c')
    # plt.grid()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('[% dx$^{{-1}}$ dy$^{{-1}}$], Sample size: {:g}'.format(Hsum))

    stop_h = timeit.default_timer()
    print('Histogram Run Time: ', stop_h - start_h)
    return fig, ds_out


if __name__ == '__main__':
    start = timeit.default_timer()

    ds = xr.open_mfdataset(["/Users/mret0001/Data/Analysis/With_Boundary/cop_threedays.nc",
                            "/Users/mret0001/Data/Analysis/With_Boundary/m1_threedays.nc",
                            ])

    # don't take scenes where convection is 1 pixel large only
    # area_max = ds.o_area_max.where(ds.o_area_max != 1)
    # h_2d = histogram_2d(area_max, ds.o_number, bins=60, x_label='Max object area', y_label='Number of objects')

    fig_h_2d, h_2d = histogram_2d(ds.cop, ds.m1, nbins=40,
                                  x_label='COP', y_label='M1')
    fig_h_2d.show()

    h_2d.to_netcdf('/Users/mret0001/Desktop/hist.nc', mode='w')

    stop = timeit.default_timer()
    print('Run Time: ', stop - start)
