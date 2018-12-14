import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import timeit
import sub as FORTRAN


def histogram_2d(x_series, y_series, bins=10, x_label='', y_label=''):
    """Computes and plots a 2D histogram."""
    start_h = timeit.default_timer()

    # Assign metric to plot and get rid of NaNs.
    x_series = x_series.fillna(-1.)
    y_series = y_series.fillna(-1.)

    l_fortran = True
    # takes seconds
    if l_fortran:
        H, xedges, yedges, x_bin_series, y_bin_series = \
            FORTRAN.histogram_2d(xseries=x_series, yseries=y_series, nbins=bins,
                                 xbound=[0, 200.], ybound=[0, 80],
                                 l_cut_off=True, cut_off=50)
        # set '-1'-values to NaN instead
        x_bin_series[x_bin_series == -1] = np.nan
        y_bin_series[y_bin_series == -1] = np.nan
        # the cut-away part
        # H = np.ma.masked_greater(H, 50)
        # percentages
        Hsum = H.sum()
        H = H / Hsum * 100.
    # takes minutes
    else:
        # range option gets rid of the original NaNs
        H, xedges, yedges = np.histogram2d(x_series, y_series, bins=bins, range=[[0, x_series.max()], [0, y_series.max()]])
        x_bin_series = pd.cut(np.array(x_series), xedges, labels=np.linspace(1, bins, bins), right=False).get_values()
        y_bin_series = pd.cut(np.array(y_series), yedges, labels=np.linspace(1, bins, bins), right=False).get_values()
        # percentages
        Hsum = H.sum()
        H = H / Hsum * 100.
        # H needs to transposed for correct plot
        H = H.T

    # Mask zeros, hence they do not show in plot
    Hmasked = np.ma.masked_where(H == 0, H)

    # create xarray dataset from 2D histogram
    abscissa = xedges[:-1] + (xedges[1:] - xedges[:-1]) / 2.
    ordinate = yedges[:-1] + (yedges[1:] - yedges[:-1]) / 2.
    ds_out = xr.Dataset(data_vars={'hist_2D': (['y', 'x'], Hmasked, {'units': '%'}),
                                   'x_series_bins': (['time'], x_bin_series, {'units': 'bin number'}),
                                   'y_series_bins': (['time'], y_bin_series, {'units': 'bin_number'})},
                        coords={'x': (['x'], abscissa),
                                'y': (['y'], ordinate),
                                'time': (['time'], x_series.time)},
                        attrs={'Sample size': '{:g}'.format(Hsum)})

    # Plot 2D histogram
    fig = plt.figure()
    plt.pcolormesh(xedges, yedges, Hmasked)  # , cmap='tab20c')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('[%], Sample size: {:g}'.format(Hsum))

    stop_h = timeit.default_timer()
    print('Histogram Run Time: ', stop_h - start_h)
    return fig, ds_out


if __name__ == '__main__':
    start = timeit.default_timer()

    ds = xr.open_mfdataset(["/Users/mret0001/Data/Analysis/No_Boundary/o_area.nc",
                            "/Users/mret0001/Data/Analysis/No_Boundary/o_number.nc",
                            ])

    # don't take scenes where convection is 1 pixel large only
    # area_max = ds.o_area_max.where(ds.o_area_max != 1)
    # h_2d = histogram_2d(area_max, ds.o_number, bins=60, x_label='Max object area', y_label='Number of objects')

    fig_h_2d, h_2d = histogram_2d(ds.o_area, ds.o_number, bins=40, x_label='Avg object area', y_label='Number of objects')
    fig_h_2d.show()

    h_2d.to_netcdf('/Users/mret0001/Desktop/o_number_area_hist.nc', mode='w')

    stop = timeit.default_timer()
    print('Run Time: ', stop - start)
