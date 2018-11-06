import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import timeit
import sub as FORTRAN


def histogram_2d(x, y, bins=10, x_label='', y_label=''):
    """Computes and plots a 2D histogram."""
    start_h = timeit.default_timer()

    # Assign metric to plot and get rid of NaNs.
    x = x.fillna(-1.)
    y = y.fillna(-1.)

    l_fortran = True
    # takes seconds
    if l_fortran:
        H, xedges, yedges = FORTRAN.histogram_2d(x, y, bins, [0, x.max()], [0, y.max()])
        # percentages
        Hsum = H.sum()
        H = H / Hsum * 100.
    # takes minutes
    else:
        # range option gets rid of the original NaNs
        H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[0, x.max()], [0, y.max()]])
        # percentages
        Hsum = H.sum()
        H = H / Hsum * 100.
        # H needs to transposed for correct plot
        H = H.T

    # Mask zeros, hence they do not show in plot
    Hmasked = np.ma.masked_where(H == 0, H)

    # create xarray dataset from 2D histogram
    abscissa = (xedges[1:] - xedges[:-1]) / 2.
    ordinate = (yedges[1:] - yedges[:-1]) / 2.
    ds_out = xr.Dataset(data_vars={'hist_2D': (['x', 'y'], Hmasked, {'units': '%'})},
                    coords={'x': (['x'], abscissa),
                            'y': (['y'], ordinate)},
                    attrs={'Sample size': '{:g}'.format(Hsum)})

    # Plot 2D histogram
    fig = plt.figure()
    plt.pcolormesh(xedges, yedges, Hmasked)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('[%], Sample size: {:g}'.format(Hsum))

    stop_h = timeit.default_timer()
    print('Histogram Run Time: ', stop_h - start_h)
    return fig, ds_out


if __name__ == '__main__':
    ds = xr.open_mfdataset(["/Users/mret0001/Data/Analysis/o_area.nc",
                            "/Users/mret0001/Data/Analysis/o_number.nc",
                            ])

    # don't take scenes where convection is 1 pixel large only
    # area_max = ds.o_area_max.where(ds.o_area_max != 1)
    # h_2d = histogram_2d(area_max, ds.o_number, bins=60, x_label='Max object area', y_label='Number of objects')

    fig_h_2d, h_2d = histogram_2d(ds.o_area, ds.o_number, bins=40, x_label='Avg object area', y_label='Number of objects')
    fig_h_2d.show()

    h_2d.to_netcdf('/Users/mret0001/Data/Analysis/o_number_area_hist.nc')
