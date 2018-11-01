import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import timeit
import sub as FORTRAN


def histogram_2d(x, y, bins=10):
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

    # Plot 2D histogram
    fig = plt.figure()
    plt.pcolormesh(xedges, yedges, Hmasked)
    plt.xlabel('Metric M1')
    plt.ylabel('Metric COP')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('[%], Sample size: {:g}'.format(Hsum))
    fig.show()

    stop_h = timeit.default_timer()
    print('Histogram Run Time: ', stop_h - start_h)


if __name__ == '__main__':
    ds = xr.open_mfdataset(["/Users/mret0001/Data/Analysis/m1.nc",
                            "/Users/mret0001/Data/Analysis/cop.nc,",
                            ])

    histogram_2d(ds.m1, ds.cop, bins=160)
