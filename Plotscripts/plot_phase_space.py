from os.path import expanduser
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import timeit
import sub as FORTRAN

home = expanduser("~")


def return_phasespace_plot():

    ds_ps = xr.open_dataset(home+'/Desktop/hist.nc')

    # rome  = xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/totalarea_km_avg6h.nc')
    rome  = xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/rom_km_max6h_avg_pm20minutes.nc')

    # subselect on the times given in the histogram data
    da = rome
    da = da[da.notnull()]
    da_sub = da[da.time.isin(ds_ps.time)]
    da = da_sub

    phase_space = ds_ps.hist_2D

    overlay = da

    # give the overlay time series information about the placements of the bins for each time step
    overlay.coords['x_bins'] = ('time', ds_ps.x_series_bins[ds_ps.time.isin(da_sub.time)].values)
    overlay.coords['y_bins'] = ('time', ds_ps.y_series_bins[ds_ps.time.isin(da_sub.time)].values)

    phase_space_stack = phase_space.stack(z=('x', 'y'))

    ind_1, ind_2 =zip(*phase_space_stack.z.values)
    phase_space_stack[:] = FORTRAN.phasespace(indices1=ind_1,
                                              indices2=ind_2,
                                              overlay=overlay,
                                              overlay_x=overlay['x_bins'],
                                              overlay_y=overlay['y_bins'],
                                              l_probability=True,
                                              upper_bound=10000.,
                                              lower_bound=np.percentile(overlay, 90))

    # set NaNs to the special values set in the Fortran-routine
    phase_space_stack = phase_space_stack.where((phase_space_stack < -1e10) ^ (-9999999998.0 < phase_space_stack))

    ps_overlay = phase_space_stack.unstack('z')

    # the actual plotting commands
    plt.rc('font'  , size=26)
    plt.rc('legend', fontsize=18)

    the_plot = ps_overlay.T.plot(cmap='rainbow',
                                 vmin=ps_overlay.min(), vmax=ps_overlay.max())

    plt.xticks((-15, -10, -5, 0))

    plt.xlabel('$\omega_{515}$ [Pa/s]')
    plt.ylabel('RH$_{515}$ [1]')

    the_plot.colorbar.set_label('Probability of ROME$_\mathrm{p90}$ [1]')

    return the_plot


if __name__ == '__main__':

    start = timeit.default_timer()
    plot = return_phasespace_plot()
    save = True
    if save:
        plt.savefig(home+'/Desktop/phase_space_rome.pdf', transparent=True, bbox_inches='tight')
        plt.show()
    else:
        plt.show()
    stop = timeit.default_timer()
    print('Script needed: {} seconds'.format(stop - start))