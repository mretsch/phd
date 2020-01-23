from os.path import expanduser
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import timeit
home = expanduser("~")

start = timeit.default_timer()

# no open_mfdataset here, since dask causes runtime-warning in loop below: "invalid value encountered in true_divide"
# ds_ps = xr.open_dataset('/Users/mret0001/Desktop/LS_PhaseSpaces_2/cape_cin_hist.nc')
ds_ps = xr.open_dataset(home+'/Documents/Plots/Phase_Space/LS_990hPa_CAPECIN/cincape_hist.nc')
ds    = xr.open_dataset(home+'/Data/LargeScaleState/CPOL_large-scale_forcing_cape990hPa_cin990hPa_rh_shear.nc')

subselect = False
if subselect:
    # subselect specific times during a day
    ds.coords['hour'] = ds.indexes['time'].hour
    ds_sub = ds.where(ds.hour.isin([6]), drop=True)
    ds = ds_sub

phase_space = ds_ps.hist_2D
for level in ds.lev[:-1]:
    overlay = ds.div.sel(lev=level) # RH.sel(lev=990) # .where(ds.cop_mod < 60.)

    # give the overlay time series information about the placements of the bins for each time step
    overlay.coords['x_bins'] = ('time', ds_ps.x_series_bins.values)
    overlay.coords['y_bins'] = ('time', ds_ps.y_series_bins.values)

    phase_space_stack = phase_space.stack(z=('x', 'y'))

    for i, indices in enumerate(phase_space_stack.z):
        ind = indices.item()
        phase_space_stack[i] = overlay.where((overlay['x_bins'] == ind[0]) & (overlay['y_bins'] == ind[1])).mean()

    ps_overlay = phase_space_stack.unstack('z')


    # ps_overlay =  ps_overlay.where(ps_overlay < 5523, other = 5523)

    # the actual plotting commands
    plt.rc('font'  , size=18)
    plt.rc('legend', fontsize=18)

    the_plot = ps_overlay.T.plot(robust=True)  # (cmap='inferno')  # (cmap='tab20c')
    # plt.ylabel('515 hPa Relative humidity '+' ['+ds.RH_srf.units+']')
    # plt.xlabel('515 hPa '+ds.omega.long_name+' ['+ds.omega.units+']')
    plt.ylabel('CAPE [J/kg]')
    plt.xlabel('CIN [J/kg]')
    the_plot.colorbar.set_label(overlay.long_name+' ['+overlay.units+']')

    save = True
    if save:
        plt.savefig(home+'/Desktop/div'+str(level.values)+'.pdf', transparent=True, bbox_inches='tight')
        plt.show()
    else:
        plt.show()

stop = timeit.default_timer()
print('Script needed: {} seconds'.format(stop - start))