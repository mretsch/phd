from os.path import expanduser
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import timeit
import sub as FORTRAN

home = expanduser("~")

start = timeit.default_timer()

# no open_mfdataset here, since dask causes runtime-warning in loop below: "invalid value encountered in true_divide"
ds_ps = xr.open_dataset(home+'/Documents/Plots/2D_Histograms/area_number_hist_ge9.nc')
ls    = xr.open_dataset(home+'/Documents/Data/LargeScaleState/' +
                        'CPOL_large-scale_forcing_cape990hPa_cin990hPa_rh_shear_dcape.nc')
ds    = ls.RH.sel(lev=515).resample(time='10min').interpolate('linear')
# ds    = xr.open_dataset(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/rom_km_avg6h_nanzero.nc')
# ds    = xr.open_dataset(home+'/Documents/Data/Analysis/With_Boundary/conv_intensity.nc')

subselect = True
if subselect:
    # subselect specific times during a day
    l_time_of_day = False
    if l_time_of_day:
        ds.coords['hour'] = ds.indexes['time'].hour
        ds_sub = ds.where(ds.hour.isin([6]), drop=True)

    # subselect on the times given in the histogram data
    l_histogram = True
    if l_histogram:
        # ds_sub = ds.sel(time=ds_ps.time)
        ds_sub = ds[ds.time.isin(ds_ps.time)]
    ds    = ds_sub

phase_space = ds_ps.hist_2D
overlay = ds # .RH.sel(lev=515) #rom_kilometres #conv_intensity #conv_intensity # div.sel(lev=845) # .where(ds.cop_mod < 60.)

# give the overlay time series information about the placements of the bins for each time step
overlay.coords['x_bins'] = ('time', ds_ps.x_series_bins[ds_ps.time.isin(ds_sub.time)].values)
overlay.coords['y_bins'] = ('time', ds_ps.y_series_bins[ds_ps.time.isin(ds_sub.time)].values)

phase_space_stack = phase_space.stack(z=('x', 'y'))

ind_1, ind_2 =zip(*phase_space_stack.z.values)
phase_space_stack[:] = FORTRAN.phasespace(indices1=ind_1,
                                          indices2=ind_2,
                                          overlay=overlay,
                                          overlay_x=overlay['x_bins'],
                                          overlay_y=overlay['y_bins'],
                                          l_probability=False,
                                          upper_bound=10000.,
                                          # lower_bound=overlay[overlay.percentile > 0.9].min(),
                                          lower_bound=0.)

# set NaNs to the special values set in the Fortran-routine
phase_space_stack = phase_space_stack.where((phase_space_stack < -1e10) ^ (-9999999998.0 < phase_space_stack))

ps_overlay = phase_space_stack.unstack('z')

# the actual plotting commands
plt.rc('font'  , size=18)
plt.rc('legend', fontsize=18)
# plt.style.use('dark_background')

the_plot = ps_overlay.T.plot(cmap='BrBG') # (robust=True)  # (cmap='inferno')  # (cmap='tab20c')
# plt.xlabel('$\omega$ at 515 hPa [hPa/h]')
# plt.ylabel('RH at 515 hPa [1]')
# the_plot.colorbar.set_label('Probability of highest ROME decile [1]')
plt.xlabel('Object mean area [km$^2$]')
plt.ylabel('Number of objects [1]')
the_plot.colorbar.set_label('RH at 515 hPa [1]')

save = True
if save:
    plt.savefig(home+'/Desktop/phase_space.pdf', transparent=True, bbox_inches='tight')
    plt.show()
else:
    plt.show()

stop = timeit.default_timer()
print('Script needed: {} seconds'.format(stop - start))