from os.path import expanduser
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import timeit
import sub as FORTRAN
from divergence_at_drymoist_rome import smallregion_in_tropics

start = timeit.default_timer()
home = expanduser("~")

# no open_mfdataset here, since dask causes runtime-warning in loop below: "invalid value encountered in true_divide"
ds_ps = xr.open_dataset(home+'/Desktop/hist.nc')

area   = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/o_area_10mmhour.nc')
number = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/o_number_10mmhour.nc')
rome   = area * number
# rome = xr.open_dataarray(home + '/Documents/Data/Simulation/r2b10/rome_10mmhour.nc')
rome_p90 = np.nanpercentile(rome, q=90)

l_take_region = True
if l_take_region:
    region = smallregion_in_tropics(rome, 'Tropic', 'coast', other_surface_fillvalue=np.nan)
    rome_copy = xr.full_like(rome, fill_value=np.nan)
    rome_copy.loc[{'lat': region['lat'], 'lon': region['lon']}] = region
    rome = rome_copy

overlay = rome.stack({'x': ('time', 'lat', 'lon')})
overlay['x'] = np.arange(len(overlay))
overlay = overlay.rename({'x': 'time'})

# subselect on the times given in the histogram data
l_histogram = True
if l_histogram:
    overlay = overlay[overlay.notnull()]
    overlay = overlay[overlay.time.isin(ds_ps.time)]

phase_space = ds_ps.hist_2D

# give the overlay time series information about the placements of the bins for each time step
overlay.coords['x_bins'] = ('time', ds_ps.x_series_bins[ds_ps.time.isin(overlay.time)].values)
overlay.coords['y_bins'] = ('time', ds_ps.y_series_bins[ds_ps.time.isin(overlay.time)].values)

phase_space_stack = phase_space.stack(z=('x', 'y'))

ind_1, ind_2 = zip(*phase_space_stack.z.values)
phase_space_stack[:] = FORTRAN.phasespace(indices1=ind_1,
                                          indices2=ind_2,
                                          overlay=overlay,
                                          overlay_x=overlay['x_bins'],
                                          overlay_y=overlay['y_bins'],
                                          l_probability=False,
                                          upper_bound=100000.,
                                          # lower_bound=np.nanpercentile(overlay, q=90))
                                          lower_bound = rome_p90)

# set NaNs to the special values set in the Fortran-routine
phase_space_stack = phase_space_stack.where((phase_space_stack < -1e10) ^ (-9999999998.0 < phase_space_stack))

ps_overlay = phase_space_stack.unstack('z')

# the actual plotting commands
plt.rc('font'  , size=26)
plt.rc('legend', fontsize=18)

the_plot = ps_overlay.T.plot(cmap='OrRd',# 'rainbow',
                             vmin=ps_overlay.min(), vmax=ps_overlay.max())
                             # vmin=0.0028409857748265717, vmax=0.4935537724163929)

plt.xticks((-15, -10, -5, 0, 5))
# plt.xticks((-30, -20, -10, 0, ))

plt.title('Tropical coast, high TCA')
plt.xlabel('$\omega_{515}$ [hPa/hour]')
plt.ylabel('RH$_{515}$ [%]')

# the_plot.colorbar.set_label('Prob. of ROME$_\mathrm{p90,trp}$ [1]')
the_plot.colorbar.set_label('TCA > TCA$_\mathrm{p90,trp}$ [km$^2$]')

plt.savefig(home + '/Desktop/phase_space.pdf', transparent=True, bbox_inches='tight')
plt.show()

stop = timeit.default_timer()
print(f'Time used: {stop - start}')
