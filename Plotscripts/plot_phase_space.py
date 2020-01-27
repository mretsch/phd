from os.path import expanduser
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import timeit
home = expanduser("~")

start = timeit.default_timer()

# no open_mfdataset here, since dask causes runtime-warning in loop below: "invalid value encountered in true_divide"
ds_ps = xr.open_dataset(home+'/Desktop/hist.nc')
ds    = xr.open_dataset(home+'/Data/Analysis/With_Boundary/conv_area.nc')

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
        ds_sub = ds.sel(time=ds_ps.time)
    ds = ds_sub

phase_space = ds_ps.hist_2D
overlay = ds.conv_area_ratio # div.sel(lev=845) # RH.sel(lev=990) # .where(ds.cop_mod < 60.)

# give the overlay time series information about the placements of the bins for each time step
overlay.coords['x_bins'] = ('time', ds_ps.x_series_bins.values)
overlay.coords['y_bins'] = ('time', ds_ps.y_series_bins.values)

phase_space_stack = phase_space.stack(z=('x', 'y'))

for i, indices in enumerate(phase_space_stack.z):
    ind = indices.item()
    phase_space_stack[i] = overlay.where((overlay['x_bins'] == ind[0]) & (overlay['y_bins'] == ind[1])).mean()

ps_overlay = phase_space_stack.unstack('z')

# the actual plotting commands
plt.rc('font'  , size=18)
plt.rc('legend', fontsize=18)

the_plot = ps_overlay.T.plot() # (robust=True)  # (cmap='inferno')  # (cmap='tab20c')
plt.xlabel('area')
plt.ylabel('number')
the_plot.colorbar.set_label('conv. area fraction [% total area]')

save = True
if save:
    plt.savefig(home+'/Desktop/phase_space.pdf', transparent=True, bbox_inches='tight')
    plt.show()
else:
    plt.show()

stop = timeit.default_timer()
print('Script needed: {} seconds'.format(stop - start))