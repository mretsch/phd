import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import timeit

start = timeit.default_timer()

# no open_mfdataset here, since dask causes runtime-warning in loop below: "invalid value encountered in true_divide"
ds_ps = xr.open_dataset('/Users/mret0001/Data/Analysis/No_Boundary/o_number_area_nB_nL_hist.nc')
ds    = xr.open_dataset('/Users/mret0001/Data/Analysis/No_Boundary/sic.nc')

phase_space = ds_ps.hist_2D
overlay = ds.sic  # .where(ds.cop_mod < 60.)

# give the overlay time series information about the placements of the bins for each time step
overlay.coords['x_bins'] = ('time', ds_ps.x_series_bins.values)
overlay.coords['y_bins'] = ('time', ds_ps.y_series_bins.values)

phase_space_stack = phase_space.stack(z=('x', 'y'))

for i, indices in enumerate(phase_space_stack.z):
    ind = indices.item()
    phase_space_stack[i] = overlay.where((overlay['x_bins'] == ind[0]) & (overlay['y_bins'] == ind[1])).mean()

ps_overlay = phase_space_stack.unstack('z')

# the actual plotting commands
the_plot = ps_overlay.T.plot()  # (cmap='inferno')  # cmap='tab20c')
plt.xlabel('Avg. no boundary object area [pixels]')
plt.ylabel('Number of no boundary objects')
the_plot.colorbar.set_label('SIC')

# plt.savefig('/Users/mret0001/Desktop/phase_space.pdf')
plt.show()

stop = timeit.default_timer()
print('Script needed: {} seconds'.format(stop - start))