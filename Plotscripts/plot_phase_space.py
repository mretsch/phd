import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import timeit
from plot_2Dhist import histogram_2d

start = timeit.default_timer()

# no open_mfdataset here, since dask causes runtime-warning in loop below: "invalid value encountered in true_divide"
ds_ps = xr.open_dataset('/Users/mret0001/Data/Analysis/o_number_area_hist.nc')
ds = xr.open_dataset('/Users/mret0001/Data/Analysis/conv_rr_mean.nc')

phase_space = ds_ps.hist_2D
overlay = ds.conv_rr_mean

# two time series with bin for each time step
x_bins = ds_ps.x_series_bins
y_bins = ds_ps.y_series_bins

# give the overlay the bin information
overlay.coords['x_bins'] = ('time', x_bins.values)
overlay.coords['y_bins'] = ('time', y_bins.values)

phase_space_stack = phase_space.stack(z=('x', 'y'))

for i, indices in enumerate(phase_space_stack.z):
    ind = indices.item()
    phase_space_stack[i] = overlay.where((overlay['x_bins'] == ind[0]) & (overlay['y_bins'] == ind[1]), drop=True).mean()

ps_overlay = phase_space_stack.unstack('z')

# the actual plotting commands
the_plot = ps_overlay.T.plot()
plt.xlabel('Average object area [pixels]')
plt.ylabel('Number of objects')
the_plot.colorbar.set_label('Mean convective rain rate [mm/h]')
plt.savefig('/Users/mret0001/Desktop/phase_space.pdf')
plt.show()

stop = timeit.default_timer()
print('Run Time: ', stop - start)