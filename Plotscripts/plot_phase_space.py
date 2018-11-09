import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import timeit
from plot_2Dhist import histogram_2d

# no open_mfdataset here, since dask causes runtime-warning in loop below: "invalid value encountered in true_divide"
ds_ps = xr.open_dataset('/Users/mret0001/Data/Analysis/o_number_area_hist.nc')
ds = xr.open_dataset('/Users/mret0001/Data/Analysis/conv_rr_mean.nc')

phase_space = ds_ps.hist_2D
overlay = ds.conv_rr_mean

# assume I have the two time series telling me at which time I have which x-y-position in the phase-space
x_bins = ds_ps.x_series_bins
y_bins = ds_ps.y_series_bins

# unstack the 2d histogramm and stack the content of the two bin series together . Compare that.

# give the overlay the bin information

overlay.coords['x_bins'] = ('time', x_bins.values)
overlay.coords['y_bins'] = ('time', y_bins.values)

phase_space_stack = phase_space.stack(z=('x', 'y')) #.load()

for i, indices in enumerate(phase_space_stack.z):
    ind = indices.item()
    phase_space_stack[i] = overlay.where((overlay['x_bins'] == ind[0]) & (overlay['y_bins'] == ind[1]), drop=True).mean()

ps_overlay = phase_space_stack.unstack('z')
ps_overlay.T.plot()
plt.show()