import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import timeit

start = timeit.default_timer()

# no open_mfdataset here, since dask causes runtime-warning in loop below: "invalid value encountered in true_divide"
ds_ps = xr.open_dataset('/Users/mret0001/Data/Analysis/o_number_area_hist.nc')
ds = xr.open_dataset('/Users/mret0001/Data/Analysis/conv_intensity.nc')

phase_space = ds_ps.hist_2D
overlay = ds.conv_intensity

# two time series with bin for each time step
x_bins = ds_ps.x_series_bins
y_bins = ds_ps.y_series_bins

# give the overlay the bin information
overlay.coords['x_bins'] = ('time', x_bins.values)
overlay.coords['y_bins'] = ('time', y_bins.values)

phase_space_stack = phase_space.stack(z=('x', 'y'))

slightly_slower = False
if slightly_slower:
    dict_ind = {}
    for i, binpair in enumerate(phase_space_stack.z):
        dict_ind[binpair.item()] = i

    # give overlay dimension consisting of the respective x and y bins
    bin_pairs = list(zip(overlay.x_bins.values, overlay.y_bins.values))
    new_coord = [dict_ind[bp] if bp in dict_ind.keys() else np.nan for bp in bin_pairs]
    overlay.coords['bins'] = ('time', new_coord)

    # groupby by the new dimension and .mean
    overlay_binned = overlay.groupby('bins').mean()

    # order for the new dimension, if not already done by groupby

    # assign to phase_stack and unstack it --> overlay in phase_space
    index = overlay_binned.bins.values.astype(int)
    phase_space_stack[index] = overlay_binned.values

else:
    for i, indices in enumerate(phase_space_stack.z):
        ind = indices.item()
        phase_space_stack[i] = overlay.where((overlay['x_bins'] == ind[0]) & (overlay['y_bins'] == ind[1])).mean()

ps_overlay = phase_space_stack.unstack('z')

# the actual plotting commands
the_plot = ps_overlay.T.plot()
plt.xlabel('Average object area [pixels]')
plt.ylabel('Number of objects')
the_plot.colorbar.set_label('Convective rain intensity [mm/h]')
plt.savefig('/Users/mret0001/Desktop/phase_space.pdf')
plt.show()

stop = timeit.default_timer()
print('Script needed: {} seconds'.format(stop - start))