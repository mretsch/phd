from os.path import expanduser
home = expanduser("~")
import timeit
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from LargeScale.ls_at_metric import large_scale_at_metric_times
from Plotscripts.colors_solarized import sol



start = timeit.default_timer()

# assemble the large scale dataset
ds_ls = xr.open_dataset(home+'/Google Drive File Stream/My Drive/Data/LargeScale/CPOL_large-scale_forcing_cape_cin_rh_shear.nc')
metric = xr.open_dataarray(home+'/Google Drive File Stream/My Drive/Data_Analysis/rom_km_avg6h.nc')

predictor, _, var_size = large_scale_at_metric_times(ds_largescale=ds_ls,
                                                     timeseries=metric,
                                                     l_take_same_time=True)

# eof-analysis itself
# see here https://stackoverflow.com/questions/13224362/principal-component-analysis-pca-in-python#13224592

# First make the data deviations from the mean and standardise it
data      = predictor - predictor.mean(dim='time')
data_norm = data / data.std(dim='time')

# where std_dev=0., dividing led to NaN, set to 0. instead
data_norm = data_norm.where(data_norm.notnull(), other=0.)

# in my case, each column 'represents a variable' (taken from np.cov-docu), e.g. omega at 900hPa in one column.
cov_matrix = np.cov(data_norm, rowvar=False)

eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

idx = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# at the moment, all eigenvectors[:, i] are scaled such that each has the l2-norm of 1.
norm_orig = np.linalg.norm(eigenvectors, axis=0)
# now scale each vector such that its l2-norm equals sqrt(eigenvalue).
eigenvalues[eigenvalues < 0.] = 0
scale_factor = np.sqrt(eigenvalues) / norm_orig
eigenvectors = scale_factor * eigenvectors

# try some time series
pc_1   = eigenvectors[:, 0] @ data_norm.T.values
# compute the principal component time series for all eigenvectors
pc_all = eigenvectors.T     @ data_norm.T.values


# the plot, want colors in the background for each 'variable'
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 4))
ax.plot(eigenvectors[:, 0], color='k', linestyle='-', lw=2.5)
ax.plot(eigenvectors[:, 1], color='k', linestyle=':', lw=1.5)
plt.axhline(0, color='grey', lw=0.5)
colors = [sol['yellow'], sol['orange'], sol['red'], sol['magenta'], sol['violet'], sol['blue'], sol['cyan'],
          sol['green'], sol['base01'], sol['base1'], sol['base00'], sol['base0']]
tick_values = []
tick_1, tick_2 = 0, 0
for i, length in enumerate(var_size):
    tick_1, tick_2 = tick_2, tick_2 + length
    # plt.axvline(x=tick_2, color='red')
    plt.axvspan(xmin=tick_1, xmax=tick_2, facecolor=colors[i], alpha=0.5)
    tick_values.append(0.5*(tick_1 + tick_2))
ax.set_xticks(tick_values)
ax.set_xticklabels(['omega', 'div',
                    'T_adv_h', 'T_adv_v',
                    'r_adv_h', 'r_adv_v',
                    's_adv_h', 's_adv_v',
                    'dsdt', 'drdt',
                    'RH', 'shear_v'])
plt.legend(['Pattern 1 (22%)', 'Pattern 2 (12%)'])
plt.savefig(home+'/Desktop/eof_plot.pdf', bbox_inches='tight', transparent=True)
plt.show()
