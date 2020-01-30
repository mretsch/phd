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
ds_ls = xr.open_dataset(home+'/Data/LargeScaleState/CPOL_large-scale_forcing_cape990hPa_cin990hPa_rh_shear.nc')
metric = xr.open_dataarray(home+'/Data/Analysis/No_Boundary/AllSeasons/rom_km_avg6h.nc')

# take only large ROME values and the according LS variables then in the subroutine
# metric = metric[metric.percentile > 0.95]

ls_vars = ['omega',
           'T_adv_h',
           'r_adv_h',
           'dsdt',
           'drdt',
           'RH'
           ]
predictor, _, var_size = large_scale_at_metric_times(ds_largescale=ds_ls,
                                                     timeseries=metric,
                                                     l_take_same_time=True)

nlev = len(predictor.lev)

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

# sorting eigenvalues with descending values
idx = np.argsort(eigenvalues)[::-1]

eigenvalues  = eigenvalues [   idx]
eigenvectors = eigenvectors[:, idx]

# how much each pattern accounts for of total variance
variance_perc =  eigenvalues / eigenvalues.sum()

# at the moment, all eigenvectors[:, i] are scaled such that each has the l2-norm of 1.
l_scale_vectors = False
if l_scale_vectors:
    norm_orig = np.linalg.norm(eigenvectors, axis=0)
    # now scale each vector such that its l2-norm equals sqrt(eigenvalue).
    eigenvalues[eigenvalues < 0.] = 0
    scale_factor = np.sqrt(eigenvalues) / norm_orig
    evec_scaled = scale_factor * eigenvectors

# add dimensions to vectors
evec = xr.DataArray(eigenvectors,
                    coords={'level': predictor.lev.values,
                            'number': list(range(nlev)),
                            'quantity': ('level', predictor.long_name.values)},
                    dims=['level', 'number'])

# try some time series
if evec.dims[0] == 'level':
    pc_1   = evec.sel(number=0).values @ data_norm.T.values / (nlev - 1)
    # compute the principal component time series for all eigenvectors
    pc_all = xr.DataArray(evec.transpose().values @ data_norm.T.values / (nlev - 1),
                          coords={'number': list(range(nlev)),
                          'time': predictor.time},
                          dims=['number', 'time'])

    # reconstruct the original data via the pc time series and the patterns (EOFs)
    pattern_0_back = pc_all.isel(time=0).values @ evec.transpose().values * (nlev - 1)
    plt.plot(data_norm.isel(time=0).values, color='k', lw=2, ls='--')
    plt.plot(pattern_0_back, color='r', lw=0.5)
    plt.legend(['original height profile data, time=0', 'Reconstructed height profile data, time=0'])
    plt.show()

# the plot, want colors in the background for each 'variable'
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 4))
ax.plot(evec.isel(number=0)     , color='k', linestyle='-', lw=2.5)
ax.plot(evec.isel(number=1) * -1, color='k', linestyle=':', lw=1.5)
ax.plot(evec.isel(number=2)     , color='r', linestyle=':', lw=1.5)
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
ls_vars.append('shear_v')
ax.set_xticklabels(ls_vars)
plt.legend(['Pattern 1 (15%)', 'Pattern 2 (9%)', 'Pattern 3 (6%)'])
plt.savefig(home+'/Desktop/eof_plot.pdf', bbox_inches='tight', transparent=True)
plt.show()
