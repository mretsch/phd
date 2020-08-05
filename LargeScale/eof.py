from os.path import expanduser
home = expanduser("~")
import timeit
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from LargeScale.ls_at_metric import large_scale_at_metric_times
from Plotscripts.colors_solarized import sol

start = timeit.default_timer()

# assemble the large scale dataset
ghome = home + '/Google Drive File Stream/My Drive'
ds_ls = xr.open_dataset(ghome + '/Data/LargeScale/CPOL_large-scale_forcing_cape990hPa_cin990hPa_rh_shear.nc')
metric = xr.open_dataarray(ghome+'/Data_Analysis/rom_km_avg6h_nanzero.nc')

# take only large ROME values and the according LS variables then in the subroutine
# metric = metric[metric.percentile > 0.95]

ls_vars = ['omega',
           'T_adv_h',
           'r_adv_h',
           'dsdt',
           'drdt',
           'RH',
           'u',
           'v',
           'dwind_dz'
           ]
predictor, _, var_size = large_scale_at_metric_times(ds_largescale=ds_ls,
                                                     timeseries=metric,
                                                     l_normalise_input=False,
                                                     chosen_vars=ls_vars, # default is all vars
                                                     large_scale_time='all_ls')

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
    eigenvectors = scale_factor * eigenvectors

# put n_add NaNs between each profile in the eigenvector
n_add = 4
add_length = (len(var_size) - 1) * n_add
evec_long = np.full(shape=(eigenvectors.shape[0] + add_length,
                           eigenvectors.shape[1]), fill_value=np.nan)
for i, size in enumerate(var_size):
    start_old  = sum(var_size[:i])
    end_old    = start_old + size
    start_long = start_old + i * n_add
    end_long   = start_long + size
    evec_long[start_long:end_long, :] = eigenvectors[start_old:end_old, :]

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
plt.rc('font', size=15)
fig, axes = plt.subplots(nrows=20, ncols=1, figsize=(15, 4*20))
ymin, ymax = 1e10, -1e10
for i, ax in enumerate(axes):
    # ax.plot(evec.isel(number=i)     , color='k', linestyle='-', lw=2.5)
    ax.plot(evec_long[:, i]     , color='k', linestyle='-', lw=2.5)
    ax.axhline(0, color='r', lw=0.5)
    # colors = [sol['yellow'], sol['blue'], sol['orange'], sol['violet'], sol['magenta'], sol['cyan'], sol['red'],
    #           sol['green'], sol['base01'], sol['base1'], sol['base00'], sol['base0'],
    #           sol['base01'], sol['base1'], sol['base00'], sol['base0']]
    colors = [sol['base01'], sol['base1']] * 7
    ax.set_xlim(0, evec_long.shape[0])
    tick_values = []
    for i, length in enumerate(var_size):
        tick_1 = sum(var_size[:i]) + i * n_add
        tick_2 = tick_1 + length
        # plt.axvline(x=tick_2, color='red')
        ax.axvspan(xmin=tick_1, xmax=tick_2, facecolor=colors[i], alpha=0.5)
        tick_values.append(0.5*(tick_1 + tick_2))
    ax.set_xticks(tick_values)
    ax.set_xticklabels(ls_vars)
    # ax.set_xticklabels(['omega', 'div', 'T_adv_h', 'T_adv_v', 'r_adv_h', 'r_adv_v',
    #                     's_adv_h', 's_adv_v', 'dsdt', 'drdt', 'RH', 'u', 'v', 'dwind_dz'])
    ax.set_xlabel('Profile quantities')
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ymin_ax, ymax_ax = ax.get_ylim()
    ymin   , ymax    = min(ymin, ymin_ax), max(ymax, ymax_ax)
    ax.set_ylabel('EOF pattern [standard deviation]')
for i, ax in enumerate(axes):
    ax.set_ylim(ymin, ymax)
    ax.legend(['Pattern ' + str(i + 1) + ' (' + str(round(variance_perc[i] * 100, 2)) + '%)'])
# plt.legend(['Pattern 1 (16%)', 'Pattern 2 (8%)', 'Pattern 3 (6%)'], fontsize=12)
plt.savefig(home+'/Desktop/eof_plot.pdf', bbox_inches='tight', transparent=True)
plt.show()
