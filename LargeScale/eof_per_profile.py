from os.path import expanduser
home = expanduser("~")
import timeit
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from LargeScale.ls_at_metric import large_scale_at_metric_times
from Plotscripts.colors_solarized import sol

start = timeit.default_timer()

def add_variable_symbol_string(dataset):
    assert type(dataset) == xr.core.dataset.Dataset

    quantity_symbol = [
        'T',
        'r',
        'u',
        'v',
        'w',
        'div(U)',
        'adv(T)',
        'conv(T)',
        'adv(r)',
        'conv(r)',
        's',
        'adv(s)',
        'conv(s)',
        'dsdt',
        'dTdt',
        'drdt',
        'xxx',
        'xxx',
        'C',
        'P',
        'LH', #"'Q_h_sfc',
        'SH', #'Q_s_sfc',
        'p',
        'p_centre',
        'T_2m',
        'T_skin',
        'RH_2m',
        'Speed_10m',
        'u_10m',
        'v_10m',
        'rad_sfc',
        'OLR', #'LW_toa',
        'SW_toa',
        'SW_dn_toa',
        'c_low',
        'c_mid',
        'c_high',
        'c_total',
        'c_thick',
        'c_top',
        'LWP',
        'dTWPdt',
        'adv(TWP)',
        'E',
        'dsdt_col',
        'adv(s)_col',
        'Q_rad',
        'Q_lat',
        'w_sfc',
        'r_2m',
        's_2m',
        'PW',
        'LW_up_sfc',
        'LW_dn_sfc',
        'SW_up_sfc',
        'SW_dn_sfc',
        'RH',
        'dUdz',
        'CAPE',
        'CIN',
        'D-CAPE',
    ]

    for i, var in enumerate(dataset):
        dataset[var].attrs['symbol'] = quantity_symbol[i]

    return dataset


# assemble the large scale dataset
ds_ls = xr.open_dataset(home +
                        '/Documents/Data/LargeScaleState/' +
                        'CPOL_large-scale_forcing_cape990hPa_cin990hPa_rh_shear_dcape_noDailyCycle.nc')
metric = xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/rom_km_avg6h_nanzero.nc')

add_variable_symbol_string(ds_ls)

# take only large ROME values and the according LS variables then in the subroutine
# metric = metric[metric.percentile > 0.95]

ls_vars = [
    'omega',
    'u',
    'v',
    's',
    'RH',
    's_adv_h',
    'r_adv_h',
    'dsdt',
    'drdt',
    'dwind_dz'
    ]

evec_list = []
for i in range(len(ls_vars)):
    predictor, _, var_size = large_scale_at_metric_times(ds_largescale=ds_ls,
                                                         timeseries=metric,
                                                         l_normalise_input=False,
                                                         chosen_vars=[ls_vars[i]], # default is all vars
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

    # add dimensions to vectors
    evec = xr.DataArray(eigenvectors,
                        coords={'level': predictor.lev.values,
                                'number': list(range(nlev)),
                                'quantity': ('level', predictor.long_name.values)},
                        dims=['level', 'number'],
                        attrs={'Variance_per_EOF': variance_perc, 'variable': ls_vars[i]})
    evec_list.append(evec)

    # try some time series
    if evec.dims[0] == 'level':
        pc_1   = evec.sel(number=0).values @ data_norm.T.values / (nlev - 1)
        # compute the principal component time series for all eigenvectors
        pc_all = xr.DataArray(evec.transpose().values @ data_norm.T.values / (nlev - 1),
                              coords={'number': list(range(nlev)), 'time': predictor.time},
                              dims=['number', 'time'],
                              attrs={'Variance_per_EOF': variance_perc, 'variable': ls_vars[i]})

        # reconstruct the original data via the pc time series and the patterns (EOFs)
        pattern_0_back = pc_all.isel(time=0).values @ evec.transpose().values * (nlev - 1)
        plt.plot(data_norm.isel(time=0).values, color='k', lw=2, ls='--')
        plt.plot(pattern_0_back, color='r', lw=0.5)
        plt.title(f'{ls_vars[i]}')
        plt.legend(['original height profile data, time=0', 'Reconstructed height profile data, time=0'])
        plt.show()

    # evec.to_netcdf(home+f'/Desktop/Profiles_to_EOF/eof_{ls_vars[i]}.nc')
    # pc_all.to_netcdf(home+f'/Desktop/Profiles_to_EOF/pc_{ls_vars[i]}.nc')

w_eof = evec_list[0]
u_eof = evec_list[1]
shear_eof = evec_list[-1]
v_eof = evec_list[2]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 10))
ax.plot(w_eof[{'number':0}], w_eof.level, label='w eof-1')
ax.plot(u_eof[{'number':0}], w_eof.level, label='u eof-1')
ax.plot(shear_eof[{'number':1}], shear_eof.level, label='shear eof-2')
ax.plot(w_eof[{'number':2}], w_eof.level, label='w eof-3')
ax.plot(v_eof[{'number':0}], w_eof.level, label='v eof-1')
ax.invert_yaxis()
plt.legend()
plt.show()


def replace_profiles_by_PCSeries_in_dataset():

    # assemble the large scale dataset
    ds_ls = xr.open_dataset(home +
                            '/Documents/Data/LargeScaleState/' +
                            'CPOL_large-scale_forcing_cape990hPa_cin990hPa_rh_shear_dcape_noDailyCycle.nc')

    # find principal component files
    search_pattern = 'pc*nc'
    pc_path = Path(home) / 'Desktop' / 'Profiles_to_EOF'
    pc_files = sorted([str(f) for f in pc_path.rglob(f'{search_pattern}')])

    numbers_needed = []
    for file in (pc_files):
        pc_all = xr.open_dataarray(file)
        explained_variance = pc_all.attrs['Variance_per_EOF']
        numbers_needed.append((explained_variance.cumsum() < 0.9).sum() + 1)

    # order such that variable with most needed EOF sets the dimension size of 'number' first, i.e. is largest
    order = np.array(numbers_needed).argsort()
    numbers_sorted = [numbers_needed[i] for i in order[::-1]]
    files_sorted = [pc_files[i] for i in order[::-1]]

    for file, n_needed in zip(files_sorted, numbers_sorted):
        pc_all = xr.open_dataarray(file)
        variable = pc_all.attrs['variable']

        pc_relevant = pc_all.sel(number=slice(None, n_needed - 1))
        # pc_relevant = pc_relevant.rename({'number': 'lev'})
        pc_relevant.attrs = ds_ls[variable].attrs
        pc_relevant.attrs['number_above_90percent'] = n_needed

        ds_ls[variable] = pc_relevant.transpose()

        print(variable, n_needed)

    # ds_ls.to_netcdf(home + '/Desktop/ls_eof.nc')
    return ds_ls
