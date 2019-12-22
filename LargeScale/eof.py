import timeit
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import keras.layers as klayers
import keras.models as kmodels
import keras.utils as kutils
import keras.callbacks as kcallbacks
from Plotscripts.plot_hist import histogram_2d
from NeuralNet.backtracking import mlp_insight
import pandas as pd

start = timeit.default_timer()

# ================================
# assemble the large scale dataset

take_same_time = True
take_only_predecessor_time = False
l_loading_model = False

ds_predictors = xr.open_dataset('/Users/mret0001/Data/LargeScaleState/CPOL_large-scale_forcing_cape_cin_rh_shear.nc')

c1 = xr.concat([
    # ds_predictors.T
    # , ds_predictors.r
    # , ds_predictors.s
    # , ds_predictors.u
    # , ds_predictors.v
    ds_predictors.omega   [:, :-1]  # ! bottom level has redundant information
    , ds_predictors.div     [:, :-1]  # ! bottom level has redundant information
    , ds_predictors.T_adv_h [:, :-1]  # ! bottom level has redundant information
    , ds_predictors.T_adv_v [:, :-1]  # ! bottom level has redundant information
    , ds_predictors.r_adv_h [:, :-1]  # ! bottom level has redundant information
    , ds_predictors.r_adv_v [:, :-1]  # ! bottom level has redundant information
    , ds_predictors.s_adv_h [:, :-1]  # ! bottom level has redundant information
    , ds_predictors.s_adv_v [:, :-1]  # ! bottom level has redundant information
    , ds_predictors.dsdt    [:, :-1]  # ! bottom level has redundant information
    , ds_predictors.drdt    [:, :-1]  # ! bottom level has redundant information
    , ds_predictors.dwind_dz[:, :-2]  # ! bottom levels filled with NaN
    , ds_predictors.RH      [:, :-1]  # ! bottom level has redundant information
], dim='lev')

c2 = xr.concat([
    ds_predictors.cin
    , ds_predictors.cld_low
    , ds_predictors.lw_dn_srf
    , ds_predictors.wspd_srf
    , ds_predictors.v_srf
    , ds_predictors.r_srf
    , ds_predictors.lw_net_toa
    , ds_predictors.SH
    , ds_predictors.LWP
])

# give c1 another coordinate to get back easily which values in concatenated array correspond to which variables
names_list =      [ds_predictors.omega.long_name    for _ in range(len(ds_predictors.omega   [:, :-1].lev))]
names_list.extend([ds_predictors.div.long_name      for _ in range(len(ds_predictors.div     [:, :-1].lev))])
names_list.extend([ds_predictors.T_adv_h.long_name  for _ in range(len(ds_predictors.T_adv_h [:, :-1].lev))])
names_list.extend([ds_predictors.T_adv_v.long_name  for _ in range(len(ds_predictors.T_adv_v [:, :-1].lev))])
names_list.extend([ds_predictors.r_adv_h.long_name  for _ in range(len(ds_predictors.r_adv_h [:, :-1].lev))])
names_list.extend([ds_predictors.r_adv_v.long_name  for _ in range(len(ds_predictors.r_adv_v [:, :-1].lev))])
names_list.extend([ds_predictors.s_adv_h.long_name  for _ in range(len(ds_predictors.s_adv_h [:, :-1].lev))])
names_list.extend([ds_predictors.s_adv_v.long_name  for _ in range(len(ds_predictors.s_adv_v [:, :-1].lev))])
names_list.extend([ds_predictors.dsdt.long_name     for _ in range(len(ds_predictors.dsdt    [:, :-1].lev))])
names_list.extend([ds_predictors.drdt.long_name     for _ in range(len(ds_predictors.drdt    [:, :-1].lev))])
names_list.extend([ds_predictors.dwind_dz.long_name for _ in range(len(ds_predictors.dwind_dz[:, :-2].lev))])
names_list.extend([ds_predictors.RH.long_name       for _ in range(len(ds_predictors.RH      [:, :-1].lev))])
c1.coords['long_name'] = ('lev', names_list)

c2_r = c2.rename({'concat_dims': 'lev'})
c2_r.coords['lev'] = np.arange(len(c2))
names_list = []
names_list.append(ds_predictors.cin.long_name)
names_list.append(ds_predictors.cld_low.long_name)
names_list.append(ds_predictors.lw_dn_srf.long_name)
names_list.append(ds_predictors.wspd_srf. long_name)
names_list.append(ds_predictors.v_srf.long_name)
names_list.append(ds_predictors.r_srf.long_name)
names_list.append(ds_predictors.lw_net_toa.long_name)
names_list.append(ds_predictors.SH.long_name)
names_list.append(ds_predictors.LWP.long_name)
c2_r.coords['long_name'] = ('lev', names_list)

# var = xr.concat([c1, c2_r], dim='lev')
var = c1
# var_itp = var# .resample(time='T9min').interpolate('linear')

# average = xr.open_dataarray('/Users/mret0001/Data/Analysis/No_Boundary/AllSeasons/rom_km_avg6h.nc')
# maximum = xr.open_dataarray('/Users/mret0001/Data/Analysis/No_Boundary/AllSeasons/rom_km_max6h.nc')
# metric = maximum - average
# metric = xr.open_dataarray('/Users/mret0001/Data/ROME_Samples/rom_avg6h_afterLS_85pct_5050sample.nc')
metric = xr.open_dataarray('/Users/mret0001/Data/Analysis/No_Boundary/AllSeasons/rom_km_avg6h.nc')

# large scale variables only where metric is defined
var_metric = var.where(metric.notnull(), drop=True)

# boolean for the large scale variables without any NaN anywhere
l_var_nonull = var_metric.notnull().all(dim='lev')

# large-scale state variables at same time as metric, or not
if take_same_time:
    predictor = var_metric[{'time': l_var_nonull}]
    target = metric.sel(time=predictor.time)

else:
    var_nonull = var_metric[l_var_nonull]
    var_nonull_6earlier = var_nonull.time - np.timedelta64(6, 'h')
    times = []
    for t in var_nonull_6earlier:
        try:
            _ = var_metric.sel(time=t)
            times.append(t.values)
        except KeyError:
            continue
    # var_sub.sel(time=[np.datetime64('2002-08-10T18'), np.datetime64('2002-08-08T12')])
    var_6earlier = var_metric.sel(time=times)
    var_6earlier_nonull = var_6earlier[var_6earlier.notnull().all(dim='lev')]

    if take_only_predecessor_time:
        # metric 6h later is necessarily a value, because back at times of var_metric, where metric is a number.
        target = metric.sel(time=(var_6earlier_nonull.time + np.timedelta64(6, 'h')).values)
        predictor = var.sel(time=target.time - np.timedelta64(6, 'h'))
    else:
        var_6later_nonull = var_nonull.sel(time=(var_6earlier_nonull.time + np.timedelta64(6, 'h')))
        # first 'create' the right array with the correct 'late' time steps
        var_both_times = xr.concat([var_6later_nonull, var_6later_nonull], dim='lev')
        half = int(len(var_both_times.lev) / 2)
        # fill one half with values from earlier time step
        var_both_times[:, half:] = var_6earlier_nonull.values
        var_both_times['long_name'][half:] = \
            [name.item() + ', 6h earlier' for name in var_both_times['long_name'][half:]]

        target = metric.sel(time=var_both_times.time.values)
        predictor = var_both_times

n_lev = len(predictor['lev'])

# ===================
# eof-analysis itself
# see here https://stackoverflow.com/questions/13224362/principal-component-analysis-pca-in-python#13224592

# First make the data deviations from the mean and standardise it
data = predictor - predictor.mean(dim='time')
data_std =   data / data.std(dim='time')

cov_matrix = np.cov(data_std, rowvar=True)

eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

idx = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]