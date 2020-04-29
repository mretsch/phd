from os.path import expanduser
import timeit
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from NeuralNet.backtracking import mlp_backtracking_maxnode, high_correct_predictions
from LargeScale.ls_at_metric import large_scale_at_metric_times, subselect_ls_vars
from basic_stats import into_pope_regimes, root_mean_square_error

home = expanduser("~")
start = timeit.default_timer()

l_testing = False
if l_testing:
    # ===== pen and paper ======
    X = np.array([[1, 5],
                  [2, 3],
                  [6, 4]])

    Y = np.array([[50],
                  [80],
                  [40]])

    k     =               X.transpose() @ X
    k_inv = np.linalg.inv(X.transpose() @ X)
    beta = k_inv @ X.transpose() @ Y
    pp_result = X @ beta
    # prints
    # array([[65.75757576],
    #        [39.03030303],
    #        [51.03030303]])

    # ===== verify with package ========
    model = sm.OLS(Y, X).fit()
    sm_result = model.predict(X)
    # prints
    # array([65.75757576, 39.03030303, 51.03030303])

# ===== the large scale state and ROME ======

ghome = home+'/Google Drive File Stream/My Drive'

# assemble the large scale dataset
ds_ls = xr.open_dataset(ghome+'/Data/LargeScale/CPOL_large-scale_forcing_cape990hPa_cin990hPa_rh_shear.nc')
metric = xr.open_dataarray(ghome+'/Data_Analysis/rom_km_avg6h_nanzero.nc')

ls_vars = ['omega',
           'T_adv_h',
           'r_adv_h',
           'dsdt',
           'drdt',
           'RH',
           'u',
           'v',
           # 'dwind_dz'
          ]
ls_times = 'same_and_earlier_time'
predictor, target, _ = large_scale_at_metric_times(ds_largescale=ds_ls,
                                                   timeseries=metric,
                                                   chosen_vars=ls_vars,
                                                   l_take_scalars=True,
                                                   large_scale_time=ls_times)

l_subselect = True
if l_subselect:
    levels = [115, 515, 990]
    predictor = subselect_ls_vars(predictor, levels=levels, large_scale_time=ls_times)


# def gen_correlation(array):
#     # the level dimension in large-scale data is much smaller than time dimension
#     assert array.shape[1] < array.shape[0]
#
#     for i in range(len(array.lev)):
#         timeseries_1 = array[:, i]
#         string_1 = timeseries_1['long_name'].values.item() + ', ' +str(int(timeseries_1['lev']))
#
#         for k in range(i + 1, len(array.lev)):
#             timeseries_2 = array[:, k]
#             string_2 = timeseries_2['long_name'].values.item() + ', ' + str(int(timeseries_2['lev']))
#             r = np.corrcoef(timeseries_1, timeseries_2)[0, 1]
#             yield string_1, string_2, r
#
# corr = list(gen_correlation(predictor))
#
# t0, t1, t2 = zip(*corr)
#
# corr_r = xr.DataArray(list(t2))
# corr_r['partner0'] = ('dim_0', list(t0))
# corr_r['partner1'] = ('dim_0', list(t1))
#
# high_corr = corr_r[abs(corr_r)>0.8]
#
# log = []
# for s in corr_r['partner0'].values:
#     log.append('H2O' in s.item())
# corr_select = corr_r[log]
#
# from org_metrics import gen_tuplelist
# tupes = list(gen_tuplelist(predictor.T))


l_load_model = True
if not l_load_model:

    mlreg_predictor = sm.add_constant(predictor.values)

    mlr_model = sm.OLS(target.values, mlreg_predictor).fit()
    mlr_predict = mlr_model.predict(mlreg_predictor)

    mlr_predicted = xr.DataArray(mlr_predict, coords={'time': predictor.time}, dims='time')

    mlr_summ = mlr_model.summary()
    with open(home+'/Desktop/mlr_coeff.csv', 'w') as csv_file:
        csv_file.write(mlr_summ.as_csv())
else:

    # mlr_coeff = pd.read_csv(csv_path, header=10, skipfooter=9)
    mlr_coeff = pd.read_csv(ghome+'/ROME_Models/NoCorrScalars/mlr_coeff.csv',
                            header=None, skiprows=12, skipfooter=7)
    mlr_coeff.rename({0: 'var', 1: 'coeff', 2: 'std_err', 3: 't', 4: 'P>|t|', 5: '[0.025', 6: '0.975]'},
                     axis='columns', inplace=True)
    mlr_coeff['var'] = predictor['long_name'].values

    n_lev = len(mlr_coeff['var'])

    plt.rc('font', size=33)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(48, 12))
    ax.plot(mlr_coeff['coeff'][:n_lev//2], marker='o', ms=19., ls='', color='k')

    # y-limits of plot with two times (0 and 6h before) as predictor
    # ax.set_ylim((-50, 50))
    ax.set_xlim((-1, 46))
    ax.axhline(y=0, color='r', lw=1.5)

    # label_list = [str(element0) + ', ' + element1 + ', ' + str(element2) for element0, element1, element2 in
    #               zip(range(n_lev//2), predictor['long_name'].values, predictor.lev.values)]
    label_list1 = [element1.replace('            ', '') + ', ' + str(int(element2)) + ' hPa ' for element1, element2 in
                  zip(predictor['long_name'][:24].values, predictor.lev.values)]
    label_list2 = [element1.replace('            ', '') + ' ' for element1, element2 in
                   zip(predictor['long_name'][24:].values, predictor.lev.values)]
    label_list = label_list1 + label_list2

    plt.xticks(list(range(n_lev//2)), label_list, rotation='vertical')#, fontsize=5)
    plt.yticks(rotation=90)
    # for tick in ax.get_xticklabels():
    #     tick.set_fontname("Andale Mono")
    # ax.axes.set_yticklabels(labels=predictor['long_name'].values, fontdict={'fontsize':8})
    # ax.tick_params(axis='both', which='major', labelsize=8)
    plt.savefig(home + '/Desktop/mlr_coeff.pdf', bbox_inches='tight', transparent=True)
