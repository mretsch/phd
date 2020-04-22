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
predictor, target, _ = large_scale_at_metric_times(ds_largescale=ds_ls,
                                                   timeseries=metric,
                                                   chosen_vars=ls_vars,
                                                   l_take_scalars=True,
                                                   l_take_same_time=True)

l_subselect = True
if l_subselect:
    levels = [115, 515, 990]
    predictor = subselect_ls_vars(predictor, levels=levels)

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
    mlr_coeff = pd.read_csv(ghome+'/Model_all_incl_scalars_cape_3levels_normb4sub_nanzero/MLR_same_time/mlr_coeff.csv',
                            header=None, skiprows=12, skipfooter=7)
    mlr_coeff.rename({0: 'var', 1: 'coeff', 2: 'std_err', 3: 't', 4: 'P>|t|', 5: '[0.025', 6: '0.975]'},
                     axis='columns', inplace=True)
    mlr_coeff['var'] = predictor['long_name'].values

    n_lev = len(mlr_coeff['var'])
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(24, 4))
    ax.plot(mlr_coeff['coeff'], marker='p', ls='', color='k')
    # y-limits of plot with two times (0 and 6h before) as predictor
    # ax.set_ylim((-40.5769, 22.772100000000002))
    ax.axhline(y=0, color='r', lw=0.5)
    label_list = [str(element0) + ', ' + element1 + ', ' + str(element2) for element0, element1, element2 in
                  zip(range(n_lev), predictor['long_name'].values, predictor.lev.values)]
    plt.xticks(list(range(n_lev)), label_list, rotation='vertical', fontsize=5)
    # ax.axes.set_yticklabels(labels=predictor['long_name'].values, fontdict={'fontsize':8})
    # ax.tick_params(axis='both', which='major', labelsize=8)
    plt.savefig(home + '/Desktop/mlr_coeff.pdf', bbox_inches='tight', transparent=True)
