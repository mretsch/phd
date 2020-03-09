from os.path import expanduser
import timeit
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from NeuralNet.backtracking import mlp_backtrack_maxnode, high_correct_predictions
from LargeScale.ls_at_metric import large_scale_at_metric_times
from basic_stats import into_pope_regimes

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

l_load_model = False

# assemble the large scale dataset
ds_ls = xr.open_dataset(ghome+'/Data/LargeScale/CPOL_large-scale_forcing_cape990hPa_cin990hPa_rh_shear.nc')
metric = xr.open_dataarray(ghome+'/Data_Analysis/rom_km_avg6h.nc')

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
                                                   l_take_same_time=False)

l_normalise_input = True
if l_normalise_input:
    predictor = (predictor - predictor.mean(dim='time')) / predictor.std(dim='time')
    # where std_dev=0., dividing led to NaN, set to 0. instead
    predictor = predictor.where(predictor.notnull(), other=0.)

if not l_load_model:

    l_subselect = False
    if l_subselect:
        # select a few levels of a few variables which might be relevant to explain ROME
        var1 = predictor.where(predictor['long_name'] == 'Surface downwelling LW            ',
                               drop=True)  # .sel(lev=[115, 940])
        var2 = predictor.where(predictor['long_name'] == 'Surface downwelling LW, 6h earlier', drop=True)  # .sel(lev=[915])
        var3 = predictor.where(predictor['long_name'] == 'TOA LW flux, upward positive            ',
                               drop=True)  # .sel(lev=[915])
        mlreg_predictor = sm.add_constant(xr.concat([var1, var2, var3], dim='lev').values)
    else:
        mlreg_predictor = sm.add_constant(predictor.values)

    mlr_model = sm.OLS(target.values, mlreg_predictor).fit()
    mlr_predict = mlr_model.predict(mlreg_predictor)

    mlr_predicted = xr.DataArray(mlr_predict, coords={'time': predictor.time}, dims='time')

    mlr_summ = mlr_model.summary()
    with open(home+'/Desktop/mlr_coeff.csv', 'w') as csv_file:
        csv_file.write(mlr_summ.as_csv())
else:

    # mlr_coeff = pd.read_csv(csv_path, header=10, skipfooter=9)
    mlr_coeff = pd.read_csv(ghome+'/Model_all_incl_scalars_cape_norm/mlr_coeff.csv',
                            header=None, skiprows=12, skipfooter=9)
    mlr_coeff.rename({0: 'var', 1: 'coeff', 2: 'std_err', 3: 't', 4: 'P>|t|', 5: '[0.025', 6: '0.975]'},
                     axis='columns', inplace=True)
    mlr_coeff['var'] = predictor['long_name'].values

    relevant_vars = [
        24,
        27,
        155,
        190,
        233,
        313,
        316,
        320,
        369,
        424,
        439,
        476,
        511,
        555,
        557,
        635,
        638,
        640,
        642
    ]
    relevant_coeff = mlr_coeff.loc[relevant_vars, :]
