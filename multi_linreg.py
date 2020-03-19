from os.path import expanduser
import timeit
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from NeuralNet.backtracking import mlp_backtracking_maxnode, high_correct_predictions
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

l_load_model = False
if not l_load_model:

    l_subselect = True
    if l_subselect:
        levels = [115, 515, 990]
        # select a few levels of a few variables which might be relevant to explain ROME
        var1  = predictor.where(predictor['long_name'] == 'vertical velocity            ',
                                drop=True).sel(lev=levels)
        var2  = predictor.where(predictor['long_name'] == 'Horizontal temperature Advection            ',
                                drop=True).sel(lev=levels)
        var3  = predictor.where(predictor['long_name'] == 'Horizontal r advection            ',
                                drop=True).sel(lev=levels)
        var4  = predictor.where(predictor['long_name'] == 'd(dry static energy)/dt            ',
                                drop=True).sel(lev=levels)
        var5  = predictor.where(predictor['long_name'] == 'd(water vapour mixing ratio)/dt            ',
                                drop=True).sel(lev=levels)
        var6  = predictor.where(predictor['long_name'] == 'Relative humidity            ',
                                drop=True).sel(lev=levels)
        var7  = predictor.where(predictor['long_name'] == 'Horizontal wind U component            ',
                                drop=True).sel(lev=levels)
        var8  = predictor.where(predictor['long_name'] == 'Horizontal wind V component            ',
                                drop=True).sel(lev=levels)
        var9  = predictor.where(predictor['long_name'] == 'Convective inhibition            ',
                                drop=True)
        var10 = predictor.where(predictor['long_name'] == 'Convective Available Potential Energy            ',
                                drop=True)
        var11 = predictor.where(predictor['long_name'] == 'Satellite-measured low cloud            ',
                                drop=True)
        var12 = predictor.where(predictor['long_name'] == 'Surface downwelling LW            ',
                                drop=True)
        var13 = predictor.where(predictor['long_name'] == '10m wind speed            ',
                                drop=True)
        var14 = predictor.where(predictor['long_name'] == '10m V component            ',
                                drop=True)
        var15 = predictor.where(predictor['long_name'] == '2m water vapour mixing ratio            ',
                                drop=True)
        var16 = predictor.where(predictor['long_name'] == 'TOA LW flux, upward positive            ',
                                drop=True)
        var17 = predictor.where(predictor['long_name'] == 'Surface sensible heat flux, upward positive            ',
                                drop=True)
        var18 = predictor.where(predictor['long_name'] == 'MWR-measured cloud liquid water path            ',
                                drop=True)
        var19 = predictor.where(predictor['long_name'] == 'vertical velocity, 6h earlier',
                                drop=True).sel(lev=levels)
        var20 = predictor.where(predictor['long_name'] == 'Horizontal temperature Advection, 6h earlier',
                                drop=True).sel(lev=levels)
        var21 = predictor.where(predictor['long_name'] == 'Horizontal r advection, 6h earlier',
                                drop=True).sel(lev=levels)
        var22 = predictor.where(predictor['long_name'] == 'd(dry static energy)/dt, 6h earlier',
                                drop=True).sel(lev=levels)
        var23 = predictor.where(predictor['long_name'] == 'd(water vapour mixing ratio)/dt, 6h earlier',
                                drop=True).sel(lev=levels)
        var24 = predictor.where(predictor['long_name'] == 'Relative humidity, 6h earlier',
                                drop=True).sel(lev=levels)
        var25 = predictor.where(predictor['long_name'] == 'Horizontal wind U component, 6h earlier',
                                drop=True).sel(lev=levels)
        var26 = predictor.where(predictor['long_name'] == 'Horizontal wind V component, 6h earlier',
                                drop=True).sel(lev=levels)
        var27 = predictor.where(predictor['long_name'] == 'Convective inhibition, 6h earlier',
                                drop=True)
        var28 = predictor.where(predictor['long_name'] == 'Convective Available Potential Energy, 6h earlier',
                                drop=True)
        var29 = predictor.where(predictor['long_name'] == 'Satellite-measured low cloud, 6h earlier',
                                drop=True)
        var30 = predictor.where(predictor['long_name'] == 'Surface downwelling LW, 6h earlier',
                                drop=True)
        var31 = predictor.where(predictor['long_name'] == '10m wind speed, 6h earlier',
                                drop=True)
        var32 = predictor.where(predictor['long_name'] == '10m V component, 6h earlier',
                                drop=True)
        var33 = predictor.where(predictor['long_name'] == '2m water vapour mixing ratio, 6h earlier',
                                drop=True)
        var34 = predictor.where(predictor['long_name'] == 'TOA LW flux, upward positive, 6h earlier',
                                drop=True)
        var35 = predictor.where(predictor['long_name'] == 'Surface sensible heat flux, upward positive, 6h earlier',
                                drop=True)
        var36 = predictor.where(predictor['long_name'] == 'MWR-measured cloud liquid water path, 6h earlier',
                                drop=True)
        mlreg_predictor = sm.add_constant(xr.concat([var1 , var2 , var3 , var4 ,
                                                     var5 , var6 , var7 , var8 ,
                                                     var9 , var10, var11, var12,
                                                     var13, var14, var15, var16,
                                                     var13, var14, var15, var16,
                                                     var17, var18, var19, var20,
                                                     var21, var22, var23, var24,
                                                     var25, var26, var27, var28,
                                                     var29, var30, var31, var32,
                                                     var33, var34, var35, var36,
                                                     ], dim='lev').values)
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
