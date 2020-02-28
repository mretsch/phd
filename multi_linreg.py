from os.path import expanduser
import timeit
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from NeuralNet.backtracking import mlp_insight, high_correct_predictions
from LargeScale.ls_at_metric import large_scale_at_metric_times
from basic_stats import into_pope_regimes

home = expanduser("~")
start = timeit.default_timer()

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

# assemble the large scale dataset
ghome = home+'/Google Drive File Stream/My Drive'
ds_ls = xr.open_dataset(ghome+'/Data/LargeScale/CPOL_large-scale_forcing_cape990hPa_cin990hPa_rh_shear.nc')
metric = xr.open_dataarray(ghome+'/Data_Analysis/rom_km_avg6h.nc')

ls_vars = ['omega',
           # 'T_adv_h',
           # 'r_adv_h',
           # 'dsdt',
           # 'drdt',
           # 'RH',
           'u',
           'v',
           # 'dwind_dz'
          ]
predictor, target, _ = large_scale_at_metric_times(ds_largescale=ds_ls,
                                                   timeseries=metric,
                                                   chosen_vars=ls_vars,
                                                   l_take_scalars=False,
                                                   l_take_same_time=False)

# select a few levels of a few variables which might be relevant to explain ROME
var1 = predictor.where(predictor['long_name'] == 'vertical velocity            ', drop=True).sel(lev=[115, 940])
var2 = predictor.where(predictor['long_name'] == 'vertical velocity, 6h earlier', drop=True).sel(lev=[915])

l_subselect = True
if l_subselect:
    mlreg_predictor = sm.add_constant(xr.concat([var1, var2], dim='lev').values)
else:
    mlreg_predictor = sm.add_constant(predictor.values)

mlr_model = sm.OLS(target.values, mlreg_predictor).fit()
mlr_predict = mlr_model.predict(mlreg_predictor)

mlr_result = xr.DataArray(mlr_predict, coords={'time': predictor.time}, dims='time')