from os.path import expanduser
import timeit
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import statsmodels.api as sm
from NeuralNet.regre_net import input_output_for_mlp
from NeuralNet.backtracking import high_correct_predictions
from Plotscripts.colors_solarized import sol
from Plotscripts.plot_contribution_whisker import contribution_whisker


def gen_correlation(array):
    # the level dimension in large-scale data is much smaller than time dimension
    assert array.shape[1] < array.shape[0]

    for i in range(len(array.lev)):
        timeseries_1 = array[:, i]
        string_1 = timeseries_1['long_name'].values.item() + ', ' +str(int(timeseries_1['lev']))

        for k in range(i + 1, len(array.lev)):
            timeseries_2 = array[:, k]
            string_2 = timeseries_2['long_name'].values.item() + ', ' + str(int(timeseries_2['lev']))
            r = np.corrcoef(timeseries_1, timeseries_2)[0, 1]
            yield string_1, string_2, r


if __name__ == "__main__":

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

    largescale_times = 'same_time'
    l_profiles_as_eof = True
    predictor, target, metric, height_dim = input_output_for_mlp(ls_times=largescale_times,
                                                                 l_profiles_as_eof=l_profiles_as_eof,
                                                                 target='tca')

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
        model_path = home + '/Desktop/'
        mlr_coeff_bias = pd.read_csv(model_path+'mlr_coeff.csv',
                                     header=None, skiprows=11, skipfooter=7) # skipfooter=9) #
        mlr_bias = mlr_coeff_bias.iloc[0, 1]
        mlr_coeff = mlr_coeff_bias.drop(index=0)
        mlr_coeff.index = list(range(mlr_coeff.shape[0]))
        mlr_coeff.rename({0: 'var', 1: 'coeff', 2: 'std_err', 3: 't', 4: 'P>|t|', 5: '[0.025', 6: '0.975]'},
                         axis='columns', inplace=True)
        # mlr_coeff['var'] = predictor['long_name'].values

        n_lev = len(mlr_coeff['var'])

        # ===== plots ==========

        # take the predictions of NN instead of MLR, to subselect high NN-predictions (apples to apples)
        predicted = xr.open_dataarray(model_path + 'predicted.nc')

        l_high_values = True
        if l_high_values:
            _, predicted_high = high_correct_predictions(target=metric, predictions=predicted,
                                                         target_percentile=0.9, prediction_offset=0.3)
            # predicted = predicted_high

        input_attribution = xr.zeros_like(predictor.sel(time=predicted.time))
        l_input_positive = xr.full_like(predictor.sel(time=predicted.time), fill_value=False, dtype='bool')
        for i, model_input in enumerate(predictor.sel(time=predicted.time)):

            single_attribution = mlr_coeff['coeff'] * model_input

            input_attribution[i, :] = single_attribution

            l_input_positive[i, :] = (model_input > 0.).values

        positive_positive_ratio = xr.zeros_like(input_attribution[:2, :])
        for i, _ in enumerate(positive_positive_ratio[height_dim]):
            positive_positive_ratio[0, i] = (l_input_positive[:, i] & (input_attribution[:, i] > 0.)).sum() \
                                            / l_input_positive[:, i].sum()
            positive_positive_ratio[1, i] = ((l_input_positive[:, i] == False) & (
                    input_attribution[:, i] < 0.)).sum() \
                                            / (l_input_positive[:, i] == False).sum()

        if l_high_values:
            input_attribution.coords['high_pred'] = (
                'time', np.full_like(input_attribution[:, 0], False, dtype='bool'))
            input_attribution['high_pred'].loc[dict(time=predicted_high.time)] = True

        p75 = xr.DataArray([np.nanpercentile(series, 75) for series in input_attribution.T])
        p50 = xr.DataArray([np.nanpercentile(series, 50) for series in input_attribution.T])
        p25 = xr.DataArray([np.nanpercentile(series, 25) for series in input_attribution.T])
        spread = p75 - p25
        # exploiting that np.unique() also sorts ascendingly, but also returns the matching index, unlike np.sort()
        high_spread_vars = input_attribution[0, np.unique(spread, return_index=True)[1][-10:]].long_name
        p50_high = xr.DataArray([np.nanpercentile(series[input_attribution['high_pred']], 50)
                                 for series in input_attribution.T])

        l_sort_input_percentage = True
        if l_sort_input_percentage:
            # sort the first time step, which is first half of data. Reverse index because I want descending order.
            first_half_order = np.unique(spread[:(n_lev // 2)], return_index=True)[1][::-1]
            # first_half_order = np.unique(abs(p50[:(n_lev // 2)]), return_index=True)[1][::-1]

            # apply same order to second time step, which is in second half of data
            second_half_order = first_half_order + (n_lev // 2)
            sort_index = np.concatenate((first_half_order, second_half_order))

            if largescale_times == 'same_time':
                # sort_index = np.unique(spread[:(n_lev)], return_index=True)[1][::-1]
                # sort_index = np.unique(p50_high[:(n_lev)], return_index=True)[1][::-1]
                sort_index = np.unique(abs(p50_high[:(n_lev)]), return_index=True)[1][::-1]

            input_attribution = input_attribution[:, sort_index]

        # ===== Plots =====================

        l_violins = True \
                    and l_high_values
        plot = contribution_whisker(input_percentages=input_attribution,
                                    levels=predictor[height_dim].values[sort_index],
                                    long_names=predictor['symbol'][sort_index],
                                    ls_times='same_time',
                                    n_lev_total=n_lev,
                                    n_profile_vars=n_lev,  # 47,#5,#13,# 50, #30, #26, #9, #23, #
                                    xlim=150,
                                    bg_color='mistyrose',
                                    l_violins=l_violins,
                                    )

        plot.savefig(home + '/Desktop/nn_whisker.pdf', bbox_inches='tight', transparent=True)
