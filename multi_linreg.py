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

    l_compute_correlations = False
    if l_compute_correlations:
        corr = list(gen_correlation(predictor))
        t0, t1, t2 = zip(*corr)
        corr_r = xr.DataArray(list(t2))
        corr_r['partner0'] = ('dim_0', list(t0))
        corr_r['partner1'] = ('dim_0', list(t1))
        high_corr = corr_r[abs(corr_r) > 0.8]
        log = []
        for s in corr_r['partner0'].values:
            log.append('CAPE' in s.item())
        corr_select = corr_r[log]

    l_load_model = False
    if not l_load_model:

        mlreg_predictor = sm.add_constant(predictor.values)

        mlr_model = sm.OLS(target.values, mlreg_predictor).fit()
        mlr_predict = mlr_model.predict(mlreg_predictor)

        mlr_predicted = xr.DataArray(mlr_predict, coords={'time': predictor.time}, dims='time')

        mlr_summ = mlr_model.summary()
        with open(home+'/Desktop/mlr_coeff.csv', 'w') as csv_file:
            csv_file.write(mlr_summ.as_csv())
    else:
        model_path = home + '/Documents/Data/NN_Models/ROME_Models/Kitchen_WithoutFirst10/'
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

        l_percentage_plots = False
        if l_percentage_plots:

            # predicted = xr.open_dataarray(model_path + 'mlr_predicted.nc')
            # take the predictions of NN instead of MLR, to subselect high NN-predictions (apples to apples)
            predicted = xr.open_dataarray(model_path + 'predicted.nc')

            l_high_values = True
            if l_high_values:
                _, predicted_high = high_correct_predictions(target=metric, predictions=predicted,
                                                             target_percentile=0.9, prediction_offset=0.3)
                # predicted = predicted_high

            input_percentages_list = []
            for model_input in predictor.sel(time=predicted.time):
                input_percentages_list.append(mlr_coeff['coeff'] * model_input /
                                              (np.dot(mlr_coeff['coeff'], model_input) + mlr_bias) * 100)

            input_percentages       = xr.zeros_like(predictor.sel(time=predicted.time))
            input_percentages[:, :] = input_percentages_list

            if l_high_values:
                input_percentages.coords['high_pred'] = (
                    'time', np.full_like(input_percentages[:, 0], False, dtype='bool'))
                input_percentages['high_pred'].loc[dict(time=predicted_high.time)] = True

            l_violins = True and l_high_values
            plot = contribution_whisker(input_percentages=input_percentages,
                                        levels=predictor.lev.values,
                                        long_names=predictor['long_name'],
                                        ls_times='same_and_earlier_time',
                                        n_lev_total=n_lev,
                                        n_profile_vars=27,
                                        xlim=100,
                                        bg_color='mistyrose',
                                        l_eof_input=l_eof_input,
                                        l_violins=l_violins,
                                        )

            plot.savefig(home + '/Desktop/mlr_whisker.pdf', bbox_inches='tight', transparent=True)

        else:
            plt.rc('font', size=28)

            n_profile_vars = 26 # 30 # 23 # 9 #
            if ls_times == 'same_and_earlier_time':
                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 29))
                n_lev_onetime =  n_lev//2 #11 #
            else:
                fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 29))
                axes = [axes]
                n_lev_onetime = n_lev

            for i, ax in enumerate(axes):

                if i == 0:
                    # var_to_plot_1 = [1, 15, 17, 18, 20, 26                    ] # profile variables
                    # var_to_plot_2 = [                       28, 34, 35, 44, 45] # scalars
                    var_to_plot_1 = list(range(n_profile_vars))
                    var_to_plot_2 = list(range(n_profile_vars, n_lev_onetime))
                    if l_eof_input:
                        var_to_plot = list(range(n_lev_onetime))
                else:
                    # var_to_plot_1 = [50, 64, 66, 67, 69, 75                    ]
                    # var_to_plot_2 = [                        77, 83, 84, 93, 94]
                    var_to_plot_1 = list(range(n_lev_onetime                 , n_lev_onetime + n_profile_vars))
                    var_to_plot_2 = list(range(n_lev_onetime + n_profile_vars, n_lev                         ))
                    if l_eof_input:
                        var_to_plot = list(range(n_lev_onetime, n_lev))
                if not l_eof_input:
                    var_to_plot = var_to_plot_1 + var_to_plot_2

                ax.plot(mlr_coeff['coeff'][var_to_plot], list(range(len(var_to_plot))), marker='p', ms=16., ls='', color='k')

                ax.axvline(x=0, color='r', lw=1.5)

                if l_eof_input:
                    label_list = [integer + 1 for integer in var_to_plot]
                else:
                    label_list1 = [element1.replace('            ', '') + ', ' + str(int(element2)) + ' hPa ' for element1, element2 in
                                  zip(predictor['long_name'][var_to_plot_1].values, predictor.lev[var_to_plot_1].values)]
                    label_list2 = [element1.replace('            ', '') + ' ' for element1, element2 in
                                   zip(predictor['long_name'][var_to_plot_2].values, predictor.lev.values)]
                    label_list = label_list1 + label_list2

                ax.set_yticks(list(range(len(var_to_plot))))
                if i == 0:
                    ax.set_yticklabels(label_list)
                    plt.text(0.75, 0.85, 'Same\ntime', transform=ax.transAxes,
                             bbox={'edgecolor': 'k', 'facecolor': 'w', 'alpha': 0.5})
                else:
                    ax.set_yticklabels([])
                    plt.text(0.75, 0.85, '6 hours\nearlier', transform=ax.transAxes,
                             bbox={'edgecolor': 'k', 'facecolor': 'w', 'alpha': 0.5})

                ax.set_xlabel('Coefficients for MLR [km$^2$]')

                ax.invert_yaxis()
                ax.set_ylim(n_lev_onetime-0.5, -0.5)

            xlim_low = min(axes[0].get_xlim()[0], axes[1].get_xlim()[0])
            xlim_upp = max(axes[0].get_xlim()[1], axes[1].get_xlim()[1])
            for ax in axes:
                ax.set_xlim(xlim_low, xlim_upp)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
                ax.grid(axis='x')

            plt.subplots_adjust(wspace=0.05)

            plt.savefig(home + '/Desktop/mlr_coeff.pdf', bbox_inches='tight', transparent=True)
