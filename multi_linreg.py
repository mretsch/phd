from os.path import expanduser
import timeit
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import statsmodels.api as sm
from NeuralNet.backtracking import mlp_backtracking_maxnode, high_correct_predictions
from LargeScale.ls_at_metric import large_scale_at_metric_times, subselect_ls_vars
from basic_stats import into_pope_regimes, root_mean_square_error
from Plotscripts.colors_solarized import sol


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

    ghome = home+'/Google Drive File Stream/My Drive'

    # assemble the large scale dataset
    ds_ls = xr.open_dataset(home+'/Documents/Data/LargeScaleState/CPOL_large-scale_forcing_cape990hPa_cin990hPa_rh_shear_dcape.nc')
    metric = xr.open_dataarray(ghome+'/Data_Analysis/rom_km_avg6h_nanzero.nc')

    ls_vars = [#'omega',
               'T_adv_h',
               'r_adv_h',
               'dsdt',
               'drdt',
               'RH',
               'u',
               'v',
               'dwind_dz'
              ]
    long_names = [ds_ls[var].long_name for var in ls_vars]
    ls_times = 'same_and_earlier_time'
    predictor, target, _ = large_scale_at_metric_times(ds_largescale=ds_ls,
                                                       timeseries=metric,
                                                       chosen_vars=ls_vars,
                                                       l_take_scalars=True,
                                                       large_scale_time=ls_times)

    l_subselect = True
    if l_subselect:
        levels = [215, 515, 990]
        predictor = subselect_ls_vars(predictor, profiles=long_names, levels_in=levels, large_scale_time=ls_times)

    l_eof_input = False
    if l_eof_input:
        pcseries = xr.open_dataarray(home + '/Documents/Data/LargeScaleState/eof_pcseries_all.nc')
        eof_late = pcseries.sel(number=list(range(20)),
                                time=predictor.time).rename({'number': 'lev'}).T
        eof_early = pcseries.sel(number=list(range(20)),
                                 time=eof_late.time - np.timedelta64(6, 'h')).rename({'number': 'lev'}).T

        predictor = xr.DataArray(np.zeros((eof_late.shape[0], eof_late.shape[1] * 2)),
                                 coords=[eof_late['time'], np.concatenate([eof_late['lev'], eof_late['lev']], axis=0)],
                                 dims=['time', 'lev'])

        predictor[:, :] = np.concatenate([eof_early, eof_late], axis=1)
        predictor = predictor[target.notnull()]
        target = target[target.notnull()]

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
            log.append('H2O' in s.item())
        corr_select = corr_r[log]

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
        mlr_coeff_bias = pd.read_csv(home+'/Documents/Data/NN_Models/ROME_Models/Kitchen_WithoutFirst10/mlr_coeff.csv',
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
            input_percentages_list = []
            for model_input in predictor.sel(time=target.time):
                input_percentages_list.append(mlr_coeff['coeff'] * model_input /
                                              (np.dot(mlr_coeff['coeff'], model_input) + mlr_bias) * 100)

            input_percentages = xr.zeros_like(predictor.sel(time=target.time))
            input_percentages[:, :] = input_percentages_list

            plt.rc('font', size=25)

            if ls_times == 'same_and_earlier_time':
                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 24))
                n_lev_onetime = n_lev//2
            else:
                fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 24))
                axes = [axes]
                n_lev_onetime = n_lev

            for i, ax in enumerate(axes):

                if i == 0:
                    # var_to_plot_1 = [1, 11, 13, 16, 18, 20                        ]
                    # var_to_plot_2 = [                       25, 30, 34, 37, 41, 42]
                    var_to_plot_1 = list(range(27))
                    var_to_plot_2 = list(range(27, n_lev_onetime))
                else:
                    # var_to_plot_1 = [47, 57, 59, 62, 64, 66                        ]
                    # var_to_plot_2 = [                        71, 76, 80, 83, 87, 88]
                    var_to_plot_1 = list(range(n_lev//2     , n_lev//2 + 27))
                    var_to_plot_2 = list(range(n_lev//2 + 27, n_lev        ))
                var_to_plot = var_to_plot_1 + var_to_plot_2

                plt.sca(ax)
                sns.boxplot(data=input_percentages[:, var_to_plot], orient='h', fliersize=1.,
                            color='darksalmon', medianprops=dict(lw=3, color='dodgerblue'))

                ax.axvline(x=0, color='r', lw=1.5)

                label_list1 = [element1.replace('            ', '') + ', ' + str(int(element2)) + ' hPa ' for element1, element2 in
                               zip(predictor['long_name'][var_to_plot_1].values, predictor.lev[var_to_plot_1].values)]
                label_list2 = [element1.replace('            ', '') + ' ' for element1, element2 in
                               zip(predictor['long_name'][var_to_plot_2].values, predictor.lev.values)]
                label_list = label_list1 + label_list2

                ax.set_yticks(list(range(len(var_to_plot))))
                if i == 0:
                    ax.set_yticklabels(label_list)
                    plt.text(0.75, 0.95, 'Same\ntime', transform=ax.transAxes,
                             bbox={'edgecolor': 'k', 'facecolor': 'w', 'alpha': 0.5})
                else:
                    ax.set_yticklabels([])
                    plt.text(0.75, 0.95, '6 hours\nearlier', transform=ax.transAxes,
                             bbox={'edgecolor': 'k', 'facecolor': 'w', 'alpha': 0.5})

                ax.set_xlabel('Contribution to predicted value [%]')

            xlim_low = min(axes[0].get_xlim()[0], axes[1].get_xlim()[0])
            xlim_upp = max(axes[0].get_xlim()[1], axes[1].get_xlim()[1])
            for ax in axes:
                # ax.set_xlim(xlim_low, xlim_upp)
                ax.set_xlim(-75, 75)

            plt.subplots_adjust(wspace=0.05)

            plt.savefig(home + '/Desktop/mlr_whisker.pdf', bbox_inches='tight', transparent=True)

        plt.rc('font', size=28)

        n_profile_vars = 23 # 9 # 27 #
        if ls_times == 'same_and_earlier_time':
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 24))
            n_lev_onetime =  n_lev//2 #11 #
        else:
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 24))
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
            # ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
            ax.grid(axis='x')

        plt.subplots_adjust(wspace=0.05)

        plt.savefig(home + '/Desktop/mlr_coeff.pdf', bbox_inches='tight', transparent=True)
