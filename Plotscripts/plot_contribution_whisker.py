import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def contribution_whisker(input_percentages, levels, long_names,
                         ls_times, n_lev_total, n_profile_vars,l_eof_input=False,
                         xlim=25, bg_color='lavender', l_violins=False):

    if ls_times not in ['same_time', 'same_and_earlier_time', 'only_earlier_time', 'only_later_time', 'all_ls']:
        raise ValueError("String ls_times to select large-scale time steps does not match or is not provided.")

    dim1, dim2 = input_percentages.dims
    if dim1 != 'time':
        height_dim = dim1
    else:
        height_dim = dim2

    plt.rc('font', size=19)

    if ls_times == 'same_and_earlier_time':
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 42 * 9/47.))  # *(12/94. + 10/94.)
        n_lev_onetime = n_lev_total // 2  # 11 #
    else:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(13.5, 82 * n_profile_vars/82 ))# * 7/47.  ))
        axes = [axes]
        n_lev_onetime = n_lev_total

    for i, ax in enumerate(axes):

        if i == 0:
            # var_to_plot_1 = [1, 15, 17, 18, 20, 26]  # profile variables
            # var_to_plot_2 = [28, 34, 35, 44, 45]  # scalars
            var_to_plot_1 = list(range(n_profile_vars))
            var_to_plot_2 = list(range(n_profile_vars, n_lev_onetime))
            if l_eof_input:
                var_to_plot = list(range(n_lev_onetime))
        else:
            # var_to_plot_1 = [50, 64, 66, 67, 69, 75]
            # var_to_plot_2 = [77, 83, 84, 93, 94]
            var_to_plot_1 = list(range(n_lev_onetime, n_lev_onetime + n_profile_vars))
            var_to_plot_2 = list(range(n_lev_onetime + n_profile_vars, n_lev_total))
            if l_eof_input:
                var_to_plot = list(range(n_lev_onetime, n_lev_total))
        if not l_eof_input:
            var_to_plot = var_to_plot_1 # + var_to_plot_2

        plt.sca(ax)

        if not l_violins:
            print("Plotting box-whisker plot.")
            sns.boxplot(data=input_percentages[:, var_to_plot], orient='h', fliersize=1.,
                        # color='mistyrose', medianprops=dict(lw=3, color='black'))
                        color=bg_color, medianprops=dict(lw=3, color='black'))
        else:
            print("Plotting violin plot.")
            # get some data as pandas dataframe
            df = input_percentages[:, var_to_plot].to_pandas()
            # rename columns
            if not l_eof_input:
                df.columns = np.char.add(input_percentages[:, var_to_plot].long_name,
                                         input_percentages[:, var_to_plot][height_dim].astype(str))
                # df.columns = input_percentages[:, var_to_plot].long_name
            else:
                df.columns = ['Pattern '+str(i) for i in range(1, len(var_to_plot) + 1)]
            # one series of data as type 'category'
            highpred_series = input_percentages[:, 0]['high_pred'].to_pandas().astype('category')
            # integrate the columns in a multi-index (representing time and variable names now). 2D-array -> 1D-array
            dfs = df.stack()
            # destroy the multi-index and place each part of the multi-index as a separate column
            dfsr = dfs.reset_index()
            # make the time column the index again
            dfsr.set_index('time', inplace=True)
            # add a category column. The length of the series will automatically adapted to the dataframe, because both
            # have time indices. Single time steps in the series will be copied to the multiple same time steps in the frame.
            # Manually this could be achieved by series.reindex_like(frame).
            dfsr['high_pred'] = highpred_series

            # copy the input data for high predictions
            df_highpred = dfsr.loc[dfsr['high_pred']]
            # set whole data to False, so that seaborn plots the distribution for the whole data set
            dfsr['high_pred'] = False
            # convert the category-type column to boolean-type column
            df_highpred.loc[:, 'high_pred'] = True
            # append the high-predictions to the whole data. I.e. the high-prediction input occurs twice,
            # once with False in the whole data and once with True in the appended data.
            df_to_plot = pd.concat([dfsr, df_highpred])

            # Plot. The dataframe has
            # column '0', which holds the contributing percentage,
            # column 'level_1', which holds the name of the input (quantity plus a unique integer) and
            # column 'high_pred', which identifies if the input contributed to a correct high prediction.
            sns.violinplot(x=0, y='level_1', hue='high_pred', data=df_to_plot, split=True, scale='width',
                           inner='quartile', palette='Set3', zorder=500)

        ax.set_xlim(-xlim, xlim)
        ax.axvline(x=0, color='lightgray', lw=2.5, zorder=-100)
        # ax.set_xticks([-60, -30, 0, 30, 60])
        ax.set_xticks([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])

        if l_eof_input:
            label_list = [integer + 1 for integer in var_to_plot]
        else:
            label_list = []
            for element1, element2 in zip(long_names.values, levels):
                # the profile-input to NN only has pressure down to 215hPa. And scalars got 'levels' below 100 earlier.
                if element2 < 5.:
                    label_list.append(
                        element1.replace('            ', '') + ', ' + str(int(element2)+1) + '. EOF'
                    )
                else:
                    label_list.append(
                        element1.replace('            ', '') + ' '
                    )
            # label_list1 = [element1.replace('            ', '') + ', ' + str(int(element2)) + ' hPa ' for
            #                element1, element2 in
            #                zip(long_names[var_to_plot_1].values, levels[var_to_plot_1])]
            # label_list2 = [element1.replace('            ', '') + ' ' for element1, element2 in
            #                zip(long_names[var_to_plot_2].values, levels)]
            # label_list = label_list1 + label_list2

        ax.set_yticks(list(range(len(var_to_plot))))
        if i == 0:
            ax.set_yticklabels(label_list[:len(var_to_plot)])
            # plt.text(0.8, 0.85, 'Same\ntime', transform=ax.transAxes,
            #          bbox={'edgecolor': 'k', 'facecolor': 'w', 'alpha': 0.5})
            # ax.axes.set_title('Time of R$_\mathrm{NN}$')
        else:
            ax.set_yticklabels([])
            # plt.text(0.7, 0.85, '6 hours\nearlier', transform=ax.transAxes,
            #          bbox={'edgecolor': 'k', 'facecolor': 'w', 'alpha': 0.5})
            ax.axes.set_title('6 hours before')

        ax.set_xlabel('Contribution to predicted value [%]', fontdict={'fontsize': 16})
        ax.set_ylabel('')

        legend = ax.legend()
        legend.remove()

    # xlim_low = min(axes[0].get_xlim()[0], axes[1].get_xlim()[0])
    # xlim_upp = max(axes[0].get_xlim()[1], axes[1].get_xlim()[1])
    # for ax in axes:
    #     ax.set_xlim(xlim_low, xlim_upp)

    plt.subplots_adjust(wspace=0.05)

    return fig
