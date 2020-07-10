import matplotlib.pyplot as plt
import seaborn as sns


def contribution_whisker(input_percentages, levels, long_names,
                         ls_times, n_lev_total, n_profile_vars,l_eof_input=False,
                         xlim=25, bg_color='lavender'):

    if ls_times not in ['same_time', 'same_and_earlier_time', 'only_earlier_time', 'only_later_time', 'all_ls']:
        raise ValueError("String ls_times to select large-scale time steps does not match or is not provided.")

    plt.rc('font', size=19)

    if ls_times == 'same_and_earlier_time':
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 24))
        n_lev_onetime = n_lev_total // 2  # 11 #
    else:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 24))
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
            var_to_plot = var_to_plot_1 + var_to_plot_2

        plt.sca(ax)
        sns.boxplot(data=input_percentages[:, var_to_plot], orient='h', fliersize=1.,
                    # color='mistyrose', medianprops=dict(lw=3, color='black'))
                    color=bg_color, medianprops=dict(lw=3, color='black'))

        ax.set_xlim(-xlim, xlim)
        ax.axvline(x=0, color='r', lw=1.5)

        if l_eof_input:
            label_list = [integer + 1 for integer in var_to_plot]
        else:
            label_list1 = [element1.replace('            ', '') + ', ' + str(int(element2)) + ' hPa ' for
                           element1, element2 in
                           zip(long_names[var_to_plot_1].values, levels[var_to_plot_1])]
            label_list2 = [element1.replace('            ', '') + ' ' for element1, element2 in
                           zip(long_names[var_to_plot_2].values, levels)]
            label_list = label_list1 + label_list2

        ax.set_yticks(list(range(len(var_to_plot))))
        if i == 0:
            ax.set_yticklabels(label_list)
            plt.text(0.8, 0.85, 'Same\ntime', transform=ax.transAxes,
                     bbox={'edgecolor': 'k', 'facecolor': 'w', 'alpha': 0.5})
        else:
            ax.set_yticklabels([])
            plt.text(0.7, 0.85, '6 hours\nearlier', transform=ax.transAxes,
                     bbox={'edgecolor': 'k', 'facecolor': 'w', 'alpha': 0.5})

        ax.set_xlabel('Contribution to predicted value [%]')

    xlim_low = min(axes[0].get_xlim()[0], axes[1].get_xlim()[0])
    xlim_upp = max(axes[0].get_xlim()[1], axes[1].get_xlim()[1])
    for ax in axes:
        ax.set_xlim(xlim_low, xlim_upp)

    plt.subplots_adjust(wspace=0.05)

    return fig
