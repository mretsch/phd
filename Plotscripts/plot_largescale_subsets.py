from os.path import expanduser
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib_venn as plt_venn
import numpy as np
import pandas as pd
import metpy.calc as mpcalc
import seaborn as sns
from Plotscripts.colors_solarized import sol
from Plotscripts.plot_phase_space import return_phasespace_plot
home = expanduser("~")
plt.rc('font', size=18)


def metrics_at_two_timesets(start_date_1, end_date_1, start_date_2, end_date_2, metric='1'):

    # rome
    metric_1 = xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/rom_kilometres.nc')
    # delta_prox
    metric_2 = metric_1 - \
               xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/rom_low_limit.nc') * 6.25
    # delta_size
    metric_3 = xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/rom_low_limit.nc') * 6.25 - \
               xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/o_area.nc') * 6.25
    # number of objects
    metric_4 = xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/o_number.nc')

    # fig, ax = plt.subplots(figsize=(15, 4))

    metric_highlow_all = []

    for dates in [(start_date_1, end_date_1), (start_date_2, end_date_2)]:

        list_all = []
        for start, end in zip(dates[0], dates[1]):
            times = slice(start, end)
            metric1_select = metric_1.sel(time=times)
            metric2_select = metric_2.sel(time=times)
            metric3_select = metric_3.sel(time=times)
            metric4_select = metric_4.sel(time=times)

            area_mean = metric1_select - metric2_select - metric3_select
            size_relative = metric_3 / area_mean
            prox_relative = metric_2 / area_mean

            if metric=='area':
                list_all.append(area_mean[metric1_select==metric1_select.max()])
            elif metric=='number':
                list_all.append(metric4_select[metric1_select==metric1_select.max()])
            else:
                list_all.append(metric3_select[metric1_select==metric1_select.max()])

            # ax.plot(range(len(size_relative)), size_relative, ls='--', color='b')
            # ax.plot(range(len(prox_relative)), prox_relative, ls='-', color='b')
            # ax.plot(range(len(prox_relative)), area_mean, ls='-', color='b')
            # plt.plot(range(len(prox_relative)), size_relative + prox_relative, ls='-', color='b')
            # plt.plot(range(len(prox_relative)), metric3_select, ls='', marker='x', lw=2, color='b', alpha=0.4)

        metric_highlow_all.append(xr.concat(list_all, dim='time'))

    return metric_highlow_all[0], metric_highlow_all[1]


if __name__ == '__main__':
    ls  = xr.open_dataset(home+'/Documents/Data/LargeScaleState/'+
                          'CPOL_large-scale_forcing_cape990hPa_cin990hPa_rh_shear_dcape.nc')

    # ROME is defined exactly at the LS time steps
    rome = xr.open_dataarray(home + '/Documents/Data/Analysis/No_Boundary/AllSeasons/rom_km_avg6h_nanzero.nc')
    totalarea = xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/totalarea_km_avg6h.nc')

    # Percentiles used for the decile-binning
    percentile_rome      = rome.percentile
    percentile_w515      = ls.omega.sel(lev=515).rank(dim='time', pct=True)
    percentile_totalarea = totalarea.rank(dim='time', pct=True)

    # What percentiles?
    percentiles = percentile_totalarea # abs(percentile_w515 - 1) #

    bins = []
    # should be 2 at least
    n_bins = 10
    # the percentile-separating numbers
    p_edges = np.linspace(0., 1., n_bins + 1)

    # taking rome-values into the bins is okay, sometimes we use the time information only, sometimes the values itself.
    # The binning itself is still done based on 'percentile' as assigned above.
    bins.append(rome.where(percentiles < p_edges[1], drop=True))
    for i in range(1, n_bins-1):
        bins.append(rome.where((p_edges[i] <= percentiles) & (percentiles < p_edges[i + 1]), drop=True))
    bins.append(rome.where(p_edges[-2] <= percentiles, drop=True))

    rome_top_w             = bins[-1][bins[-1].notnull()].where(ls.time, drop=True)
    rome_top_decile        = rome[rome.percentile > 0.9]

    # SCAI has low numbers for high degrees of organisation, ROME vice versa, this is important for the sorting
    l_use_scai = False
    l_sort_ascending = False if l_use_scai else True
    rome_top_w_sorted      = rome_top_w.     sortby(rome_top_w,      ascending=l_sort_ascending)
    rome_top_decile_sorted = rome_top_decile.sortby(rome_top_decile, ascending=l_sort_ascending)[-len(rome_top_w):]

    # Take large-scale variable based on a subset
    rh500 = ls.RH.sel(lev=515).where(rome_top_w)
    rh500_sorted =      rh500.sortby(rome_top_w, ascending=l_sort_ascending)

    # Choose how many of the last (highest) ROME values to take
    n = 220 # 700 #
    rh_at_highROME = ls.RH.sel(lev=515, time=rh500_sorted.time[-n:].time.values)
    rh_at_highROME_sorted = rh_at_highROME.sortby(rh_at_highROME)

    # logical array masking ROME values which are not at top or low end of the sorted array
    m = 40
    l_rh_high = rh500_sorted.time.isin(rh_at_highROME_sorted.time[-m:])
    l_rh_low  = rh500_sorted.time.isin(rh_at_highROME_sorted.time[:m ])

    l_subselect_low_org = False
    if l_subselect_low_org:
        # do the subselecting again, but for high RH but at the lowest ROMEs in the top w-decile
        rh_at_lowROME = ls.RH.sel(lev=515, time=rh500_sorted.time[:n].time.values)
        rh_at_lowROME_sorted = rh_at_lowROME.sortby(rh_at_lowROME)
        l_org_low = rh500_sorted.time.isin(rh_at_lowROME_sorted.time[-m:])

    # time slices for high and low RH values at high ROME values in highest w-decile
    start_highRH = rh500_sorted.where(l_rh_high, drop=True).time - np.timedelta64(170, 'm')
    stop_highRH  = rh500_sorted.where(l_rh_high, drop=True).time + np.timedelta64(3, 'h')
    start_lowRH  = rh500_sorted.where(l_rh_low,  drop=True).time - np.timedelta64(170, 'm')
    stop_lowRH   = rh500_sorted.where(l_rh_low,  drop=True).time + np.timedelta64(3, 'h')
    if l_subselect_low_org:
        start_lowOrg = rh500_sorted.where(l_org_low, drop=True).time - np.timedelta64(170, 'm')
        stop_lowOrg  = rh500_sorted.where(l_org_low, drop=True).time + np.timedelta64(3, 'h')

    ####### PLOTS ########

    l_plot_venn= False
    if l_plot_venn:
        omega_in_rome = rome_top_w.time.isin(rome_top_decile.time)
        plt_venn.venn2(subsets=(len(rome_top_decile), len(rome_top_w), omega_in_rome.sum()),
                       set_labels=('Lowest ROME decile', 'Lowest w decile'))
        plt.savefig(home+'/Desktop/omega_highest_venn.pdf', bbox_inches='tight')

    l_plot_rome_vs_rome = False
    if l_plot_rome_vs_rome:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(rome_top_decile_sorted, rome_top_w_sorted, ls='', marker='+')
        ax.set_ylim((0, 600))
        ax.set_xlim((0, 600))
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
        ax.set_ylabel('ROME in highest omega-decile')
        ax.set_xlabel('Highest ROME-decile')
        plt.show()
        plt.close()

    l_plot_scatter = True
    if l_plot_scatter:

        l_neither_subset = np.logical_not(np.logical_or(l_rh_high, l_rh_low))
        if l_subselect_low_org:
            l_neither_subset = np.logical_and(l_neither_subset, np.logical_not(l_org_low))

        fig, ax = plt.subplots(figsize=(15, 5))
        # ax.plot(range(len(rh500)), rh500_sorted.where(l_neither_subset),
        #         ls='', marker='x', mew=2, color=sol['cyan'])

        ax.plot(rome_top_w_sorted, rh500_sorted.where(l_neither_subset),
                ls='', marker='x', mew=2, color=sol['cyan'])

        # ax.set_title('Highest decile of ROME.')
        ax.set_title('Highest decile of total convective area.')
        # ax.set_title('Highest decile of omega_515. Less than -6.6 hPa/hour.')
        # ax.set_xlabel('Ascending ranks of ROME in highest decile of $\omega_{515}$ [1]')
        ax.set_xlabel('ROME [km$^2$]')
        ax.set_ylabel('Relative humidity at 515 hPa [1]')

        # again plot the non-masked ROME values in the previous figure-axes
        # ax.plot(range(len(rh500)), rh500_sorted.where(l_rh_high), ls='', marker='^', color=sol['magenta'])
        # ax.plot(range(len(rh500)), rh500_sorted.where(l_rh_low) , ls='', marker='s', color=sol['violet'])
        # if l_subselect_low_org:
        #     ax.plot(range(len(rh500)), rh500_sorted.where(l_org_low), ls='', marker='o', color=sol['yellow'])

        ax.plot(rome_top_w.where(rh_at_highROME_sorted[-m:]),
                rh500.     where(rh_at_highROME_sorted[-m:]), ls='', marker='^', color=sol['magenta'])
        ax.plot(rome_top_w.where(rh_at_highROME_sorted[:m ]),
                rh500.     where(rh_at_highROME_sorted[:m ]), ls='', marker='s', color=sol['violet'])
        if l_subselect_low_org:
            ax.plot(rome_top_w.where(rh_at_lowROME_sorted.time[-m:]),
                    rh500.     where(rh_at_lowROME_sorted.time[-m:]), ls='', marker='o', color=sol['yellow'])

        # ax.axvline(x=67, color='gray', ls='--', lw=1.5, zorder=0)
        # ax.axvline(x=199, color='gray', ls='--', lw=1.5, zorder=0)
        plt.savefig(home+'/Desktop/omega_rh.pdf', bbox_inches='tight')
        # plt.show()

        # Christian's plot of violins in each ROME-bin of the plot above
        l_christians_violins = False
        if l_christians_violins:
            # df = pd.DataFrame(
            #     [rome_top_w_sorted[:240].values, rome_top_w_sorted[240:480].values, rome_top_w_sorted[480:].values])
            dummy = rh500_sorted.copy(deep=True)
            subset = dummy.to_dataframe('quantity')
            cat = dummy.values
            cat[:240] = 1
            cat[240:480] = 2
            cat[480:] = 3
            subset['category'] = cat
            sns.violinplot(x=subset['category'], y=subset['quantity'], data=subset)

            rh = ls.RH.sel(lev=515).where(rome)
            rh = rh[rh.notnull()]

            ds_long = rh.to_pandas()
            longset = ds_long.to_frame()
            # cat_long = subset.reindex_like(longset)
            longset.columns = ['rh']
            longset['category'] = 0.
            longset.loc[[np.datetime64('2001-11-02T12:00:00'), np.datetime64('2001-11-02T18:00:00')]]
            idx = longset[longset.index.isin(dummy[   :240].time.values)].index
            longset.loc[idx, 'category'] = 1
            idx = longset[longset.index.isin(dummy[240:480].time.values)].index
            longset.loc[idx, 'category'] = 2
            idx = longset[longset.index.isin(dummy[480:   ].time.values)].index
            longset.loc[idx, 'category'] = 3
            sns.violinplot(x=longset['category'], y=longset['rh'], data=longset)

    l_plot_histo = False
    if l_plot_histo:
        set_a, set_b = metrics_at_two_timesets(start_highRH, stop_highRH, start_lowRH, stop_lowRH)
        plt.figure(figsize=(10, 5))
        sns.distplot(set_a, bins=np.arange(0, 600, 20), kde=False)#, kde_kws=dict(cut=0) bins=20,
        sns.distplot(set_b, bins=np.arange(0, 600, 20), kde=False)#, kde_kws=dict(cut=0) bins=20,
        plt.legend(['High RH', 'Low RH'])
        plt.ylabel('Count')
        plt.xlabel('Avg. object area in radar scene')
        # plt.xlim(0, 600)
        plt.title('Radar scenes with high ROME in top $\omega$-decile.')
        plt.savefig(home+'/Desktop/x_highROME_highW_diffRH.pdf', bbox_inches='tight')
        plt.close()

    l_plot_phasespace = False
    if l_plot_phasespace:
        # high_rh_xaxis, low_rh_xaxis = metrics_at_two_timesets(start_highRH, stop_highRH, start_lowRH, stop_lowRH,
        #                                                         metric='area')
        # high_rh_yaxis, low_rh_yaxis = metrics_at_two_timesets(start_highRH, stop_highRH, start_lowRH, stop_lowRH,
        #                                                         metric='number')
        #
        # if l_subselect_low_org:
        #     low_org_xaxis, _ = metrics_at_two_timesets(start_lowOrg, stop_lowOrg, start_lowOrg, stop_lowOrg,
        #                                                metric='area')
        #     low_org_yaxis, _ = metrics_at_two_timesets(start_lowOrg, stop_lowOrg, start_lowOrg, stop_lowOrg,
        #                                                metric='number')

        low_rh_xaxis  = ls.s. sel(lev=990, time=rh500_sorted.where(l_rh_low, drop=True).time.values)
        low_rh_yaxis  = ls.h2o_adv_col.sel(time=rh500_sorted.where(l_rh_low, drop=True).time.values)
        high_rh_xaxis = ls.s. sel(lev=990, time=rh500_sorted.where(l_rh_high, drop=True).time.values)
        high_rh_yaxis = ls.h2o_adv_col.sel(time=rh500_sorted.where(l_rh_high, drop=True).time.values)
        low_org_xaxis = ls.s. sel(lev=990, time=rh500_sorted.where(l_org_low, drop=True).time.values)
        low_org_yaxis = ls.h2o_adv_col.sel(time=rh500_sorted.where(l_org_low, drop=True).time.values)

        phasespace_plot = return_phasespace_plot()
        plt.plot(low_rh_xaxis,  low_rh_yaxis,  ls='', marker='s', color=sol['violet'], alpha=1.0)
        plt.plot(high_rh_xaxis, high_rh_yaxis, ls='', marker='^', color=sol['magenta'], alpha=1.0)
        plt.legend(['Low RH, high ROME', 'High RH, high ROME'])
        if l_subselect_low_org:
            plt.plot(low_org_xaxis, low_org_yaxis, ls='', marker='o', color=sol['yellow'], alpha=1.0)
            plt.legend(['Low RH, high ROME', 'High RH, high ROME', 'High RH, low ROME'], fontsize=14)

        save = True
        if save:
            plt.savefig(home+'/Desktop/phase_space_annotate.pdf', transparent=True, bbox_inches='tight')
            plt.show()
        else:
            plt.show()




    var_strings = [
        'T'
        ,'r'
        ,'s'
        ,'u'
        ,'v'
        ,'omega'
        ,'div'
        ,'T_adv_h'
        ,'T_adv_v'
        ,'r_adv_h'
        ,'r_adv_v'
        ,'s_adv_h'
        ,'s_adv_v'
        ,'dTdt'
        ,'dsdt'
        ,'drdt'
        ,'RH'
        ,'dwind_dz'
        # 'wind_dir'
    ]

    l_plot_profiles = False
    if l_plot_profiles:
        for var in var_strings: # ['omega']:#
            ref_profile = ls[var].where(rome.notnull(), drop=True)[:, :-1].mean(dim='time')
            daily_cycle = ls[var].where(rome.notnull(), drop=True)[:, :-1].groupby(group='time.time').mean(dim='time')
            del daily_cycle['percentile']

            # allocate proper array
            quantity = ls[var][:5, :-1]

            # fill array
            times = rome_top_decile.time
            quantity[0, :] = ls[var].sel(lev=slice(None, 990),
                                         time=times.where(
                                             times.isin(ls.time), drop=True
                                         ).values).mean(dim='time')

            for i, hours in enumerate([6, 12, 18, 24]):
                times = rome_top_decile.time - np.timedelta64(hours, 'h')
                quantity[i+1, :] = ls[var].sel(lev=slice(None, 990), time=times.where(
                                                 times.isin(ls.time), drop=True
                                             ).values).mean(dim='time')

            l_relative_profiles = False
            if l_relative_profiles:
                quantity -= quantity[0, :]
                daily_cycle -= daily_cycle.mean(dim='time')

            colormap = cm.BuGn
            # for i, q in enumerate(quantity[::-1]):
            #     plt.plot(q, q.lev, lw=2, color=colormap(i * 60 + 60))
            plt.plot(quantity[4], quantity.lev, lw=2, color=colormap(0 * 60 + 60))
            plt.plot(quantity[3], quantity.lev, lw=2, color=colormap(1 * 60 + 50))
            plt.plot(quantity[2], quantity.lev, lw=2, color=colormap(2 * 60 + 40))
            plt.plot(quantity[1], quantity.lev, lw=2, color=colormap(3 * 60 + 30))
            plt.plot(quantity[0], quantity.lev, lw=2, color='k'                  )
            # plt.plot(ref_profile, q.lev, color='k', ls='--')
            colormap = cm.Purples
            plt.plot(daily_cycle[0], daily_cycle.lev, lw=1, ls='-', color=colormap(1 * 60 + 60))
            plt.plot(daily_cycle[1], daily_cycle.lev, lw=1, ls='--', color=colormap(1 * 60 + 60))
            plt.plot(daily_cycle[2], daily_cycle.lev, lw=1, ls='-.', color=colormap(1 * 60 + 60))
            plt.plot(daily_cycle[3], daily_cycle.lev, lw=1, ls='dotted', color=colormap(1 * 60 + 60))
            plt.legend([
                '-24 h', '-18 h', '-12 h', '-6 h', 't=0',
                '9:30 h', '15:30 h', '21:30 h', '3:30 h', ])

            plt.gca().invert_yaxis()
            plt.ylabel('Pressure [hPa]')
            plt.xlabel(quantity.long_name+', ['+quantity.units+']')
            plt.savefig('/Users/mret0001/Desktop/P/'+var+'_before_highW_ROME.pdf', bbox_inches='tight', transparent=True)
            plt.close()
