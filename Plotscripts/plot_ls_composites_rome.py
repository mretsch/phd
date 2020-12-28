from os.path import expanduser
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib_venn as plt_venn
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import metpy.calc as mpcalc
import seaborn as sns
from Plotscripts.colors_solarized import sol
from Plotscripts.plot_phase_space import return_phasespace_plot
home = expanduser("~")
plt.rc('font', size=18)

colours = ['yellow', 'orange', 'red', 'magenta', 'violet', 'blue', 'cyan', 'green', 'base01', 'base03']
colours = ['yellow', 'base01']
# colours = ['violet', 'magenta']

ls  = xr.open_dataset(home+'/Documents/Data/LargeScaleState/'+
                      # 'CPOL_large-scale_forcing_cape990hPa_cin990hPa_rh_shear_dcape_NoDailyCycle.nc')
                      'CPOL_large-scale_forcing_cape990hPa_cin990hPa_rh_shear_dcape.nc')

ls_noday  = xr.open_dataset(home+'/Documents/Data/LargeScaleState/'+
                      'CPOL_large-scale_forcing_cape990hPa_cin990hPa_rh_shear_dcape_NoDailyCycle.nc')

# ROME is defined exactly at the LS time steps
rome = xr.open_dataarray(home + '/Documents/Data/Analysis/No_Boundary/AllSeasons/rom_km_avg6h_nanzero.nc')
totalarea = xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/totalarea_km_avg6h.nc')

# Percentiles used for the decile-binning
percentile_rome      = rome.percentile
percentile_w515      = ls.omega.sel(lev=515).rank(dim='time', pct=True)
percentile_totalarea = totalarea.rank(dim='time', pct=True)

# What percentiles?
percentiles = percentile_rome # abs(percentile_w515 - 1) # percentile_totalarea #

bins = []
# should be 2 at least
n_bins = 2#10
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

l_plot_scalars = True
if l_plot_scalars:
    vars = [
        (ls['omega'].sel(lev=515)    , 'w' ,   'hPa/hour'),
        (ls['lw_net_toa'],             'OLR',  'W/m${{^2}}$'),
        (ls['PW'],                     'PW',   'cm'),
        (ls['r_srf'],                  'r_2m', 'g/kg'),
        (ls['u']    .sel(lev=990)    , 'u' ,   'm/s'),
        (ls['RH']   .sel(lev=990)*100, 'RH',   '%'),
        (ls['T_srf'],                  'T_2m', 'K'),
        (ls['p_srf_aver'],                  'p_sfc', 'hPa'),
        # (ls['RH']   .sel(lev=215)*100, 'RH',   '%'),
        # (ls['RH']   .sel(lev=515)*100, 'RH',   '%'),
        # (ls['s_srf'],                  's_2m', 'K'),
        ]
    vars_noday = [
        (ls_noday['omega'].sel(lev=515)    , 'w' ,   'hPa/hour'),
        (ls_noday['lw_net_toa'],             'OLR',  'W/m${{^2}}$'),
        (ls_noday['PW'],                     'PW',   'cm'),
        (ls_noday['r_srf'],                  'r_2m', 'g/kg'),
        (ls_noday['u']    .sel(lev=990)    , 'u' ,   'm/s'),
        (ls_noday['RH']   .sel(lev=990)*100, 'RH',   '%'),
        (ls_noday['T_srf'],                  'T_2m', 'K'),
        (ls_noday['p_srf_aver'],                  'p_sfc', 'hPa'),
        # (ls_noday['RH']   .sel(lev=215)*100, 'RH',   '%'),
        # (ls_noday['RH']   .sel(lev=515)*100, 'RH',   '%'),
        # (ls_noday['s_srf'],                  's_2m', 'K'),
    ]

    fig, axes = plt.subplots(ncols=1, nrows=len(vars), sharex=True, figsize=(6, len(vars)*3))

    for m, ((var, symbol, unit), ax) in enumerate(zip(vars, axes)):

        # for j in range(1):
        for j in [0, 1]:#, 2]:#[0, 5, 9]:#range(len(bins)):
        # for j, selecting_var in enumerate([l_rh_low, l_rh_high]):

            ref_profile = var.where(rome.notnull(), drop=True).mean(dim='time')
            # daily_cycle = var.where(rome.notnull(), drop=True).groupby(group='time.time').mean(dim='time')
            daily_cycle = var.groupby(group='time.time').mean(dim='time')
            # del daily_cycle['percentile']
            # ax.plot(daily_cycle, ls='--', color=sol[colours[j]])

            # allocate proper array
            n_timesteps = 5
            quantity = var[:2*n_timesteps+1]

            # fill array
            # basetime = rome_top_decile.time
            basetime = bins[j].time
            # basetime = rh500_sorted.where(selecting_var, drop=True).time

            # do it once for dataset with daily cycle and once without
            for k in range(2):

                times = basetime

                if k == 0:
                    (var, symbol, unit) = vars[m]
                else:
                    (var, symbol, unit) = vars_noday[m]

                quantity[n_timesteps] = var.sel(time=times.where(
                                              times.isin(ls.time), drop=True
                                          ).values).mean(dim='time')

                for i, hours in enumerate([6, 12, 18, 24, 30]):
                    # times before the high-ROME time -> '-'
                    times = basetime - np.timedelta64(hours, 'h')
                    quantity[n_timesteps-1-i] = var.sel(time=times.where(
                        times.isin(ls.time), drop=True
                    ).values).mean(dim='time')

                    # times after the high-ROME time -> '+'
                    times = basetime + np.timedelta64(hours, 'h')
                    quantity[n_timesteps+1+i] = var.sel(time=times.where(
                        times.isin(ls.time), drop=True
                    ).values).mean(dim='time')

                if k == 0:
                    ax.plot(quantity, lw=2.5, color=sol[colours[j]])
                    # keep data with daily cycle for next iter
                    quantity_day = quantity.copy(deep=True)
                else:
                    ax.plot(quantity_day - quantity, lw=2, ls='--', color=sol[colours[j]])

        ax.axvline(x=n_timesteps, color='grey', ls='--', lw=1, zorder=-100)
        # plt.axes().xaxis.set_major_locator(ticker.MultipleLocator(1))

        try:
            # ax.set_ylabel(f'$\Delta(${symbol}$_{{{str(int(quantity.lev.values))}}} , \Phi)$ [{unit}]')
            ax.set_ylabel(f'{symbol}$_{{{str(int(quantity.lev.values))}}}$ [{unit}]')
        except AttributeError:
            # ax.set_ylabel(f'$\Delta(${symbol}$, \Phi)$ [{unit}]')
            ax.set_ylabel(f'{symbol} [{unit}]')

        ax.axes.spines['top'].set_visible(False)
        l_yaxis_on_left = True
        if l_yaxis_on_left:
            ax.axes.spines['right'].set_visible(False)
        else:
            ax.axes.spines['left'].set_visible(False)
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()

    ax.set_xlabel('Time [h]')
    ax.set_xticklabels(['xxx', '-30', '-18', '-6',
                                '+6', '+18', '+30'])
    # plt.axes().set_xticklabels(['xxx', '-30', '-24', '-18', '-12', '-6',  't(ROME)',
    #                             '+6', '+12', '+18', '+24', '+30'])

    plt.sca(axes[0])
    plt.legend(['1. Decile (D1)', 'Diurnal cycle in D1',
                '5. Decile (D5)', 'Diurnal cycle in D5',
                '10. Decile (D10)', 'Diurnal cycle in D10',
               ], fontsize=8, loc='lower right', markerfirst=False)

    plt.subplots_adjust(hspace=0.13)

    # plt.savefig('/Users/mret0001/Desktop/'+var.long_name[:3]+'_afterbefore_ROME.pdf', bbox_inches='tight', transparent=True)
    plt.savefig('/Users/mret0001/Desktop/afterbefore_ROME.pdf', bbox_inches='tight', transparent=True)
    plt.close()
