from os.path import expanduser
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import metpy.calc as mpcalc
from Plotscripts.colors_solarized import sol
home = expanduser("~")
plt.rc('font', size=18)

# ls = xr.open_dataset('/Users/mret0001/Data/LargeScaleState/CPOL_large-scale_forcing_cape_cin_rh_shear.nc')
ls  = xr.open_dataset(home+
                      # '/Documents/Data/LargeScaleState/CPOL_large-scale_forcing_cape990hPa_cin990hPa_rh_shear_dcape.nc')
                      '/Documents/Data/LargeScaleState/'+
                      'CPOL_large-scale_forcing_cape990hPa_cin990hPa_rh_shear_dcape_wperc.nc')

                      # ROME is defined exactly at the LS time steps
metric = xr.open_dataarray(home + '/Documents/Data/Analysis/No_Boundary/AllSeasons/rom_km_avg6h_nanzero.nc')

# percentiles = metric.percentile
percentiles = abs(ls.percentile_w515 - 1)

bins = []
# i should be 2 at least
n_bins = 10
# the percentile-separating numbers
p_edges = np.linspace(0., 1., n_bins + 1)

# taking the metric-values into the bin-list is okay, later we only use the time information, not the values itself.
bins.append(metric.where(percentiles < p_edges[1], drop=True))
for i in range(1, n_bins-1):
    bins.append(metric.where((p_edges[i] <= percentiles) & (percentiles < p_edges[i + 1]), drop=True))
bins.append(metric.where(p_edges[-2] <= percentiles, drop=True))

l_plot_divers = True
if l_plot_divers:
    rome_top_w = bins[-1][bins[-1].notnull()]
    rome_top_decile = metric[metric.percentile > 0.9]
    rome_top_w_sorted = rome_top_w.sortby(rome_top_w)
    rome_top_decile_sorted = rome_top_decile.sortby(rome_top_decile)[-len(rome_top_w):]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rome_top_decile_sorted, rome_top_w_sorted, ls='', marker='+')
    ax.set_ylim((0, 600))
    ax.set_xlim((0, 600))
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    ax.set_ylabel('ROME in highest omega-decile')
    ax.set_xlabel('Highest ROME-decile')
    plt.show()

    metric_w = metric.where(ls.percentile_w515.notnull() & metric.notnull(), drop=True)
    metric_w_sorted = metric_w.sortby(abs(metric_w.percentile_w515 - 1))
    metric_sorted = metric.sortby(metric).where(metric.notnull(), drop=True)[-len(metric_w_sorted):]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(metric_sorted, metric_w_sorted, ls='', marker='+')
    ax.set_ylabel('ROME sorted by omega-percentiles')
    ax.set_xlabel('ROME sorted ascending')
    plt.show()

    rome = metric.where(metric.notnull(), drop=True)
    omega = ls.omega.sel(lev=515).where(ls.omega.sel(lev=515).notnull(), drop=True)
    plt.plot(rome.where(omega), omega.where(rome), ls='', marker='+')

    fig, ax = plt.subplots(figsize=(15, 5))
    rh500 = ls.RH.sel(lev=515).where(rome_top_w)
    ax.plot(range(len(rome_top_w)), rh500.sortby(rome_top_w), ls='', marker='*')
    ax.set_title('Highest decile of omega_515. Less than -6.6 hPa/hour.')
    ax.set_ylabel('Relative humidity at 515 hPa')
    ax.set_xlabel('Ranks of ROME ascending')
    plt.savefig(home+'/Desktop/omega_rh.pdf', bbox_inches='tight')

    rh = ls.RH[:5, :]
    rh[0, :] = ls.RH.sel(time=rome_top_w.time.values).mean(dim='time')
    rh[1, :] = ls.RH.sel(time=rome_top_w.time.values - np.timedelta64(6, 'h')).mean(dim='time')
    rh[2, :] = ls.RH.sel(time=rome_top_w.time.values - np.timedelta64(12, 'h')).mean(dim='time')
    rh[3, :] = ls.RH.sel(time=rome_top_w.time.values - np.timedelta64(18, 'h')).mean(dim='time')
    rh[4, :] = ls.RH.sel(time=rome_top_w.time.values - np.timedelta64(24, 'h')).mean(dim='time')

var_strings = [
# 'T'
# ,'r'
# 's'
# ,'u'
# ,'v'
'omega'
# ,'div'
# ,'T_adv_h'
# ,'T_adv_v'
# ,'r_adv_h'
# ,'r_adv_v'
# ,'s_adv_h'
# ,'s_adv_v'
# ,'dTdt'
# ,'dsdt'
# ,'drdt'
# ,'RH'
# ,'dwind_dz'
# 'wind_dir'
]

colours = ['yellow', 'orange', 'red', 'magenta', 'violet', 'blue', 'cyan', 'green', 'base01', 'base03']
for var in var_strings:

    l_percentage_profiles = False
    if l_percentage_profiles:
        ref_profile = ls[var].mean(dim='time')

    for i in range(n_bins):
        print(var)

        l_earlier_time = False
        if not l_earlier_time:
            ls_sub = ls.where(bins[i])
        else:
            earlytime = (bins[i].time - np.timedelta64(6, 'h')).values
            is_in_ls = [True if time in ls.time else False for time in earlytime]
            ls_sub = ls.sel(time=earlytime[is_in_ls])

        if var == 'wind_dir':
            u_mean = ls_sub['u'][:, :-1].mean(dim='time')
            v_mean = ls_sub['v'][:, :-1].mean(dim='time')
            direction    = xr.full_like(v_mean, np.nan)
            direction[:] = mpcalc.wind_direction(u_mean, v_mean)
            dir_diff = direction.sel(lev=slice(None, 965)).values - direction.sel(lev=slice(65, None)).values
            l_diff_gt_180 = abs(dir_diff) > 180.
            # once there was a change of direction .gt. 180, all values higher up in atmosphere are True also
            for j in range(len(l_diff_gt_180) - 1, 0, -1):
                l_diff_gt_180[j - 1] = l_diff_gt_180[j] ^ l_diff_gt_180[j - 1]
            direction.loc[40:965] = xr.where(l_diff_gt_180, (direction.loc[40:965]+360.) % 540., direction.loc[40:965])

            # manual adjustments
            if i==4: # and l_earlier_time:
                direction.loc[40:440] = direction.loc[40:440] + 360.
            if i==6 and not l_earlier_time:
                direction.loc[40:815] = direction.loc[40:815] -  360.
            ## if i==4 and not l_earlier_time:
            ##     direction.loc[465:540] = direction.loc[465:540] - 180

            speed = np.sqrt((u_mean**2 + v_mean**2))
            l_wind_speed = False
            if l_wind_speed:
                plot_var = speed
            else:
                plot_var = direction
            plt.plot(plot_var, ls_sub.lev[:-1], color=sol[colours[i]])
            if not l_wind_speed:
                tick_degrees = np.arange(-45, 540, 45)
                tick_labels = ['NW', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N', 'NE', 'E', 'SE']
                plt.axes().set_xticks(tick_degrees)
                plt.axes().set_xticklabels(tick_labels)
        else:
            profile_to_plot =  ls_sub[var][:, :-1].mean(dim='time')

            if l_percentage_profiles:
                profile_to_plot = (ls_sub[var][:, :-1].mean(dim='time') / ref_profile) - 1

            plt.plot(profile_to_plot, ls_sub.lev[:-1], color=sol[colours[i]])

    plt.gca().invert_yaxis()
    if l_earlier_time:
        plt.title('6 hours before prediction')
    else:
        plt.title('Same time as prediction')
    plt.ylabel('Pressure [hPa]')
    plt.xlabel(ls_sub[var].long_name+', ['+ls_sub[var].units+']')
    # plt.xlabel(ls_sub[var].long_name+' deviation from average, [K/K]')
    # plt.xlabel('Wind direction, [degrees]')
    plt.legend(['1. decile', '2. decile', '3. decile',
                '4. decile', '5. decile', '6. decile',
                '7. decile', '8. decile', '9. decile',
                '10. decile'], fontsize=9)

    plt.savefig('/Users/mret0001/Desktop/'+var+'.pdf', bbox_inches='tight', transparent=True)
    plt.close()
