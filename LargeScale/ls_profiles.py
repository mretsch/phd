from os.path import expanduser
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import metpy.calc as mpcalc
from Plotscripts.colors_solarized import sol
home = expanduser("~")
plt.rc('font', size=18)

ls  = xr.open_dataset(home+
                      '/Documents/Data/LargeScaleState/CPOL_large-scale_forcing_cape990hPa_cin990hPa_rh_shear_dcape.nc')

                      # ROME is defined exactly at the LS time steps
rome = xr.open_dataarray(home + '/Documents/Data/Analysis/No_Boundary/AllSeasons/rom_km_avg6h_nanzero.nc')

percentiles = rome.percentile

bins = []
# i should be 2 at least
n_bins = 10
# the percentile-separating numbers
p_edges = np.linspace(0., 1., n_bins + 1)

# taking the metric-values into the bin-list is okay, later we only use the time information, not the values itself.
bins.append(rome.where(percentiles < p_edges[1], drop=True))
for i in range(1, n_bins-1):
    bins.append(rome.where((p_edges[i] <= percentiles) & (percentiles < p_edges[i + 1]), drop=True))
bins.append(rome.where(p_edges[-2] <= percentiles, drop=True))

var_strings = [
# 'T'
# ,'r'
# 's'
# ,'u'
# ,'v'
# 'omega'
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
'RH'
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
    plt.xlim((0.28, 0.82))

    plt.legend(['1. decile', '2. decile', '3. decile',
                '4. decile', '5. decile', '6. decile',
                '7. decile', '8. decile', '9. decile',
                '10. decile'], fontsize=9)

    plt.savefig('/Users/mret0001/Desktop/'+var+'.pdf', bbox_inches='tight', transparent=True)
    plt.close()
