from os.path import expanduser
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import metpy.calc as mpcalc
from Plotscripts.colors_solarized import sol
home = expanduser("~")
plt.rc('font', size=18)

ls  = xr.open_dataset(home+
                      '/Documents/Data/LargeScaleState/CPOL_large-scale_forcing_cape990hPa_cin990hPa_rh_shear_dcape_noDailyCycle.nc')

ls_day = xr.open_dataset(home + '/Documents/Data/LargeScaleState/' +
                         'CPOL_large-scale_forcing_cape990hPa_cin990hPa_rh_shear_dcape.nc')
# remove false data in precipitable water
ls_day['PW'].loc[{'time': slice(None, '2002-02-27T12')}] = np.nan
ls_day['PW'].loc[{'time': slice('2003-02-07T00', '2003-10-19T00')}] = np.nan
ls_day['PW'].loc[{'time': slice('2005-11-14T06', '2005-12-09T12')}] = np.nan
ls_day['PW'].loc[{'time': slice('2006-02-25T00', '2006-04-07T00')}] = np.nan
ls_day['PW'].loc[{'time': slice('2011-11-09T18', '2011-12-01T06')}] = np.nan
ls_day['PW'].loc[{'time': slice('2015-01-05T00', None)}] = np.nan
ls_day['LWP'].loc[{'time': slice(None, '2002-02-27T12')}] = np.nan
ls_day['LWP'].loc[{'time': slice('2003-02-07T00', '2003-10-19T00')}] = np.nan
ls_day['LWP'].loc[{'time': slice('2005-11-14T06', '2005-12-09T12')}] = np.nan
ls_day['LWP'].loc[{'time': slice('2006-02-25T00', '2006-04-07T00')}] = np.nan
ls_day['LWP'].loc[{'time': slice('2011-11-09T18', '2011-12-01T06')}] = np.nan
ls_day['LWP'].loc[{'time': slice('2015-01-05T00', None)}] = np.nan

xr.set_options(keep_attrs=True)
for v in ls.data_vars:
    ls[v] = ls[v] + ls_day[v].mean(dim='time')
xr.set_options(keep_attrs=False)

# ROME is defined exactly at the LS time steps
rome = xr.open_dataarray(home + '/Documents/Data/Analysis/No_Boundary/AllSeasons/totalarea_km_avg6h.nc')
# rome = xr.open_dataarray(home + '/Documents/Data/Analysis/No_Boundary/AllSeasons/rom_km_max6h_avg_pm20minutes.nc')

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
# 'u'
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
# 'dsdt'
'drdt'
# 'RH'
# ,'dwind_dz'
# 'wind_dir'
]

colours = ['yellow', 'orange', 'red', 'magenta', 'violet', 'blue', 'cyan', 'green', 'base01', 'base03']
# colours = ['orange','yellow',  'red', 'magenta','blue',  'violet', 'green', 'base01', 'base03', 'cyan']
for var in var_strings:

    l_percentage_profiles = False
    if l_percentage_profiles:
        ref_profile = ls[var].mean(dim='time')

    wind_vector_plot_ratio = (105.68/18.22)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(3, 3 * 0.25*wind_vector_plot_ratio))
    for i in [0, 4, 9]:#range(n_bins):#
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
            u_std  = ls_sub['u'][:, :-1].std (dim='time')
            v_std  = ls_sub['v'][:, :-1].std (dim='time')
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
            speed_std = np.sqrt((u_std**2 + v_std**2))

            l_wind_speed = False
            if l_wind_speed:
                plot_var = speed
            else:
                plot_var = direction

            # plt.plot(plot_var, ls_sub.lev[:-1], color=sol[colours[i]])
            for level in ls_sub.lev[:-1]:
                # the minus for dy is necessary when later we apply invert_yaxis() but want to retain arrow direction
                ax.arrow(x=0, y=level/20., dx=u_mean.sel(lev=level), dy=-v_mean.sel(lev=level),
                         width=0.08,
                         length_includes_head=True,
                         head_width=0.15,
                         overhang=0.2,
                         color=sol[colours[i]])

            # lower_band = speed - 0.5*speed_std
            # upper_band = speed + 0.5*speed_std
            # plt.fill_betweenx(y=ls_sub.lev[:-1], x1=lower_band, x2=upper_band, alpha=0.1)

            # if not l_wind_speed:
            #     tick_degrees = np.arange(-45, 540, 45)
            #     tick_labels = ['NW', 'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N', 'NE', 'E', 'SE']
            #     plt.axes().set_xticks(tick_degrees)
            #     plt.axes().set_xticklabels(tick_labels)

            ax.set_aspect('equal')
            ax.set_yticks(list(range(0, 60, 10)))
            ax.set_yticklabels(list(range(0, 1200, 200)))
            ax.set_xticks([-10, -5, 0])
            ax.set_xticklabels([10, 5, 0])
            # plt.ylabel('Pressure [hPa]')
            plt.xlabel('|$\\vec{{u}}$| [m/s]')
        else:
            data_to_plot = ls_sub[var][:, :-1]
            profile_to_plot =  data_to_plot.mean(dim='time')

            if l_percentage_profiles:
                profile_to_plot = data_to_plot.mean(dim='time') - ref_profile

            # lower_band = [np.nanpercentile(series, q=25) for series in data_to_plot.transpose()]
            # upper_band = [np.nanpercentile(series, q=75) for series in data_to_plot.transpose()]
            lower_band = [series.mean() - 0.5*series.std() for series in data_to_plot.transpose()]
            upper_band = [series.mean() + 0.5*series.std() for series in data_to_plot.transpose()]

            ax.plot(profile_to_plot, ls_sub.lev[:-1], color=sol[colours[i]], lw=2.5)
            ax.fill_betweenx(y=ls_sub.lev[:-1],
                             x1=lower_band,#- ref_profile[:39],
                             x2=upper_band,#- ref_profile[:39],
                             alpha=0.1, color=sol[colours[i]])

    plt.xlim((-0.05, 0.05))
    # plt.xlim((-1,1))
    # plt.xlim((-12.875084799528121, 3.117282193899155))
    # ax.xaxis.set_ticks((-10, -5, 0))
    ax.set_yticklabels([])

    plt.ylim(0, 1000)
    # plt.ylim(0, 51)

    # plt.ylabel('Pressure [hPa]')
    # plt.xlabel(ls_sub[var].long_name+', ['+ls_sub[var].units+']')
    # plt.xlabel('$\omega$, ['+ls_sub[var].units+']')
    plt.xlabel('dr/dt, ['+ls_sub[var].units+']')
    # plt.xlabel('$\Delta(s)$ from average profile, [K]')

    # plt.legend(['1. decile', '5. decile', '10. decile',], fontsize=10)
    # plt.legend(['1. decile', '2. decile', '3. decile',
    #             '4. decile', '5. decile', '6. decile',
    #             '7. decile', '8. decile', '9. decile',
    #             '10. decile'], fontsize=9)

    ax.axvline(x=0, lw=1.5, color='darkgrey', zorder=1)
    ax.invert_yaxis()

    plt.savefig('/Users/mret0001/Desktop/'+var+'.pdf', bbox_inches='tight', transparent=True)
    plt.close()
