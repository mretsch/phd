from os.path import expanduser
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import metpy.calc as mpcalc
from Plotscripts.colors_solarized import sol
home = expanduser("~")

# ls = xr.open_dataset('/Users/mret0001/Data/LargeScaleState/CPOL_large-scale_forcing_cape_cin_rh_shear.nc')
ls = xr.open_dataset(home+'/Data/LargeScaleState/CPOL_large-scale_forcing_cape990hPa_cin990hPa_rh_shear_dir.nc')

# ROME is defined exactly at the LS time steps
metric = xr.open_dataarray('/Users/mret0001/Data/Analysis/No_Boundary/AllSeasons/rom_km_avg6h.nc')

metric_bins = []
# i should be 2 at least
n_bins = 10
# the percentile-separating numbers
p_edges = np.linspace(0., 1., n_bins + 1)
metric_bins.append(metric.where(metric.percentile < p_edges[1], drop=True))
for i in range(1, n_bins-1):
    metric_bins.append(metric.where((p_edges[i] <= metric.percentile) & (metric.percentile < p_edges[i+1]), drop=True) )
metric_bins.append(metric.where(p_edges[-2] <= metric.percentile, drop=True))

var_strings = [
#'T'
# ,'r'
# ,'s'
# ,'u'
# ,'v'
# ,'omega'
# ,'div'
# ,'T_adv_h'
# ,'T_adv_v'
# ,'r_adv_h'
# ,'r_adv_v'
# ,'s_adv_h'
# ,'s_adv_v'
# ,'dsdt'
# ,'drdt'
# ,'RH'
# 'dwind_dz'
'wind_dir'
]

colours = ['yellow', 'orange', 'red', 'magenta', 'violet', 'blue', 'cyan', 'green', 'base01', 'base03']
for var in var_strings:

    for i in range(n_bins):
        print(var)
        ls_sub = ls.where(metric_bins[i])
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
            direction.loc[40:965] = xr.where(l_diff_gt_180, direction.loc[40:965]+360., direction.loc[40:965])

            speed = np.sqrt((u_mean**2 + v_mean**2))
            l_wind_speed = False
            if l_wind_speed:
                plot_var = speed
            else:
                plot_var = direction
            plt.plot(plot_var, ls_sub.lev[:-1], color=sol[colours[i]])
            if not l_wind_speed:
                tick_degrees = np.arange(0, 540, 45)
                tick_labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N', 'NE', 'E', 'SE']
                plt.axes().set_xticks(tick_degrees)
                plt.axes().set_xticklabels(tick_labels)
        else:
            plt.plot(ls_sub[var][:, :-1].mean(dim='time'), ls_sub.lev[:-1], color=sol[colours[i]])

    plt.gca().invert_yaxis()
    plt.ylabel('Pressure [hPa]')
    plt.xlabel(ls_sub[var].long_name+', ['+ls_sub[var].units+']')
    plt.legend(['1st decile', '2st decile', '3st decile', '4st decile', '5st decile', '6st decile', '7st decile', '8st decile', '9st decile', '10st decile'])

    plt.savefig('/Users/mret0001/Desktop/'+var+'.pdf', bbox_inches='tight')
    plt.close()
