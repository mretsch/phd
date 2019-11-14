import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from Plotscripts.colors_solarized import sol

ls = xr.open_dataset('/Users/mret0001/Data/LargeScaleState/CPOL_large-scale_forcing_cape_cin_rh_shear.nc')

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
'dwind_dz'
]

colours = ['yellow', 'orange', 'red', 'magenta', 'violet', 'blue', 'cyan', 'green', 'base01', 'base03']
for var in var_strings:

    for i in range(n_bins):
        print(var)
        ls_sub = ls.where(metric_bins[i])
        plt.plot(ls_sub[var][:, 1:].mean(dim='time'), ls_sub.lev[1:], color=sol[colours[i]])

    plt.gca().invert_yaxis()
    plt.ylabel('Pressure [hPa]')
    # plt.xlabel(ls_low[var].long_name+', ['+ls_low[var].units+']')
    plt.xlabel(ls_sub[var].long_name+', ['+ls_sub[var].units+']')
    # plt.legend(['Low Tercile', 'Mid Tercile', 'High Tercile'])
    plt.legend(['1st decile', '2st decile', '3st decile', '4st decile', '5st decile', '6st decile', '7st decile', '8st decile', '9st decile', '10st decile'])

    plt.savefig('/Users/mret0001/Desktop/'+var+'.pdf', bbox_inches='tight')
    plt.close()
