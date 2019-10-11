import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from Plotscripts.colors_solarized import sol

ls = xr.open_dataset('/Volumes/GoogleDrive/My Drive/Data/LargeScale/CPOL_large-scale_forcing_cape_cin_rh.nc')


# ROME is defined exactly at the LS time steps
metric = xr.open_dataarray('/Users/mret0001/Data/Analysis/No_Boundary/AllSeasons/rom_kilometres_avg6h.nc')
# drop some labels that Ellie and Valentin found to be anomalous propagation in radar data
metric.drop(labels=np.arange('2003-03-14', '2003-03-15', dtype='datetime64[h]', step=6), dim='time')
metric.drop(labels=np.arange('2003-03-16', '2003-03-17', dtype='datetime64[h]', step=6), dim='time')
metric.drop(labels=np.arange('2003-10-28', '2003-10-29', dtype='datetime64[h]', step=6), dim='time')
metric.drop(labels=np.arange('2003-10-29', '2003-10-30', dtype='datetime64[h]', step=6), dim='time')
metric.drop(labels=np.arange('2003-11-24', '2003-11-25', dtype='datetime64[h]', step=6), dim='time')
metric.drop(labels=np.arange('2006-11-09', '2006-11-10', dtype='datetime64[h]', step=6), dim='time')
metric.drop(labels=np.arange('2006-11-10', '2006-11-11', dtype='datetime64[h]', step=6), dim='time')

lower_tercile = metric.where(metric.percentile < 0.33, drop=True)
middl_tercile = metric.where((0.33 <= metric.percentile) & (metric.percentile <= 0.66), drop=True)
upper_tercile = metric.where(0.66 < metric.percentile, drop=True)

ls_low = ls.where(lower_tercile)
ls_mid = ls.where(middl_tercile)
ls_upp = ls.where(upper_tercile)

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
,'dsdt'
,'drdt'
,'RH'
]


for var in var_strings:
    plt.plot(ls_low[var][:, 1:].mean(dim='time'), ls_low.lev[1:], color=sol['blue'])
    plt.plot(ls_mid[var][:, 1:].mean(dim='time'), ls_mid.lev[1:], color=sol['violet'])
    plt.plot(ls_upp[var][:, 1:].mean(dim='time'), ls_upp.lev[1:], color=sol['magenta'])
    plt.gca().invert_yaxis()
    plt.ylabel('Pressure [hPa]')
    plt.xlabel(ls_low[var].long_name+', ['+ls_low[var].units+']')
    plt.legend(['Low Tercile', 'Mid Tercile', 'High Tercile'])

    plt.savefig('/Users/mret0001/Desktop/ROME_LS_profiles/'+var+'.pdf', bbox_inches='tight')
    plt.close()
