from os.path import expanduser
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib_venn as plt_venn
import numpy as np
import metpy.calc as mpcalc
from Plotscripts.colors_solarized import sol
home = expanduser("~")
plt.rc('font', size=18)

ls  = xr.open_dataset(home+'/Documents/Data/LargeScaleState/'+
                      'CPOL_large-scale_forcing_cape990hPa_cin990hPa_rh_shear_dcape.nc')

# ROME is defined exactly at the LS time steps
rome = xr.open_dataarray(home + '/Documents/Data/Analysis/No_Boundary/AllSeasons/rom_km_avg6h_nanzero.nc')

# What percentiles?
percentile_rome = rome.percentile
percentile_w515 = ls.omega.sel(lev=515).rank(dim='time', pct=True)
# What percentiles?
percentiles = abs(percentile_w515 - 1)

bins = []
# should be 2 at least
n_bins = 10
# the percentile-separating numbers
p_edges = np.linspace(0., 1., n_bins + 1)

# always taking rome-values into the bin-list is okay, later we only use the time information, not the values itself.
bins.append(rome.where(percentiles < p_edges[1], drop=True))
for i in range(1, n_bins-1):
    bins.append(rome.where((p_edges[i] <= percentiles) & (percentiles < p_edges[i + 1]), drop=True))
bins.append(rome.where(p_edges[-2] <= percentiles, drop=True))

rome_top_w             = bins[-1][bins[-1].notnull()].where(ls.time, drop=True)
rome_top_decile        = rome[rome.percentile > 0.9]
rome_top_w_sorted      = rome_top_w.sortby(rome_top_w)
rome_top_decile_sorted = rome_top_decile.sortby(rome_top_decile)[-len(rome_top_w):]

omega_in_rome = np.array([t.time.values in rome_top_decile.time for t in rome_top_w])
# the same
# rome_in_omega = np.array([t.time.values in rome_top_w.time for t in rome_top_decile])
plt_venn.venn2(subsets=(len(rome_top_decile), len(rome_top_w), omega_in_rome.sum()),
               set_labels=('Top ROME decile', 'Top w decile'))

l_plot_divers = False
if l_plot_divers:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rome_top_decile_sorted, rome_top_w_sorted, ls='', marker='+')
    ax.set_ylim((0, 600))
    ax.set_xlim((0, 600))
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
    ax.set_ylabel('ROME in highest omega-decile')
    ax.set_xlabel('Highest ROME-decile')
    plt.show()

    rome_w = rome.where(ls.percentile_w515.notnull() & rome.notnull(), drop=True)
    rome_w_sorted = rome_w.sortby(abs(rome_w.percentile_w515 - 1))
    rome_sorted = rome.sortby(rome).where(rome.notnull(), drop=True)[-len(rome_w_sorted):]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rome_sorted, rome_w_sorted, ls='', marker='+')
    ax.set_ylabel('ROME sorted by omega-percentiles')
    ax.set_xlabel('ROME sorted ascending')
    plt.show()

    rome = rome.where(rome.notnull(), drop=True)
    omega = ls.omega.sel(lev=515).where(ls.omega.sel(lev=515).notnull(), drop=True)
    plt.plot(rome.where(omega), omega.where(rome), ls='', marker='+')

    fig, ax = plt.subplots(figsize=(15, 5))
    rh500 = ls.RH.sel(lev=515).where(rome_top_w)
    ax.plot(range(len(rome_top_w)), rh500.sortby(rome_top_w), ls='', marker='*')
    ax.set_title('Highest decile of omega_515. Less than -6.6 hPa/hour.')
    ax.set_ylabel('Relative humidity at 515 hPa')
    ax.set_xlabel('Ranks of ROME ascending')
    plt.savefig(home+'/Desktop/omega_rh.pdf', bbox_inches='tight')

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

for var in ['omega']:# var_strings:
    ref_profile = ls[var].where(rome.notnull(), drop=True)[:, :-1].mean(dim='time')
    daily_cycle = ls[var].where(rome.notnull(), drop=True)[:, :-1].groupby(group='time.time').mean(dim='time')
    del daily_cycle['percentile']

    quantity = ls[var][:5, :-1]
    quantity[0, :] = ls[var].sel(lev=slice(None, 990), time=rome_top_w.time.values).mean(dim='time')
    quantity[1, :] = ls[var].sel(lev=slice(None, 990), time=rome_top_w.time.values - np.timedelta64(6, 'h')).mean(dim='time')
    quantity[2, :] = ls[var].sel(lev=slice(None, 990), time=rome_top_w.time.values - np.timedelta64(12, 'h')).mean(dim='time')
    quantity[3, :] = ls[var].sel(lev=slice(None, 990), time=rome_top_w.time.values - np.timedelta64(18, 'h')).mean(dim='time')
    quantity[4, :] = ls[var].sel(lev=slice(None, 990), time=rome_top_w.time.values - np.timedelta64(24, 'h')).mean(dim='time')
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
