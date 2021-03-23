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
home = expanduser("~")
plt.rc('font', size=18)

colours = ['yellow', 'orange', 'red', 'magenta', 'violet', 'blue', 'cyan', 'green', 'base01', 'base03']

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
# rome = xr.open_dataarray(home + '/Documents/Data/Analysis/No_Boundary/AllSeasons/rom_km_max6h_avg_pm20minutes.nc')
rome = xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/totalarea_km_avg6h.nc')
totalarea = xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/totalarea_km_avg6h.nc')

# Percentiles used for the decile-binning
percentile_rome      = rome.percentile
percentile_w515      = ls.omega.sel(lev=515).rank(dim='time', pct=True)
percentile_totalarea = totalarea.rank(dim='time', pct=True)

# What percentiles?
percentiles = percentile_rome # abs(percentile_w515 - 1) # percentile_totalarea #

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

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[j:j + lv // 3], 16) for j in range(0, lv, lv // 3))


df = pd.DataFrame()

var = ls['PW']
dataseries = []
for i in range(10):
    times = bins[i].time

    dataseries.append( var.sel(time=times.where(
        times.isin(ls.time), drop=True
    ).values).to_pandas())

df = pd.DataFrame(dataseries).transpose()
# sns.boxplot(data=df)
data_list = [df[i][df[i].notnull()] for i in range(df.shape[1])]
fig, ax = plt.subplots(figsize=(4, 3))
ax.boxplot(data_list, medianprops={'lw': 2.5, 'color': sol['magenta']}, showmeans=True,
           meanprops={'mfc': sol['cyan'],
                      'marker': '*',
                      'mec': sol['cyan']},
           flierprops={'ms': 0.2})

plt.ylabel('PW [cm]')
plt.xlabel('TCA decile')
# plt.ylim(-10, 105)
# plt.show()
plt.savefig(home+'/Desktop/whisker_romedeciles.pdf', bbox_inches='tight', transparent=True)



