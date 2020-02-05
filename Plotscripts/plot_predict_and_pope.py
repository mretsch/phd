from os.path import expanduser
home = expanduser("~")
import timeit
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from Plotscripts.colors_solarized import sol
from basic_stats import into_pope_regimes

start = timeit.default_timer()

metric = xr.open_dataarray(home+'/Data/Analysis/No_Boundary/AllSeasons/rom_km_avg6h.nc')
predicted = xr.open_dataarray(home+'/Data/NN_models/Model_300x3_avg_wholeROME_bothtimes_reducedinput/predicted.nc')

metric = metric.where(predicted.time)

# the plot of predicted versus true ROME values with Pope regimes in the background
ds_pope = into_pope_regimes(metric, l_upsample=True, l_all=True)
p_regime = xr.full_like(ds_pope.var_all, np.nan)
p_regime[:] = xr.where(ds_pope.var_p1.notnull(), 1, p_regime)
p_regime[:] = xr.where(ds_pope.var_p2.notnull(), 2, p_regime)
p_regime[:] = xr.where(ds_pope.var_p3.notnull(), 3, p_regime)
p_regime[:] = xr.where(ds_pope.var_p4.notnull(), 4, p_regime)
p_regime[:] = xr.where(ds_pope.var_p5.notnull(), 5, p_regime)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(48, 4))
ax.plot(metric[-1200:], color='white')
ax.plot(predicted[-1200:], color='black')
plt.legend(['target', 'predicted'])
plt.title('less input to NN')

colors = [sol['yellow'], sol['red'], sol['magenta'], sol['violet'], sol['cyan']]
# colors = [sol['violet'], sol['red'], sol['cyan'], sol['green'], sol['yellow']]

tick_1, tick_2 = -1.5, -0.5
for i in range (1200):
    tick_1, tick_2 = tick_2, tick_2 + 1
    plt.axvspan(xmin=tick_1, xmax=tick_2, facecolor=colors[int(p_regime[i]) - 1], alpha=0.5)
ax.text(x=1210, y=300, s='(Pope 1, DE)', verticalalignment='top', color=colors[0], fontdict={'fontsize': 16})
ax.text(x=1210, y=250, s=' Pope 2, DW ', verticalalignment='top', color=colors[1], fontdict={'fontsize': 16})
ax.text(x=1210, y=200, s='(Pope 3,  E)', verticalalignment='top', color=colors[2], fontdict={'fontsize': 16})
ax.text(x=1210, y=150, s=' Pope 4, SW ', verticalalignment='top', color=colors[3], fontdict={'fontsize': 16})
ax.text(x=1210, y=100, s=' Pope 5, ME ', verticalalignment='top', color=colors[4], fontdict={'fontsize': 16})

plt.savefig(home+'/Desktop/last1200.pdf', bbox_inches='tight')

stop = timeit.default_timer()
print('This script needed {} seconds.'.format(stop-start))
