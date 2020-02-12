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
predicted = xr.open_dataarray(
    home + '/Data/NN_models/Model_300x3_avg_wholeROME_bothtimes_reducedinput_uvwind/predicted.nc')

# only times that could be predicted (via large-scale set). Sample size: 26,000 -> 6,000
metric = metric.where(predicted.time)

l_high_values = True
if l_high_values:
    # only interested in high ROME values. Sample size: O(100)
    metric_high = metric[metric['percentile'] > 0.90]
    diff = predicted - metric_high
    off_percent = (abs(diff) / metric_high).values
    # allow x% of deviation from true value
    correct_pred = xr.where(abs(diff) < 0.3 * metric, True, False)
    predicted = predicted.sel(time=metric_high[correct_pred].time.values)
    metric    = metric   .sel(time=metric_high[correct_pred].time.values)

# the plot of predicted versus true ROME values with Pope regimes in the background
ds_pope = into_pope_regimes(metric, l_upsample=True, l_all=True)
p_regime = xr.full_like(ds_pope.var_all, np.nan)
p_regime[:] = xr.where(ds_pope.var_p1.notnull(), 1, p_regime)
p_regime[:] = xr.where(ds_pope.var_p2.notnull(), 2, p_regime)
p_regime[:] = xr.where(ds_pope.var_p3.notnull(), 3, p_regime)
p_regime[:] = xr.where(ds_pope.var_p4.notnull(), 4, p_regime)
p_regime[:] = xr.where(ds_pope.var_p5.notnull(), 5, p_regime)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(48, 4))
ax.plot(metric   [:], color='white')
ax.plot(predicted[:], color='black')
plt.legend(['target', 'predicted'])
plt.title('reduced predictors with uv-wind. 90-percentile ROME with prediction within 30%.')
plt.ylim(0, None)

colors = [sol['yellow'], sol['red'], sol['magenta'], sol['violet'], sol['cyan']]
# colors = [sol['violet'], sol['red'], sol['cyan'], sol['green'], sol['yellow']]

tick_1, tick_2 = -1.5, -0.5
for thistime in metric_high[correct_pred].time.values:
    tick_1, tick_2 = tick_2, tick_2 + 1
    plt.axvspan(xmin=tick_1, xmax=tick_2, facecolor=colors[int(p_regime.sel(time=thistime)) - 1], alpha=0.5)
ax.text(x=323, y=300, s='(Pope 1, DE)', verticalalignment='top', color=colors[0], fontdict={'fontsize': 16})
ax.text(x=323, y=250, s=' Pope 2, DW ', verticalalignment='top', color=colors[1], fontdict={'fontsize': 16})
ax.text(x=323, y=200, s='(Pope 3,  E)', verticalalignment='top', color=colors[2], fontdict={'fontsize': 16})
ax.text(x=323, y=150, s=' Pope 4, SW ', verticalalignment='top', color=colors[3], fontdict={'fontsize': 16})
ax.text(x=323, y=100, s=' Pope 5, ME ', verticalalignment='top', color=colors[4], fontdict={'fontsize': 16})

plt.savefig(home+'/Desktop/last1200.pdf', bbox_inches='tight')

stop = timeit.default_timer()
print('This script needed {} seconds.'.format(stop-start))
