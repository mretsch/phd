from os.path import expanduser
home = expanduser("~")
import timeit
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from Plotscripts.colors_solarized import sol
from basic_stats import into_pope_regimes
from NeuralNet.backtracking import high_correct_predictions

start = timeit.default_timer()

ghome = home+'/Google Drive File Stream/My Drive'

rome = xr.open_dataarray(home + '/Documents/Data/Analysis/No_Boundary/AllSeasons/rom_km_avg6h_nanzero.nc')
# area   = xr.open_dataarray(home+'/Documents/Data/Analysis/o_area_avg6h_nanzero.nc') * 6.25
predicted = xr.open_dataarray(
    home + '/Documents/Data/NN_Models/ROME_Models/KitchenSink/predicted.nc')
mlr_predicted = xr.open_dataarray(
    home + '/Documents/Data/NN_Models/ROME_Models/KitchenSink/mlr_predicted.nc')

l_high_values = False
if l_high_values:
    rome, predicted = high_correct_predictions(target=rome, predictions=predicted,
                                               target_percentile=0.9, prediction_offset=0.3)
else:
    # only times that could be predicted (via large-scale set). Sample size: 26,000 -> 6,000
    rome = rome.where(predicted.time)
    # area = area.where(predicted.time)

# the plot of predicted versus true ROME values with Pope regimes in the background
ds_pope = into_pope_regimes(rome, l_upsample=True, l_all=True)

p_regime = xr.full_like(ds_pope.var_all, np.nan)
p_regime[:] = xr.where(ds_pope.var_p1.notnull(), 1, p_regime)
p_regime[:] = xr.where(ds_pope.var_p2.notnull(), 2, p_regime)
p_regime[:] = xr.where(ds_pope.var_p3.notnull(), 3, p_regime)
p_regime[:] = xr.where(ds_pope.var_p4.notnull(), 4, p_regime)
p_regime[:] = xr.where(ds_pope.var_p5.notnull(), 5, p_regime)

plt.rc('font', size=24)

n_last = 400
predicted_list = [mlr_predicted, predicted]
legend_both = ['MLR', 'NN']
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(48, 8), sharex=True, sharey=True)
for i, ax in enumerate(axes):
    ax.plot(rome             [-n_last:], color='white', lw=4.5  )
    # ax.plot(area             [-n_last:], color='red'  , lw=1.5)
    ax.plot(predicted_list[i][-n_last:], color='black', lw=3.5)
    # ax.plot(mlr_predicted[-1200:], color='black', lw=1.5  )
    # ax.plot(    predicted[-1200:], color='red')
    ax.legend(['ROME', 'Earlier & same time '+legend_both[i]])
    # plt.title('reduced predictors with uv-wind. 90-percentile ROME with prediction within 30%.')
    # plt.title('reduced predictors with uv-wind. Input to NN normalised and given as standard-deviation.')
    # ax.set_ylim(0, 442.8759794239834)
    ax.set_xlim(0, n_last)

    colors = [sol['yellow'], sol['red'], sol['magenta'], sol['violet'], sol['cyan']]
    # colors = [sol['violet'], sol['red'], sol['cyan'], sol['green'], sol['yellow']]

    tick_1, tick_2 = -1.5, -0.5
    for thistime in rome[-n_last:].time.values: #metric_high[correct_pred].time.values:
        tick_1, tick_2 = tick_2, tick_2 + 1
        ax.axvspan(xmin=tick_1, xmax=tick_2, facecolor=colors[int(p_regime.sel(time=thistime)) - 1], alpha=0.5)

    # ax.text(x=-50, y=300, s='(Pope 1, DE)', verticalalignment='top', color=colors[0], fontdict={'fontsize': 16})
    # ax.text(x=-50, y=250, s=' Pope 2, DW ', verticalalignment='top', color=colors[1], fontdict={'fontsize': 16})
    # ax.text(x=-50, y=200, s='(Pope 3,  E)', verticalalignment='top', color=colors[2], fontdict={'fontsize': 16})
    # ax.text(x=-50, y=150, s=' Pope 4, SW ', verticalalignment='top', color=colors[3], fontdict={'fontsize': 16})
    # ax.text(x=-50, y=100, s=' Pope 5, ME ', verticalalignment='top', color=colors[4], fontdict={'fontsize': 16})

# add a big axis, hide frame
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.xlabel('Time [6h intervals]')
plt.ylabel('6h-average ROME [km$^2$]', labelpad=15)
plt.savefig(home+'/Desktop/last1200.pdf', bbox_inches='tight', transparent=True)

l_area_plot = False
if l_area_plot:
    rome_area_diff = rome      - area
    pred_area_diff = predicted - area
    romediff_order = np.   sort(rome_area_diff)
    arg_order      = np.argsort(rome_area_diff)
    preddiff_order = pred_area_diff[arg_order.values]
    preddiff_roll = preddiff_order.rolling(time=30, center=True).mean()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 9))
    ax.plot(preddiff_order, color='lightgrey', lw=1)
    ax.plot(romediff_order, color=sol['cyan'], lw=2)
    ax.plot(preddiff_roll,  color=sol['violet'], lw=2)
    ax.legend(['Prediction - Area', 'ROME - Area', '30-wide avg(Prediction - Area)'])
    ax.set_xlim(0, len(romediff_order))
    ax.axhline(y=0, color=sol['red'], lw=1.5)
    plt.savefig(home+'/Desktop/a.pdf', bbox_inches='tight', transparent=True)

stop = timeit.default_timer()
print('This script needed {} seconds.'.format(stop-start))
