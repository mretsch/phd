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

# rome = xr.open_dataarray(home + '/Documents/Data/Analysis/No_Boundary/AllSeasons/totalarea_km_avg6h.nc')
rome = xr.open_dataarray(home + '/Documents/Data/Analysis/No_Boundary/AllSeasons/rom_km_max6h_avg_pm20minutes.nc')
# area   = xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/o_area_km_max6h_avg_pm20minutes.nc')
# timedelta = xr.open_dataarray(home + '/Documents/Data/Analysis/No_Boundary/AllSeasons/timedelta_maxrome_largescale.nc')

# model_path = '/Documents/Data/NN_Models/ROME_Models/Kitchen_EOFprofiles/No_lowcloud_rstp2m/SecondModel/'
model_path = '/Desktop/'
predicted     = xr.open_dataarray(home + model_path + 'predicted.nc')
mlr_predicted = xr.open_dataarray(home + model_path + 'predicted.nc')

ls = xr.open_dataset(home + '/Documents/Data/LargeScaleState/' +
                     'CPOL_large-scale_forcing_cape990hPa_cin990hPa_rh_shear_dcape_NoDailyCycle.nc')

omega = ls['omega'].sel(lev=515).where(predicted, drop=True)

l_high_values = False
if l_high_values:
    rome, predicted = high_correct_predictions(target=rome, predictions=predicted,
                                               target_percentile=0.9, prediction_offset=0.3)
else:
    # only times that could be predicted (via large-scale set). Sample size: 26,000 -> 6,000
    rome = rome.where(predicted.time)
    # area = area.where(predicted.time)
    # timedelta = timedelta.where(predicted.time) /1e9

# rome = (rome - rome.mean()) / rome.std()
# rome = rome['percentile']

# the plot of predicted versus true ROME values with Pope regimes in the background
ds_pope = into_pope_regimes(rome, l_upsample=True, l_all=True)

p_regime = xr.full_like(ds_pope.var_all, np.nan)
p_regime[:] = xr.where(ds_pope.var_p1.notnull(), 1, p_regime)
p_regime[:] = xr.where(ds_pope.var_p2.notnull(), 2, p_regime)
p_regime[:] = xr.where(ds_pope.var_p3.notnull(), 3, p_regime)
p_regime[:] = xr.where(ds_pope.var_p4.notnull(), 4, p_regime)
p_regime[:] = xr.where(ds_pope.var_p5.notnull(), 5, p_regime)

special_time = np.array(['2002-03-16T00:00:00.000000000', '2003-01-19T12:00:00.000000000',
                         '2003-02-06T00:00:00.000000000', '2003-10-22T18:00:00.000000000',
                         '2003-11-05T18:00:00.000000000', '2003-12-04T06:00:00.000000000',
                         '2003-12-14T00:00:00.000000000', '2004-02-10T00:00:00.000000000',
                         '2004-11-24T18:00:00.000000000', '2004-11-30T00:00:00.000000000',
                         '2005-02-09T12:00:00.000000000', '2005-03-08T06:00:00.000000000',
                         '2005-11-13T18:00:00.000000000', '2005-12-12T06:00:00.000000000',
                         '2005-12-23T18:00:00.000000000', '2006-01-09T18:00:00.000000000',
                         '2006-02-09T12:00:00.000000000', '2006-11-13T12:00:00.000000000',
                         '2007-01-22T12:00:00.000000000', '2007-02-14T06:00:00.000000000',
                         '2007-03-11T06:00:00.000000000', '2007-03-29T18:00:00.000000000',
                         '2010-02-03T12:00:00.000000000', '2010-02-07T12:00:00.000000000',
                         '2010-02-09T12:00:00.000000000', '2010-12-03T06:00:00.000000000',
                         '2011-01-31T00:00:00.000000000', '2011-11-05T18:00:00.000000000',
                         '2011-12-15T06:00:00.000000000', '2012-02-14T06:00:00.000000000',
                         '2012-03-21T06:00:00.000000000', '2013-03-10T12:00:00.000000000',
                         '2013-03-11T12:00:00.000000000', '2014-01-02T06:00:00.000000000',
                         '2014-01-04T06:00:00.000000000', '2014-01-06T06:00:00.000000000',
                         '2014-01-09T06:00:00.000000000', '2014-01-09T12:00:00.000000000',
                         '2014-02-08T18:00:00.000000000', '2014-02-11T18:00:00.000000000',
                         '2014-03-04T12:00:00.000000000', '2014-03-09T18:00:00.000000000',
                         '2014-03-12T12:00:00.000000000', '2014-03-16T12:00:00.000000000',
                         '2014-03-24T12:00:00.000000000', '2014-03-25T06:00:00.000000000',
                         '2014-11-19T12:00:00.000000000', '2014-12-23T06:00:00.000000000',
                         '2014-12-28T12:00:00.000000000', '2014-12-28T18:00:00.000000000'],
                        dtype='datetime64[ns]')
# special_time = np.array(['2012-02-14T06:00:00.000000000'],
#                         dtype='datetime64[ns]')
# special_time = np.array(['2010-02-03T12:00:00.000000000'],
#                         dtype='datetime64[ns]')

plt.rc('font', size=40)

plot_length = 200

# plot_index = slice(None, 1200)
# plot_index = slice(-1200, -1000)
plot_index = slice(2400+260, 2400+460)

predicted_list = [predicted, mlr_predicted]
legend_both = ['NN', 'MLR']
# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(148, 4), sharex=True, sharey=True)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(36, 4), sharex=True, sharey=True)
colors = [sol['yellow'], sol['red'], sol['magenta'], sol['violet'], sol['cyan']]

for i, ax in enumerate([axes]):
    ax.plot(rome             [plot_index], color='k', lw=7)
    ax.plot(predicted_list[i][plot_index], color=sol['blue'], lw=5)
    # ax.plot(omega            [plot_index]*10, color='g'  , lw=1.5)
    # ax.plot(area            [plot_index], color='pink'  , lw=1.5)
    # ax.plot(rome.where(rome.time.isin(special_time)), ls='', marker='D', color='r')

    # ax.legend(['TCA', 'T$_\mathrm{NN}$'])
    ax.legend(['ROME', 'R$_\mathrm{NN}$'])
    # plt.title('reduced predictors with uv-wind. 90-percentile ROME with prediction within 30%.')
    # plt.title('reduced predictors with uv-wind. Input to NN normalised and given as standard-deviation.')

    # ax.set_ylim(0, None)
    ax.set_xlim(0, plot_length)

    # tick_1, tick_2 = -1.5, -0.5
    # for thistime in rome[plot_index].time.values: #metric_high[correct_pred].time.values:
    #     tick_1, tick_2 = tick_2, tick_2 + 1
        # ax.axvspan(xmin=tick_1, xmax=tick_2, facecolor=colors[int(p_regime.sel(time=thistime)) - 1], alpha=0.5)

    # ax.text(x=-50, y=300, s='(Pope 1, DE)', verticalalignment='top', color=colors[0], fontdict={'fontsize': 16})
    # ax.text(x=-50, y=250, s=' Pope 2, DW ', verticalalignment='top', color=colors[1], fontdict={'fontsize': 16})
    # ax.text(x=-50, y=200, s='(Pope 3,  E)', verticalalignment='top', color=colors[2], fontdict={'fontsize': 16})
    # ax.text(x=-50, y=150, s=' Pope 4, SW ', verticalalignment='top', color=colors[3], fontdict={'fontsize': 16})
    # ax.text(x=-50, y=100, s=' Pope 5, ME ', verticalalignment='top', color=colors[4], fontdict={'fontsize': 16})

    # ax.axhline(y=-180, color='grey')
    # ax.axhline(y=0, color='grey')
    # ax.axhline(y=180, color='grey')

    ax.yaxis.set_label_position("right")
    # ax.set_ylabel('Total conv. area [km$^2$]', labelpad=15)
    ax.set_ylabel('ROME [km$^2$]', labelpad=15)

    ax.set_xlabel('Time [6h intervals]')

    ax.tick_params(left=False, labelleft=False)
    ax.tick_params(right=False, labelright=True)

# plt.savefig(home+'/Desktop/first1200.pdf', bbox_inches='tight', transparent=True)
plt.savefig(home+'/Desktop/last1200.pdf', bbox_inches='tight', transparent=True)

l_scatter_plot = True
if l_scatter_plot:
    # make a 'scatter'-plot via a 2d-histogram
    from Plotscripts.plot_hist import histogram_2d
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    ax, h_2d = histogram_2d(rome, predicted, nbins=100, ax=ax,
                            y_label='R$_\mathrm{NN}$ [km$^2$]',
                            x_label='ROME [km$^2$]',
                            cbar_label='[%]')
    ax.set_aspect(abs((rome.max()-rome.min())/(predicted.max()-predicted.min())))
    fig.savefig(home+'/Desktop/predicted_scatter.pdf', bbox_inches='tight')

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
