from os.path import expanduser
import timeit
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import keras.layers as klayers
import keras.models as kmodels
import keras.utils as kutils
import keras.callbacks as kcallbacks
from NeuralNet.backtracking import mlp_insight, high_correct_predictions
from LargeScale.ls_at_metric import large_scale_at_metric_times
from basic_stats import into_pope_regimes
import pandas as pd

home = expanduser("~")
start = timeit.default_timer()

l_loading_model = True

# assemble the large scale dataset
ghome = home+'/Google Drive File Stream/My Drive'
ds_ls  = xr.open_dataset(ghome+'/Data/LargeScale/CPOL_large-scale_forcing_cape990hPa_cin990hPa_rh_shear.nc')
metric = xr.open_dataarray(ghome+'/Data_Analysis/rom_km_avg6h.nc')

ls_vars = ['omega',
           # 'T_adv_h',
           # 'r_adv_h',
           # 'dsdt',
           # 'drdt',
           # 'RH',
           'u',
           'v',
           # 'dwind_dz'
           ]
predictor, target, _ = large_scale_at_metric_times(ds_largescale=ds_ls,
                                                   timeseries=metric,
                                                   chosen_vars=ls_vars,
                                                   l_take_same_time=False)

n_lev = len(predictor['lev'])

l_normalise_input = False
if l_normalise_input:
    predictor = (predictor - predictor.mean(dim='time')) / predictor.std(dim='time')
    # where std_dev=0., dividing led to NaN, set to 0. instead
    predictor = predictor.where(predictor.notnull(), other=0.)

if not l_loading_model:
    # building the model
    model = kmodels.Sequential()
    model.add(klayers.Dense(300, activation='relu', input_shape=(n_lev,)))
    model.add(klayers.Dense(300, activation='relu'))
    model.add(klayers.Dense(300, activation='relu'))
    model.add(klayers.Dense(1))

    # compiling the model
    model.compile(optimizer='adam', loss='mean_squared_error')  # , metrics=['accuracy'])

    # checkpoint
    filepath = home+'/Desktop/weights-improvement-{epoch:02d}-{val_loss:.2f}.h5'
    checkpoint = kcallbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only=False)
    callbacks_list = [checkpoint]

    # fit the model
    # predictor = predictor.transpose()
    model.fit(x=predictor, y=target, validation_split=0.2, epochs=10, batch_size=10, callbacks=callbacks_list)

    l_predict = True
    if l_predict:
        print('Predicting...')
        pred = []
        for i, entry in enumerate(predictor):
            pred.append(model.predict(np.array([entry])))
        pred_array = xr.DataArray(pred).squeeze()
        pred_array.coords['time'] = ('dim_0', target.time)
        predicted = pred_array.swap_dims({'dim_0': 'time'})

        fig, ax_host = plt.subplots(nrows=1, ncols=1, figsize=(48, 4))
        ax_host.plot(target[-1200:])
        ax_host.plot(predicted[-1200:])
        plt.legend(['target', 'predicted'])
        # ax_host.xaxis.set_major_locator(ticker.MultipleLocator(1))
        # ax_host.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
        # plt.grid(which='both')
        plt.savefig(home+'/Desktop/last1200.pdf', bbox_inches='tight')

else:
    # load a model
    model_path = ghome + '/Model_omega_uv/'
    model = kmodels.load_model(model_path + 'model.h5')

    input_length = len(predictor[0])
    w = model.get_weights()
    needed_input_size = len(w[0])

    assert needed_input_size == input_length, 'Provided input to model does not match needed input size.'

    l_high_values = True
    if l_high_values:
        predicted = xr.open_dataarray(model_path + 'predicted.nc')
        metric, predicted = high_correct_predictions(target=metric, predictions=predicted,
                                                     target_percentile=0.9, prediction_offset=0.3)
    else:
        predicted = xr.open_dataarray(model_path + 'predicted.nc')
        # only times that could be predicted (via large-scale set). Sample size: 26,000 -> 6,000
        metric = metric.where(predicted.time)

    ds_pope = into_pope_regimes(metric, l_upsample=True, l_all=True)
    p_regime = xr.full_like(ds_pope.var_all, np.nan)
    p_regime[:] = xr.where(ds_pope.var_p1.notnull(), 1, p_regime)
    p_regime[:] = xr.where(ds_pope.var_p2.notnull(), 2, p_regime)
    p_regime[:] = xr.where(ds_pope.var_p3.notnull(), 3, p_regime)
    p_regime[:] = xr.where(ds_pope.var_p4.notnull(), 4, p_regime)
    p_regime[:] = xr.where(ds_pope.var_p5.notnull(), 5, p_regime)

    for n_node in range(1, 2):
        maximum_nodes, first_conn = [], []

        for input in predictor.sel(time=metric.time.values):
            max_node, firstconn = mlp_insight(model, input, n_highest_node=n_node, return_firstconn=True)
            maximum_nodes.append(max_node)
            first_conn.append(firstconn)

        # ========= Plotting ==============================
        mn = xr.DataArray(maximum_nodes)
        first_node = mn[:, 0]

        # plt.hist(first_node, bins=np.arange(0, input_length+1))

        fig, ax_host = plt.subplots(nrows=1, ncols=1, figsize=(60, 4))
        ax_host.hist(first_node, bins=np.arange(0, input_length + 1))
        ax_host.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax_host.xaxis.set_minor_locator(ticker.MultipleLocator(2))
        # ax_host.set_xlim(1, None)
        # ax_host.set_yscale('log')
        plt.savefig(home+'/Desktop/'+str(n_node)+'_input_histo.pdf', transparent=True, bbox_inches='tight')

    # u-wind at 65hPa at the predicted time steps where it is most-contributing first-layer node
    # for NN Model_300x3_avg_wholeROME_bothtimes_reducedinput_uvwind
    # l_u65hPa = mn[:, 0] == 235
    # predictor.sel(time=metric.time[l_u65hPa.values].values, lev=65)[:, -10]

    plt.close()
    fig, axes = plt.subplots(nrows=5, ncols=1)
    # axes[0].plot(first_conn[-1])
    # fig.show()
    for i, conn in enumerate(first_conn):
        thetime = metric.time[i]
        axes[int(p_regime.sel(time=thetime))-1].plot(conn, alpha=0.1)
    fig.show()
    # plt.close()
    # plt.savefig(home+'/Desktop/firstconn.pdf')

stop = timeit.default_timer()
print('This script needed {} seconds.'.format(stop-start))
