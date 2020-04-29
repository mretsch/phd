from os.path import expanduser
import timeit
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import keras.layers as klayers
import keras.models as kmodels
import keras.utils as kutils
import keras.callbacks as kcallbacks
from NeuralNet.backtracking import mlp_backtracking_maxnode, mlp_backtracking_percentage, high_correct_predictions
from LargeScale.ls_at_metric import large_scale_at_metric_times, subselect_ls_vars
from basic_stats import into_pope_regimes, root_mean_square_error
import pandas as pd

home = expanduser("~")
start = timeit.default_timer()

# assemble the large scale dataset
ghome = home+'/Google Drive File Stream/My Drive'
ds_ls  = xr.open_dataset(ghome+'/Data/LargeScale/CPOL_large-scale_forcing_cape990hPa_cin990hPa_rh_shear.nc')
metric = xr.open_dataarray(ghome+'/Data_Analysis/rom_km_avg6h_nanzero.nc')

ls_vars = ['omega',
           'T_adv_h',
           'r_adv_h',
           'dsdt',
           'drdt',
           'RH',
           'u',
           'v',
           # 'dwind_dz'
           ]
ls_times = 'same_and_earlier_time'
predictor, target, _ = large_scale_at_metric_times(ds_largescale=ds_ls,
                                                   timeseries=metric,
                                                   chosen_vars=ls_vars,
                                                   l_take_scalars=True,
                                                   large_scale_time=ls_times)

l_subselect = True
if l_subselect:
    predictor = subselect_ls_vars(predictor, levels=[115, 515, 990], large_scale_time=ls_times)

n_lev = len(predictor['lev'])

l_loading_model = True
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
    filepath = home+'/Desktop/model-{epoch:02d}-{val_loss:.2f}.h5'
    checkpoint = kcallbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only=False)
    callbacks_list = [checkpoint]

    # fit the model
    model.fit(x=predictor, y=target, validation_split=0.2, epochs=10, batch_size=10, callbacks=callbacks_list)

    l_predict = False
    if l_predict:
        print('Predicting...')
        pred = []
        for i, entry in enumerate(predictor):
            pred.append(model.predict(np.array([entry])))
        pred_array = xr.DataArray(pred).squeeze()
        predicted = xr.DataArray(pred_array.values, coords={'time': predictor.time}, dims='time')

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
    model_path = ghome + '/ROME_Models/NoCorrScalars/'
    model = kmodels.load_model(model_path + 'model.h5')

    input_length = len(predictor[0])
    w = model.get_weights()
    needed_input_size = len(w[0])

    assert needed_input_size == input_length, 'Provided input to model does not match needed input size.'

    l_high_values = False
    if l_high_values:
        predicted = xr.open_dataarray(model_path + 'predicted.nc')
        metric, predicted = high_correct_predictions(target=metric, predictions=predicted,
                                                     target_percentile=0.9, prediction_offset=0.3)
    else:
        predicted = xr.open_dataarray(model_path + 'predicted.nc')
        # only times that could be predicted (via large-scale set). Sample size: 26,000 -> 6,000
        metric = metric.where(predicted.time)

    input_percentages_list = []
    for input in predictor.sel(time=predicted.time):

        node_contribution = mlp_backtracking_percentage(model, input)[0]
        input_percentages_list.append(node_contribution)

    input_percentages = xr.zeros_like(predictor.sel(time=predicted.time))
    input_percentages[:, :] = input_percentages_list

    plt.rc('font', size=22)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(48, 4))
    # ax.set_ylim(-25, 25)
    ax.axhline(y=0, color='r', lw=0.5)
    sns.boxplot(data=input_percentages)
    label_list = [str(element0)+', '+element1+', '+str(element2) for element0, element1, element2 in
                  zip(range(n_lev), predictor['long_name'].values, predictor.lev.values)]
    plt.xticks(list(range(n_lev)), label_list, rotation='vertical', fontsize=5)
    # ax.axes.set_yticklabels(labels=predictor['long_name'].values, fontdict={'fontsize':8})
    # ax.tick_params(axis='both', which='major', labelsize=8)
    plt.savefig(home + '/Desktop/whisker.pdf', bbox_inches='tight', transparent=True)

stop = timeit.default_timer()
print('This script needed {} seconds.'.format(stop-start))
