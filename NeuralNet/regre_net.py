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
from Plotscripts.plot_hist import histogram_2d
from NeuralNet.backtracking import mlp_insight
from LargeScale.ls_at_metric import large_scale_at_metric_times
import pandas as pd

home = expanduser("~")
start = timeit.default_timer()

l_loading_model = False

# assemble the large scale dataset
ds_ls  = xr.open_dataset(home+'/Data/LargeScaleState/CPOL_large-scale_forcing_cape990hPa_cin990hPa_rh_shear.nc')
metric = xr.open_dataarray(home+'/Data/Analysis/No_Boundary/AllSeasons/rom_km_avg6h.nc')

ls_vars = ['omega',
           'T_adv_h',
           'r_adv_h',
           'dsdt',
           'drdt',
           'RH',
           'u',
           'v',
           ]
predictor, target, _ = large_scale_at_metric_times(ds_largescale=ds_ls,
                                                   timeseries=metric,
                                                   chosen_vars=ls_vars,
                                                   l_take_same_time=False)

n_lev = len(predictor['lev'])

if not l_loading_model:
    # building the model
    model = kmodels.Sequential()
    model.add(klayers.Dense(300, activation='relu', input_shape=(n_lev,)))
    model.add(klayers.Dense(300, activation='relu'))
    model.add(klayers.Dense(300, activation='relu'))
    model.add(klayers.Dense(1))

    # compiling the model
    model.compile(optimizer='adam', loss='mean_squared_error')  # , metrics=['accuracy'])

    # fit the model
    # predictor = predictor.transpose()
    model.fit(x=predictor, y=target, validation_split=0.2, epochs=10, batch_size=10)

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
        plt.savefig('/Users/mret0001/Desktop/last1200.pdf', bbox_inches='tight')

else:
    # load a model
    model = kmodels.load_model('/Users/mret0001/Desktop/Model_300x3_avg_wholeROME_RH_bothtimes_again/model.h5')

    input_length = len(predictor[0])
    w = model.get_weights()
    needed_input_size = len(w[0])

    assert needed_input_size == input_length, 'Provided input to model does not match needed input size.'

    for n_node in range(12, 101):
        maximum_nodes = []
        for input in predictor:
            maximum_nodes.append(mlp_insight(model, input, n_highest_node=n_node))

        # =================================================
        mn = xr.DataArray(maximum_nodes)
        first_node = mn[:, 0]

        # plt.hist(first_node, bins=np.arange(0, input_length+1))

        fig, ax_host = plt.subplots(nrows=1, ncols=1, figsize=(60, 4))
        ax_host.hist(first_node, bins=np.arange(0, input_length + 1))
        ax_host.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax_host.xaxis.set_minor_locator(ticker.MultipleLocator(2))
        # ax_host.set_xlim(1, None)
        # ax_host.set_yscale('log')
        plt.savefig('/Users/mret0001/Desktop/histos/'+str(n_node)+'_input_histo.pdf', transparent=True, bbox_inches='tight')

stop = timeit.default_timer()
print('This script needed {} seconds.'.format(stop-start))
