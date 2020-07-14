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
from Plotscripts.plot_contribution_whisker import contribution_whisker
import pandas as pd


def nth_percentile(x, p):
    assert x.dims[0] == 'time'
    assert (0 < p) & (p < 1)
    return x[abs(x.rank(dim='time', pct=True) - p).argmin().item()].item()


home = expanduser("~")
start = timeit.default_timer()

# assemble the large scale dataset
ghome = home+'/Google Drive File Stream/My Drive'
ds_ls  = xr.open_dataset(home+
                         '/Documents/Data/LargeScaleState/CPOL_large-scale_forcing_cape990hPa_cin990hPa_rh_shear_dcape.nc')
metric = xr.open_dataarray(ghome+'/Data_Analysis/rom_km_avg6h_nanzero.nc')

ls_vars = [#'omega',
           'T_adv_h',
           'r_adv_h',
           'dsdt',
           'drdt',
           'RH',
           'u',
           'v',
           'dwind_dz'
           ]
long_names = [ds_ls[var].long_name for var in ls_vars]
ls_times = 'same_and_earlier_time'
predictor, target, _ = large_scale_at_metric_times(ds_largescale=ds_ls,
                                                   timeseries=metric,
                                                   chosen_vars=ls_vars,
                                                   l_take_scalars=True,
                                                   large_scale_time=ls_times)

l_subselect = True
if l_subselect:
    predictor = subselect_ls_vars(predictor, long_names, levels_in=[215, 515, 990], large_scale_time=ls_times)

l_eof_input = False
if l_eof_input:
    pcseries = xr.open_dataarray(home + '/Documents/Data/LargeScaleState/eof_pcseries_all.nc')
    eof_late  = pcseries.sel(number=list(range(20)),
                             time=predictor.time                        ).rename({'number': 'lev'}).T
    eof_early = pcseries.sel(number=list(range(20)),
                             time=eof_late.time - np.timedelta64(6, 'h')).rename({'number': 'lev'}).T

    predictor = xr.DataArray(np.zeros((eof_late.shape[0], eof_late.shape[1]*2)),
                             coords=[eof_late['time'], np.concatenate([eof_late['lev'], eof_late['lev']], axis=0)],
                             dims=['time', 'lev'])

    predictor[:, :] = np.concatenate([eof_early, eof_late], axis=1)
    predictor = predictor[target.notnull()]
    target    = target   [target.notnull()]

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
    filepath = home+'/Desktop/M/model-{epoch:02d}-{val_loss:.2f}.h5'
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

else:
    # load a model
    model_path = home + '/Documents/Data/NN_Models/ROME_Models/Kitchen_WithoutFirst10/'
    model = kmodels.load_model(model_path + 'model.h5')

    input_length = len(predictor[0])
    w = model.get_weights()
    needed_input_size = len(w[0])

    assert needed_input_size == input_length, 'Provided input to model does not match needed input size.'

    predicted = xr.open_dataarray(model_path + 'predicted.nc')

    l_high_values = False
    if l_high_values:
        _, predicted = high_correct_predictions(target=metric, predictions=predicted,
                                                target_percentile=0.9, prediction_offset=0.3)

    input_percentages_list = []
    for model_input in predictor.sel(time=predicted.time):
        node_contribution = mlp_backtracking_percentage(model, model_input)[0]
        input_percentages_list.append(node_contribution)

    input_percentages = xr.zeros_like(predictor.sel(time=predicted.time))
    input_percentages[:, :] = input_percentages_list

    p75 = xr.DataArray([nth_percentile(series, 0.75) for series in input_percentages.T])
    p25 = xr.DataArray([nth_percentile(series, 0.25) for series in input_percentages.T])
    spread = p75 - p25
    high_spread_vars = input_percentages[0, np.unique(spread, return_index=True)[1][-10:]].long_name

    # ===== Plots =====================

    plot = contribution_whisker(input_percentages=input_percentages,
                                levels=predictor.lev.values,
                                long_names=predictor['long_name'],
                                ls_times='same_and_earlier_time',
                                n_lev_total=n_lev,
                                n_profile_vars=23, #27, #9, #
                                xlim=25,
                                bg_color='mistyrose',
                                )

    plot.savefig(home + '/Desktop/nn_whisker.pdf', bbox_inches='tight', transparent=True)

stop = timeit.default_timer()
print('This script needed {} seconds.'.format(stop-start))
