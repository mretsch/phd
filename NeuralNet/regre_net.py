from os.path import expanduser
import timeit
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import keras.layers as klayers
import keras.models as kmodels
import keras.callbacks as kcallbacks
import NeuralNet.backtracking as bcktrck
from LargeScale.ls_at_metric import large_scale_at_metric_times, subselect_ls_vars
from Plotscripts.plot_contribution_whisker import contribution_whisker


def add_variable_symbol_string(dataset):
    assert type(dataset) == xr.core.dataset.Dataset

    quantity_symbol = [
        'T',
        'r',
        'u',
        'v',
        'w',
        'div(U)',
        'adv(T)',
        'conv(T)',
        'adv(r)',
        'conv(r)',
        's',
        'adv(s)',
        'conv(s)',
        'dsdt',
        'dTdt',
        'drdt',
        'xxx',
        'xxx',
        'C',
        'P',
        'LH', #"'Q_h_sfc',
        'SH', #'Q_s_sfc',
        'p',
        'p_centre',
        'T_2m',
        'T_skin',
        'RH_2m',
        'Speed_10m',
        'u_10m',
        'v_10m',
        'rad_sfc',
        'OLR', #'LW_toa',
        'SW_toa',
        'SW_dn_toa',
        'c_low',
        'c_mid',
        'c_high',
        'c_total',
        'c_thick',
        'c_top',
        'LWP',
        'dTWPdt',
        'adv(TWP)',
        'E',
        'dsdt_col',
        'adv(s)_col',
        'Q_rad',
        'Q_lat',
        'w_sfc',
        'r_2m',
        's_2m',
        'PW',
        'LW_up_sfc',
        'LW_dn_sfc',
        'SW_up_sfc',
        'SW_dn_sfc',
        'RH',
        'dUdz',
        'CAPE',
        'CIN',
        'D-CAPE',
    ]

    for i, var in enumerate(dataset):
        dataset[var].attrs['symbol'] = quantity_symbol[i]

    return dataset


def nth_percentile(x, p):
    assert x.dims[0] == 'time'
    assert (0 < p) & (p < 1)
    return x[abs(x.rank(dim='time', pct=True) - p).argmin().item()].item()


home = expanduser("~")
start = timeit.default_timer()

l_profiles_as_eof = True
if l_profiles_as_eof:
    height_dim = 'number'
else:
    height_dim = 'lev'

# get the large scale dataset
ds_ls  = xr.open_dataset(home +
                         '/Documents/Data/LargeScaleState/' +
                         'CPOL_large-scale_forcing_noDailyCycle_profilesEOF.nc')

# Choose to predict TCA or ROME
metric = xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/rom_km_max6h_avg_pm20minutes.nc')
# metric = xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/totalarea_km_avg6h.nc')

# add quantity symbols to large-scale dataset
ds_ls = add_variable_symbol_string(ds_ls)

# remove false data in precipitable water
ds_ls['PW'].loc[{'time': slice(None           , '2002-02-27T12')}] = np.nan
ds_ls['PW'].loc[{'time': slice('2003-02-07T00', '2003-10-19T00')}] = np.nan
ds_ls['PW'].loc[{'time': slice('2005-11-14T06', '2005-12-09T12')}] = np.nan
ds_ls['PW'].loc[{'time': slice('2006-02-25T00', '2006-04-07T00')}] = np.nan
ds_ls['PW'].loc[{'time': slice('2011-11-09T18', '2011-12-01T06')}] = np.nan
ds_ls['PW'].loc[{'time': slice('2015-01-05T00', None           )}] = np.nan
ds_ls['LWP'].loc[{'time': slice(None           , '2002-02-27T12')}] = np.nan
ds_ls['LWP'].loc[{'time': slice('2003-02-07T00', '2003-10-19T00')}] = np.nan
ds_ls['LWP'].loc[{'time': slice('2005-11-14T06', '2005-12-09T12')}] = np.nan
ds_ls['LWP'].loc[{'time': slice('2006-02-25T00', '2006-04-07T00')}] = np.nan
ds_ls['LWP'].loc[{'time': slice('2011-11-09T18', '2011-12-01T06')}] = np.nan
ds_ls['LWP'].loc[{'time': slice('2015-01-05T00', None           )}] = np.nan

ls_vars = [
           'omega',
           'u',
           'v',
           's',
           'RH',
           's_adv_h',
           'r_adv_h',
           'dsdt',
           'drdt',
           'dwind_dz'
           ]
long_names = [ds_ls[var].long_name for var in ls_vars]
ls_times = 'same_time'

predictor, target, _ = large_scale_at_metric_times(ds_largescale=ds_ls,
                                                   timeseries=metric,
                                                   chosen_vars=ls_vars,
                                                   l_take_scalars=True,
                                                   large_scale_time=ls_times,
                                                   l_profiles_as_eof=l_profiles_as_eof)

l_subselect = True
if l_subselect:
    predictor = subselect_ls_vars(predictor,
                                  long_names,
                                  levels_in=[215, 515, 990],
                                  large_scale_time=ls_times,
                                  l_profiles_as_eof=l_profiles_as_eof)

n_lev = len(predictor[height_dim])

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
    filepath = home+'/Desktop/M/model-{epoch:02d}-{val_loss:.5f}.h5'
    checkpoint = kcallbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_weights_only=False)
    callbacks_list = [checkpoint]

    # fit the model
    model.fit(x=predictor, y=target, validation_split=0.2, epochs=5, batch_size=1, callbacks=callbacks_list)

    l_predict = False
    if l_predict:
        print('Predicting...')
        model2 = kmodels.load_model(home + '/Desktop/model.h5')
        pred = []
        for i, entry in enumerate(predictor):
            pred.append(model2.predict(np.array([entry])))
        pred_array = xr.DataArray(pred).squeeze()
        predicted = xr.DataArray(pred_array.values, coords={'time': predictor.time}, dims='time')
        predicted.to_netcdf(home + '/Desktop/predicted.nc')

else:
    # load a model
    model_path = home + '/Desktop/'
    model = kmodels.load_model(model_path + 'model.h5')

    input_length = len(predictor[0])
    w = model.get_weights()
    needed_input_size = len(w[0])

    assert needed_input_size == input_length, 'Provided input to model does not match needed input size.'

    predicted = xr.open_dataarray(model_path + 'predicted.nc')

    l_high_values = True
    if l_high_values:
        _, predicted_high = bcktrck.high_correct_predictions(target=metric, predictions=predicted,
                                                             target_percentile=0.9, prediction_offset=0.3)

    # percentage-backtracking
    input_percentages = xr.zeros_like(predictor.sel(time=predicted.time))
    l_input_positive  = xr.full_like (predictor.sel(time=predicted.time), fill_value=False, dtype='bool')

    for i, model_input in enumerate  (predictor.sel(time=predicted.time)):
        input_percentages[i, :] = bcktrck.mlp_backtracking_percentage(model, model_input)[0]
        l_input_positive [i, :] = (model_input > 0.).values

    positive_positive_ratio = xr.zeros_like(input_percentages[:2, :])
    for i, _ in enumerate(positive_positive_ratio[height_dim]):
        positive_positive_ratio[0, i] = (l_input_positive[:, i] & (input_percentages[:, i] > 0.)).sum() \
                                      /  l_input_positive[:, i].sum()
        positive_positive_ratio[1, i] = ((l_input_positive[:, i] == False) & (input_percentages[:, i] < 0.)).sum() \
                                      /  (l_input_positive[:, i] == False).sum()

    if l_high_values:
        input_percentages.coords['high_pred'] = ('time', np.full_like(input_percentages[:, 0], False, dtype='bool'))
        input_percentages['high_pred'].loc[dict(time=predicted_high.time)] = True

    p75 = xr.DataArray([nth_percentile(series, 0.75) for series in input_percentages.T])
    p50 = xr.DataArray([nth_percentile(series, 0.50) for series in input_percentages.T])
    p25 = xr.DataArray([nth_percentile(series, 0.25) for series in input_percentages.T])
    spread = p75 - p25
    # exploiting that np.unique() also sorts ascendingly, but also returns the matching index, unlike np.sort()
    high_spread_vars = input_percentages[0, np.unique(spread, return_index=True)[1][-10:]].long_name
    p50_high = xr.DataArray([nth_percentile(series[input_percentages['high_pred']], 0.50)
                             for series in input_percentages.T])

    l_sort_input_percentage = True
    if l_sort_input_percentage:
        # sort and reverse index because I want descending order.
        sort_index = np.unique(p50_high[:(n_lev)], return_index=True)[1][::-1]

        input_percentages = input_percentages[:, sort_index]

    # ===== Plots =====================

    l_violins = True \
                and l_high_values
    plot = contribution_whisker(input_percentages=input_percentages,
                                levels=predictor[height_dim].values[sort_index],
                                long_names=predictor['symbol'][sort_index],
                                ls_times='same_time',
                                n_profile_vars=n_lev,
                                xlim=150,
                                bg_color='mistyrose',
                                l_violins=l_violins,
                                )

    l_show_correlationmatrix = False
    if l_show_correlationmatrix:
        df = predictor.to_pandas()

        symbl_list = [string.strip() for string in predictor.symbol.values]
        n_list = [str(n) for n in predictor[height_dim].values]

        column_list = [a + b for a, b in zip(symbl_list, n_list)]
        df.columns = column_list

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(80, 70))
        corrmatrix = df.corr()
        sns.heatmap(abs(corrmatrix), annot=corrmatrix, fmt='.2f', cmap='Spectral')
        plt.savefig(home + '/Desktop/corrmatrix.pdf', bbox_inches='tight')

    plot.savefig(home + '/Desktop/nn_whisker.pdf', bbox_inches='tight', transparent=True)

stop = timeit.default_timer()
print('This script needed {} seconds.'.format(stop-start))
