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
import pandas as pd

start = timeit.default_timer()

real_data = True
testing = False
manual_sampling = False

if real_data:
    ds_predictors =\
        xr.open_dataset('/Users/mret0001/Data/LargeScaleState/CPOL_large-scale_forcing_cape_cin_rh_shear.nc')

    c1 = xr.concat([
          # ds_predictors.T
        # , ds_predictors.r
        # , ds_predictors.s
        # , ds_predictors.u
        # , ds_predictors.v
         ds_predictors.omega    [:, :-1] #!
        , ds_predictors.div     [:, :-1] #!
        , ds_predictors.T_adv_h [:, :-1] #!
        , ds_predictors.T_adv_v [:, :-1] #
        , ds_predictors.r_adv_h [:, :-1] #!
        , ds_predictors.r_adv_v [:, :-1] #
        , ds_predictors.s_adv_h [:, :-1] #!
        , ds_predictors.s_adv_v [:, :-1] #!
        , ds_predictors.dsdt    [:, :-1] #!
        , ds_predictors.drdt    [:, :-1] #!
        , ds_predictors.dwind_dz[:, :-2] #! bottom levels filled with NaN
        , ds_predictors.RH      [:, :-1] #!
    ], dim='lev')
    c2 = xr.concat([
          ds_predictors.cin
        , ds_predictors.cld_low
        , ds_predictors.lw_dn_srf
        , ds_predictors.wspd_srf
        , ds_predictors.v_srf
        , ds_predictors.r_srf
        , ds_predictors.lw_net_toa
        , ds_predictors.SH
        , ds_predictors.LWP
    ])
    c2_r = c2.rename({'concat_dims': 'lev'})
    c2_r.coords['lev'] = np.arange(len(c2))

    # var = xr.concat([c1, c2_r], dim='lev')
    var = c1
    # var_itp = var# .resample(time='T9min').interpolate('linear')

    # metric = xr.open_dataarray('/Users/mret0001/Data/ROME_Samples/rom_avg6h_afterLS_85pct_5050sample.nc')
    metric = xr.open_dataarray('/Users/mret0001/Data/Analysis/No_Boundary/AllSeasons/rom_km_avg6h.nc')

    # metric has no unique times atm, so cannot be used as a dimension
    # m = metric.where(metric.time==np.unique(metric.time))

    # large scale variables only where metric is defined
    var_metric = var.where(metric.notnull(), drop=True)

    # boolean for the large scale variables without any NaN anywhere
    l_var_nonull = var_metric.notnull().all(dim='lev')

    take_same_time = False
    if take_same_time:
        predictor = var_metric[{'time': l_var_nonull}]
        target = metric.sel(time=predictor.time)

    if not take_same_time:
        take_only_predecessor_time = False

        if take_only_predecessor_time:
            var_nonull = var_metric[l_var_nonull]
            var_nonull_6earlier = var_nonull.time - np.timedelta64(6, 'h')
            times = []
            for t in var_nonull_6earlier:
                try:
                    _ = var_metric.sel(time=t)
                    times.append(t.values)
                except KeyError:
                    continue
            # var_sub.sel(time=[np.datetime64('2002-08-10T18'), np.datetime64('2002-08-08T12')])
            var_6earlier = var_metric.sel(time=times)
            var_6earlier_nonull = var_6earlier[var_6earlier.notnull().all(dim='lev')]

            # m_later = metric.sel(time=(var_6earlier_nonull.time + np.timedelta64(6, 'h')).values)
            # metric six hour later not necessarily a value. Actually it is necessarily a value, because now we are back at
            # times of var_metric, where metric is a number.
            # m_nonull = m_later.where(m_later.notnull(), drop=True)
            # target = metric.sel(time=m_nonull.time)
            target = metric.sel(time=(var_6earlier_nonull.time + np.timedelta64(6, 'h')).values)

            predictor = var.sel(time=target.time - np.timedelta64(6, 'h'))

        else:
            # the timesteps have to be consecutive, not the indizes (time between indices can jump)

            var_nonull = var_metric[l_var_nonull]
            var_nonull_6earlier = var_nonull.time - np.timedelta64(6, 'h')
            times = []
            for t in var_nonull_6earlier:
                try:
                    _ = var_metric.sel(time=t)
                    times.append(t.values)
                except KeyError:
                    continue
            # var_sub.sel(time=[np.datetime64('2002-08-10T18'), np.datetime64('2002-08-08T12')])
            var_6earlier = var_metric.sel(time=times)
            var_6earlier_nonull = var_6earlier[var_6earlier.notnull().all(dim='lev')]

            var_6later_nonull = var_nonull.sel(time=(var_6earlier_nonull.time + np.timedelta64(6, 'h')))
            # first 'create' the right array with the correct 'late' time steps
            xr.concat([var_6later_nonull, var_6later_nonull], dim='lev')

            # m_later = metric.sel(time=(var_6earlier_nonull.time + np.timedelta64(6, 'h')).values)
            # metric six hour later not necessarily a value. Actually it is necessarily a value, because now we are back at
            # times of var_metric, where metric is a number.
            # m_nonull = m_later.where(m_later.notnull(), drop=True)
            # target = metric.sel(time=m_nonull.time)
            target = metric.sel(time=(var_6earlier_nonull.time + np.timedelta64(6, 'h')).values)


            predictor = var.sel(time=target.time - np.timedelta64(6, 'h'))





    n_lev = len(predictor['lev'])

    # building the model
    model = kmodels.Sequential()
    model.add(klayers.Dense( 300, activation='relu', input_shape=(n_lev,)))
    model.add(klayers.Dense( 300, activation='relu'))
    model.add(klayers.Dense( 300, activation='relu'))
    # model.add(klayers.Dense( 300, activation='relu'))
    # model.add(klayers.Dense( 300, activation='relu'))
    model.add(klayers.Dense(1))

    # compiling the model
    model.compile(optimizer='adam', loss='mean_squared_error')#, metrics=['accuracy'])

    # fit the model
    # predictor = predictor.transpose()
    model.fit(x=predictor, y=target, validation_split=0.2, epochs=10, batch_size=10)

    l_predict = True
    if l_predict:
        print('Predicting...')
        pred = []
        for i, entry in enumerate(predictor):
            pred.append(model.predict(np.array([entry])) )
        p = xr.DataArray(pred)
        pp = p.squeeze()
        pp.coords['time'] = ('dim_0', target.time)
        predicted = pp.swap_dims({'dim_0': 'time'})

        fig, ax_host = plt.subplots(nrows=1, ncols=1, figsize=(48, 4))
        ax_host.plot(target[-1200:])
        ax_host.plot(predicted[-1200:])
        plt.legend(['target', 'predicted'])
        # ax_host.xaxis.set_major_locator(ticker.MultipleLocator(1))
        # ax_host.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
        # plt.grid(which='both')
        plt.savefig('/Users/mret0001/Desktop/last1200.pdf', bbox_inches='tight')

if testing:
    if real_data:
        c = target[2:4]
        c[0:2] = [3., 5.]
        cre = c.resample(time='T0min').interpolate('linear')
        print(cre.time)
        print(c.time)
        # is it a bug? I want 10min frequency but have to say 10-1 = 9
        a = predictor[:2]
        b = a.resample(time='T9min').interpolate('linear')

    convolving = False
    if convolving:
        # https://datascience.stackexchange.com/questions/27506/back-propagation-in-cnn
        from scipy import signal
        o = np.array([(0.51, 0.9, 0.88, 0.84, 0.05),
                      (0.4, 0.62, 0.22, 0.59, 0.1),
                      (0.11, 0.2, 0.74, 0.33, 0.14),
                      (0.47, 0.01, 0.85, 0.7, 0.09),
                      (0.76, 0.19, 0.72, 0.17, 0.57)])
        d = np.array([(0, 0, 0.0686, 0),
                      (0, 0.0364, 0, 0),
                      (0, 0.0467, 0, 0),
                      (0, 0, 0, -0.0681)])

        gradient = signal.convolve(np.rot90(np.rot90(d)), o, 'valid')
        I = np.array([[3,0], [1,0]])
        K = np.array([[0,2], [0,3]])
        signal.convolve(I, K)

    l_model1 = False
    if l_model1:
        x = np.random.randint(1, 50, size=(200))
        y = np.square(x)
        model = kmodels.Sequential()
        model.add(klayers.Dense(2, activation='relu', input_shape=(1,)))
        for _ in range(64):
            model.add(klayers.Dense(2, activation='relu'))
        model.add(klayers.Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_absolute_error')
        early_stopping_monitor = kcallbacks.EarlyStopping(patience=15)
        model.fit(x, y, batch_size=1, epochs=2000, validation_split=0.3, callbacks=[early_stopping_monitor])

    l_model2 = False
    if l_model2:
        x = np.random.randint(1, 50, size=(200))
        y = np.square(x)
        model = kmodels.Sequential()
        model.add(klayers.Dense(150, activation='sigmoid', input_shape=(1,)))
        model.add(klayers.Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_absolute_error')
        model.fit(x, y, batch_size=1, epochs=2000)

    l_model3 = False
    if l_model3:
        x = np.random.randint(1, 50, size=(500, 3))
        y = x[:, 0] * 10
        model = kmodels.Sequential()
        model.add(klayers.Dense(150, activation='relu', input_shape=(x.shape[1],)))
        model.add(klayers.Dense(150, activation='relu'))
        model.add(klayers.Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x, y, batch_size=1, epochs=20, validation_split=0.3)

    l_model4 = False
    if l_model4:
        x = np.random.randint(10, 30, size=(500, 6))
        y = x[:, 3] * 10
        model = kmodels.Sequential()
        model.add(klayers.Dense(150, activation='relu', input_shape=(x.shape[1],)))
        model.add(klayers.Dense(150, activation='relu'))
        model.add(klayers.Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x, y, batch_size=1, epochs=20, validation_split=0.3)

    l_model5 = False
    if l_model5:
        x = np.random.randint(1, 50, size=(500, 3))
        y = x.mean(axis=1)
        model = kmodels.Sequential()
        model.add(klayers.Dense(150, activation='relu', input_shape=(x.shape[1],)))
        model.add(klayers.Dense(150, activation='relu'))
        model.add(klayers.Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x, y, batch_size=1, epochs=20, validation_split=0.3)
        # model.predict(np.array([[10, 20, 30]]))
        # array([[19.90018]], dtype=float32)
        # model.predict(np.array([[10, 20, 60]]))
        # array([[30.204426]], dtype=float32)
        # model.predict(np.array([[5, 5, 5]]))
        # array([[4.967877]], dtype=float32)
        # model.predict(np.array([[5, 7, 9]]))
        # array([[6.959164]], dtype=float32)
        # model.predict(np.array([[5, 7, 18]]))
        # array([[10.033775]], dtype=float32)

    l_model6 = True
    if l_model6:
        x = np.random.randint(-50, 50, size=(500))
        y = np.square(x)
        model = kmodels.Sequential()
        model.add(klayers.Dense(150, activation='relu', input_shape=(1,)))
        model.add(klayers.Dense(150, activation='relu'))
        model.add(klayers.Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x, y, batch_size=1, epochs=200, validation_split=0.3)

    model_insight = False
    if model_insight:

        def mlp_insight(model, data_in):
            output = np.array(data_in)
            weight_list = model.get_weights()
            # each layer has weights and biases
            n_layers = int(len(weight_list) / 2)

            # cycle through the layers, a forward pass
            results = []
            for i in range(n_layers):
                # get appropriate trained parameters, first are weights, second are biases
                weights = weight_list[i*2]
                bias = weight_list[i*2 + 1]
                # the @ is a matrix multiplication, first output is actually the mlp's input
                output = weights.transpose() @ output + bias
                output[output < 0] = 0
                # append output, so it can be overwritten in next iteration
                results.append(output)

            # after forward pass, recursively find chain of nodes with maximum value in each layer
            last_layer = results[-2] * weight_list[-2].transpose()
            max_nodes = [last_layer.argmax()]
            # concatenate the original NN input, data_in, and the output from the remaining layers
            iput = [np.array(data_in)] + results[:-2]
            for i in range(n_layers - 1)[::-1]:
                # weights are stored in array of shape (# nodes in layer n, # nodes in layer n+1)
                layer_to_maxnode = iput[i] * weight_list[2*i][:, max_nodes[-1]]
                max_nodes.append(layer_to_maxnode.argmax())

            return np.array(max_nodes[::-1])


        model = kmodels.load_model('/Users/mret0001/Desktop/correlationmodel.h5')
        # some arbitrary input
        x = [40, 40, 20]
        output = np.array(x)
        weight_list = model.get_weights()
        # each layer has weights and biases
        n_layers = int(len(weight_list) / 2)

        # cycle through the layers, a forward pass
        results = []
        for i in range(n_layers):
            # get appropriate trained parameters, first are weights, second are biases
            weights = weight_list[i*2]
            bias = weight_list[i*2 + 1]
            # the @ is a matrix multiplication, first output is actually the mlp's input
            output = weights.transpose() @ output + bias
            output[output < 0] = 0
            # append output, so it can be overwritten in next iteration
            results.append(output)

        t = results[-2] * weight_list[-2].transpose()
        # the correct predicition
        print(model.predict(np.array([x])))
        print(t.sum() + bias)
        print(t.argmax())
        t_maxind = t.argmax()
        # weights stored in array of shape (# nodes in layer n, # nodes in layer n+1)
        s = results[-3] * weight_list[-4][:, t_maxind]
        print(s.argmax())
        s_maxind = s.argmax()
        r = x * weight_list[-6][:, s_maxind]
        print(r.argmax())

        # maximum_nodes = np.zeros(shape=(n_layers, 20**6))
        maximum_nodes = np.zeros(shape=(n_layers, 50**3))
        index = 0
        for k in range(1,51):
            for l in range(1,51):
                for m in range(1,51):
                    # for n in range(10,30):
                    #     for o in range(10,30):
                    #         for p in range(10,30):
                                x = [k, l, m]#, n, o, p]
                                maximum_nodes[:, index] = mlp_insight(model=model, data_in=x)
                                index += 1

    plotting_model = False
    if plotting_model:
        kutils.plot_model(model, to_file='a.png')
        data = plt.imread('a.png')
        plt.imshow(data)
        plt.show()

    plotting_result = False
    if plotting_result:
        model = kmodels.load_model('/Users/mret0001/Desktop/long_squaremodel_0_200.h5')
        n = 50
        predictions = np.zeros(shape=n)
        true = np.array(list(range(n)))**2
        for i in range(n):
            predictions[i] = model.predict([i])
        plt.plot(list(range(n)), true)
        plt.plot(list(range(n)), predictions, color='red')
        plt.show()


if manual_sampling:
    l_resample = True
    if l_resample:
        metric = xr.open_dataarray('/Users/mret0001/Data/Analysis/No_Boundary/AllSeasons/rom_kilometres.nc')
        # take means over 6 hours each, starting at 3, 9, 15, 21 h. The time labels are placed in the middle of
        # the averaging period. Thus the labels are aligned to the large scale data set.
        m_avg = metric.resample(indexer={'time': '6H'}, skipna=False, closed='left', label='left', base=3,
                                loffset='3H').max()
        m_avg.coords['percentile'] = m_avg.rank(dim='time', pct=True)

    # metric = xr.open_dataarray('/Volumes/GoogleDrive/My Drive/Data_Analysis/rom_kilometres_avg6h.nc')
    metric = m_avg

    # ROME-value at given percentile
    threshold = metric[abs((metric.percentile - 0.85)).argmin()]
    n_above_thresh = (metric > threshold).sum().item()
    sample_ind = xr.DataArray(np.zeros(shape=2*n_above_thresh))
    sample_ind[:] = -1

    # find arguments (meaning indizes) for the highest ROME-values
    m_present = metric.where(metric.notnull(), drop=True)
    sort_ind = m_present.argsort()
    sample_ind[-n_above_thresh:] = sort_ind[-n_above_thresh:]

    # stride through ROME-values (not the percentiles or sorted indizes) linearly
    # With this method some indizes might be taken twice as they have shortest distance to two ROME-values.
    # We sort that out below.
    check_values = np.linspace(6.25, threshold, n_above_thresh)
    for i, v in enumerate(check_values):
        ind = abs((m_present - v)).argmin()
        sample_ind[i] = ind

    unique, indizes, inverse, count = np.unique(sample_ind, return_index=True, return_inverse=True, return_counts=True)

    # take the samples in samples_ind which recreate the outcome of the unique function, which orders the unique values.
    # Here thats the order of indizes we use to get a sample from ROME-values. Hence they will be timely ordered. Okay.
    sample_ind_unique = sample_ind[indizes]

    metric_sample = m_present[sample_ind_unique.astype(int)]
    sample = metric_sample.rename({'dim_0': 'time'})
    sample.to_netcdf('/Users/mret0001/Desktop/rom_sample.nc')


stop = timeit.default_timer()
print('This script needed {} seconds.'.format(stop-start))
