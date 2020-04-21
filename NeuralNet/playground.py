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
from NeuralNet.backtracking import mlp_forward_pass, mlp_backtracking_maxnode, mlp_backtracking_percentage
import pandas as pd
import seaborn as sns

home = expanduser("~")
ghome = home+'/Google Drive File Stream/My Drive'
start = timeit.default_timer()

testing = False
manual_sampling = True

if testing:
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

    l_model6 = False
    if l_model6:
        x = np.random.randint(-50, 50, size=(500))
        y = np.square(x)
        model = kmodels.Sequential()
        model.add(klayers.Dense(150, activation='relu', input_shape=(1,)))
        model.add(klayers.Dense(150, activation='relu'))
        model.add(klayers.Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x, y, batch_size=1, epochs=200, validation_split=0.3)

    l_model7 = False
    if l_model7:
        x = np.random.randint(1, 50, size=(1000, 3))
        y = x[:, 1] ** 2
        model = kmodels.Sequential()
        model.add(klayers.Dense(750, activation='relu', input_shape=(x.shape[1],)))
        model.add(klayers.Dense(750, activation='relu'))
        model.add(klayers.Dense(750, activation='relu'))
        model.add(klayers.Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x, y, batch_size=10, epochs=150, validation_split=0.3)

    l_model8 = False
    if l_model8:
        x = np.random.randint(1, 50, size=(500, 3))
        y = 0.7 * x[:, 1] + 0.3 * x[:, 2]
        model = kmodels.Sequential()
        model.add(klayers.Dense(150, activation='relu', input_shape=(x.shape[1],)))
        model.add(klayers.Dense(150, activation='relu'))
        model.add(klayers.Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x, y, batch_size=1, epochs=2, validation_split=0.3)

    l_model9 = False
    if l_model9:
        x = np.random.randint(1, 50, size=(500, 3))
        # y = x[:, 1] * x[:, 2]
        y = x[:, 1] * x[:, 2] + x[:, 0]

        model = kmodels.Sequential()
        model.add(klayers.Dense(750, activation='relu', input_shape=(x.shape[1],)))
        model.add(klayers.Dense(750, activation='relu'))
        model.add(klayers.Dense(750, activation='relu'))
        model.add(klayers.Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        checkpoint = kcallbacks.ModelCheckpoint(home+'/Desktop/Models/model-{epoch:02d}-{val_loss:.2f}.h5',
                                                monitor='val_loss', verbose=1, save_weights_only=False)
        model.fit(x, y, batch_size=10, epochs=150, validation_split=0.3, callbacks=[checkpoint])

    l_model10 = False
    if l_model10:
        x = np.random.randint(1, 50, size=(1000, 3))
        y = np.sqrt(x[:, 1]) + 3 * x[:, 2]
        model = kmodels.Sequential()
        model.add(klayers.Dense(750, activation='relu', input_shape=(x.shape[1],)))
        model.add(klayers.Dense(750, activation='relu'))
        model.add(klayers.Dense(750, activation='relu'))
        model.add(klayers.Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x, y, batch_size=10, epochs=5, validation_split=0.3)

    l_model11 = False
    if l_model11:
        x = np.random.randint(1, 50, size=(1000, 3))
        y = np.sqrt(x[:, 1]) + np.log(x[:, 2])
        model = kmodels.Sequential()
        model.add(klayers.Dense(750, activation='relu', input_shape=(x.shape[1],)))
        model.add(klayers.Dense(750, activation='relu'))
        model.add(klayers.Dense(750, activation='relu'))
        model.add(klayers.Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x, y, batch_size=10, epochs=20, validation_split=0.3)

    model_insight = True
    if model_insight:

        model = kmodels.load_model(
            ghome+'/Data/NN_Models/BasicUnderstanding/SqrtLn_Model/sqrtlnmodel.h5')
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

        # maximum_nodes = np.zeros(shape=(n_layers, 50**3))
        # index = 0
        # for k in range(1,51):
        #     for l in range(1,51):
        #         for m in range(1,51):
        #             x = [k, l, m]#, n, o, p]
        #             maximum_nodes[:, index] = mlp_backtracking_maxnode(model=model, data_in=x,
        #                                                                            n_highest_node=1,
        #                                                                            return_firstconn=False)

        percentage_input = np.zeros(shape=(len(x), 50 ** 3))
        percentage_input_full = []
        index = 0
        for k in range(1, 51):
            for l in range(1, 51):
                for m in range(1, 51):
                    x = [k, l, m]
                    percentage_input[:, index] = mlp_backtracking_percentage(model=model, data_in=x)[0]
                    # percentage_input_full.append(mlp_backtracking_percentage(model=model, data_in=x))
                    # percentage_input[:, index] = percentage_input_full[-1][0]
                    index += 1

        sns.boxplot(data=percentage_input .T)
        # plt.ylim(90, 110)
        plt.title('Backtracking for sqrtln-net (target is sqrt(node1) + ln(node2).')
        plt.xlabel('# input node')
        plt.ylabel('Contributing percentage [%]')
        plt.savefig(home + '/Desktop/backtrack_corrnet_zoom_0.pdf', bbox_inches='tight')
        # np.unravel_index(maximum_nodes.argmax(), shape=maximum_nodes.shape)

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
        metric = xr.open_dataarray(ghome+'/Data_Analysis/rom_kilometres.nc')
        # take means over 6 hours each, starting at 3, 9, 15, 21 h. The time labels are placed in the middle of
        # the averaging period. Thus the labels are aligned to the large scale data set.
        # For reasons unknown, averages crossing a day of no data, not even NaN, into a normal day have wrongly
        # calculated averages. Overwrite manually with correct values.
        # Take sum and divide by 36 time steps (for 10 min intervals in 6 hours), to avoid one single value in 6 hours
        # (with the rest NaNs) to have that value as the average. sum()/36. treats NaNs as Zeros basically.
        m_avg = metric.resample(indexer={'time': '6H'}, skipna=False, closed='left', label='left', base=3,
                                loffset='3H').sum()/36. # mean() # max() # std()**2
        # .sum() applied to NaNs in xarray yields Zero, not NaN as with .mean().
        # put NaNs where .mean() yields NaNs.
        m_mean = metric.resample(indexer={'time': '6H'}, skipna=False, closed='left', label='left', base=3,
                                loffset='3H').mean()
        m_avg = m_avg.where(m_mean.notnull(), other=np.nan)
        manual_overwrite = True
        if manual_overwrite:
            m_avg.loc[
                [np.datetime64('2003-03-15T00:00'), np.datetime64('2003-03-17T00:00'), np.datetime64('2003-10-30T00:00'),
                 np.datetime64('2003-11-25T00:00'), np.datetime64('2006-11-11T00:00')]] = \
                [metric.sel(time=slice('2003-03-14T21', '2003-03-15T02:50')).sum()/36.,
                 metric.sel(time=slice('2003-03-16T21', '2003-03-17T02:50')).sum()/36.,
                 metric.sel(time=slice('2003-10-29T21', '2003-10-30T02:50')).sum()/36.,
                 metric.sel(time=slice('2003-11-24T21', '2003-11-25T02:50')).sum()/36.,
                 metric.sel(time=slice('2006-11-10T21', '2006-11-11T02:50')).sum()/36.]
        m_avg.coords['percentile'] = m_avg.rank(dim='time', pct=True)

    l_split_and_redistribute = False
    if l_split_and_redistribute:
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
        sample.to_netcdf(home+'/Desktop/rom_sample.nc')


stop = timeit.default_timer()
print('This script needed {} seconds.'.format(stop-start))
