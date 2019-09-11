import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import keras.layers as klayers
import keras.models as kmodels
import keras.utils as kutils
import keras.callbacks as kcallbacks

real_data = False
if real_data:
    ds_predictors = xr.open_dataset('/Volumes/GoogleDrive/My Drive/Data/LargeScale/CPOL_large-scale_forcing.nc')
    var1 = ds_predictors.omega
    var2 = ds_predictors.T
    var3 = ds_predictors.div
    var4 = ds_predictors.r
    var5 = ds_predictors.u
    var6 = ds_predictors.v
    var7 = ds_predictors.r_adv_h
    var = xr.concat([var1, var2, var3, var4, var5, var6, var7], dim='lev')
    var_itp = var.resample(time='T9min').interpolate('linear')
    #metric = xr.open_dataarray('/Users/mret0001/Data/Analysis/No_Boundary/rom.nc')
    metric = xr.open_dataarray('/Users/mret0001/Desktop/rom_sample_nn.nc')

    # same sample size for both data sets
    var_itp_sub = var_itp.where(metric[metric.notnull()])
    predictor = var_itp_sub.where(var_itp_sub.notnull(), drop=True)
    target = metric.where(predictor.time)

    n_lev = predictor.shape[1]

    # building the model
    model = kmodels.Sequential()
    model.add(klayers.Dense(20, activation='relu', input_shape=(n_lev,)))
    model.add(klayers.Dense(20, activation='relu'))
    model.add(klayers.Dense(1))

    # compiling the model
    model.compile(optimizer='adam', loss='mean_absolute_error')#, metrics=['accuracy'])

    # fit the model
    model.fit(x=predictor, y=target, validation_split=0.3, epochs=5, batch_size=100)

    pred = []
    for i, entry in enumerate(predictor):
        pred.append( model.predict(np.array([entry])) )

testing = True
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

    model_insight = True
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


manual_sampling = True
if manual_sampling:
    metric = xr.open_dataarray('/Users/mret0001/Data/Analysis/No_Boundary/rom_kilometres.nc')

    # ROME-value at 97 percentile
    threshold = metric[abs((metric.percentile - 0.97)).argmin()]
    n_above_thresh = (metric > threshold).sum().item()
    sample_ind = xr.DataArray(np.zeros(shape=2*n_above_thresh))
    sample_ind[:] = -1

    # find arguments (meaning indizes) for the highest ROME-values
    m_present = metric.where(metric.notnull(), drop=True)
    sort_ind = m_present.argsort()
    sample_ind[-n_above_thresh:] = sort_ind[-n_above_thresh:]
    # stride through ROME-values (not the percentiles or sorted indizes) linearly
    rome_values = np.linspace(6.25, threshold, n_above_thresh)
    for i, v in enumerate(rome_values):
        ind = abs((m_present - v)).argmin()
        sample_ind[i] = ind

    metric_sample = m_present[sample_ind.astype(int)]
