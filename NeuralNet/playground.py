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
    var = xr.concat([var1, var2, var3], dim='lev')
    var_itp = var.resample(time='T9min').interpolate('linear')
    metric = xr.open_dataarray('/Users/mret0001/Data/Analysis/No_Boundary/AllSeasons/rom.nc')

    # same sample size for both data sets
    var_itp_sub = var_itp.where(metric[metric.notnull()])
    predictor = var_itp_sub.where(var_itp_sub.notnull(), drop=True)
    target = metric.where(predictor.time)

    n_lev = predictor.shape[1]

    # building the model
    model = kmodels.Sequential()
    model.add(klayers.Dense(100, activation='relu', input_shape=(n_lev,)))
    model.add(klayers.Dense(100, activation='relu'))
    model.add(klayers.Dense(1))

    # compiling the model
    model.compile(optimizer='adam', loss='mean_squared_error')#, metrics=['accuracy'])

    # fit the model
    model.fit(x=predictor, y=target, validation_split=0.3, epochs=1, batch_size=1)

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

    l_model3 = True
    if l_model3:
        x = np.random.randint(1, 50, size=(500, 3))
        y = x[:, 0] * 10
        model = kmodels.Sequential()
        model.add(klayers.Dense(150, activation='relu', input_shape=(x.shape[1],)))
        model.add(klayers.Dense(150, activation='relu'))
        model.add(klayers.Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x, y, batch_size=1, epochs=20, validation_split=0.3)
        # model.predict(np.array([[3, 4, 5]]))
        # array([[30.2555]], dtype=float32)
        # model.predict(np.array([[7, 4, 5]]))
        # array([[69.987595]], dtype=float32)
        # model.predict(np.array([[10, 4, 5]]))
        # array([[99.86937]], dtype=float32)
        # model.predict(np.array([[66, 4, 5]]))
        # array([[660.58734]], dtype=float32)

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
