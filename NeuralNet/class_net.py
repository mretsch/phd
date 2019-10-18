import timeit
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from Plotscripts.colors_solarized import sol
import keras.layers as klayers
import keras.models as kmodels
import keras.utils as kutils
import keras.callbacks as kcallbacks

start = timeit.default_timer()

ds_predictors = xr.open_dataset('/Volumes/GoogleDrive/My Drive/Data/LargeScale/CPOL_large-scale_forcing_cape_cin_rh.nc')
var_itp = ds_predictors.omega[:, 1:]  # .resample(time='T9min').interpolate('linear')

metric = xr.open_dataarray('/Users/mret0001/Data/Analysis/No_Boundary/AllSeasons/rom_kilometres_avg6h.nc')

mpp = metric.percentile.to_pandas()
tercile_tuple = pd.cut(mpp, 3, labels=[1, 2, 3], retbins=True)

metric.coords['tercile'] = tercile_tuple[0]
# metric = metric.rename({'dim_0':'time'})

var_itp_sub = var_itp.where(metric[metric.notnull()])
# predictor = var_itp_sub.where(var_itp_sub.notnull(), drop=True)

lst = var_itp_sub.notnull().all(dim='lev')
predictor = var_itp_sub[lst]

classes = metric.tercile.sel(time=predictor.time)
# keras wants classes 0-based, not 1-based, hence -1
target = kutils.to_categorical(classes.astype(int)-1)

n_lev = predictor.shape[1]

# building the model
# in regression models, activation would be 'linear', so a whole R is possible as outcome.
# in classification models, the activation switches to 'softmax'
input_tensor  = klayers.Input(shape=(n_lev,))
l1_tensor     = klayers.Dense(400, activation='relu'   )(input_tensor)
# l2_tensor     = klayers.Dense(1200, activation='relu'   )(   l1_tensor)
# l3_tensor     = klayers.Dense(100, activation='relu'   )(   l2_tensor)
output_tensor = klayers.Dense(  3, activation='softmax')(   l1_tensor)
model = kmodels.Model(input_tensor, output_tensor)

early_stopping_monitor = kcallbacks.EarlyStopping(patience=3)

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# fit the model
model.fit(x=predictor, y=target, batch_size=10, validation_split=0.2, epochs=20, callbacks=[early_stopping_monitor])

plotting_model = False
if plotting_model:
    kutils.plot_model(model, to_file='a.png')
    data = plt.imread('a.png')
    plt.imshow(data)
    plt.show()

l_predict = False
if l_predict:
    pred = []
    for i, entry in enumerate(predictor):
        pred.append(model.predict(np.array([entry])))

    p = xr.DataArray(pred)
    pp = p.squeeze()
    class_predicted = pp.argmax(dim='dim_2') + 1
    certainty_predicted = pp.max(dim='dim_2')

    # got class right
    certainty_gcr   = certainty_predicted.where( class_predicted == classes.astype(int).values, other=np.nan)
    certainty_gcr_1 = certainty_gcr.where((class_predicted == 1), other=np.nan)
    certainty_gcr_2 = certainty_gcr.where((class_predicted == 2), other=np.nan)
    certainty_gcr_3 = certainty_gcr.where((class_predicted == 3), other=np.nan)

    # got class wrong
    got_class_wrong = certainty_predicted.where(class_predicted != classes.astype(int).values, other=np.nan) * 0. + 0.33
    certainty_gcw   = certainty_predicted.where(class_predicted != classes.astype(int).values, other=np.nan)
    certainty_gcw_1 = certainty_gcw.where(class_predicted == 1, other=np.nan)
    certainty_gcw_2 = certainty_gcw.where(class_predicted == 2, other=np.nan)
    certainty_gcw_3 = certainty_gcw.where(class_predicted == 3, other=np.nan)

    fig, ax_host = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
    ax_r1 = ax_host.twinx()

    ax_host.step(np.arange(100) + 0.5, classes[:100], color='grey')
    ax_r1.plot(np.arange(100), certainty_gcr_3[:100], linestyle=' ', marker='p', color=sol['magenta'])
    ax_r1.plot(np.arange(100), certainty_gcr_2[:100], linestyle=' ', marker='p', color=sol['green'])
    ax_r1.plot(np.arange(100), certainty_gcr_1[:100], linestyle=' ', marker='p', color=sol['blue'])
    ax_r1.plot(np.arange(100), got_class_wrong[:100], linestyle=' ', marker='x', color=sol['orange'])


stop = timeit.default_timer()
print('This script needed {} seconds.'.format(stop-start))
