from os.path import expanduser
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
from LargeScale.ls_at_metric import large_scale_at_metric_times

home = expanduser("~")
start = timeit.default_timer()

metric = xr.open_dataarray('/Volumes/GoogleDrive/My Drive/Data_Analysis/rom_km_avg6h.nc')
# cut metric into classes, based on percentiles
mpp = metric.percentile.to_pandas()
# tercile_tuple = pd.cut(mpp, 10, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], retbins=True)
tercile_tuple = pd.cut(mpp, 3, labels=[1, 2, 3], retbins=True)
metric.coords['tercile'] = tercile_tuple[0]

# assemble large-scale state
ds_ls = xr.open_dataset('/Volumes/GoogleDrive/My Drive/Data/LargeScale/CPOL_large-scale_forcing_cape_cin_rh_shear.nc')
predictor, target = large_scale_at_metric_times(ds_largescale=ds_ls,
                                                timeseries=metric,
                                                l_take_same_time=True)

# convert classes in class-net digestible categories
classes = target.tercile
target_cat = kutils.to_categorical(classes.astype(int)-1)
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
model.fit(x=predictor, y=target_cat, batch_size=10, validation_split=0.2, epochs=7)#, callbacks=[early_stopping_monitor])

plotting_model = False
if plotting_model:
    kutils.plot_model(model, to_file='a.png')
    data = plt.imread('a.png')
    plt.imshow(data)
    plt.show()

l_predict = True
if l_predict:
    pred = []
    for i, entry in enumerate(predictor):
        pred.append(model.predict(np.array([entry])))

    p = xr.DataArray(pred)
    pp = p.squeeze()
    class_predicted     = pp.argmax(dim='dim_2') + 1
    certainty_predicted = pp.   max(dim='dim_2')

    # got class right
    certainty_gcr    = certainty_predicted.where( class_predicted == classes.astype(int).values, other=np.nan)
    certainty_gcr_1  = certainty_gcr.where((class_predicted ==  1), other=np.nan)
    certainty_gcr_2  = certainty_gcr.where((class_predicted ==  2), other=np.nan)
    certainty_gcr_3  = certainty_gcr.where((class_predicted ==  3), other=np.nan)
    # certainty_gcr_4  = certainty_gcr.where((class_predicted ==  4), other=np.nan)
    # certainty_gcr_5  = certainty_gcr.where((class_predicted ==  5), other=np.nan)
    # certainty_gcr_6  = certainty_gcr.where((class_predicted ==  6), other=np.nan)
    # certainty_gcr_7  = certainty_gcr.where((class_predicted ==  7), other=np.nan)
    # certainty_gcr_8  = certainty_gcr.where((class_predicted ==  8), other=np.nan)
    # certainty_gcr_9  = certainty_gcr.where((class_predicted ==  9), other=np.nan)
    # certainty_gcr_10 = certainty_gcr.where((class_predicted == 10), other=np.nan)

    # got class wrong
    got_class_wrong = certainty_predicted.where(class_predicted != classes.astype(int).values, other=np.nan) * 0. + 0.3
    certainty_gcw   = certainty_predicted.where(class_predicted != classes.astype(int).values, other=np.nan)
    # certainty_gcw_1  = certainty_gcw.where((class_predicted ==  1), other=np.nan)
    # certainty_gcw_2  = certainty_gcw.where((class_predicted ==  2), other=np.nan)
    # certainty_gcw_3  = certainty_gcw.where((class_predicted ==  3), other=np.nan)
    # certainty_gcw_4  = certainty_gcw.where((class_predicted ==  4), other=np.nan)
    # certainty_gcw_5  = certainty_gcw.where((class_predicted ==  5), other=np.nan)
    # certainty_gcw_6  = certainty_gcw.where((class_predicted ==  6), other=np.nan)
    # certainty_gcw_7  = certainty_gcw.where((class_predicted ==  7), other=np.nan)
    # certainty_gcw_8  = certainty_gcw.where((class_predicted ==  8), other=np.nan)
    # certainty_gcw_9  = certainty_gcw.where((class_predicted ==  9), other=np.nan)
    # certainty_gcw_10 = certainty_gcw.where((class_predicted == 10), other=np.nan)

    fig, ax_host = plt.subplots(nrows=1, ncols=1, figsize=(18, 4))
    ax_r1 = ax_host.twinx()

    # colours = ['yellow', 'orange', 'red', 'magenta', 'violet', 'blue', 'cyan', 'green', 'base01', 'base03']
    colours = ['yellow', 'magenta', 'base03']

    n_length = 100 # 6387
    ax_host.step(np.arange(n_length) + 0.5,   classes[:n_length], color='grey')
    ax_r1.plot  (np.arange(n_length), certainty_gcr_1 [:n_length], linestyle=' ', marker='p', color=sol[colours[0]])
    ax_r1.plot  (np.arange(n_length), certainty_gcr_2 [:n_length], linestyle=' ', marker='p', color=sol[colours[1]])
    ax_r1.plot  (np.arange(n_length), certainty_gcr_3 [:n_length], linestyle=' ', marker='p', color=sol[colours[2]])
    # ax_r1.plot  (np.arange(n_length), certainty_gcr_4 [:n_length], linestyle=' ', marker='p', color=sol[colours[3]])
    # ax_r1.plot  (np.arange(n_length), certainty_gcr_5 [:n_length], linestyle=' ', marker='p', color=sol[colours[4]])
    # ax_r1.plot  (np.arange(n_length), certainty_gcr_6 [:n_length], linestyle=' ', marker='p', color=sol[colours[5]])
    # ax_r1.plot  (np.arange(n_length), certainty_gcr_7 [:n_length], linestyle=' ', marker='p', color=sol[colours[6]])
    # ax_r1.plot  (np.arange(n_length), certainty_gcr_8 [:n_length], linestyle=' ', marker='p', color=sol[colours[7]])
    # ax_r1.plot  (np.arange(n_length), certainty_gcr_9 [:n_length], linestyle=' ', marker='p', color=sol[colours[8]])
    # ax_r1.plot  (np.arange(n_length), certainty_gcr_10[:n_length], linestyle=' ', marker='p', color=sol[colours[9]])
    ax_r1.plot  (np.arange(n_length), got_class_wrong [:n_length], linestyle=' ', marker='x', color=sol['orange'])
    plt.savefig(home+'/Desktop/class_hitmiss.pdf', transparent=True, bbox_inches='tight')


stop = timeit.default_timer()
print('This script needed {} seconds.'.format(stop-start))
