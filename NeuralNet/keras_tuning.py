from os.path import expanduser
import keras.layers as klayers
from tensorflow import keras
from keras_tuner import RandomSearch
from keras_tuner import HyperModel
from regre_net import input_output_for_mlp
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

home = expanduser("~")

class MyHyperModel(HyperModel):
    def __init__(self, n_inputs):
        self.n_inputs = n_inputs

    def build(self, hp):
        model = keras.Sequential()
        n_nodes = hp.Choice(f"nodes", [30, 100, 300, 1000, 3000,])  # 300
        for i in range(hp.Int("num_layers", 2, 5)):  # 3

            if i == 1:
                model.add(klayers.Dense(units=n_nodes, activation="relu", input_shape=(self.n_inputs,)))
            else:
                model.add(klayers.Dense(units=n_nodes, activation="relu"))

        model.add(klayers.Dense(1, activation="linear"))

        model.compile(
            # optimizer=keras.optimizers.Adam(hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4, 1e-5, ])),
            optimizer=keras.optimizers.Adam(1e-4),
            loss="mean_squared_error",
            # metrics=["accuracy"],
        )
        return model


largescale_times = 'same_time'
l_profiles_as_eof = True
predictor, target, metric, height_dim = input_output_for_mlp(ls_times=largescale_times,
                                                             l_profiles_as_eof=l_profiles_as_eof,
                                                             target='tca')

##################################
# Slice and dice the whole dataset
##################################

# target = target.where(target<300, other=300)

# predictor = xr.concat((predictor[:2450], predictor[2750:]), dim='time')
# target = xr.concat((target[:2450], target[2750:]), dim='time')
# predictor = xr.concat((predictor[:2496], predictor[2635:]), dim='time')
# target = xr.concat((target[:2496], target[2635:]), dim='time')

# predictor = xr.concat((predictor.sel(time=slice(None        , '2009-12-15')),
#                        predictor.sel(time=slice('2010-01-21', None))), dim='time')
# target = xr.concat((target.sel(time=slice(None        , '2009-12-15')),
#                     target.sel(time=slice('2010-01-21', None))), dim='time')

##################
# Set up the tuner
##################

n_lev = len(predictor[height_dim])

hypermodel = MyHyperModel(n_inputs=n_lev)

tuner = RandomSearch(
    hypermodel,
    objective="val_loss",
    max_trials=25,
    executions_per_trial=3,
    overwrite=False,
    directory=home+"/Desktop/KerasTuning/",
    project_name="project1",
)

################################################################
# Chunk the whole dataset into training, validation and test set
################################################################

n_samples      = len(predictor)
n_trainsamples = round(0.8*n_samples)
n_valsamples   = n_samples - n_trainsamples
n_testsamples  = round(0.1*n_samples)

all_indices        = np.arange(n_samples)
l_every_tenth      = (all_indices % 10) == 3
l_20percent_of_all = np.isin(all_indices % 10, [8, 9])
l_70percent_of_all = np.logical_and(np.logical_not(l_every_tenth), np.logical_not(l_20percent_of_all))
assert l_every_tenth.sum() + l_20percent_of_all.sum() + l_70percent_of_all.sum() == n_samples

testset  = (predictor[l_every_tenth]     , target[l_every_tenth])
valset   = (predictor[l_20percent_of_all], target[l_20percent_of_all])
trainset = (predictor[l_70percent_of_all], target[l_70percent_of_all])

tuner.search(trainset[0], trainset[1],
             epochs=100,
             validation_data=(valset[0], valset[1]))

# tuner.search(predictor[:n_samples//2], target[:n_samples//2],
#              epochs=20,
#              validation_data=(predictor[n_samples//2:], target[n_samples//2:]))
# tuner.search(predictor[n_samples//2:], target[n_samples//2:],
#              epochs=20,
#              validation_data=(predictor[:n_samples//2], target[:n_samples//2]))
# tuner.search(predictor[-n_valsamples:], target[-n_valsamples:],
#              epochs=20,
#              validation_data=(predictor[-n_valsamples:], target[-n_valsamples:]))
# tuner.search(predictor[np.arange(0, len(predictor), 2)], target[np.arange(0, len(predictor), 2)],
#              epochs=100,
#              validation_data=(predictor[np.arange(1, len(predictor), 2)], target[np.arange(1, len(predictor), 2)]))
# tuner.search(predictor[n_testsamples:n_trainsamples], target[n_testsamples:n_trainsamples],
#              epochs=20,
#              validation_data=(predictor[-n_valsamples:], target[-n_valsamples:]))

# n_sample_per_day = 4
# period_indicator = np.arange(n_samples) // (3 * n_sample_per_day)
# l_even_odd = (period_indicator % 2).astype('bool')
# tuner.search(predictor[l_even_odd], target[l_even_odd],
#              epochs=100,
#              validation_data=(predictor[np.logical_not(l_even_odd)], target[np.logical_not(l_even_odd)]))

# random_generator = np.random.default_rng(50063171011904362696069285316491504896)
# random_generator = np.random.default_rng(254687641397713365954774546647777894534)
# rand_indices = random_generator.permutation(np.arange(n_samples))
# indices_training   = rand_indices[:n_trainsamples ]
# indices_validation = rand_indices[ n_trainsamples:]
# tuner.search(predictor[indices_training], target[indices_training],
#              epochs=10,
#              validation_data=(predictor[indices_validation], target[indices_validation]))
