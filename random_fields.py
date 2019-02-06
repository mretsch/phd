import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import timeit

start = timeit.default_timer()

# parameters to indirectly control the number and size of random objects
# times to play Game of Life, growing objects at each iteration
# n_steps = 10  # 60
# take randomly distributed pixels of interval [0,1) above this threshold
# threshold = 0.90

# radar = xr.open_dataarray('/Users/matthiasretsch/Google Drive File Stream/My Drive/Data/CPOL_STEINER_ECHO_CLASSIFICATION_threedays.nc')
steiner = xr.open_dataarray('/Users/mret0001/Data/Steiner/CPOL_STEINER_ECHO_CLASSIFICATION_threedays.nc')

np.random.seed(seed=71897190)
real_rand = np.random.random(steiner.shape)
rand = np.empty_like(real_rand, dtype='bool')

for i, array in enumerate(real_rand):
    threshold = np.random.random() * 0.15 + 0.85  # random number between 0.85 and 1
    rand[i, :, :] = np.where(array > threshold, True, False)

# implement Game of Life rules, to extinct solo positive pixels. Yet spare the rule
# of overpopulation, to generate as large objects as possible.
# Code from https://jakevdp.github.io/blog/2013/08/07/conways-game-of-life/
def game_of_life_step(array, fill=False):
    """Modified Game of life step using generator expressions. Cells do not die due to overpopulation."""
    # np.roll shifts an array (toroidal) in different directions. Hence we shift each adjacent pixel onto
    # our pixel of interest and then create an iterable of (shifted) arrays and sum() them element-wise together.
    array_nbrs_count = sum(np.roll(np.roll(array, i, 0), j, 1)
                           for i in (-1, 0, 1)
                           for j in (-1, 0, 1)
                           if (i != 0 or j != 0))
    if fill:
        return array_nbrs_count >= 4
    else:
        # modified rules, do not die due to overpopulation (more than 3 neighbours)
        return (array_nbrs_count == 3) | (array & (array_nbrs_count >= 2))


for i, array in enumerate(rand):
    n_steps = round(np.random.random() * 45 + 5)  # random number between 5 and 50
    for _ in range(n_steps):
        # rand_o[i], field = field, game_of_life_step(field)
        rand[i, :, :], array = array, game_of_life_step(array, fill=False)

    rand[i, :, :] = game_of_life_step(array, fill=True)

nan_mask, dummy = xr.broadcast(steiner[28, :, :], xr.zeros_like(steiner))
nan_mask_t = nan_mask.transpose('time', 'lat', 'lon')
rand_objects = xr.where(nan_mask_t.isnull(), nan_mask_t, rand*2)

save = False
if save:
    rand_objects.to_netcdf('/Users/mret0001/Desktop/random_scenes.nc')

stop = timeit.default_timer()
print('Random field script needed {} seconds.'.format(stop - start))
