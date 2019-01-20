import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# parameters to indirectly control the number and size of random objects
# times to play Game of Life, growing objects at each iteration
n_steps = 60
# take randomly distributed pixels of interval [0,1) above this threshold
cut_off = 0.95

radar = xr.open_dataarray('/Users/matthiasretsch/Google Drive File Stream/My Drive/Data/CPOL_STEINER_ECHO_CLASSIFICATION_threedays.nc')
# radar = xr.open_dataarray('/Users/mret0001/Data/Steiner/CPOL_STEINER_ECHO_CLASSIFICATION_threedays.nc')

rand = np.random.random(radar.shape)
rand = np.where(rand > cut_off, True, False)

# implement Game of Life rules, to extinct solo positive pixels. Yet spare the rule
# of overpopulation, to generate as large objects as possible.
# Code from https://jakevdp.github.io/blog/2013/08/07/conways-game-of-life/
def game_of_life_step(array):
    """Modified Game of life step using generator expressions. Cells do not die due to overpopulation."""
    # np.roll shifts an array (toroidal) in different directions. Hence we shift each adjacent pixel onto
    # our pixel of interest and then create an iterable of (shifted) arrays and sum() them element-wise together.
    array_nbrs_count = sum(np.roll(np.roll(array, i, 0), j, 1)
                           for i in (-1, 0, 1)
                           for j in (-1, 0, 1)
                           if (i != 0 or j != 0))
    # modified rules, do not die due to overpopulation (more than 3 neighbours)
    return (array_nbrs_count == 3) | (array & (array_nbrs_count >= 2))


for i, field in enumerate(rand):
    for _ in range(n_steps):
        # rand_o[i], field = field, game_of_life_step(field)
        rand[i, :, :], field = field, game_of_life_step(field)

# rand_objects = xr.where(radar.isnull(), radar, rand*2)


# t1, t2 = xr.broadcast(radar[300, :, :], xr.zeros_like(radar))
# t1.shape
# t1[:, :, 0].isnull().sum()