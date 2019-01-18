import xarray as xr
import numpy as np

# parameters to indirectly control the number and size of random objects
# times to play Game of Life, growing objects at each iteration
n_steps = 10
# take randomly distributed pixels of interval [0,1) above this threshold
cut_off = 0.9

# radar = xr.open_dataarray('/Users/matthiasretsch/Google Drive File Stream/My Drive/Data/CPOL_STEINER_ECHO_CLASSIFICATION_threedays.nc')
radar = xr.open_dataarray('/Users/mret0001/Data/Steiner/CPOL_STEINER_ECHO_CLASSIFICATION_threedays.nc')

rand = np.random.random(radar.shape)
rand = np.where(rand > cut_off, True, False)

# implement Game of Life rules, to extinct the solo positive pixels. Yet spare the rule
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
    # Game of Life rules: create for 3 neighbours, maintain for 2 or 3, die for everything else.
    # return (array_nbrs_count == 3) | (array & (array_nbrs_count == 2))
    # modified rules, do not die due to overpopulation (more than 3 neighbours)
    return (array_nbrs_count == 3) | (array & (array_nbrs_count >= 2))


# test field
# arr = np.array([[[False, False, False, False, False],
#                  [False, True , True , True , False],
#                  [False, False, True , True , False],
#                  [False, False, False, True , False],
#                  [True , False, False, False, False]],
#                 [[False, False, False, False, False],
#                  [False, True , True , True , False],
#                  [False, False, True , True , False],
#                  [False, False, False, True , False],
#                  [True , False, False, False, False]]])

for i, field in enumerate(rand):
    for _ in range(n_steps):
        # rand_o[i], field = field, game_of_life_step(field)
        rand[i, :, :], field = field, game_of_life_step(field)

rand_objects = xr.where(radar.isnull(), radar, rand*2)