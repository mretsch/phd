import xarray as xr
import numpy as np


radar = xr.open_dataarray('/Users/matthiasretsch/Google Drive File Stream/My Drive/Data/CPOL_STEINER_ECHO_CLASSIFICATION_threedays.nc')

rand = np.random.random(radar.shape)
rand = np.where(rand > 0.7, 2., 0.)
field = xr.where(radar.isnull(), radar, rand)

# implement Game of Life rules, to extinct the solo positive pixels. Yet spare the rule
# of overpopulation, to generate as large objects as possible.
# Code from https://jakevdp.github.io/blog/2013/08/07/conways-game-of-life/


def life_step_1(X):
    """Game of life step using generator expressions"""
    nbrs_count = sum(np.roll(np.roll(X, i, 0), j, 1)
                     for i in (-1, 0, 1)
                     for j in (-1, 0, 1)
                     if (i != 0 or j != 0))
    return (nbrs_count == 3) | (X & (nbrs_count == 2))
