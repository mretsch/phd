import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import timeit
from plot_2Dhist import histogram_2d

ds = xr.open_mfdataset('Data/Analysis/o_number_area_hist.nc')

phase_space = ds.hist_2D