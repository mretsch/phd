import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import timeit
from plot_2Dhist import histogram_2d

ds_ps = xr.open_mfdataset('/Users/mret0001/Data/Analysis/o_number_area_hist.nc')
ds = xr.open_mfdataset('/Users/mret0001/Data/Analysis/conv_rr_mean.nc')

phase_space = ds_ps.hist_2D
overlay = ds.conv_rr_mean

# assume I have the two time series telling me at which time I have which x-y-position in the phase-space
x_bins = ds_ps.x_series_bins
y_bins = ds_ps.y_series_bins




