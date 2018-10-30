import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import timeit

start = timeit.default_timer()

ds = xr.open_mfdataset(["/Users/mret0001/Data/Analysis/m1_season0910.nc",
                        "/Users/mret0001/Data/Analysis/cop_season0910.nc"])

# get rid of NaNs.
x = ds.m1.fillna(-1.)
y = ds.cop.fillna(-1.)

# Estimate the 2D histogram, range option gets rid of the original NaNs
nbins = 55
H, xedges, yedges = np.histogram2d(x, y, bins=nbins, range=[[0, x.max()], [0, y.max()]])

# H needs to transposed for correct plot
H = H.T

# Mask zeros, hence they do not show in plot
Hmasked = np.ma.masked_where(H == 0, H)

# Plot 2D histogram
fig = plt.figure()
plt.pcolormesh(xedges, yedges, Hmasked)
plt.xlabel('Metric M1')
plt.ylabel('Metric COP')
cbar = plt.colorbar()
cbar.ax.set_ylabel('Counts')

fig.show()
stop = timeit.default_timer()
print('Run Time: ', stop - start)
