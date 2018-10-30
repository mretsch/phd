import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import timeit


# # Even in conv_area_ratio are NaNs, because there are days without raw data.
# # con_area_ratio has them filled with NaNs.
# x = conv_area_ratio.fillna(0.)
# y = conv_intensity.fillna(0.)
# # Plot data
# fig1 = plt.figure()
# plt.plot(x, y, '.r')
# plt.xlabel('Conv area ratio')
# plt.ylabel('Conv intensity')
# # Estimate the 2D histogram
# nbins = 10
# H, xedges, yedges = np.histogram2d(x, y, bins=nbins)
# # H needs to be rotated and flipped
# H = np.rot90(H)
# H = np.flipud(H)
# # Mask zeros
# Hmasked = np.ma.masked_where(H == 0, H)  # Mask pixels with a value of zero
# # Plot 2D histogram using pcolor
# fig2 = plt.figure()
# plt.pcolormesh(xedges, yedges, Hmasked)
# plt.xlabel('Conv area ratio')
# plt.ylabel('Conv intensity')
# cbar = plt.colorbar()
# cbar.ax.set_ylabel('Counts')


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
