import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np
import artificial_fields as af

fig = plt.figure(figsize=(6, 4))
grid = AxesGrid(fig, 111,
                nrows_ncols=(3, 2),
                axes_pad=0.05,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.1
                )
for i, ax in enumerate(grid):
    ax.set_axis_off()
    # im = ax.imshow(np.random.random((16, 16)), vmin=0, vmax=1)
    im = ax.imshow(af.art[i, :, :])
    # wtf = ax.contour(af.art.lat,af.art.lon,af.art[i, :, :])

plt.show()