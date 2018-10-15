import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np
import artificial_fields as af
import org_metrics as om

fig = plt.figure(figsize=(6, 4))
grid = AxesGrid(fig, 111,
                nrows_ncols=(3, 2),
                axes_pad=0.05
                # cbar_mode='single',
                # cbar_location='right',
                # cbar_pad=0.1
                )

for i, axis in enumerate(grid):
    # ax.set_axis_off()
    # im = ax.imshow(af.art[i, :, :])
    plot = axis.contourf(af.art.lon, af.art.lat, af.art[i, :, :], levels=[0.5, 1], colors=('k'))
    axis.tick_params(bottom=False, labelbottom=False,
                     left=False, labelleft=False)
    axis.text(x=130, y=-12, s='HELLO')

cop, m1 = om.run_metrics(artificial=True)

plt.show()