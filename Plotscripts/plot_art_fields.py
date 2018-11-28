import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np
import artificial_fields as af
import org_metrics as om

ds_metric = om.run_metrics(artificial=True)

cop = ds_metric.cop
m1 = ds_metric.m1
cop_mod = ds_metric.cop_mod

fig = plt.figure(figsize=(9, 8))
grid = AxesGrid(fig, 111,
                nrows_ncols=(4, 2),
                axes_pad=0.05
                # cbar_mode='single',
                # cbar_location='right',
                # cbar_pad=0.1
                )

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
for i, axis in enumerate(grid):
    # ax.set_axis_off()
    # im = ax.imshow(af.art[i, :, :])

    plot = axis.contourf(af.art.lon, af.art.lat, af.art[i, :, :], levels=[0.5, 1], colors='k')

    axis.tick_params(bottom=False, labelbottom=False,
                     left=False, labelleft=False)
    axis.text(x=129.8, y=-11.05, s=alphabet[i] + ')', verticalalignment='top')
    if i != 4:
        axis.text(x=129.8, y=-13, s='COP = ' + str(cop[i].round(decimals=2).values)
                                    + '\nCAM = ' + str(cop_mod[i].round(decimals=2).values),
                  verticalalignment='top')
    else:
        axis.text(x=129.8, y=-13, s='COP ='
                                    + '\nCAM =',
                  verticalalignment='top')
        axis.text(x=131.5, y=-13, s=str(cop[i].round(decimals=2).values)
                                    + '\n' + str(cop_mod[i].round(decimals=2).values),
                  verticalalignment='top')
        # axis.tick_params(labelbottom=True, labelleft=True)

plt.savefig('/Users/mret0001/Desktop/cop_m1_artificial.pdf')
plt.show()