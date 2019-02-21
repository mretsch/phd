from os.path import expanduser
home = expanduser("~")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np
import artificial_fields as af
# import random_fields as rf
import org_metrics as om


ds_metric = om.run_metrics(switch={'artificial': True, 'boundary': True, 'rom': True, 'rome': True})

m1 = ds_metric.rom
m2 = ds_metric.rome

fig = plt.figure(figsize=(9, 16))  # width 9, height 10 (2 per row)
grid = AxesGrid(fig, 111,
                nrows_ncols=(8, 2),
                axes_pad=0.05
                # cbar_mode='single',
                # cbar_location='right',
                # cbar_pad=0.1
                )

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']
for i, axis in enumerate(grid):
    # ax.set_axis_off()
    # im = ax.imshow(af.art[i, :, :])

    plot = axis.contourf(af.art.lon, af.art.lat, af.art[i, :, :], levels=[1.5, 2], colors='k')

    axis.tick_params(bottom=False, labelbottom=False,
                     left=False, labelleft=False)
    axis.text(x=129.8, y=-11.05, s=alphabet[i] + ')', verticalalignment='top')
    if i != 5:
        axis.text(x=129.8, y=-13, s='ROM = ' + str(m1[i].round(decimals=2).values)
                                    + '\nROME = ' + str(m2[i].round(decimals=2).values),
                  verticalalignment='top', color='r')
    else:
        axis.text(x=129.8, y=-13, s='ROM ='
                                    + '\nROME =',
                  verticalalignment='top')
        axis.text(x=131.5, y=-13, s=str(m1[i].round(decimals=2).values)
                                    + '\n' + str(m2[i].round(decimals=2).values),
                  verticalalignment='top')
        # axis.tick_params(labelbottom=True, labelleft=True)

plt.savefig(home+'/Desktop/artificial.pdf')
plt.show()