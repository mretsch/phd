from os.path import expanduser
home = expanduser("~")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import artificial_fields as af
# import random_fields as rf
import org_metrics as om
from Plotscripts.colors_solarized import sol

plt.rc('font'  , size=12)
plt.rc('legend', fontsize=12)

ds_metric = om.run_metrics(switch={'artificial': True, 'boundary': True, 'rom': True, 'rom_li': True,
                                   'iorg': True, 'cop': True, 'scai': True, 'basics': False})

m1 = ds_metric.rom
m2 = ds_metric.cop

fig = plt.figure(figsize=(12, 9))  # width 9, height 10 (2 per row)
grid = AxesGrid(fig, 111,
                nrows_ncols=(2, 6),
                axes_pad=0.05
                # cbar_mode='single',
                # cbar_location='right',
                # cbar_pad=0.1
                )

alphabet = ['a$_1$', 'a$_2$', 'b$_1$', 'b$_2$', 'c$_1$', 'c$_2$', 'd$_1$', 'd$_2$',
            'e$_1$', 'e$_2$', 'f$_1$', 'f$_2$', 'g$_1$', 'g$_2$', 'h$_1$', 'h$_2$']
for i, _ in enumerate(grid):
    # order of plots when numbered row through row
    h = int(2*6/2 - 1)  # half the plots are in one of two rows
    idx = [h-h, h+1, h-4, h+2, h-3, h+3, h-2, h+4, h-1, h+5, h, h+6]
    k = idx[i]
    axis = grid[k]

    plot = axis.contourf(af.art.lon, af.art.lat, af.art[i, :, :], levels=[1.5, 2], colors='k')

    axis.tick_params(bottom=False, labelbottom=False,
                     left=False, labelleft=False)

    axis.text(x=129.8, y=-11.05, s=alphabet[i] + ')', verticalalignment='top')
    #    if i != 5:
    #        axis.text(x=129.8, y=-11.1, s='ROME = ' + str(m1[i].astype('int').values)
    #                                    + '\nCOP  = ' + str(m2[i].round(decimals=2).values),
    #                  verticalalignment='top', color=sol['red'], fontdict={'family' : 'monospace'})
    #    else:
    #        axis.text(x=129.8, y=-13, s='ROM ='
    #                                    + '\nROME =',
    #                  verticalalignment='top')
    #        axis.text(x=131.5, y=-13, s=str(m1[i].round(decimals=2).values)
    #                                    + '\n' + str(m2[i].round(decimals=2).values),
    #                  verticalalignment='top')
    # axis.tick_params(labelbottom=True, labelleft=True)

plt.savefig(home+'/Desktop/artificial.pdf', bbox_inches='tight', transparent=True)
plt.show()