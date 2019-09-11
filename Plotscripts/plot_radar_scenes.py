from os.path import expanduser
home = expanduser("~")
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from datetime import datetime
import timeit

start = timeit.default_timer()

plt.rc('font'  , size=16)     # 22 # 18
plt.rc('legend', fontsize=16) # 22 # 18

metric_1   = xr.open_dataarray(home+'/Data/Analysis/No_Boundary/rom_kilometres.nc')
metric_2   = xr.open_dataarray(home+'/Data/Analysis/No_Boundary/iorg.nc')
metric_3   = xr.open_dataarray(home+'/Data/Analysis/No_Boundary/cop.nc')
metric_4   = xr.open_dataarray(home+'/Data/Analysis/No_Boundary/scai.nc')
ds_steiner = xr.open_mfdataset(home+'/Data/Steiner/*season*', chunks=40)

timeselect = False
contiguous = False
if timeselect:
    start_date = '2015-11-10T03:00:00' # '2017-03-30T14:50:00' # '2009-12-07T09:10:00'
    end_date   = '2015-11-10T06:10:00' # '2017-03-30T18:00:00' # '2009-12-07T12:20:00'
    if contiguous:
        times = slice(start_date, end_date)
    else:
        # the order in next lines is not relevant, steiner_select is sorted along time coordinate
        #times = ds_steiner.indexes['time'].intersection(['2009-12-04T10:30:00', '2009-12-07T11:00:00'])
        #times = ds_steiner.indexes['time'].intersection(['2009-12-04T10:30:00', '2015-11-10T05:10:00'])
        times = ds_steiner.indexes['time'].intersection(['2013-02-24T08:40:00', '2016-11-19T14:50:00',
                                                         '2011-03-15T06:10:00', '2017-02-27T13:00:00'])

    steiner_select = ds_steiner.steiner_echo_classification.sel(time=times)
    time_select    = ds_steiner.time.sel(time=times)
    metric1_select = metric_1.sel(time=times)
    metric2_select = metric_2.sel(time=times)
    metric3_select = metric_3.sel(time=times)
    metric4_select = metric_4.sel(time=times)

    # for the 2x2 plots of radar scenes to describe limits
    limit_plots = False
    if limit_plots:
        steiner_select.load()
        dummy = steiner_select.copy(deep=True)
        steiner_select[0], steiner_select[1], steiner_select[2], steiner_select[3] = \
                 dummy[1],          dummy[2],          dummy[0],          dummy[3]

else:
    steiner = ds_steiner.steiner_echo_classification

    m_real = metric_1.where(metric_1.notnull(), drop=True)
    m_sort = m_real.sortby(m_real)

    idx_mean      = int(abs(m_sort - m_sort.mean()).argmin())
    idx_25percent = round(0.25 * len(m_sort))
    idx_33percent = round(0.33 * len(m_sort))
    idx_median    = int(len(m_sort) / 2)
    idx_66percent = round(0.66 * len(m_sort))
    idx_75percent = round(0.75 * len(m_sort))
    idx_90percent = round(0.9 * len(m_sort))
    idx_97percent = round(0.97 * len(m_sort))
    idx_99percent = round(0.99 * len(m_sort))
    idx_value     = abs(m_sort - 10.8).argmin().item()
    idx_my_select = [2, idx_25percent+1, idx_median, idx_75percent+2, -2]
    metric1_select = m_sort[idx_99percent-10:idx_99percent+10]  #[idx_my_select]  #[idx_value-10:idx_value+10]  # [idx_median-10:idx_median+10]  #[:20]  #[idx_mean-10:idx_mean+10]  #[-20:]  #[-60:-40] #

    steiner_select = steiner.loc        [metric1_select.time]
    time_select    = ds_steiner.time.loc[metric1_select.time]
    metric2_select = metric_2.loc       [metric1_select.time]

# ########
# plotting
# ########

# create mask to have line around nan-region
radar_mask = xr.where(steiner_select[0].isnull(), 1, 0)

# create own colormap. matplotlib.org/tutorials/colors/colormap-manipulation.html
vals = np.ones((256, 4))
for i in range(3):
    vals[   :100, i] = 1.0   # white
    vals[100:200, i] = 0.95  # light grey
    vals[200:   , i] = 0.0   # black
newcmp = ListedColormap(vals)

# letters to print on plots
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']
alphabet_numbered = ['a$_1$', 'b$_1$', 'a$_2$', 'b$_2$', 'c$_1$', 'c$_2$']
darwin_time = np.timedelta64(570, 'm')  # UTC + 9.5 hours

n_per_row = 5
# aspect is a hack based on measuring pixels on my screen. aspect=1 for a square plot did not work as intended.
if n_per_row == 2:
    fontsize = 19
    #p = steiner_select.plot(col='time', col_wrap=n_per_row, add_colorbar=False, aspect=700./880., size=4, cmap=newcmp)
    p = steiner_select.plot(col='time', col_wrap=n_per_row, add_colorbar=False, aspect=260./300., size=4, cmap=newcmp)
if n_per_row == 5:
    fontsize = 19
    p = steiner_select.plot(col='time', col_wrap=n_per_row, add_colorbar=False, aspect=614./754, size=4, cmap=newcmp)
    #fontsize = 22
    #p = steiner_select.plot(col='time', col_wrap=n_per_row, add_colorbar=False, aspect=684./754, size=4, cmap=newcmp)

for i, ax in enumerate(p.axes.flat):
    plt.sca(ax)  # sets the current axis ('plot') to draw to
    radar_mask.plot.contour(colors='k', linewidths=0.5, levels=1)

    #ax.set_axis_off()
    ax.set_title('')
    ax.axes.set_xlabel('')
    ax.axes.set_ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    #    # hatch the nan-region
    #    # stackoverflow.com/questions/18390068/hatch-a-nan-region-in-a-contourplot-in-matplotlib#18403408
    #    xmin, xmax = ax.get_xlim()
    #    ymin, ymax = ax.get_ylim()
    #    xy = (xmin, ymin)
    #    width = xmax - xmin
    #    height = ymax - ymin
    #    patch = patches.Rectangle(xy, width, height, hatch='xxxx', color='lightgrey', fill=None, zorder=-10)
    #    ax.add_patch(patch)

    if not timeselect:
        ax.text(x=129.8, y=-11.0, s=alphabet[0] + ')', verticalalignment='top', fontdict={'fontsize': fontsize})
    if not contiguous:
        ax.text(x=129.8, y=-11.0, s=str(alphabet_numbered[0]) + ')', verticalalignment='top', fontdict={'fontsize': fontsize})
        #if i == 0:
        #    ax.text(x=129.8, y=-11.0, s='a)', verticalalignment='top', fontdict={'fontsize': 18})
        # ax.text(x=131.75, y=-11.05, s=str((time_select[i] + darwin_time).values)[11:16]+' h', verticalalignment='top'
        #         , fontdict={'fontsize': 16})
        #ax.set_title(str((time_select[i]).values)[11:16]+'h')

# all the bottom plots
for ax in p.axes.flat[-n_per_row:]:
    ax.spines['bottom'].set_visible(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.axes.set_xlabel('Longitude [$^\circ$E]', fontdict={'fontsize': fontsize})
    ax.axes.set_xticks([130, 130.5, 131, 131.5, 132])
    ax.axes.set_xticklabels(labels=['130', '', '131', '', '132'], fontdict={'fontsize': fontsize})

# all the left plots
for i in np.arange(0, len(p.axes.flat), n_per_row):
    p.axes.flat[i].spines['left'].set_visible(True)
    p.axes.flat[i].yaxis.set_ticks_position('left')
    p.axes.flat[i].axes.set_ylabel('Latitude [$^\circ$S]', fontdict={'fontsize': fontsize})
    p.axes.flat[i].axes.set_yticklabels(labels=['xxx', '', '13', '', '12', '', '11'], fontdict={'fontsize': fontsize})

# Print some information on plots
percentiles = True
as_coordinate = True
if percentiles:
    if as_coordinate:
        perct1 = metric1_select.percentile * 100
        perct2 = metric2_select.percentile * 100
    else:
        perct1 = metric_1.rank(dim='time', pct=True).sel(time=slice(start_date, end_date)) * 100
        perct2 = metric_2.rank(dim='time', pct=True).sel(time=slice(start_date, end_date)) * 100

print1 = metric1_select
print2 = metric2_select

print_numbers = False
if print_numbers:
    for i, ax in enumerate(p.axes.flat):
        ax.annotate('ROM: {:5.1f}\n'
                    'SIC: {:5.1f}'.format(print1[i].item(),
                                          print2[i].item()), (131.78, -11.2), color='blue')
        if percentiles:
            ax.annotate('ROM: {:3.0f}%\n'
                        'SIC: {:3.0f}%'.format(perct1[i].item(),
                                               perct2[i].item()), (131.78, -13.5), color='blue')

save = False
if save:
    plt.savefig(home+'/Desktop/radar_scenes1.pdf', transparent=True, bbox_inches='tight')
else:
    plt.show()

# Have all scenes separately as a pdf to create a gif
# for i, scene in enumerate(steiner_select):
#     plt.close()
#     scene.plot()
#     plt.savefig('/Users/mret0001/Desktop/'+str(i)+'.pdf')

stop = timeit.default_timer()
print('This script needed {} seconds.'.format(stop - start))