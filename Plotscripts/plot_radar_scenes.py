from os.path import expanduser
home = expanduser("~")
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import timeit

start = timeit.default_timer()

plt.rc('font'  , size=12)
plt.rc('legend', fontsize=12)

try:
    metric_1   = xr.open_dataarray(home+'/Data/Analysis/No_Boundary/rom.nc')
    metric_2   = xr.open_dataarray(home+'/Data/Analysis/No_Boundary/sic.nc')
    ds_steiner = xr.open_mfdataset(home+'/Data/Steiner/*season*', chunks=40)
except FileNotFoundError:
    metric_1 = xr.open_dataarray(home+'/Google Drive File Stream/My Drive/Data_Analysis/rom.nc')#\
    #sel({'time': slice('2009-10-01', '2010-03-31')})
    metric_2 = xr.open_dataarray(home+'/Google Drive File Stream/My Drive/Data_Analysis/sic.nc')
    ds_steiner = xr.open_mfdataset(home+'/Google Drive File Stream/My Drive/Data/Steiner/*season*')

consecutive = True
if consecutive:
    start_date = '2009-11-30T04:40:00'
    end_date   = '2009-11-30T23:50:00'
    #times = slice(start_date, end_date)
    times = ds_steiner.indexes['time'].intersection(['2009-12-04T10:30:00','2009-12-07T11:00:00'])

    steiner_select = ds_steiner.steiner_echo_classification.sel(time=times)
    metric1_select = metric_1.sel(time=times)
    metric2_select = metric_2.sel(time=times)

else:
    steiner = ds_steiner.steiner_echo_classification

    m_real = metric_1.where(metric_1.notnull(), drop=True)
    m_sort = m_real.sortby(m_real)

    idx_median = int(len(m_sort) / 2)
    idx_mean = int(abs(m_sort - m_sort.mean()).argmin())
    idx_33percent = round(0.33 * len(m_sort))
    idx_66percent = round(0.66 * len(m_sort))
    idx_90percent = round(0.9 * len(m_sort))
    idx_99percent = round(0.99 * len(m_sort))
    idx_value = abs(m_sort - 10.8).argmin().item()
    idx_my_select = [0, idx_33percent-1, idx_66percent-1, -1]
    metric1_select = m_sort[idx_my_select]  #[idx_99percent-10:idx_99percent+10]  #[idx_value-10:idx_value+10]  # [idx_median-10:idx_median+10]  #[:20]  #[idx_mean-10:idx_mean+10]  #[-20:]  #[-60:-40] #

    steiner_select = steiner.loc[metric1_select.time]
    metric2_select = metric_2.loc[metric1_select.time]

# plotting

# create mask to have line around nan-region
radar_mask = xr.where(steiner_select[0].isnull(), 1, 0)

# create own colormap. matplotlib.org/tutorials/colors/colormap-manipulation.html
vals = np.ones((256, 4))
for i in range(3):
    vals[   :100, i] = 1.0  # white
    vals[100:200, i] = 0.95  # light grey
    vals[200:   , i] = 0.0  # black
newcmp = ListedColormap(vals)

# letters to print on plots
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']

# aspect is a hack based on measuring pixels on my screen. aspect=1 for a square plot did not work as intended.
p = steiner_select.plot(col='time', col_wrap=2, add_colorbar=False, aspect=450./558, size=4,
                        cmap=newcmp)
for i, ax in enumerate(p.axes.flat):
    plt.sca(ax)  # sets the current axis ('plot') to draw to
    radar_mask.plot.contour(colors='k', linewidths=0.5, levels=1)

    ax.set_title('')
    ax.axes.set_ylabel('')
    ax.axes.set_xlabel('Longitude [$^\circ$E]')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.yaxis.set_ticks_position('none')

    #    # stackoverflow.com/questions/18390068/hatch-a-nan-region-in-a-contourplot-in-matplotlib#18403408
    #    xmin, xmax = ax.get_xlim()
    #    ymin, ymax = ax.get_ylim()
    #    xy = (xmin, ymin)
    #    width = xmax - xmin
    #    height = ymax - ymin
    #    patch = patches.Rectangle(xy, width, height, hatch='xxxx', color='lightgrey', fill=None, zorder=-10)
    #    ax.add_patch(patch)

    if not consecutive:
        ax.text(x=129.8, y=-11.0, s=alphabet[i] + ')', verticalalignment='top')

p.axes.flat[0].spines['left'].set_visible(True)
p.axes.flat[0].yaxis.set_ticks_position('left')
p.axes.flat[0].axes.set_ylabel('Latitude [$^\circ$S]')
p.axes.flat[0].axes.set_yticklabels(labels=['14.0', '13.5', '13.0', '12.5', '12.0', '11.5', '11.0', '10.5'])

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

save = True
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