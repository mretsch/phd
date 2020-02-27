from os.path import expanduser
home = expanduser("~")
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import cartopy.io as cio
from datetime import datetime
import timeit

start = timeit.default_timer()

plt.rc('font'  , size=16)     # 22 # 18
plt.rc('legend', fontsize=16) # 22 # 18

metric_1   = xr.open_dataarray(home+'/Data/Analysis/No_Boundary/rom_kilometres.nc')
metric_2   = metric_1 # xr.open_dataarray(home+'/Data/Analysis/No_Boundary/iorg.nc')
metric_3   = metric_1 # xr.open_dataarray(home+'/Data/Analysis/No_Boundary/cop.nc')
metric_4   = metric_1 # xr.open_dataarray(home+'/Data/Analysis/No_Boundary/scai.nc')
ds_steiner = xr.open_mfdataset(home+'/Data/Steiner/*season*', chunks=40)

timeselect = True
contiguous = True
if timeselect:
    start_date = '2015-11-10T03:00' # '2005-03-11T15:10:00' # '2017-03-30T14:50:00' # '2009-12-07T09:10:00'
    end_date   = '2015-11-10T06:10' # '2005-03-11T20:50:00' # '2017-03-30T18:00:00' # '2009-12-07T12:20:00'
    if contiguous:
        times = slice(start_date, end_date)
    else:
        # the order in next lines is not relevant, intersection is sorted along time coordinate
        # times = ds_steiner.indexes['time'].intersection(['2009-12-22T06:20:00', '2009-12-07T11:00:00'])
        times = [np.datetime64('2009-12-22T06:20:00'), np.datetime64('2009-12-07T11:00:00')]
        # times = [np.datetime64('2013-02-24T08:40:00'), np.datetime64('2016-11-19T14:50:00'),
        #          np.datetime64('2011-03-15T06:10:00'), np.datetime64('2017-02-27T13:00:00')]

    steiner_select = ds_steiner.steiner_echo_classification.sel(time=times)
    time_select    = ds_steiner.time.sel(time=times)
    metric1_select = metric_1.sel(time=times)
    metric2_select = metric_2.sel(time=times)
    metric3_select = metric_3.sel(time=times)
    metric4_select = metric_4.sel(time=times)

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
    metric1_select = m_sort[idx_my_select]  #[idx_99percent-10:idx_99percent+10]  #[idx_value-10:idx_value+10]
                                            # [idx_median-10:idx_median+10]  #[:20]
                                            #[idx_mean-10:idx_mean+10]  #[-20:]  #[-60:-40] #

    steiner_select = steiner.loc        [metric1_select.time]
    time_select    = ds_steiner.time.loc[metric1_select.time]
    metric2_select = metric_2.loc       [metric1_select.time]

# ########
# plotting
# ########

# create mask to have line around nan-region
radar_mask = xr.where(steiner_select[0].isnull(), 1, 0)

# create own colormap. matplotlib.org/tutorials/colors/colormap-manipulation.html
vals = np.ones((256, 4)) # 256 different colors in this colormap
for i in range(3):
    vals[   :100, i] = 1.0        # white
    vals[100:200, i] = 1.0        # white
    vals[200:   , 0] =   0 / 255  # blue, r
    vals[200:   , 1] = 162 / 255  # blue, g
    vals[200:   , 2] = 255 / 255  # blue, b
vals[:200, 3] = 0. # transparent white
transparent_blue = ListedColormap(vals)

# another colormap
vals = np.ones((256, 4)) # 256 different colors in this colormap
for i in range(3):
    vals[   :100, i] = 1.0  # white
    vals[100:200, i] = 1.0  # white
    vals[200:   , 0] = 1.0  # white, r
    vals[200:   , 1] = 1.0  # white, g
    vals[200:   , 2] = 1.0  # white, b
vals[:200, 3] = 0. # transparent white
transparent_white = ListedColormap(vals)

# letters to print on plots
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']
alphabet_numbered = ['a$_1$', 'b$_1$', 'a$_2$', 'b$_2$', 'c$_1$', 'c$_2$']
title_numbers = ['$\mathcal{R}_\mathrm{NI}$=130, ROME=157', '$\mathcal{R}_\mathrm{min}$=129, ROME=191',
                 '$\mathcal{R}_\mathrm{NI}$=126, ROME=127', '$\mathcal{R}_\mathrm{min}$=130, ROME=250']
darwin_time = np.timedelta64(570, 'm')  # UTC + 9.5 hours

n_per_row = 4

fontsize = 19
fig, axes_list = plt.subplots(nrows=5, ncols=4, figsize=(1.4* 13.4, 1.4* 16),
                              subplot_kw=dict(projection=ccrs.PlateCarree()))

plt.subplots_adjust(hspace = 0.05, wspace=0.01)

for i, ax in enumerate(axes_list.flatten()):
    plt.sca(ax)  # sets the current axis ('plot') to draw to

    # Out of radar range, paint over in white. Inside leave transparent.
    ax.pcolormesh(ds_steiner.lon, ds_steiner.lat, radar_mask, zorder=2000, cmap=transparent_white)

    # Paint a line at the radar range boundary
    radar_mask.plot.contour(colors='k', linewidths=0.5, levels=1, zorder=5000)

    # draw the coast-line
    lons = ds_steiner.lon[:]
    lats = ds_steiner.lat[:]
    width = 0  # degrees east,west,north,south
    extent = [min(lons) - width, max(lons) + width, min(lats) - width, max(lats) + width]  # Darwin [lon,lon,lat,lat]
    # set_extent speeds up the drawing of all the land polygons
    # ax.set_extent(extent)

    shp = cio.shapereader.Reader(home+'/Data/OpenStreetMaps/land-polygons-complete-4326/Australia_polygons')
    for record, geometry in zip(shp.records(), shp.geometries()):
        ax.add_geometries([geometry], ccrs.PlateCarree(), facecolor='white', edgecolor='black')

    # Plot the actual convective objects in blue and leave the rest transparent, so coastlines are visible
    ax.pcolormesh(lons, lats, steiner_select[i], zorder=1000, cmap=transparent_blue)
                   # color=[(0.1, 0.8, 0.5, 0.2), (0.9, 0.1, 0.5, 0.2)])

    # ax.set_axis_off()

    ax.set_title('')
    ax.axes.set_xlabel('')
    ax.axes.set_ylabel('')
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    # ax.xaxis.set_ticks_position('none')
    # ax.yaxis.set_ticks_position('none')

    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    if timeselect and not contiguous:
        # ax.text(x=129.8, y=-11.0, s=str(alphabet_numbered[i]) + ')', verticalalignment='top', fontdict={'fontsize': fontsize})
        # ax.text(x=129.8, y=-11.0, s=alphabet[i] + ')', verticalalignment='top', fontdict={'fontsize': fontsize})
        ax.set_title(str(alphabet_numbered[i]) + ')   ' + title_numbers[i])
    if contiguous:
        ax.text(x=131.75, y=-11.05, s=str((time_select[i] + darwin_time).values)[11:16]+' h', verticalalignment='top'
                , fontdict={'fontsize': 16}, zorder=6000)

# all the bottom plots
for ax in axes_list.flatten()[-n_per_row:]:
    ax.spines['bottom'].set_visible(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.axes.set_xlabel('Longitude [$^\circ$E]', fontdict={'fontsize': fontsize})
    ax.axes.set_xticks([130, 130.5, 131, 131.5, 132])
    ax.axes.set_xticklabels(labels=['130', '', '131', '', '132'], fontdict={'fontsize': fontsize})

# all the left plots
for i in np.arange(0, len(axes_list.flatten()), n_per_row):
    axes_list.flatten()[i].spines['left'].set_visible(True)
    axes_list.flatten()[i].yaxis.set_ticks_position('left')
    axes_list.flatten()[i].axes.set_ylabel('Latitude [$^\circ$S]', fontdict={'fontsize': fontsize})
    axes_list.flatten()[i].axes.set_yticks([-11, -11.5, -12, -12.5, -13, -13.5])
    # axes_list.flatten()[i].axes.set_yticklabels(labels=['', '13', '', '12', '', '11'], fontdict={'fontsize': fontsize})
    axes_list.flatten()[i].axes.set_yticklabels(labels=['11', '', '12', '', '13', ''], fontdict={'fontsize': fontsize})

# Print some information on plots
percentiles = False
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
    for i, ax in enumerate(axes_list.flatten()):
        ax.annotate('ROME: {:5.1f}'.format(print1[i].item()), (131.78, -11.2), color='blue')
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