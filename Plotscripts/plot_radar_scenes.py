from os.path import expanduser
from pathlib import Path
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


def plot_multiple_radar_scenes(large_scale_date):

    print(large_scale_date.astype(str)[:13])

    day_string = str(large_scale_date)[:10]
    lookup_string = day_string.replace('-', '')
    assert lookup_string.isdigit()
    lookup_daybefore = str(np.datetime64(day_string) - np.timedelta64(1, 'D')).replace('-', '')
    lookup_dayafter  = str(np.datetime64(day_string) + np.timedelta64(1, 'D')).replace('-', '')

    # find steiner files
    steiner_path = Path(home) / 'Documents' / 'Data' / 'Steiner'
    steiner_files = []
    for string in [lookup_daybefore, lookup_string, lookup_dayafter]:
        search_pattern = '*' + string + '.nc'
        steiner_files.append(next(steiner_path.rglob(f'{search_pattern}')))

    metric_1   = xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/rom_kilometres.nc')
    # delta_prox
    metric_2   = metric_1 - \
                 xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/rom_low_limit.nc') * 6.25
    # delta_size
    metric_3   = xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/rom_low_limit.nc') * 6.25 - \
                 xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/o_area.nc') * 6.25
    ds_steiner = xr.open_mfdataset(steiner_files, chunks=40)

    timeselect = True
    contiguous = True
    if timeselect:
        # start_date = '2015-11-10T03:00:00' # '2006-01-09T09:00' # '2017-03-30T14:50:00' # '2009-12-07T09:10:00'
        # end_date   = '2015-11-10T06:10:00' # '2006-01-09T14:50' # '2017-03-30T18:00:00' # '2009-12-07T12:20:00'
        # start_date = '2003-03-13T15:10:00' # '2004-11-14T03:10:00' # '2015-03-14T21:10:00' # '2004-12-12T03:10:00' #
        # end_date   = '2003-03-13T21:00:00' # '2004-11-14T09:00:00' # '2015-03-15T03:00:00' # '2004-12-12T09:00:00' #
        # start_date = '2007-03-02T09:10:00' # '2007-01-30T09:10:00' # '2003-12-19T09:10:00' # '2006-01-19T21:10' #
        # end_date   = '2007-03-02T15:00:00' # '2007-01-30T15:00:00' # '2003-12-19T15:00:00' # '2006-01-20T03:00' #
        start_date = large_scale_date - np.timedelta64(170, 'm') # '2003-02-06T03:10:00' # '2004-02-23T03:10:00'
        end_date   = large_scale_date + np.timedelta64(  3, 'h') # '2003-02-06T09:00:00' # '2004-02-23T09:00:00'

        if contiguous:
            times = slice(start_date, end_date)
        else:
            # the order in next lines is not relevant, intersection is sorted along time coordinate
            # times = ds_steiner.indexes['time'].intersection(['2009-12-04T10:30:00', '2009-12-07T11:00:00'])
            times = [np.datetime64('2009-12-22T06:20:00'), np.datetime64('2009-12-07T11:00:00')]
            # times = [np.datetime64('2013-02-24T08:40:00'), np.datetime64('2016-11-19T14:50:00'),
            #          np.datetime64('2011-03-15T06:10:00'), np.datetime64('2017-02-27T13:00:00')]

        steiner_select = ds_steiner.steiner_echo_classification.sel(time=times)
        time_select    = ds_steiner.time.sel(time=times)

        metric1_select = metric_1.sel(time=times)
        metric2_select = metric_2.sel(time=times)
        metric3_select = metric_3.sel(time=times)

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
        metric1_select = m_sort[idx_my_select]  #[idx_99percent-10:idx_99percent+10]  #[idx_value-10:idx_value+10]  # [idx_median-10:idx_median+10]  #[:20]  #[idx_mean-10:idx_mean+10]  #[-20:]  #[-60:-40] #

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
        vals[100:200, i] = 0.9  # light grey
        # vals[100:200, i] = 1.0   # white
        # vals[200:   , i] = 0.0   # black
        vals[200:, 0] =   0 / 255  # blue, r
        vals[200:, 1] = 162 / 255  # blue, g
        vals[200:, 2] = 255 / 255  # blue, b
    newcmp = ListedColormap(vals)

    # letters to print on plots
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p']
    alphabet_numbered = ['a$_1$', 'b$_1$', 'a$_2$', 'b$_2$', 'c$_1$', 'c$_2$']
    # title_numbers = ['ROME=157 km$^2$, R$_\mathrm{NI}$=130 km$^2$', 'ROME=127 km$^2$, R$_\mathrm{NI}$=126 km$^2$',
    #                  'ROME=191 km$^2$, R$_\mathrm{min}$=129 km$^2$', 'ROME=250 km$^2$, R$_\mathrm{min}$=130 km$^2$']
    title_numbers = ['$\mathcal{R}_\mathrm{NI}$=130, ROME=157', '$\mathcal{R}_\mathrm{min}$=129, ROME=191',
                     '$\mathcal{R}_\mathrm{NI}$=126, ROME=127', '$\mathcal{R}_\mathrm{min}$=130, ROME=250']
    darwin_time = np.timedelta64(570, 'm')  # UTC + 9.5 hours

    n_per_row = 4
    # aspect is a hack based on measuring pixels on my screen. aspect=1 for a square plot did not work as intended.
    if n_per_row == 2:
        fontsize = 16
        p = steiner_select.plot(col='time', col_wrap=n_per_row, add_colorbar=False, aspect=700./880., size=4, cmap=newcmp)
        # fontsize = 19
        # p = steiner_select.plot(col='time', col_wrap=n_per_row, add_colorbar=False, aspect=260./300., size=4, cmap=newcmp)
    if n_per_row == 5:
        fontsize = 19
        p = steiner_select.plot(col='time', col_wrap=n_per_row, add_colorbar=False, aspect=614./754, size=4, cmap=newcmp)
        # fontsize = 22
        # p = steiner_select.plot(col='time', col_wrap=n_per_row, add_colorbar=False, aspect=684./754, size=4, cmap=newcmp)
    if n_per_row == 4:
        fontsize = 19
        p = steiner_select.plot(col='time', col_wrap=n_per_row, add_colorbar=False, aspect=684./754, size=4, cmap=newcmp)

    for i, ax in enumerate(p.axes.flat):
        plt.sca(ax)  # sets the current axis ('plot') to draw to

    # for i in range(36):
    #     scene, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

        try:
            ax.pcolormesh(steiner_select.lon, steiner_select.lat, steiner_select[i], cmap=newcmp)
        except AttributeError:
            ax.pcolormesh(steiner_select.longitude, steiner_select.latitude, steiner_select[i], cmap=newcmp)

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

        l_print_alphabet = False
        if l_print_alphabet:
            # ax.text(x=129.8, y=-11.0, s=str(alphabet_numbered[i]) + ')', verticalalignment='top', fontdict={'fontsize': fontsize})
            # ax.text(x=129.8, y=-11.0, s=alphabet[i] + ')', verticalalignment='top', fontdict={'fontsize': fontsize})
            ax.set_title(str(alphabet_numbered[i]) + ')   ' + title_numbers[i])

        l_print_time = True
        if l_print_time:
            ax.text(x=131.75, y=-11.05, s=str((time_select[i] + darwin_time).values)[11:16]+' h', verticalalignment='top'
                    , fontdict={'fontsize': 16})

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

    print_numbers = True
    if print_numbers:
        for i, ax in enumerate(p.axes.flat):
            ax.annotate('ROME: {:5.1f}'        .format(metric1_select[i].item()), (131.73, -13.5 ), color='blue',  fontsize=12)
            ax.annotate('$\Delta$prox: {:5.1f}'.format(metric2_select[i].item()), (129.8 , -13.5 ), color='green', fontsize=12)
            ax.annotate('$\Delta$size: {:5.1f}'.format(metric3_select[i].item()), (129.8 , -11.05), color='red',   fontsize=12)
            if percentiles:
                ax.annotate('ROM: {:3.0f}%\n'
                            'SIC: {:3.0f}%'.format(perct1[i].item(),
                                                   perct2[i].item()), (131.78, -13.5), color='blue')

    save = True
    if save:
        plt.savefig(home+'/Desktop/radar_scenes'+large_scale_date.astype(str)[:13]+'.pdf',
                    transparent=True, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

        # # Have all scenes separately as a pdf to create a gif
        # plt.savefig('/Users/mret0001/Desktop/R/'+str(i)+'.pdf', bbox_inches='tight', transparent=True)
        # plt.close()

if __name__=='__main__':
    plot_multiple_radar_scenes(np.datetime64('2017-03-31T18:00:00'))
    # plot_multiple_radar_scenes(np.datetime64('2006-01-24T00')) # max value at 2006-01-23T21:50:00
    # plot_multiple_radar_scenes(np.datetime64('2003-12-20T09')) # max value at 2003-12-20T09:40:00


stop = timeit.default_timer()
print('This script needed {} seconds.'.format(stop - start))