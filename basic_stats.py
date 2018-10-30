import math as m
import xarray as xr
from dask.distributed import Client
import numpy as np
import matplotlib.pyplot as plt
import timeit


def precip_stats(rain, stein, period='', group=''):
    """Calculates area and rate of stratiform and convective precipitation and area ratio to scan area."""
    grouped = (period == '')
    rain_conv = rain.where(stein == 2)
    rain_stra = rain.where(stein == 1)
    rain_conv = rain_conv.where(rain_conv != 0.)
    rain_stra = rain_stra.where(rain_stra != 0.)

    i = 0
    while rain[i, :, :].isnull().all():
        i += 1
    else:
        first_scene = rain[i, :, :]

    # The total area (number of pixels) of one radar scan. All cells have same area.
    area_scan = first_scene.notnull().sum()

    # Total rain in one time slice. And the corresponding number of cells.
    if grouped:
        group = group if (group == '') else '.' + group
        conv_rain = rain_conv.groupby('time' + group).sum(skipna=True)
        stra_rain = rain_stra.groupby('time' + group).sum(skipna=True)
        conv_area = rain_conv.notnull().groupby('time' + group).sum()
        stra_area = rain_stra.notnull().groupby('time' + group).sum()
    else:
        conv_rain = rain_conv.resample(time=period).sum(skipna=True)
        stra_rain = rain_stra.resample(time=period).sum(skipna=True)
        conv_area = rain_conv.notnull().resample(time=period).sum()
        stra_area = rain_stra.notnull().resample(time=period).sum()

    if not conv_area._in_memory:
        conv_area.load()
        stra_area.load()
    if not conv_rain._in_memory:
        conv_rain.load()
        stra_rain.load()

    # conv_intensity gives mean precip over its area in a time interval.
    # Summation of it hence would be a sum of means. Beware, mean(a) + mean(b) != mean(a + b).
    # xarray automatically throws NaN if division by zero
    conv_intensity = conv_rain / conv_area
    stra_intensity = stra_rain / stra_area

    area_period = area_scan * (len(rain.time) / len(conv_rain))
    conv_mean = conv_rain / area_period
    stra_mean = stra_rain / area_period
    conv_area_ratio = conv_area / area_period * 100
    stra_area_ratio = stra_area / area_period * 100

    return conv_intensity, conv_mean, conv_area_ratio, stra_intensity, stra_mean, stra_area_ratio


if __name__ == '__main__':

    start = timeit.default_timer()

    files = "Steiner/CPOL_STEINER_ECHO_CLASSIFICATION_season*.nc"
    ds_st = xr.open_mfdataset("/Users/mret0001/Data/"+files, chunks={'time': 1000})
    files = "RainRate/CPOL_RADAR_ESTIMATED_RAIN_RATE_season*.nc"
    ds_rr = xr.open_mfdataset("/Users/mret0001/Data/"+files, chunks={'time': 1000})

    # run with 4 parallel threads on my local laptop
    # c = Client()

    # Rain rate units are mm/hour, dividing by 86400 yields mm/s == kg/m^2s. No, the factor is 3600, not 86400.
    # And 6 (=600/3600) for 10 minutes, the measurement interval.
    conv_intensity, conv_mean, conv_area_ratio,\
    stra_intensity, stra_mean, stra_area_ratio = precip_stats(rain=ds_rr.radar_estimated_rain_rate,  # / 6.,
                                                              stein=ds_st.steiner_echo_classification,
                                                              period='',
                                                              group='')

    # add some attributes for convenience to the stats
    conv_intensity.attrs['units'] = 'mm/hour'
    conv_mean.attrs['units'] = 'mm/hour'
    conv_area_ratio.attrs['units'] = ['%']
    stra_intensity.attrs['units'] = 'mm/hour'
    stra_mean.attrs['units'] = 'mm/hour'
    stra_area_ratio.attrs['units'] = ['%']

    # save as netcdf-files
    path = '/Users/mret0001/Data/Analysis/'
    xr.save_mfdataset([xr.Dataset({'conv_intensity': conv_intensity}), xr.Dataset({'conv_rr_mean': conv_mean}),
                       xr.Dataset({'conv_area_ratio': conv_area_ratio}),
                       xr.Dataset({'stra_intensity': stra_intensity}), xr.Dataset({'stra_rr_mean': stra_mean}),
                       xr.Dataset({'stra_area_ratio': stra_area_ratio})],
                      [path+'conv_intensity.nc', path+'conv_rr_mean.nc',
                       path+'conv_area_ratio.nc',
                       path+'stra_intensity.nc', path+'stra_rr_mean.nc',
                       path+'stra_area_ratio.nc'])

    # sanity check
    check = False
    if check:
        r = ds_rr.radar_estimated_rain_rate / 6.
        cr = r.where(ds_st.steiner_echo_classification == 2)
        cr = cr.where(cr != 0.)
        cr_1h = cr[9774:9780, :, :].load()  # the most precip hour in the 09/10-season
        # cr_1h = cr[9774, :, :].load()  # '2010-02-25T21:00:00'
        npixels = cr_1h.notnull().sum()
        cr_intens_appro = cr_1h.sum() / npixels
        cr_intens_orig = conv_intensity.sel(time='2010-02-25T21:00:00')
        print('Simple 2.5 x 2.5 km square assumption approximated the most precipitating hour by {} %.'
              .format(str((cr_intens_appro/cr_intens_orig).values * 100)))

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

    stop = timeit.default_timer()
    print('Run Time: ', stop - start)
