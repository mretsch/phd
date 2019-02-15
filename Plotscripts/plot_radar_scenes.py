import xarray as xr
import matplotlib.pyplot as plt
import timeit

start = timeit.default_timer()

metric_1 = xr.open_dataarray('/Users/mret0001/Data/Analysis/No_Boundary/rom.nc')
metric_2 = xr.open_dataarray('/Users/mret0001/Data/Analysis/No_Boundary/sic.nc')
ds_steiner = xr.open_mfdataset('/Users/mret0001/Data/Steiner/*season*', chunks=40)

# metric_1 = xr.open_dataarray('/Users/matthiasretsch/Google Drive File Stream/My Drive/Data_Analysis/sic.nc').\
#     sel({'time': slice('2009-10-01', '2010-03-31')})
# metric_2 = xr.open_dataarray('/Users/matthiasretsch/Google Drive File Stream/My Drive/Data_Analysis/iorg_season0910.nc')
# ds_steiner = xr.open_mfdataset('/Users/matthiasretsch/Google Drive File Stream/My Drive/Data/steiner*')

consecutive = False
if consecutive:
    start_date = '2013-03-16T09:00:00'
    end_date   = '2013-03-16T12:10:00'
    steiner_select = ds_steiner.steiner_echo_classification.sel(time=slice(start_date, end_date))
    metric1_select = metric_1.sel(time=slice(start_date, end_date))
    metric2_select = metric_2.sel(time=slice(start_date, end_date))

else:
    steiner = ds_steiner.steiner_echo_classification

    m_real = metric_1.where(metric_1.notnull(), drop=True)
    m_sort = m_real.sortby(m_real)

    idx_median = int(len(m_sort) / 2)
    idx_mean = int(abs(m_sort - m_sort.mean()).argmin())
    idx_90percent = round(0.9 * len(m_sort))
    metric1_select = m_sort[-20:]  #[-60:-40] #[idx_90percent-10:idx_90percent+10] #[idx_mean-10:idx_mean+10] #[idx_median-10:idx_median+10] #[:20] #

    steiner_select = steiner.loc[metric1_select.time]
    metric2_select = metric_2.loc[metric1_select.time]

# aspect is a hack based on measuring pixels on my screen. aspect=1 for a square plot did not work as intended.
# p = steiner_select.plot(col='time', col_wrap=4, add_colorbar=False, aspect=1, size=4)
p = steiner_select.plot(col='time', col_wrap=4, add_colorbar=False, aspect=1 - abs(1 - 679./740), size=4)

percentiles = False
as_coordinate = True
if percentiles:
    if as_coordinate:
        print1 = metric1_select.percentile * 100
        print2 = metric2_select.percentile * 100
    else:
        print1 = metric_1.rank(dim='time', pct=True).sel(time=slice(start_date, end_date)) * 100
        print2 = metric_2.rank(dim='time', pct=True).sel(time=slice(start_date, end_date)) * 100
else:
    print1 = metric1_select
    print2 = metric2_select

for i, ax in enumerate(p.axes.flat):
    ax.annotate('ROM: {:5.0f}\n'
                'SIC: {:5.0f}'.format(print1[i].item(),
                                      print2[i].item()), (131.78, -11.2), color='blue')

plt.savefig('/Users/mret0001/Desktop/radar_scenes.pdf')
#plt.show()

# Have all scenes separately as a pdf to create a gif
# for i, scene in enumerate(steiner_select):
#     plt.close()
#     scene.plot()
#     plt.savefig('/Users/mret0001/Desktop/'+str(i)+'.pdf')

stop = timeit.default_timer()
print('This script needed {} seconds.'.format(stop - start))