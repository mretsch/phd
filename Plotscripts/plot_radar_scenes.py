import xarray as xr
import matplotlib.pyplot as plt
import timeit

start = timeit.default_timer()

metric_1 = xr.open_dataarray('/Users/mret0001/Data/Analysis/No_Boundary/eso_random.nc')
metric_2 = xr.open_dataarray('/Users/mret0001/Data/Analysis/No_Boundary/sic_random.nc')
metric_3 = xr.open_dataarray('/Users/mret0001/Data/Analysis/No_Boundary/cop.nc')
ds_steiner = xr.open_mfdataset('/Users/mret0001/Data/Steiner/STEINER_random_scenes_noBound.nc')


consecutive = False
if consecutive:
    start_date = '2013-03-16T09:00:00'
    end_date   = '2013-03-16T12:10:00'
    steiner_select = ds_steiner.steiner_echo_classification.sel(time=slice(start_date, end_date))
    metric1_select = metric_1.sel(time=slice(start_date, end_date))
    metric2_select = metric_2.sel(time=slice(start_date, end_date))
    metric3_select = metric_3.sel(time=slice(start_date, end_date))

else:
    steiner = ds_steiner.steiner_echo_classification

    m_real = metric_1.where(metric_1.notnull(), drop=True)
    m_sort = m_real.sortby(m_real)

    idx_median = int(len(m_sort) / 2)
    idx_mean = int(abs(m_sort - m_sort.mean()).argmin())
    idx_90percent = round(0.9 * len(m_sort))
    metric1_select = m_sort[-20:] #[-60:-40] #[idx_90percent-10:idx_90percent+10] #[idx_mean-10:idx_mean+10] #[idx_median-10:idx_median+10] #[:20] #

    steiner_select = steiner.loc[metric1_select.time]
    metric2_select = metric_2.loc[metric1_select.time]
    metric3_select = metric_3.loc[metric1_select.time]

#for i, scene in enumerate(steiner_select):
#    plt.close()
#    scene.plot()
#    plt.savefig('/Users/mret0001/Desktop/'+str(i)+'.pdf')

# aspect is a hack based on measuring pixels on my screen. aspect=1 for a square plot did not work as intended.
# p = steiner_select.plot(col='time', col_wrap=4, add_colorbar=False, aspect=1, size=4)
p = steiner_select.plot(col='time', col_wrap=4, add_colorbar=False, aspect=1 - abs(1 - 679./740), size=4)

for i, ax in enumerate(p.axes.flat):
    ax.annotate('ESO: {:5.1f}\nSIC: {:5.1f}'.format(metric1_select[i].item(),
                                                    metric2_select[i].item()), (131.78, -11.2), color='blue')
    # ax.annotate('COP: {:5.1f}'.format(metric3_select[i].item()), (131.78, -13.4), color='blue')

plt.savefig('/Users/mret0001/Desktop/test.pdf')
#plt.show()

stop = timeit.default_timer()
print('This script needed {} seconds.'.format(stop - start))