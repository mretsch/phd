import xarray as xr
import matplotlib.pyplot as plt
import timeit

start = timeit.default_timer()

metric = xr.open_dataarray('/Users/mret0001/Data/Analysis/No_Boundary/sic.nc')
metric_2 = xr.open_dataarray('/Users/mret0001/Data/Analysis/No_Boundary/eso.nc')
ds_steiner = xr.open_mfdataset('/Users/mret0001/Data/Steiner/*season*')


consecutive = False
if consecutive:
    steiner = ds_steiner.steiner_echo_classification.sel(time=slice('2010-01-07T13:00:00', '2010-01-07T14:00:00'))
    p = steiner.plot(col='time', col_wrap=4, add_colorbar=False, aspect=1, size=4)

else:
    steiner = ds_steiner.steiner_echo_classification

    m_real = metric.where(metric.notnull(), drop=True)
    m_sort = m_real.sortby(m_real)

    idx_median = int(len(m_sort) / 2)
    idx_mean = int(abs(m_sort - m_sort.mean()).argmin())
    idx_90percent = round(0.9 * len(m_sort))
    m_select = m_sort[-60:-40] #[-20:] #[idx_90percent-10:idx_90percent+10] #[idx_mean-10:idx_mean+10] #[idx_median-10:idx_median+10] #[:20] #

    steiner_select = steiner.loc[m_select.time]
    metric2_select = metric_2.loc[m_select.time]

    #for i, scene in enumerate(steiner_select):
    #    plt.close()
    #    scene.plot()
    #    plt.savefig('/Users/mret0001/Desktop/'+str(i)+'.pdf')

    # aspect is a hack based on measuring pixels on my screen. aspect=1 for a square plot did not work as intended.
    # p = steiner_select.plot(col='time', col_wrap=4, add_colorbar=False, aspect=1, size=4)
    p = steiner_select.plot(col='time', col_wrap=4, add_colorbar=False, aspect=1 - abs(1 - 679./740), size=4)

    for i, ax in enumerate(p.axes.flat):
        ax.annotate('SIC: {:5.1f}\nESO: {:5.1f}'.format(m_select[i].item(), metric2_select[i].item()), (131.78, -11.2), color='blue')

plt.savefig('/Users/mret0001/Desktop/test.pdf')
#plt.show()

stop = timeit.default_timer()
print('This script needed {} seconds.'.format(stop - start))