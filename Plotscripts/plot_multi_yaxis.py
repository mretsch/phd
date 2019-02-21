import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

var1 = xr.open_dataarray('/Users/mret0001/Data/Analysis/No_Boundary/rom.nc')
var2 = xr.open_dataarray('/Users/mret0001/Data/Analysis/With_Boundary/conv_intensity.nc')
var3 = xr.open_dataarray('/Users/mret0001/Data/Analysis/With_Boundary/conv_rr_mean.nc') * 100


var_actual = []
for var in [var1, var2, var3]:
    var_day = var.groupby('time.hour').mean()
    var_local = var_day.roll(shifts={'hour': 9}, roll_coords=False)

    dti = pd.date_range('2019-01-07T00:30:00', periods=24, freq='H')

    var_local.coords['time'] = ('hour', dti)
    var_actual.append(var_local.swap_dims({'hour': 'time'}))

    # groupby takes all values following one timestamp. The 1400 time has all values between 1400 and 1500.
    # oneday = xr.open_dataarray('/Users/mret0001/Data/Analysis/No_Boundary/rom.nc').sel({'time': '2009-11-30'})
    # for i, _ in enumerate(oneday):
    #     oneday[i] = i - i % 6
    # grouped = oneday.groupby('time.hour').mean()

    p = var_actual[-1].plot()
# from: https://matplotlib.org/gallery/text_labels_and_annotations/date.html?highlight=ticks
hourFmt = mdates.DateFormatter('%H')
# finally I understood how to access axis properties without "fig, ax = plt.subplot()"
p[0].axes.xaxis.set_major_formatter(hourFmt)
plt.grid()
plt.show()

# d_rom = var_actual[0].to_pandas()
# d_int = var_actual[1].to_pandas()
# d_mean = var_actual[2].to_pandas()
# d_rom.plot()
# d_int.plot(secondary_y=True)
# d_mean.plot(secondary_y=True)

