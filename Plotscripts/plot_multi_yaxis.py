import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

var1 = xr.open_dataarray('/Users/mret0001/Data/Analysis/No_Boundary/rom.nc')
var2 = xr.open_dataarray('/Users/mret0001/Data/Analysis/With_Boundary/conv_intensity.nc')
var3 = xr.open_dataarray('/Users/mret0001/Data/Analysis/With_Boundary/conv_rr_mean.nc')  # * 100


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

    # p = var_actual[-1].plot()

# # from: https://matplotlib.org/gallery/text_labels_and_annotations/date.html?highlight=ticks
# hourFmt = mdates.DateFormatter('%H')
# # finally I understood how to access axis properties without "fig, ax = plt.subplot()"
# p[0].axes.xaxis.set_major_formatter(hourFmt)
# plt.grid()
# plt.show()

# d_rom = var_actual[0].to_pandas()
# d_int = var_actual[1].to_pandas()
# d_mean = var_actual[2].to_pandas()
# d_rom.plot()
# d_int.plot(secondary_y=True)
# d_mean.plot(secondary_y=True)

# from: https://matplotlib.org/examples/pylab_examples/multiple_yaxis_with_spines.html


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


fig, host = plt.subplots()
fig.subplots_adjust(right=0.75)

par1 = host.twinx()
par2 = host.twinx()

# Offset the right spine of par2.  The ticks and label have already been
# placed on the right by twinx above.
par2.spines["right"].set_position(("axes", 1.2))
# Having been created by twinx, par2 has its frame off, so the line of its
# detached spine is invisible.  First, activate the frame but make the patch
# and spines invisible.
make_patch_spines_invisible(par2)
# Second, show the right spine.
par2.spines["right"].set_visible(True)

p1, = host.plot(var_actual[0].time, var_actual[0], "b-", label="Density")
p2, = par1.plot(var_actual[1].time, var_actual[1], "r-", label="Temperature")
p3, = par2.plot(var_actual[2].time, var_actual[2], "g-", label="Velocity")

#host.set_xlim(0, 2)
#host.set_ylim(0, 2)
#par1.set_ylim(0, 4)
#par2.set_ylim(1, 65)

host.set_xlabel("Distance")
host.set_ylabel("Density")
par1.set_ylabel("Temperature")
par2.set_ylabel("Velocity")

host.yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())
par2.yaxis.label.set_color(p3.get_color())

tkw = dict(size=4, width=1.5)
host.tick_params(axis='y', colors=p1.get_color(), **tkw)
par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
par2.tick_params(axis='y', colors=p3.get_color(), **tkw)
host.tick_params(axis='x', **tkw)

lines = [p1, p2, p3]

host.legend(lines, [l.get_label() for l in lines])

plt.show()