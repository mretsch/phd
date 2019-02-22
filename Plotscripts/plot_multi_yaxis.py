from os.path import expanduser
home = expanduser("~")
import timeit
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import solarized_colors as colo

start = timeit.default_timer()

plt.rc('font'  , size=12)
plt.rc('legend', fontsize=12)

var1 = xr.open_dataarray('/Users/mret0001/Data/Analysis/No_Boundary/rom.nc')#.sel({'time':slice('2009-11-30','2009-12-02')})
var2 = xr.open_dataarray('/Users/mret0001/Data/Analysis/With_Boundary/conv_intensity.nc')#.sel({'time':slice('2009-11-30','2009-12-02')})
var3 = xr.open_dataarray('/Users/mret0001/Data/Analysis/With_Boundary/conv_rr_mean.nc')#.sel({'time':slice('2009-11-30','2009-12-02')})  # * 100


def identity(x):
    return x


var_actual = []
for var in [var1, var2, var3]:
    try:
        del var['percentile']
    except KeyError:
        pass
    var_day = var.groupby('time.time').mean()#apply(identity)  # mean()

    dti = pd.date_range('2019-01-07T00:00:00', periods=144, freq='10T')

    var_day.coords['new_time'] = ('time', dti)
    var_day = var_day.swap_dims({'time': 'new_time'})
    del var_day['time']
    var_day = var_day.rename({'new_time': 'time'})
    var_local = var_day.roll(shifts={'time': 9*6+3}, roll_coords=False)
    var_actual.append(var_local)

    # groupby takes all values following one timestamp. The 1400 time has all values between 1400 and 1500.
    # See here:
    # oneday = xr.open_dataarray('/Users/mret0001/Data/Analysis/No_Boundary/rom.nc').sel({'time': '2009-11-30'})
    # for i, _ in enumerate(oneday):
    #     oneday[i] = i - i % 6
    # grouped = oneday.groupby('time.hour').mean()

    # p = var_actual[-1].plot()

# # from: https://matplotlib.org/gallery/text_labels_and_annotations/date.html?highlight=ticks
hourFmt = mdates.DateFormatter('%H')
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


fig, ax_host = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
#fig.subplots_adjust(right=0.75)

ax_r1 = ax_host.twinx()
ax_r2 = ax_host.twinx()
ax_r3 = ax_host.twinx() # this one draws the line from ax_host again (visible)

# Offset the right spine of ax_r2.  The ticks and label have already been
# placed on the right by twinx above.
ax_r2.spines["right"].set_position(("axes", 1.11))
# Having been created by twinx, ax_r2 has its frame off, so the line of its
# detached spine is invisible.  First, activate the frame but make the patch
# and spines invisible.
#make_patch_spines_invisible(ax_r2)
# Second, show the right spine.
#ax_r2.spines["right"].set_visible(True)

# one-element unpacking
p0, = ax_host.plot(var_actual[0].time, var_actual[0], "k",          lw=2, label="ROM", alpha=0.)
p1, =   ax_r1.plot(var_actual[1].time, var_actual[1], colo.orange,  lw=1, label="Conv. rain intensity")
p2, =   ax_r2.plot(var_actual[2].time, var_actual[2], colo.violet,  lw=1, label="Conv. mean rain")
p3, =   ax_r3.plot(var_actual[0].time, var_actual[0], "k",          lw=2, label="ROM")

ax_host.set_xlim('2019-01-07T00:00:00', '2019-01-07T23:50:00')
ax_r3.  set_xlim('2019-01-07T00:00:00', '2019-01-07T23:50:00')
ax_host.set_ylim(8, 18)
ax_r3.  set_ylim(8, 18)
ax_r1.set_ylim(bottom=11.5)
ax_r2.set_ylim(bottom=0.08)

ax_host.set_xlabel("Time of day [hour]")
ax_host.set_ylabel("ROM [1]")
ax_r1.  set_ylabel("Conv. rain intensity [mm/hour]")
ax_r2.  set_ylabel("Conv. mean rain [mm/hour]")

#ax_host.yaxis.label.set_color(p0.get_color())
#ax_r1.yaxis.label.set_color(p1.get_color())
#ax_r2.yaxis.label.set_color(p2.get_color())

#ax_r3.spines["right"].set_color('blue')
#ax_r3.spines["right"].set_visible(False)
#ax_r3.yaxis.set_ticks_position('none')
#make_patch_spines_invisible(ax_r3)
#ax_r3.yaxis.set_alpha(0.)
#ax_r3.yaxis.remove()

# no spines, labels, ticks, ticklabels (at top, bottom, left, right) whatsoever
ax_r3.set_axis_off()

ax_host.xaxis.set_major_formatter(hourFmt)

for ax in [ax_host, ax_r1, ax_r2]:
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # this only tells if ticks are at top, bottom, left, right spine!
    #ax.xaxis.set_ticks_position('none')
    ax.tick_params(axis='x', bottom=False)
    # remove one single tick
    xticks = ax.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)

# direction brings ticklabels closer, the boolean turns the ticks off
ax_host.tick_params(axis='y', direction='in', left=False)
ax_r1.tick_params(axis='y', direction='in', right=False)
ax_r2.tick_params(axis='y', direction='in', right=False)

#tkw = dict(size=4, width=1.5)
#ax_host.tick_params(axis='y', colors=p1.get_color(), **tkw)
#ax_r1.  tick_params(axis='y', colors=p2.get_color(), **tkw)
#ax_r2.  tick_params(axis='y', colors=p3.get_color(), **tkw)
#ax_host.tick_params(axis='x', **tkw)

plots = [p3, p1, p2]
ax_host.legend(plots, [p.get_label() for p in plots], framealpha=1.)

ax_host.grid(which='both', axis='x', **dict(linestyle='--'))

save = False
if save:
    # bbox_inches captures the right portion of the plot
    plt.savefig(home+'/Desktop/plot.pdf', bbox_inches='tight', transparent=True)
else:
    plt.show()

stop = timeit.default_timer()
print('This script needed {} seconds.'.format(stop-start))
