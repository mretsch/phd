from os.path import expanduser
home = expanduser("~")
import timeit
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from Plotscripts.colors_solarized import sol as col
from basic_stats import diurnal_cycle

start = timeit.default_timer()

plt.rc('font'  , size=12)
plt.rc('legend', fontsize=12)

var1 = xr.open_dataarray(home+'/Data/Analysis/No_Boundary/rome.nc')# .percentile * 100 # .sel({'time':slice('2009-11-30','2009-12-02')})
var2 = xr.open_dataarray(home+'/Data/Analysis/With_Boundary/conv_intensity.nc')#.sel({'time':slice('2009-11-30','2009-12-02')})
var3 = xr.open_dataarray(home+'/Data/Analysis/With_Boundary/conv_rr_mean.nc')#.sel({'time':slice('2009-11-30','2009-12-02')})  # * 100

var_day = []
for var in [var1, var2, var3]:
    var_day.append(diurnal_cycle(var))

# # from: https://matplotlib.org/gallery/text_labels_and_annotations/date.html?highlight=ticks
hourFmt = mdates.DateFormatter('%H')

# # finally I understood how to access axis properties without "fig, ax = plt.subplot()"
# p[0].axes.xaxis.set_major_formatter(hourFmt)

# from: https://matplotlib.org/examples/pylab_examples/multiple_yaxis_with_spines.html
fig, ax_host = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))

ax_r1 = ax_host.twinx()
ax_r2 = ax_host.twinx()
ax_r3 = ax_host.twinx() # this one draws the line from ax_host again (visible)

# Offset the right spine of ax_r2.  The ticks and label have already been
# placed on the right by twinx above.
ax_r2.spines["right"].set_position(("axes", 1.11))

# one-element unpacking
p0, = ax_host.plot(var_day[0].time, var_day[0], "k",           lw=2, label="ROME", alpha=0.)
p1, =   ax_r1.plot(var_day[1].time, var_day[1], col['orange'], lw=1, label="Conv. rain intensity")
p2, =   ax_r2.plot(var_day[2].time, var_day[2], col['violet'], lw=1, label="Conv. mean rain")
p3, =   ax_r3.plot(var_day[0].time, var_day[0], "k",           lw=2, label="ROME")

ax_host.set_xlim('2019-01-07T00:00:00', '2019-01-07T23:50:00')
ax_r3.  set_xlim('2019-01-07T00:00:00', '2019-01-07T23:50:00')
#ax_host.set_ylim(bottom=8)
#ax_r3.  set_ylim(bottom=8)
ax_r1.  set_ylim(bottom=11.5)
ax_r2.  set_ylim(bottom=0.08)

ax_host.set_xlabel("Time of day [hour]")
ax_host.set_ylabel("ROME [1]")
ax_r1.  set_ylabel("Conv. rain intensity [mm/hour]")
ax_r2.  set_ylabel("Conv. mean rain [mm/hour]")

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
ax_r1.  tick_params(axis='y', direction='in', right=False)
ax_r2.  tick_params(axis='y', direction='in', right=False)

plots = [p3, p1, p2]
lg = ax_host.legend(plots, [p.get_label() for p in plots], framealpha=1.)  # ,frameon=False)
lg.get_frame().set_facecolor('none')

ax_host.grid(which='both', axis='x', **dict(linestyle='--'))

save = True
if save:
    # bbox_inches captures the right portion of the plot
    plt.savefig(home+'/Desktop/plot.pdf', bbox_inches='tight', transparent=True)
else:
    plt.show()

stop = timeit.default_timer()
print('This script needed {} seconds.'.format(stop-start))
