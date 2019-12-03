from os.path import expanduser
home = expanduser("~")
import timeit
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from Plotscripts.colors_solarized import sol as col
from basic_stats import diurnal_cycle

start = timeit.default_timer()

plt.rc('font'  , size=14)
plt.rc('legend', fontsize=14)

var11 = xr.open_dataarray(home+'/Data/Analysis/No_Boundary/rom_kilometres.nc')
var22 = xr.open_dataarray(home+'/Data/Analysis/No_Boundary/rom_low_limit.nc') * 6.25
var33 = xr.open_dataarray(home+'/Data/Analysis/No_Boundary/o_area_kilometres.nc')
var44 = xr.open_dataarray(home+'/Data/Analysis/No_Boundary/o_number.nc')
var55 = xr.open_dataarray(home+'/Data/Analysis/With_Boundary/conv_intensity.nc')
var66 = xr.open_dataarray(home+'/Data/Analysis/With_Boundary/conv_rr_mean.nc')

# # from: https://matplotlib.org/gallery/text_labels_and_annotations/date.html?highlight=ticks
hourFmt = mdates.DateFormatter('%H')

# # finally I understood how to access axis properties without "fig, ax = plt.subplot()"
# p[0].axes.xaxis.set_major_formatter(hourFmt)

# from: https://matplotlib.org/examples/pylab_examples/multiple_yaxis_with_spines.html
fig, ax_host = plt.subplots(nrows=3, ncols=1, figsize=(9, 11))

# reduce vertical white space between subplots, default is 0.2
plt.subplots_adjust(hspace = 0.1)

clr1= ['k'           , col['green'] , 'k'          ]
clr2= [col['magenta'], col['blue']  , col['cyan']  ]
clr3= [col['violet'] , col['orange'], col['yellow']]
lw1 = [2, 2, 1]
lw2 = [1, 2, 2]
lw3 = [1, 1, 2]
legend1    = ['ROME'                , 'Mean object area'       , 'ROME'                  ]
legend2    = ['Conv. rain intensity', 'Number of conv. objects', '$\Delta_\mathrm{prox}$']
legend3    = ['Conv. mean rain'     , 'Total object area'      , '$\Delta_\mathrm{size}$']
y_label_l1 = ['ROME [km$^2$]'                 , 'Mean object area [km$^2$]' , 'ROME [km$^2$]'                  ]
y_label_r1 = ['Conv. rain intensity [mm/hour]', 'Number of conv. objects'   , '$\Delta_\mathrm{prox}$ [km$^2$]']
y_label_r2 = ['Conv. mean rain [mm/hour]'     , 'Total object area [km$^2$]', '$\Delta_\mathrm{size}$ [km$^2$]']
alphabet =   ['a', 'b', 'c']

for i in range(3):

    if i == 0:
        var1 =  var11
        var2 =  var55
        var3 =  var66
    if i == 1:
        var1 =  var33
        var2 =  var44
        var3 =  var33 * var44 # total area
    if i == 2:
        var1 =  var11
        var2 =  var11 - var22 # delta_prox
        var3 =  var22 - var33 # delta_size

    var_day = []
    for var in [var1, var2, var3]:
        var_day.append(diurnal_cycle(var))

    ax_r1 = ax_host[i].twinx()
    ax_r2 = ax_host[i].twinx()
    ax_r3 = ax_host[i].twinx() # this one draws the line from ax_host again (visible)

    # Offset the right spine of ax_r2.  The ticks and label have already been
    # placed on the right by twinx above.
    ax_r2.spines["right"].set_position(("axes", 1.11))

    # one-element unpacking
    p0, = ax_host[i].plot(var_day[0].time, var_day[0], clr1[i], lw=lw1[i], label="", alpha=0.)
    p1, =   ax_r1.plot(var_day[1].time, var_day[1],    clr2[i], lw=lw2[i], label=legend2[i])
    p2, =   ax_r2.plot(var_day[2].time, var_day[2],    clr3[i], lw=lw3[i], label=legend3[i])
    p3, =   ax_r3.plot(var_day[0].time, var_day[0],    clr1[i], lw=lw1[i], label=legend1[i])

    ax_host[i].set_xlim('2019-01-07T00:00:00', '2019-01-07T23:50:00')
    ax_r3.  set_xlim('2019-01-07T00:00:00', '2019-01-07T23:50:00')
    #ax_host.set_ylim(bottom=8)
    #ax_r3.  set_ylim(bottom=8)
    # ax_r1.  set_ylim(bottom=11.5)
    # ax_r2.  set_ylim(bottom=0.08)

    if i == 2:
        ax_host[i].set_xlabel("Time of day [hour]")
    ax_host[i].set_ylabel(y_label_l1[i], color=clr1[i])
    ax_r1.     set_ylabel(y_label_r1[i], color=clr2[i])
    ax_r2.     set_ylabel(y_label_r2[i], color=clr3[i])

    # no spines, labels, ticks, ticklabels (at top, bottom, left, right) whatsoever
    ax_r3.set_axis_off()

    ax_host[i].xaxis.set_major_formatter(hourFmt)

    for ax in [ax_host[i], ax_r1, ax_r2]:
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        # this only tells if ticks are at top, bottom, left, right spine!
        #ax.xaxis.set_ticks_position('none')
        ax.tick_params(axis='x', bottom=False)
        # remove one single tick
        xticks = ax.xaxis.get_major_ticks()
        xticks[0].label1.set_visible(False)
        if i != 2:
            for xtick in xticks:
                xtick.label1.set_visible(False)

    # direction brings ticklabels closer, the boolean turns the ticks off
    ax_host[i].tick_params(axis='y', direction='in', left =False) #, colors=clr1[i])
    ax_r1.     tick_params(axis='y', direction='in', right=False) #, colors=clr2[i])
    ax_r2.     tick_params(axis='y', direction='in', right=False) #, colors=clr3[i])
    if i == 0:
        ax_r2.axes.set_yticks([0.10, 0.15, 0.20, 0.25])

    plots = [p3, p1, p2]
    if i == 2:
        lg = ax_host[i].legend(plots, [p.get_label() for p in plots], framealpha=1., loc=4)  # ,frameon=False)
    else:
        lg = ax_host[i].legend(plots, [p.get_label() for p in plots], framealpha=1.       )  # ,frameon=False)
    lg.get_frame().set_facecolor('none')

    ax_host[i].grid(which='both', axis='x', **dict(linestyle='--'))

    ax_host[i].text(-0.1, 1, alphabet[i] + ')', transform=ax.transAxes, verticalalignment='top')

save = True
if save:
    # bbox_inches captures the right portion of the plot
    plt.savefig(home+'/Desktop/plot.pdf', bbox_inches='tight', transparent=True)
else:
    plt.show()

stop = timeit.default_timer()
print('This script needed {} seconds.'.format(stop-start))
