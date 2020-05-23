from os.path import expanduser
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

home = expanduser("~")
files = "Steiner/CPOL_STEINER_ECHO_CLASSIFICATION_season16*.nc"
ds_st = xr.open_mfdataset("/Users/mret0001/Data/" + files, chunks={'time': 1000})
files = "RainRate/CPOL_RADAR_ESTIMATED_RAIN_RATE_season16*.nc"
ds_rr = xr.open_mfdataset("/Users/mret0001/Data/" + files, chunks={'time': 1000})
rain=ds_rr.radar_estimated_rain_rate
stein=ds_st.steiner_echo_classification

# create mask to have line around nan-region
radar_mask = xr.where(stein[1300].isnull(), 1, 0)

plt.rc('font'  , size=19)
plt.style.use('dark_background')
for i, now in enumerate(rain.sel(time=slice('2017-03-30T15:00','2017-03-30T19:00')).time.values):
    rs = rain.sel(time=now).where(stein.sel(time=now)==2)
    rs.plot(cmap='rainbow', vmin=0, vmax=100)
    radar_mask.plot.contour(colors='w', linewidths=0.5, levels=1)

    ax = plt.gca()
    ax.set_title('')
    ax.axes.set_xlabel('')
    ax.axes.set_ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    ax.yaxis.set_ticks_position('left')
    ax.axes.set_ylabel('Latitude [$^\circ$S]')
    ax.axes.set_yticklabels(labels=['xxx', '', '13', '', '12', '', '11'])

    ax.xaxis.set_ticks_position('bottom')
    ax.axes.set_xlabel('Longitude [$^\circ$E]')
    ax.axes.set_xticks([130, 130.5, 131, 131.5, 132])
    ax.axes.set_xticklabels(labels=['130', '', '131', '', '132'])

    plt.savefig(home+'/Desktop/C/'+str(i)+'.pdf', bbox_inches='tight', transparent=True)
    plt.close()
