from os.path import expanduser
home = expanduser("~")
import timeit
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from Plotscripts.plot_hist import histogram_2d
from divergence_at_drymoist_rome import smallregion_in_tropics

start = timeit.default_timer()
plt.rc('font'  , size=26)
plt.rc('legend', fontsize=18)


area   = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/o_area_10mmhour.nc')
number = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/o_number_10mmhour.nc')
rome   = area * number
# rome = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/rome_10mmhour.nc')
rh500 = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/rh500.nc')
w500 = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/omega500.nc')
# convert Pa/s to hPa/hour
w500 = w500 * (1/100.) * 3600

rome_p90 = np.nanpercentile(rome, q=90)

l_take_region_and_high_rome = True
if l_take_region_and_high_rome:
    region = smallregion_in_tropics(rome, 'Tropic', 'ocean', other_surface_fillvalue=np.nan)
    rome_copy = xr.full_like(rome, fill_value=np.nan)
    rome_copy.loc[{'lat': region['lat'], 'lon': region['lon']}] = region
    rome = rome_copy

    rome = rome.where(rome > rome_p90)

rh_where_conv = xr.where(rome.notnull(), rh500, np.nan)
rh = rh_where_conv.stack({'x': ('time', 'lat', 'lon')})
rh['x'] = np.arange(len(rh))

w_where_conv = xr.where(rome.notnull(), w500, np.nan)
w = w_where_conv.stack({'x': ('time', 'lat', 'lon')})
w['x'] = np.arange(len(w))

# make a 'scatter'-plot via a 2d-histogram
fig, ax = plt.subplots(nrows=1, ncols=1)#, figsize=(5, 5))
ax, h_2d = histogram_2d(w,
                        rh,
                        nbins=9,
                        l_cut_off=True,
                        ax=ax,
                        y_label='rh',
                        x_label='omega',
                        cbar_label='[%]')

ax.set_xticks((-15, -10, -5, 0, 5))
# ax.set_xticks((-30, -20, -10, 0, ))
ax.set_title('Tropical ocean, high TCA')
ax.set_xlabel('$\omega_{515}$ [hPa/hour]')
ax.set_ylabel('RH$_{515}$ [%]')

fig.savefig(home + '/Desktop/hist.pdf', bbox_inches='tight')
plt.show()

h_2d.to_netcdf(home+'/Desktop/hist.nc', mode='w')
# fig.show()

stop = timeit.default_timer()
print(f'Time used: {stop - start}')