from os.path import expanduser
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import timeit
import sub as FORTRAN

start = timeit.default_timer()
home = expanduser("~")

# no open_mfdataset here, since dask causes runtime-warning in loop below: "invalid value encountered in true_divide"
ds_ps = xr.open_dataset(home+'/Desktop/hist.nc')

rome = xr.open_dataarray(home + '/Documents/Data/Simulation/r2b10/rome_14mmhour.nc')
rome_p90 = np.nanpercentile(rome, q=90)

l_pick_surface = True
if l_pick_surface:
    land_sea = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/land_sea_avg.nc')
    coast_mask =   land_sea < 0.2 # ocean
    # coast_mask =  (0.2 < land_sea) & (land_sea < 0.8) # coast
    # coast_mask =   land_sea > 0.8 # land
    coast_mask['lat'] = rome['lat']
    coast_mask['lon'] = rome['lon']
    rome = rome.where(coast_mask, other=np.nan)

# rome = rome.where(((160 < rome['lon']) | (rome['lon'] < -90)) & (0 < rome['lat']), other=np.nan) # North Pacific
# rome = rome.where((-52 < rome['lon']) & (rome['lon'] < -44) & (-1 < rome['lat']) & (rome['lat'] < 5), other=np.nan) # Amazonas delta
# rome = rome.where((120 < rome['lon']) & (rome['lon'] < 134) & (-20 < rome['lat']) & (rome['lat'] < -10), other=np.nan) # West Australia
rome = rome.where(( 77 < rome['lon']) & (rome['lon'] <  82) & (-12 < rome['lat']) & (rome['lat'] <  -6), other=np.nan)

da = rome.stack({'x': ('time', 'lat', 'lon')})
da['x'] = np.arange(len(da))
da = da.rename({'x': 'time'})

subselect = True
if subselect:
    # subselect specific times during a day
    l_time_of_day = False
    if l_time_of_day:
        da.coords['hour'] = da.indexes['time'].hour
        da_sub = da.where(da.hour.isin([6]), drop=True)

    # subselect on the times given in the histogram data
    l_histogram = True
    if l_histogram:
        # da_sub = da.sel(time=ds_ps.time)
        da = da[da.notnull()]
        da_sub = da[da.time.isin(ds_ps.time)]

    da = da_sub
    # rome = rome.where(da_sub)

phase_space = ds_ps.hist_2D

overlay = da

# give the overlay time series information about the placements of the bins for each time step
overlay.coords['x_bins'] = ('time', ds_ps.x_series_bins[ds_ps.time.isin(da_sub.time)].values)
overlay.coords['y_bins'] = ('time', ds_ps.y_series_bins[ds_ps.time.isin(da_sub.time)].values)

phase_space_stack = phase_space.stack(z=('x', 'y'))

ind_1, ind_2 =zip(*phase_space_stack.z.values)
phase_space_stack[:] = FORTRAN.phasespace(indices1=ind_1,
                                          indices2=ind_2,
                                          overlay=overlay,
                                          overlay_x=overlay['x_bins'],
                                          overlay_y=overlay['y_bins'],
                                          l_probability=True,
                                          upper_bound=100000.,
                                          lower_bound=np.nanpercentile(overlay, q=90))
                                          # upper_bound=np.percentile(overlay, q=10),
                                          # lower_bound=-10000.)

# set NaNs to the special values set in the Fortran-routine
phase_space_stack = phase_space_stack.where((phase_space_stack < -1e10) ^ (-9999999998.0 < phase_space_stack))

ps_overlay = phase_space_stack.unstack('z')

# the actual plotting commands
plt.rc('font'  , size=26)
plt.rc('legend', fontsize=18)

the_plot = ps_overlay.T.plot(cmap='rainbow', #'gray_r', #'gnuplot2', #'gist_yarg_r', # 'inferno',# (robust=True)  # (cmap='coolwarm_r', 'tab20c')
                             vmin=ps_overlay.min(), vmax=ps_overlay.max())
                             # vmin=0.0028409857748265717, vmax=0.4935537724163929)

# plt.xticks((-15, -10, -5, 0))

plt.xlabel('$\omega_{515}$ [hPa/hour]')
plt.ylabel('RH$_{515}$ [%]')

the_plot.colorbar.set_label('Prob. of ROME$_\mathrm{p90,ind}$ [1]')

plt.savefig(home + '/Desktop/phase_space.pdf', transparent=True, bbox_inches='tight')
plt.show()

stop = timeit.default_timer()
print(f'Time used: {stop - start}')
