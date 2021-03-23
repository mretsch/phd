from os.path import expanduser
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import timeit
import sub as FORTRAN

home = expanduser("~")


def return_phasespace_plot():

    # no open_mfdataset here, since dask causes runtime-warning in loop below: "invalid value encountered in true_divide"
    # ds_ps = xr.open_dataset(home+'/Documents/Plots/2D_Histograms/area_number_hist.nc')
    # ds_ps = xr.open_dataset(home+'/Documents/Plots/Phase_Space/515rh_515w/valentines_hist.nc')
    # ds_ps = xr.open_dataset(home+'/Documents/Plots/Phase_Space/515rh_515w/Change_Bin_and_Cutoff/hist_18bin4per.nc')
    # ds_ps = xr.open_dataset(home+'/Documents/Plots/Phase_Space/w_pw_9bin40per_4to96perc_hist.nc')
    ds_ps = xr.open_dataset(home+'/Documents/Plots/Phase_Space/515rh_515w/Change_Bin_and_Cutoff/9bin_1to99perc_55per.nc')
    # ds_ps = xr.open_dataset(home+'/Desktop/hist.nc')

    # rome  = xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/rom_km_avg6h_nanzero.nc')
    rome  = xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/rom_km_max6h_avg_pm20minutes.nc')
    # rome  = xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/rom_kilometres.nc')

    ls    = xr.open_dataset(home+'/Documents/Data/LargeScaleState/' +
                                'CPOL_large-scale_forcing_cape990hPa_cin990hPa_rh_shear_dcape_noDailyCycle.nc')

    ls_day    = xr.open_dataset(home+'/Documents/Data/LargeScaleState/' +
                            'CPOL_large-scale_forcing_cape990hPa_cin990hPa_rh_shear_dcape.nc')
    # remove false data in precipitable water
    ls_day['PW'].loc[{'time': slice(None, '2002-02-27T12')}] = np.nan
    ls_day['PW'].loc[{'time': slice('2003-02-07T00', '2003-10-19T00')}] = np.nan
    ls_day['PW'].loc[{'time': slice('2005-11-14T06', '2005-12-09T12')}] = np.nan
    ls_day['PW'].loc[{'time': slice('2006-02-25T00', '2006-04-07T00')}] = np.nan
    ls_day['PW'].loc[{'time': slice('2011-11-09T18', '2011-12-01T06')}] = np.nan
    ls_day['PW'].loc[{'time': slice('2015-01-05T00', None)}] = np.nan
    ls_day['LWP'].loc[{'time': slice(None, '2002-02-27T12')}] = np.nan
    ls_day['LWP'].loc[{'time': slice('2003-02-07T00', '2003-10-19T00')}] = np.nan
    ls_day['LWP'].loc[{'time': slice('2005-11-14T06', '2005-12-09T12')}] = np.nan
    ls_day['LWP'].loc[{'time': slice('2006-02-25T00', '2006-04-07T00')}] = np.nan
    ls_day['LWP'].loc[{'time': slice('2011-11-09T18', '2011-12-01T06')}] = np.nan
    ls_day['LWP'].loc[{'time': slice('2015-01-05T00', None)}] = np.nan

    # da    = ls['cin']#.resample(time='10min').interpolate('linear')
    # da    = ls['lw_net_toa'].resample(time='10min').interpolate('linear')
    # da    = xr.open_dataarray(home+'/Documents/Data/Analysis/With_Boundary/conv_intensity.nc')
    # da    = xr.open_dataarray(home+'/Documents/Data/Analysis/With_Boundary/o_area.nc') * 6.25
    # da    = xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/o_area.nc') * 6.25
    # da    = xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/o_area.nc') * 6.25 \
    #       * xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/o_number.nc')
    # da    = xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/o_area_avg6h.nc') * 6.25# \
    #       * xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/o_number_avg6h.nc')
    # model_path = '/Documents/Data/NN_Models/ROME_Models/Kitchen_NoDiurnal/'
    # da = xr.open_dataarray(home + model_path + 'predicted.nc')
    # da    = xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/totalarea_km_avg6h.nc')
    da=rome

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

    overlay = da#.rom_kilometres #.RH.sel(lev=515) #conv_intensity #conv_intensity # div.sel(lev=845) # .where(da.cop_mod < 60.)
    # overlay = rome

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
                                              upper_bound=10000.,
                                              lower_bound=np.percentile(overlay, 90))
                                              # upper_bound=np.percentile(overlay, q=10),
                                              # lower_bound=-10000.)

    # set NaNs to the special values set in the Fortran-routine
    phase_space_stack = phase_space_stack.where((phase_space_stack < -1e10) ^ (-9999999998.0 < phase_space_stack))

    ps_overlay = phase_space_stack.unstack('z')

    # the actual plotting commands
    plt.rc('font'  , size=26)
    plt.rc('legend', fontsize=18)
    # plt.style.use('dark_background')

    the_plot = ps_overlay.T.plot(cmap='rainbow', #'gray_r', #'gnuplot2', #'gist_yarg_r', # 'inferno',# (robust=True)  # (cmap='coolwarm_r', 'tab20c')
                                 vmin=ps_overlay.min(), vmax=ps_overlay.max())
                                 # vmin=-150, vmax=ps_overlay.max())

    plt.xticks((-15, -10, -5, 0))

    # plt.title('6h large-scale')
    plt.xlabel('$\omega_{515}$ [hPa/h]')
    # plt.xlabel('$\Delta(\omega, \Phi)$ at 515 hPa [hPa/h]')
    # plt.xlabel('Avg. object area [km$^2$]')
    # plt.xlabel('Dry static energy, 990 hPa [K]')
    # plt.xlabel('CAPE')

    # plt.ylabel('$\Delta(\mathrm{RH}, \Phi)$ at 515 hPa [1]')
    # plt.ylabel('$\Delta(\mathrm{PW}, \Phi)$ [cm]')
    # plt.ylabel('Number of objects [1]')
    plt.ylabel('RH$_{515}$ [1]')
    # plt.ylabel('PW')

    # the_plot.colorbar.set_label('Probability of R$_\mathrm{NN}$ > p$_{90}$(R$_\mathrm{NN}$) [1]')
    the_plot.colorbar.set_label('Probability of ROME$_\mathrm{p90}$ [1]')
    # the_plot.colorbar.set_label('Probability of ROME < p$_{10}$(ROME) [1]')
    # the_plot.colorbar.set_label('Total conv. area [km$^2$]')
    # the_plot.colorbar.set_label('ROME [km$^2$]')
    # the_plot.colorbar.set_label('max ROME ($\pm$20min avg)')
    # the_plot.colorbar.set_label('Number of objects [1]')
    # the_plot.colorbar.set_label('Mean object area [km$^2$]')
    # the_plot.colorbar.set_label(da.long_name+' ['+da.units+']')
    # the_plot.colorbar.set_label(da.long_name+', 515 hPa [1]')


    # x = ls['omega'].sel(lev=515, time=ds_ps.time) + ls_day['omega'].sel(lev=515).mean(dim='time')
    # y = ls['PW'   ].sel(         time=ds_ps.time) + ls_day['PW']                .mean(dim='time')
    # x_mean, x_std = x.mean(dim='time'), x.std(dim='time')
    # y_mean, y_std = y.mean(dim='time'), y.std(dim='time')
    # plt.vlines(x=x_mean, ymin=y_mean-0.5*y_std, ymax=y_mean+0.5*y_std)
    # plt.hlines(y=y_mean, xmin=x_mean-0.5*y_std, xmax=x_mean+0.5*y_std)

    return the_plot


if __name__ == '__main__':

    start = timeit.default_timer()
    plot = return_phasespace_plot()
    save = True
    if save:
        plt.savefig(home+'/Desktop/phase_space_rome.pdf', transparent=True, bbox_inches='tight')
        plt.show()
    else:
        plt.show()
    stop = timeit.default_timer()
    print('Script needed: {} seconds'.format(stop - start))