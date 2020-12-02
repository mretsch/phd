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
    # ds_ps = xr.open_dataset(home+'/Documents/Plots/Phase_Space/515rh_515w/515rh_515omega_hist_gt8.nc')
    # ds_ps = xr.open_dataset(home+'/Documents/Plots/Phase_Space/adv_h2o_s_hist.nc')
    ds_ps = xr.open_dataset(home+'/Desktop/hist.nc')

    rome  = xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/rom_km_avg6h_nanzero.nc')

    # ls    = xr.open_dataset(home+'/Documents/Data/LargeScaleState/' +
    #                         'CPOL_large-scale_forcing_cape990hPa_cin990hPa_rh_shear_dcape.nc')
    # da    = xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/rom_km_avg6h_nanzero.nc')
    # da    = ls.RH.sel(lev=515).resample(time='10min').interpolate('linear')
    # da    = xr.open_dataarray(home+'/Documents/Data/Analysis/With_Boundary/conv_intensity.nc')
    # da    = xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/o_area.nc') * 6.25 \
    #       * xr.open_dataarray(home+'/Documents/Data/Analysis/No_Boundary/AllSeasons/o_number.nc')
    model_path = '/Documents/Data/NN_Models/ROME_Models/Kitchen_NoDiurnal/'
    da = xr.open_dataarray(home + model_path + 'predicted.nc')

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
            # ds_sub = da.sel(time=ds_ps.time)
            da_sub = da[da.time.isin(ds_ps.time)]

        da = da_sub
        rome = rome.where(da)

    phase_space = ds_ps.hist_2D

    # overlay = da #.rom_kilometres #.RH.sel(lev=515) #conv_intensity #conv_intensity # div.sel(lev=845) # .where(da.cop_mod < 60.)
    overlay = rome

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
                                              # lower_bound=overlay[overlay.percentile > 0.9].min())
                                              lower_bound=np.percentile(overlay, 90))
                                              # lower_bound=np.percentile(rome, 90))
                                              # lower_bound=0.)

    # set NaNs to the special values set in the Fortran-routine
    phase_space_stack = phase_space_stack.where((phase_space_stack < -1e10) ^ (-9999999998.0 < phase_space_stack))

    ps_overlay = phase_space_stack.unstack('z')

    # the actual plotting commands
    plt.rc('font'  , size=18)
    plt.rc('legend', fontsize=18)
    # plt.style.use('dark_background')

    the_plot = ps_overlay.T.plot(cmap='rainbow', # 'gist_yarg_r', #'gray_r', #'inferno',# (robust=True)  # (cmap='coolwarm_r', 'gnuplot2', 'tab20c')
                                 vmin=ps_overlay.min(), vmax=ps_overlay.max())
                                 # vmin=0., vmax=1.)

    plt.xlabel('$\Delta(\omega, \Phi)$ at 515 hPa [hPa/h]')
    # plt.xlabel('Total conv. area [km$^2$]')
    # plt.xlabel('Dry static energy, 990 hPa [K]')
    # plt.ylabel('$\Delta(\mathrm{RH}, \Phi)$ at 515 hPa [1]')
    plt.ylabel('$\Delta(\mathrm{PW}, \Phi)$ [cm]')
    # plt.ylabel('Number of objects [1]')
    # plt.ylabel('OLR [W/m2]')
    # the_plot.colorbar.set_label('Probability of R$_\mathrm{NN}$ > p$_{90}$(R$_\mathrm{NN}$) [1]')
    the_plot.colorbar.set_label('Probability of ROME > p$_{90}$(ROME) [1]')
    # the_plot.colorbar.set_label('Total conv. area [km$^2$]')
    # the_plot.colorbar.set_label(da.long_name+' ['+da.units+']')
    # the_plot.colorbar.set_label(da.long_name+', 515 hPa [1]')

    return the_plot


if __name__ == '__main__':

    start = timeit.default_timer()
    plot = return_phasespace_plot()
    save = True
    if save:
        # plt.savefig(home+'/Desktop/phase_space_rnn.pdf', transparent=True, bbox_inches='tight')
        plt.savefig(home+'/Desktop/phase_space_rome.pdf', transparent=True, bbox_inches='tight')
        plt.show()
    else:
        plt.show()
    stop = timeit.default_timer()
    print('Script needed: {} seconds'.format(stop - start))