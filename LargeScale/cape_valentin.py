from os.path import expanduser
import netCDF4
import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
import xarray as xr
import matplotlib.pyplot as plt
home = expanduser("~")

with netCDF4.Dataset(home+'/Data/LargeScaleState/CPOL_large-scale_forcing.nc', 'r') as ncid:
    time     = netCDF4.num2date(ncid['time'][:], ncid['time'].units)
    lev      = ncid['lev'][:].filled(np.NaN)  # Pressure hPa
    sfc_pres = ncid['p_srf_aver'][:].filled(np.NaN)
    temp     = ncid['T'][:].filled(np.NaN)  # Kelvin
    sfc_temp = ncid['T_srf'][:].filled(np.NaN)
    wvmr     = 1e-3 * ncid['r'][:].filled(np.NaN)  # convert to kg/kg
    sfc_wvmr = 1e-3 * ncid['r_srf'][:].filled(np.NaN)

print('Time length: ', len(time))

# From dimension to variable.
press = np.zeros_like(temp)
for cnt in range(temp.shape[0]):
    press[cnt, :] = lev

# replace bottommost level with surface variables
press[:, 0] = sfc_pres
temp [:, 0] = sfc_temp + 273.15
wvmr [:, 0] = sfc_wvmr

pressure     = press * units.hPa
temperature  = temp  * units.K
mixing_ratio = wvmr  * units('kg/kg')

# get dew point
relative_humidity = mpcalc.relative_humidity_from_mixing_ratio(mixing_ratio, temperature, pressure)
e                 = mpcalc.vapor_pressure(pressure, mixing_ratio)
dew_point         = mpcalc.dewpoint(e)


def get_cape(inargs, return_parcel_profile=False):
    pres_prof, temp_prof, dp_prof = inargs
    try:
        prof = mpcalc.parcel_profile(pres_prof, temp_prof[0], dp_prof[0])
        cape, cin = mpcalc.cape_cin(pres_prof, temp_prof, dp_prof, prof)
    except Exception:
        cape, cin, prof = np.NaN, np.NaN, np.NaN
        print('Problem during CAPE-calculation. Likely NaN-related.')
    if return_parcel_profile:
        return cape, cin, prof
    else:
        return cape, cin


arg_list = []
for cnt in range(len(time)):
    arg_list.append((pressure[cnt, :], temperature[cnt, :], dew_point[cnt, :]))

single_time = True
if single_time:
    time_idx = (time == np.datetime64('2009-12-18T06:00:00')).argmax()
    idx = time_idx # max(cape)=16444 # min(cape)=16815 # min(cin)=2934 # max(sfc_RH)=6570 # min(sfc_RH)=16748 # max(cin)=7248
    cape, cin, parcel_profile = get_cape(arg_list[idx], return_parcel_profile=True)

    plot_profiles = True
    if plot_profiles:
        plt.plot(temperature[idx, :], pressure[idx, :],  lw=2)# range(0, 40),
        plt.plot(dew_point  [idx, :], pressure[idx, :],  lw=2)# range(0, 40),
        plt.plot(parcel_profile     , pressure[idx, :],  lw=1)# range(0, 40),
        plt.gca().invert_yaxis()
        plt.grid()
        plt.title(time[idx].isoformat())
        plt.text(x=140, y=500, s='CIN='+str(cin.magnitude)+'\n'+'CAPE='+str(cape.magnitude))
        plt.legend(['Env. temp.', 'Dew point temp.', 'Parcel temp.'])
        plt.savefig(home+'/Desktop/profile.pdf', bbox_inches='tight', transparent=True)

else:
    cape_cin = [get_cape(arg) for arg in arg_list]
    t_cape, t_cin = zip(*cape_cin)

    # create arrays for cape and cin
    ls = xr.open_dataset(home+'/Data/LargeScaleState/CPOL_large-scale_forcing.nc')
    array_1d = ls.T_srf
    da_cape = xr.zeros_like(array_1d)
    da_cin  = xr.zeros_like(array_1d)


    def gen_tupleitem_return(tuple):
        for item in tuple:
            try:
                yield item.magnitude
            except AttributeError:
                yield item


    da_cape[:] = xr.DataArray(list(gen_tupleitem_return(t_cape)))
    da_cin [:] = xr.DataArray(list(gen_tupleitem_return(t_cin)))

    da_cape.attrs['long_name'] = 'Convective Available Potential Energy'
    da_cape.attrs['units'] = 'J/kg'
    cape = da_cape.rename('cape')

    da_cin.attrs['long_name'] = 'Convective Inhibition'
    da_cin.attrs['units'] = 'J/kg'
    cin = da_cin.rename('cin')

    # ls_cc = xr.merge([ls, cape, cin])
    # ls_cc.to_netcdf(home+'/Desktop/CPOL_large-scale_forcing_cape_cin.nc')