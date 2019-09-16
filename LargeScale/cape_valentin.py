# %%


import os
import glob
import warnings
import traceback

from multiprocessing.dummy import Pool

import netCDF4
import numpy as np
import matplotlib.pyplot as pl
import metpy.calc as mpcalc

from metpy.units import units
from matplotlib.colors import LogNorm

# %%

with netCDF4.Dataset('/Users/mret0001/Data/LargeScaleState/CPOL_large-scale_forcing.nc', 'r') as ncid:
    time = netCDF4.num2date(ncid['time'][:], ncid['time'].units)
    temp = ncid['T'][:].filled(np.NaN)  # Kelvin
    sfc_temp = ncid['T_srf'][:].filled(np.NaN) # MHR
    lev = ncid['lev'][:].filled(np.NaN)  # Pressure hPa
    sfc_pres = ncid['p_srf_aver'][:].filled(np.NaN) # MHR
    u = ncid['u'][:].filled(np.NaN)  # m/s
    v = ncid['v'][:].filled(np.NaN)  # m/s
    wvmr = 1e-3 * ncid['r'][:].filled(np.NaN)  # convert to kg/kg
    sfc_wvmr = 1e-3 * ncid['r_srf'][:].filled(np.NaN) # MHR

# %%

len(time)

# %%

# From dimension to variable.
press = np.zeros_like(temp)
for cnt in range(temp.shape[0]):
    press[cnt, :] = lev

# %%

press[:, 0] = sfc_pres # MHR
temp[:, 0] = sfc_temp + 273.15# MHR
wvmr[:, 0] = sfc_wvmr # MHR

pressure = press * units.hPa
temperature = temp * units.K
mixing_ratio = wvmr * units('kg/kg')

# %%

relative_humidity = mpcalc.relative_humidity_from_mixing_ratio(mixing_ratio, temperature, pressure)

# %%

e = mpcalc.vapor_pressure(pressure, mixing_ratio)

# %%

dew_point = mpcalc.dewpoint(e)


# %%

def get_cape(inargs):
    pres_prof, temp_prof, dp_prof = inargs
    try:
        prof = mpcalc.parcel_profile(pres_prof, temp_prof[0], dp_prof[0])
        cape, cin = mpcalc.cape_cin(pres_prof, temp_prof, dp_prof, prof)
    except Exception:
        cape, cin = np.NaN, np.NaN
    #         print('Problem')

    return cape, cin


# %%

arg_list = []
for cnt in range(50):
    arg_list.append((pressure[cnt, :], temperature[cnt, :], dew_point[cnt, :]))

# %%

cape, cin = get_cape(arg_list[0])

# %%

cape

# %%


