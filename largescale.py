import xarray as xr
import numpy as np

ls = xr.open_dataset('/Users/matthiasretsch/Google Drive File Stream/My Drive/Data/LargeScale/CPOL_large-scale_forcing.nc')

# level 0 is not important because quantities at level 0 have repeated value from level 1 there.
lev_pres  = ls.lev[1:]    # [hPa] constant level heights
# srf_pres  = ls.p_srf_aver  # [hPa]
# srf_mix_ratio = ls.r_srf  # [g/kg]
# srf_temp  = ls.T_srf + 273.15 # [K]
temp      = ls.T[:, 1:]   # [K]
mix_ratio = ls.r[:, 1:]   # [g/kg] water vapour mixing ratio

# a date where surface pressure is below the pressure of the bottommost level (below 1015 hPa)
# srf_pres[srf_pres == srf_pres.min()]
# in that case the value for the actually non-existent bottommost level is copied from the level above
# temp.sel(time='2014-02-11T06:00:00')
# spec_hum.sel(time='2014-02-11T06:00:00')
# and surface pressure is always below 1015, hence the bottommost level is always a redundant value...
# no it should just be a unique value which then is valid for a different pressure than 1015...

global R_d, g
g = 9.81  # gravitational acceleration
R_d = 287.1  # dry air gas constant
R_w = 461.5 # water vapour gas constant
c_p = 1005
gamma_d = - g / c_p
l_ent = 2.501e6 # specific vaporisation enthalpy

def mixratio_to_spechum(mixing_ratio, pressure):
    """Given water vapour mixing ratio [g/kg] and ambient pressure [hPa] return specific humidity [kg/kg]."""
    # convert to SI units
    p = pressure * 100
    r = mixing_ratio * 1e-3

    vapour_pres = r * p / 0.622
    return 0.622 * vapour_pres / (p - 0.378 * vapour_pres)


def temp_to_virtual(temperature, spec_hum):
    """Calculate virtual temperature [K] given ambient temperature [K] and specific humidity [kg/kg]."""
    return temperature * (1 + 0.608 * spec_hum)


def delta_height(p_levels, virt_temp):
    # xarray first matches the coordinate/dimension values and then divides at each matching coordinate value
    # pressure_ratio = p_levels[:-1] / p_levels[1:]
    pressure_ratio = p_levels[:-1].values / p_levels[1:].values

    mean_temp_v = 0.5 * (virt_temp[:, :-1].values + virt_temp[:, 1:].values)
    dz    = xr.zeros_like(virt_temp[:, 1:])
    dz[:] = np.log(pressure_ratio) * (R_d / g) * mean_temp_v
    dz    = dz.assign_attrs({'units':'m', 'Bottom pressure': '990 hPa'})
    return dz


# def total_height(surface_pressure, surface_temperature, surface_mixratio, p_levels, virt_temp):
    # virtual temperature at 2 meters height
    # q_srf = mixratio_to_spechum(surface_mixratio, surface_pressure)
    # temp_v_srf = temp_to_virtual(surface_temperature, q_srf)

    # pressure_ratio = surface_pressure.values / p_levels[0].values
    # mean_temp_v = 0.5 * (temp_v_srf.values + virt_temp[:, 0].values)
    # delta_z_bottom = np.log(pressure_ratio) * (R / g) * mean_temp_v

#     # all the surface variables are mostly NaN...


def mixratio_to_dewpoint(mixing_ratio, pressure):
    # convert to SI units
    p = pressure * 100
    r = mixing_ratio * 1e-3

    vapour_pres = r * p / 0.622
    ln_term = np.log(vapour_pres / 610.78)
    # the inverted Magnus-formula e_s(T)
    return (235 * ln_term / (17.1 - ln_term)) + 273.15


def d_dewpoint_d_height(dewpoint):
    """Takes dewpoint at a lower level (no layer mean value) and returns the change of dewpoint with height [K/m]"""
    return - R_w * g * dewpoint / (R_d * l_ent)


def lifting_condensation_level(temperature, dewpoint, ddew_dheight):
    diff_temp = temperature[:, 0] - dewpoint[:, 0]
    diff_gradients = ddew_dheight[:, 0] - gamma_d
    return diff_temp / diff_gradients


# preliminaries
spec_hum = mixratio_to_spechum(mix_ratio, lev_pres)
temp_v = temp_to_virtual(temp, spec_hum)
delta_z = delta_height(lev_pres, temp_v)


# let a parcel ascent (forced), starting at 990hPa,
# first adiabatically to lifting condensation level, then moist-adiabatically
temp_dew = mixratio_to_dewpoint(mix_ratio, lev_pres)
dTd_dz = d_dewpoint_d_height(temp_dew)
lcl = lifting_condensation_level(temp, temp_dew, dTd_dz)

