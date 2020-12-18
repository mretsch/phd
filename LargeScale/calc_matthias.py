from os.path import expanduser
home = expanduser("~")
import xarray as xr
import numpy as np
import metpy.calc as mpcalc
import metpy.constants as mpconsts
from metpy.units import units
import matplotlib.pyplot as plt

# global R_d, g
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

    vapour_pres = r * p / (0.622 + r)
    return 0.622 * vapour_pres / (p - 0.378 * vapour_pres)


def mixingratio_to_relativehumidity(pressure, temperature, mixing_ratio):
        actual_vapour_pressure = mpcalc.vapor_pressure(pressure * units['hPa'],
                                                       mixing_ratio * units['g/kg']).to_base_units()
        sat_vapour_pressure = mpcalc.saturation_vapor_pressure(temperature * units['degC']).to_base_units()
        rh = actual_vapour_pressure / sat_vapour_pressure
        return rh


def temp_to_virtual(temperature, spec_hum):
    """Calculate virtual temperature [K] given ambient temperature [K] and specific humidity [kg/kg]."""
    return temperature * (1 + 0.608 * spec_hum)


def delta_height(p_levels, virt_temp):
    # xarray first matches the coordinate/dimension values and then divides at each matching coordinate value
    # if only given: pressure_ratio = p_levels[:-1] / p_levels[1:]

    l_isotherm = False
    if l_isotherm:
        if take_lower_level:
            pressure_ratio = p_levels[:, :-1].values / p_levels[:, 1:].values
        else:
            pressure_ratio = p_levels[:-1].values / p_levels[1:].values

        mean_temp_v = 0.5 * (virt_temp[:, :-1].values + virt_temp[:, 1:].values)
        dz    = xr.zeros_like(virt_temp[:, 1:])
        dz[:] = np.log(pressure_ratio) * (R_d / g) * mean_temp_v
        dz    = dz.assign_attrs({'units':'m', 'Bottom pressure': 'above/at 990 hPa'})

    l_lineargradient = not l_isotherm
    if l_lineargradient:
        if take_lower_level:
            pressure_ratio = p_levels[:, 1:].values / p_levels[:, :-1].values
        else:
            pressure_ratio = p_levels[1:].values / p_levels[:-1].values

        # the mean temperature gradient of the atmosphere [K/m]
        gamma = -0.0065
        dz    = xr.zeros_like(virt_temp[:, 1:])
        dz[:] = (virt_temp[:, :-1].values / gamma) * (1 - pressure_ratio**(gamma * R_d / g))
        dz    = dz.assign_attrs({'units':'m', 'Bottom pressure': 'above/at 990 hPa'})
    return dz


def total_height(height_delta, surface_pressure, surface_temperature, surface_mixratio, p_levels, virt_temp):
    # first calculate the total heights from the levels we already have
    absolute_height = xr.zeros_like(virt_temp)
    for i in range(0, height_delta.shape[1]):
        absolute_height[:, i+1] = height_delta[:, :i+1].sum(dim='lev')

    if take_lower_level:
        height = absolute_height
    else:
        # now for below the lowest level, via quantities at 2 meters height
        q_srf = mixratio_to_spechum(surface_mixratio, surface_pressure)
        temp_v_srf = temp_to_virtual(surface_temperature, q_srf)

        pressure_ratio = surface_pressure.values / p_levels[0].values
        mean_temp_v = 0.5 * (temp_v_srf.values + virt_temp[:, 0].values)
        dz_bottom = np.log(pressure_ratio) * (R_d / g) * mean_temp_v
        delta_z_bottom = xr.zeros_like(surface_pressure)
        delta_z_bottom[:] = dz_bottom

        # add the height below the bottommost level to the total heights
        height = absolute_height + delta_z_bottom.transpose()
    return height

    # all the surface variables are mostly NaN... nope that's the same for all variables, it's the dry seasons
    # and missing years in between 2001 and 2015. Many 6-hours points are NaN therefore.


def mixratio_to_dewpoint(mixing_ratio, pressure):
    # convert to SI units
    p = pressure * 100
    r = mixing_ratio * 1e-3

    vapour_pres = r * p / (0.622 + r)
    ln_term = np.log(vapour_pres / 610.78)
    # the inverted Magnus-formula e_s(T)
    return (235 * ln_term / (17.1 - ln_term)) + 273.15


def temperature_to_satmixratio(temperature, pressure):
    """Given temperature [K] and ambient pressure [hPa] return saturation mixing ratio [kg/kg]."""
    # convert to SI units
    p = pressure * 100

    # get saturation vapour pressure for given temperature via Magnus formula
    e_s = 610.78 * np.exp((17.1 * (temperature - 273.15)) / (235 + (temperature - 273.15)))
    r_sat = 0.622 * e_s / (p - e_s)
    return r_sat


def d_satmixratio_d_height(satmixratio, layer_thickness):
    # delta_rs    = xr.full_like(layer_thickness, np.nan)
    delta_rs = satmixratio[1:].values - satmixratio[:-1].values
    return delta_rs / layer_thickness.values


def moistadiabat_temp_gradient(drs_dz):
    return gamma_d - l_ent / c_p * drs_dz


def d_dewpoint_d_height(dewpoint):
    """Takes dewpoint at a lower level (no layer mean value) and returns the change of dewpoint with height [K/m]"""
    return - R_w * g * dewpoint / (R_d * l_ent)


def lifting_condensation_level(temperature, dewpoint, ddew_dheight):
    diff_temp = temperature[:, 0] - dewpoint[:, 0]
    diff_gradients = ddew_dheight[:, 0] - gamma_d
    lcl_height = diff_temp / diff_gradients
    # disregard negative LCLs. The pressure at 2m might be too high for the given vapour pressure, such that
    # dewpoint > temperature, which is actually not possible. Possibly an error due to model extrapolation and/or
    # averaging the bottom pressure.
    lcl_height[lcl_height < 0] = 0
    return lcl_height


def find_lcl_level(lcl_height, level_height):
    negative_positive = level_height - lcl_height
    nlev = xr.where(negative_positive < 0, True, False).sum(dim='lev')
    return nlev.where(lcl_height.notnull(), other=np.nan)


def parcel_ascent(temperature, level_thickness, nlevel_below_lcl, pressure, mixratio, lcl_height):
    # -1 because of zero-based indexing.
    level_to_switch = nlevel_below_lcl - 1
    temp_ascent = xr.full_like(temperature, fill_value=np.nan)
    # bottom (starting) temperature stays the same
    temp_ascent[:, 0] = temperature[:, 0]
    # switch to moist-adiabatic ascent is at a different level each time step, so loop through time
    for i, _ in enumerate(temp_ascent[:, 0]):
        # also loop through levels, from bottom to top
        # for j, _ in enumerate(temp_ascent.lev):
        j = 0
        while j <= level_to_switch[i]:
            # TODO dont restart at temp for every level again, take the newly coputed temp_ascent instead
            temp_ascent[i, j+1] = temp[i, j] + level_thickness[i, j] * gamma_d
            j += 1
        # now moist-adiabatic ascent for the switching level
        # the satmixratio for the upper level here is actually wrong, because in between the
        # switch to moistadiabat takes place.
        mixratio_sat = temperature_to_satmixratio(temp_ascent[i, j-1:j+1], pressure[i, j-1:j+1])
        # make a switch from dry to moist-adiabatic according to height-ratio
        # for moistadiabat gamma_w I have to get drs_dz. At the point where moist adiabat start it is still
        # the dry one (could get it from the calculation above). But then it changes.
        drs_dz = d_satmixratio_d_height(mixratio_sat, level_thickness[i, j-1])
        gamma_w = moistadiabat_temp_gradient(drs_dz)
        moist_ascent_temp = temp_ascent[i, j-1] + level_thickness[i, j-1] * gamma_w
        k=2

    return temp_ascent


def vertical_wind_shear(u, v):
    diff_u = xr.zeros_like(u[:, 1:])
    diff_u.attrs['long_name'] = 'Horizontal wind U component delta to level below'
    diff_v = xr.zeros_like(v[:, 1:])
    diff_v.attrs['long_name'] = 'Horizontal wind V component delta to level below'

    diff_u[:, :] = u[:, 1:].values - u[:, :-1].values
    diff_v[:, :] = v[:, 1:].values - v[:, :-1].values

    # mpcalc.bulk_shear(ls.lev[1:3].values*units.hPa, ls.u[:1, 1:3].squeeze().values*units('m/s'),
    #                   ls.v[:1, 1:3].squeeze().values*units('m/s'), depth=25.*units.hPa)

    velo_change = np.sqrt(diff_u ** 2 + diff_v ** 2)

    # change of velocity is [m/s], but I want it per meter, which is dwind_dz => [m/s/m] = [1/s]
    thickness = delta_height(ls.lev[1:], ls.T[:, 1:])
    thickness_my = thickness.copy(deep=True)
    thickness_metpy = thickness.copy(deep=True)
    thickness_metpy[:, :] = 0

    for i in range(len(thickness.lev)):
        thickness_metpy[0, i] = mpcalc.thickness_hydrostatic(ls.lev[i + 1:i + 3], ls.T[:1, i + 1:i + 3])[0]

    shear = velo_change / thickness
    shear.attrs['long_name'] = 'Vertical wind shear'
    shear.attrs['units'] = '1/s'

    return xr.merge([ls, xr.Dataset({'dwind_dz': shear})])


def wind_direction(u, v):
    # depending on input dataset, the lev dimension is transposed or not. I want all levels above 990 hPa,
    # not the 1015-level below 990 hPa.
    if ls.u.sel(lev=slice(990, None)).lev.max() == 990:
        wind_dir = xr.full_like(ls.u.sel(lev=slice(990, None)), np.nan)
        wind_dir[:, :] = mpcalc.wind_direction(ls.u.sel(lev=slice(990, None)), ls.v.sel(lev=slice(990, None)))
    else:
        wind_dir = xr.full_like(ls.u.sel(lev=slice(None, 990)), np.nan)
        wind_dir[:, :] = mpcalc.wind_direction(ls.u.sel(lev=slice(None, 990)), ls.v.sel(lev=slice(None, 990)))
    wind_dir.attrs['long_name'] = 'wind direction'
    wind_dir.attrs['units'] = 'degrees'
    return xr.merge([ls, xr.Dataset({'wind_dir': wind_dir})])


def down_cape(p_start=None):

    if p_start not in ls.lev:
        raise ValueError("Please provide pressure of one level of large-scale dataset to start calculating DCAPE from.")

    # find index of p_start
    start_level = int((abs(ls.lev - p_start)).argmin())

    # get temperature and humidity from large-scale state
    temp     = ls.T  .sel(lev=slice(None, 990)).metpy.unit_array
    mix      = ls.r  .sel(lev=slice(None, 990)).metpy.unit_array.to('kg/kg')
    p_vector = ls.lev.sel(lev=slice(None, 990)).metpy.unit_array

    # get dew-point temperature
    vap_pres = mpcalc.vapor_pressure(p_vector, mix)
    dew_temp = mpcalc.dewpoint(vap_pres)

    # pressure levels to integrate over
    p_down = ls.lev.sel(lev=slice(p_start, 990))

    # find NaNs
    l_valid = ls.T[:, start_level].notnull().values

    d_cape   = xr.full_like(ls.cape, np.nan)
    x        = p_down.values
    temp     = temp    [l_valid, :]
    dew_temp = dew_temp[l_valid, :]
    # loop over all non-NaN times in large-scale state
    for i, this_time in enumerate(ls.T[l_valid].time):
        print(i)
        # bug: p_start has to be multiplied with units when given as argument, not beforehand
        wb_temp = mpcalc.wet_bulb_temperature(p_start*units['hPa'], temp[i, start_level], dew_temp[i, start_level])

        # create placeholder for moist adiabat temperature
        moist_adiabat = temp[i, start_level:].to('degC')

        moist_adiabat_below = mpcalc.moist_lapse(p_vector[start_level+1:], wb_temp, p_start*units['hPa'])
        moist_adiabat[0]    = wb_temp
        moist_adiabat[1:]   = moist_adiabat_below

        env_temp  = temp[i, start_level:]
        temp_diff = moist_adiabat - env_temp

        y = temp_diff.magnitude
        d_cape.loc[this_time] = (mpconsts.Rd * (np.trapz(y, np.log(x)) * units.degK)).to(units('J/kg'))

    d_cape.attrs['long_name'] = 'Downward CAPE'
    return xr.merge([ls, xr.Dataset({'d_cape': d_cape})])


if __name__ == '__main__':
    home = home + '/Documents/Data/LargeScaleState/'
    # ls = xr.open_dataset(ghome + '/Data/LargeScale/CPOL_large-scale_forcing_cape990hPa_cin990hPa_rh_shear.nc')
    ls = xr.open_dataset(home + 'CPOL_large-scale_forcing_cape990hPa_cin990hPa_rh_shear_dcape.nc')

    # level 0 is not important because quantities at level 0 have repeated value from level 1 there.
    take_lower_level = True
    if take_lower_level:
        level_pres = ls.lev  # [hPa] constant level heights
        temp = ls.T  # [K]
        mix_ratio = ls.r  # [g/kg] water vapour mixing ratio
    else:
        lev_pres = ls.lev[1:]  # [hPa] constant level heights
        temp = ls.T[:, 1:]  # [K]
        mix_ratio = ls.r[:, 1:]  # [g/kg] water vapour mixing ratio

    srf_pres = ls.p_srf_aver  # [hPa]
    srf_mix_ratio = ls.r_srf  # [g/kg]
    srf_temp = ls.T_srf + 273.15  # [K]

    if take_lower_level:
        temp[:, 0] = srf_temp
        mix_ratio[:, 0] = srf_mix_ratio
        lev_pres = xr.zeros_like(temp)
        lev_pres[:, 0] = srf_pres
        lev_pres[:, 1:] = level_pres[1:]

    # a date where surface pressure is below the pressure of the bottommost level (below 1015 hPa)
    # srf_pres[srf_pres == srf_pres.min()]
    # in that case the value for the actually non-existent bottommost level is copied from the level above
    # temp.sel(time='2014-02-11T06:00:00')
    # spec_hum.sel(time='2014-02-11T06:00:00')
    # and surface pressure is always below 1015, hence the bottommost level is always a redundant value...
    # no it should just be a unique value which then is valid for a different pressure than 1015...

    l_cape = False
    if l_cape:
        # preliminaries
        spec_hum = mixratio_to_spechum(mix_ratio, lev_pres)
        temp_v = temp_to_virtual(temp, spec_hum)
        delta_z = delta_height(lev_pres, temp_v)

        # let a parcel ascent (forced), starting at 990hPa,
        # first adiabatically to lifting condensation level, then moist-adiabatically
        temp_dew = mixratio_to_dewpoint(mix_ratio, lev_pres)
        dTd_dz = d_dewpoint_d_height(temp_dew)
        # the next lines are for the ascending parcel not for the measured ambient quantities
        # mix_ratio_sat = temperature_to_satmixratio(temp, lev_pres)
        # drs_dz = d_satmixratio_d_height(mix_ratio_sat, delta_z)
        # gamma_w = moistadiabat_temp_gradient(drs_dz)
        lcl = lifting_condensation_level(temp, temp_dew, dTd_dz)
        absolute_z = total_height(delta_z, srf_pres, srf_temp, srf_mix_ratio, lev_pres, temp_v)

        # dewpoint higher than actual temp at 2 meter eg here: 2001-11-02T18:00:00

        nlev_below_lcl = find_lcl_level(lcl, absolute_z)

        t = parcel_ascent(temp[:2, :], delta_z[:2, :], nlev_below_lcl[:2], lev_pres, mix_ratio, lcl)

    # ls_new = vertical_wind_shear(ls.u, ls.v)
    # ls_new = wind_direction(ls.u, ls.v)
    # ls_new = down_cape(p_start=515)

    # What happens to moistening (in RH terms) for different starting levels of mixing ratio and temperature?
    mix_diffs = []
    r = np.arange(0, 50)
    for mix in r:
        warmRH = mixingratio_to_relativehumidity(1003.75, 29.3, mix)
        coldRH = mixingratio_to_relativehumidity(1003.75, 28.3, mix)
        mix_diffs.append(coldRH - warmRH)

    temp_diffs = []
    t = np.arange(10, 35)
    for temp in t:
        warmRH = mixingratio_to_relativehumidity(1003.75, temp  , 20)
        coldRH = mixingratio_to_relativehumidity(1003.75, temp-1, 20)
        temp_diffs.append(coldRH - warmRH)

