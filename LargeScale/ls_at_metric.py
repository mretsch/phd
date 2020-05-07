import numpy as np
import xarray as xr

def large_scale_at_metric_times(ds_largescale, timeseries,
                                chosen_vars=None,
                                l_normalise_input=True,
                                l_take_scalars=False,
                                large_scale_time=None):
    """Returns a concatenated array of the large-scale variables and the time series, at times of both being present.
    chosen_vars selects some variables out of the large-scale state dataset."""

    if large_scale_time not in ['same_time', 'same_and_earlier_time', 'only_earlier_time', 'only_later_time']:
        raise ValueError("String large_scale_time to select large-scale time steps does not match or is not provided.")

    if chosen_vars is None:
        chosen_vars = ['omega', 'div', 'T_adv_h', 'T_adv_v', 'r_adv_h', 'r_adv_v',
                       's_adv_h', 's_adv_v', 'dsdt', 'drdt', 'RH', 'u', 'v', 'dwind_dz']
    # bottom level has redundant information and two bottom levels filled with NaN for dwind_dz
    var_list = [ds_largescale[var][:, :-1] if var != 'dwind_dz' else ds_largescale[var][:, :-2]
                for var in chosen_vars ]

    c1 = xr.concat(var_list, dim='lev')

    if l_take_scalars:
        scalars = [
            'cin',
            'cape',
            'cld_low',
            'lw_dn_srf',
            'wspd_srf',
            'v_srf',
            'r_srf',
            'lw_net_toa',
            'SH',
            'LWP',

            'LH',
            'p_srf_aver',
            'T_srf',
            # 'T_skin', # correlation to other variables higher than 0.8
            'RH_srf',
            'u_srf',
            # 'rad_net_srf', # correlation to other variables higher than 0.8
            # 'sw_net_toa', # correlation to other variables higher than 0.8
            'cld_mid',
            'cld_high',
            # 'cld_tot',
            'dh2odt_col',
            'h2o_adv_col',
            # 'evap_srf', # correlation to other variables too high (according to statsmodels)
            'dsdt_col',
            # 's_adv_col', # correlation to other variables too high (according to statsmodels)
            # 'rad_heat_col', # correlation to other variables too high (according to statsmodels)
            # 'LH_col', # correlation to other variables higher than 0.8
            # 'r_srf', # correlation to other variables too high (according to statsmodels)
            's_srf',
            'PW',
            # 'lw_up_srf', # correlation to other variables higher than 0.8
            # 'lw_dn_srf', # correlation to other variables too high (according to statsmodels)
            # 'sw_up_srf', # has same long_name as sw_dn_srf (according to statsmodels)
            # 'sw_dn_srf', # correlation to other variables higher than 0.8
        ]

        c2 = xr.concat([ds_largescale[scalar] for scalar in scalars])

    # give c1 another coordinate to look up 'easily' which values in concatenated array correspond to which variables
    # Also count how long that variable is in the resulting array.
    names_list, variable_size = [], []
    for var_string in chosen_vars:
        if var_string != 'dwind_dz':
            last_index = -1
        else:
            last_index = -2
        # add extra length to variable name, such that additional info can be added later
        names_list.extend([ds_largescale[var_string].long_name + '            ' for
                           _ in range(len(ds_largescale[var_string][:, :last_index].lev))])
        variable_size.append(len(names_list) - sum(variable_size))

    c1.coords['long_name'] = ('lev', names_list)

    if l_take_scalars:
        c2_r = c2.rename({'concat_dims': 'lev'})
        c2_r.coords['lev'] = np.arange(len(c2))
        names_list = [ds_largescale[scalar].long_name + '            ' for scalar in scalars]
        c2_r.coords['long_name'] = ('lev', names_list)

        var = xr.concat([c1, c2_r], dim='lev')
    else:
        var = c1

    if l_normalise_input:
        var_copy = var.copy(deep=True)
        var_std = (var - var.mean(dim='time')) / var.std(dim='time')
        # where std_dev=0., dividing led to NaN, set to 0. instead
        var = var_std.where(var_std.notnull(), other=0.)
        # put the 'original' NaN back into array
        var = xr.where(var_copy.isnull(), var_copy, var)

    # large scale variables only where timeseries is defined
    var_metric = var.where(timeseries.notnull(), drop=True)

    # boolean for the large scale variables without any NaN anywhere
    l_var_nonull = var_metric.notnull().all(dim='lev')

    # large-scale state variables at same time as timeseries, or not
    if large_scale_time == 'same_time':
        predictor = var_metric[{'time': l_var_nonull}]
        target = timeseries.sel(time=predictor.time)

    else:
        var_nonull = var_metric[l_var_nonull]
        time_nonull_6earlier = var_nonull.time - np.timedelta64(6, 'h')
        times = []
        for t in time_nonull_6earlier:
            try:
                _ = var_metric.sel(time=t)
                times.append(t.values)
            except KeyError:
                continue
        var_6earlier = var_metric.sel(time=times)
        var_6earlier_nonull = var_6earlier[var_6earlier.notnull().all(dim='lev')]

        if large_scale_time == 'only_earlier_time':
            # timeseries at +6h is necessarily a value,
            # because it is back at times of var_metric, where timeseries is a number.
            target = timeseries.sel(time=(var_6earlier_nonull.time + np.timedelta64(6, 'h')).values)
            predictor = var.sel(time=target.time - np.timedelta64(6, 'h'))
            predictor['long_name'][:] = \
                [name.item().replace('            ', ', 6h earlier') for name in predictor['long_name']]

        if large_scale_time == 'same_and_earlier_time':
            # choose times of var_nonull which follow a time in var_6earlier_nonull
            var_time0_nonull = var_nonull.sel(time=(var_6earlier_nonull.time + np.timedelta64(6, 'h')))
            # first create the right array with the correct 'time=0' time steps
            var_both_times = xr.concat([var_time0_nonull, var_time0_nonull], dim='lev')
            half = int(len(var_both_times.lev) / 2)
            # fill one half with values from earlier time step
            var_both_times[:, half:] = var_6earlier_nonull.values
            var_both_times['long_name'][half:] = \
                [name.item().replace('            ', ', 6h earlier') for name in var_both_times['long_name'][half:]]

            target = timeseries.sel(time=var_both_times.time.values)
            predictor = var_both_times

        if large_scale_time == 'only_later_time':
            time_nonull_6later = var_nonull.time + np.timedelta64(6, 'h')
            times = []
            for t in time_nonull_6later:
                try:
                    _ = var_metric.sel(time=t)
                    times.append(t.values)
                except KeyError:
                    continue
            var_6later = var_metric.sel(time=times)
            var_6later_nonull = var_6later[var_6later.notnull().all(dim='lev')]

            # timeseries at -6h is necessarily a value,
            # because it is back at times of var_metric, where timeseries is a number.
            target = timeseries.sel(time=(var_6later_nonull.time - np.timedelta64(6, 'h')).values)
            predictor = var.sel(time=target.time + np.timedelta64(6, 'h'))
            predictor['long_name'][:] = \
                [name.item().replace('            ', ', 6h   later') for name in predictor['long_name']]

    return predictor, target, variable_size


def subselect_ls_vars(large_scale, profiles, levels_in=None, large_scale_time=None):

    if large_scale_time not in ['same_time', 'same_and_earlier_time', 'only_earlier_time', 'only_later_time']:
        raise ValueError("String large_scale_time to select large-scale time steps does not match or is not provided.")

    scalars = [
        'Convective Inhibition',
        'Convective Available Potential Energy',
        'Satellite-measured low cloud',
        'Surface downwelling LW',
        '10m wind speed',
        '10m V component',
        '2m water vapour mixing ratio',
        'TOA LW flux, upward positive',
        'Surface sensible heat flux, upward positive',
        'MWR-measured cloud liquid water path',

        'Surface latent heat flux, upward positive',
        'Surface pressure averaged over the domain',
        '2m air temperature',
        # 'Surface skin temperature', # correlation to other variables higher than 0.8
        '2m air relative humidity',
        '10m U component',
        # 'Surface net radiation, downward positive', # correlation to other variables higher than 0.8
        # 'TOA net SW flux, downward positive', # correlation to other variables higher than 0.8
        'Satellite-measured middle cloud',
        'Satellite-measured high cloud',
        # 'Satellite-measured total cloud', # correlation to other variables higher than 0.8
        'Column-integrated dH2O/dt',
        'Column-integrated H2O advection',
        # 'Surface evaporation', # correlation to other variables too high (according to statsmodels)
        'Column d(dry static energy)/dt',
        # 'Column dry static energy advection', # correlation to other variables too high (according to statsmodels)
        # 'Column radiative heating', # correlation to other variables too high (according to statsmodels)
        # 'Column latent heating', # correlation to other variables higher than 0.8
        # '2m water vapour mixing ratio', # correlation to other variables too high (according to statsmodels)
        '2m dry static energy',
        'MWR-measured column precipitable water',
        # 'Surface upwelling LW', # correlation to other variables higher than 0.8
        # 'Surface downwelling LW', # correlation to other variables too high (according to statsmodels)
        # 'Surface downwelling SW', # has same long_name as sw_dn_srf (according to statsmodels)
        # 'Surface downwelling SW', # correlation to other variables higher than 0.8
    ]

    ls_list = []

    if large_scale_time == 'same_and_earlier_time':
        for profile_string in profiles:
            if profile_string != 'Vertical wind shear':
                levels = levels_in
            else:
                levels = [115, 515, 965]
            ls_list.append(
                large_scale.where(large_scale['long_name'] == profile_string+'            ',
                                  drop=True).sel(lev=levels)
            )

        for scalar_string in scalars:
            ls_list.append(
                large_scale.where(large_scale['long_name'] == scalar_string+'            ',
                                  drop=True)
            )

        for profile_string in profiles:
            if profile_string != 'Vertical wind shear':
                levels = levels_in
            else:
                levels = [115, 515, 965]
            ls_list.append(
                large_scale.where(large_scale['long_name'] == profile_string+', 6h earlier',
                                  drop=True).sel(lev=levels)
            )

        for scalar_string in scalars:
            ls_list.append(
                large_scale.where(large_scale['long_name'] == scalar_string+', 6h earlier',
                                  drop=True)
            )

    else:

        if large_scale_time == 'same_time':
            string_selection = '            '
        if large_scale_time == 'only_earlier_time':
            string_selection = ', 6h earlier'
        if large_scale_time == 'only_later_time':
            string_selection = ', 6h   later'

        for profile_string in profiles:
            if profile_string != 'Vertical wind shear':
                levels = levels_in
            else:
                levels = [115, 515, 965]
            ls_list.append(
                large_scale.where(large_scale['long_name'] == profile_string+string_selection,
                                  drop=True).sel(lev=levels)
            )

        for scalar_string in scalars:
            ls_list.append(
                large_scale.where(large_scale['long_name'] == scalar_string+string_selection,
                                  drop=True)
            )

    return xr.concat(ls_list, dim='lev')
