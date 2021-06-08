import numpy as np
import xarray as xr

def large_scale_at_metric_times(ds_largescale, timeseries,
                                chosen_vars=None,
                                l_normalise_input=True,
                                l_take_scalars=False,
                                large_scale_time=None,
                                l_profiles_as_eof=False):
    """Returns a concatenated array of the large-scale variables and the time series, at times of both being present.
    chosen_vars selects some variables out of the large-scale state dataset."""

    if large_scale_time not in ['same_time', 'same_and_earlier_time', 'only_earlier_time', 'only_later_time', 'all_ls']:
        raise ValueError("String large_scale_time to select large-scale time steps does not match or is not provided.")

    if chosen_vars is None:
        chosen_vars = [
            'T',
            'dTdt',
            'T_adv_h',
            'T_adv_v',
            's',
            'dsdt',
            's_adv_h',
            's_adv_v',
            'r',
            'drdt',
            'r_adv_h',
            'r_adv_v',
            'omega',
            'div',
            'u',
            'v',
            'RH',
            'dwind_dz',
            ]

    if not l_profiles_as_eof:
        # bottom level has redundant information and two bottom levels filled with NaN for dwind_dz
        var_list = [ds_largescale[var][:, :-1] if var != 'dwind_dz' else ds_largescale[var][:, :-2]
                    for var in chosen_vars ]
        height_dim = 'lev'
    else:
        var_list = [ds_largescale[var][:, :] for var in chosen_vars]
        height_dim = 'number'


    c1 = xr.concat(var_list, dim=height_dim)

    if l_take_scalars:
        scalars = [
            'cin',
            'cape',
            'lw_net_toa',
            'SH',
            'LH',
            'PW',
        ]

        c2 = xr.concat([ds_largescale[scalar] for scalar in scalars], dim='concat_dims')

    # give c1 another coordinate to look up 'easily' which values in concatenated array correspond to which variables
    # Also count how long that variable is in the resulting array.
    names_list, symbl_list, variable_size = [], [], []
    for var_string in chosen_vars:
        if not l_profiles_as_eof:
            if var_string != 'dwind_dz':
                last_index = -1
            else:
                last_index = -2
            # add extra length to variable name, such that additional info can be added later
            names_list.extend([ds_largescale[var_string].long_name + '            ' for
                               _ in range(len(ds_largescale[var_string][:, :last_index][height_dim]))])
            symbl_list.extend([ds_largescale[var_string].symbol    + '            ' for
                               _ in range(len(ds_largescale[var_string][:, :last_index][height_dim]))])
            variable_size.append(len(names_list) - sum(variable_size))
        else:
            # add extra length to variable name, such that additional info can be added later
            names_list.extend([ds_largescale[var_string].long_name + '            ' for
                               _ in range(len(ds_largescale[var_string][height_dim]))])
            symbl_list.extend([ds_largescale[var_string].symbol    + '            ' for
                               _ in range(len(ds_largescale[var_string][height_dim]))])
            variable_size.append(len(names_list) - sum(variable_size))

    c1.coords['long_name'] = (height_dim, names_list)
    c1.coords['symbol']    = (height_dim, symbl_list)

    if l_take_scalars:
        c2_r = c2.rename({'concat_dims': height_dim})
        c2_r.coords[height_dim] = np.arange(len(c2))
        symbl_list = [ds_largescale[scalar].symbol    + '            ' for scalar in scalars]
        names_list = [ds_largescale[scalar].long_name + '            ' for scalar in scalars]
        c2_r.coords['symbol']    = (height_dim, symbl_list)
        c2_r.coords['long_name'] = (height_dim, names_list)

        var = xr.concat([c1, c2_r], dim=height_dim)
    else:
        var = c1

    if l_normalise_input:
        var_copy = var.copy(deep=True)
        var_std = (var - var.mean(dim='time')) / var.std(dim='time')
        # where std_dev=0., dividing led to NaN, set to 0. instead
        var = var_std.where(var_std.notnull(), other=0.)
        # put the 'original' NaN back into array
        var = xr.where(var_copy.isnull(), var_copy, var)

    if large_scale_time == "all_ls":
        # only return the large scale variables, without considering anything else
        predictor = var.where(var.notnull(), drop=True)
        target = timeseries.sel(time=predictor.time)

    if l_profiles_as_eof:
        # remove all 'series' at a level/number which consists exclusively of NaN
        var = var.where(var.notnull(), drop=True)

    # large scale variables only where timeseries is defined
    var_metric = var.where(timeseries.notnull(), drop=True)

    # boolean for the large scale variables without any NaN anywhere
    l_var_nonull = var_metric.notnull().all(dim=height_dim)

    # large-scale state variables at same time as timeseries, or not
    if large_scale_time == 'same_time':
        predictor = var_metric[{'time': l_var_nonull}]
        target = timeseries.sel(time=predictor.time)

    elif large_scale_time in ['only_earlier_time', 'same_and_earlier_time', 'only_later_time']:
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


def subselect_ls_vars(large_scale, profiles, levels_in=None, large_scale_time=None, l_profiles_as_eof=False):

    if large_scale_time not in ['same_time', 'same_and_earlier_time', 'only_earlier_time', 'only_later_time']:
        raise ValueError("String large_scale_time to select large-scale time steps does not match or is not provided.")

    if not l_profiles_as_eof:
        height_dim='lev'
    else:
        height_dim='number'

    scalars = [
        'Convective Inhibition',
        'Convective Available Potential Energy',
        'TOA LW flux, upward positive',
        'Surface sensible heat flux, upward positive',
        'Surface latent heat flux, upward positive',
        'MWR-measured column precipitable water',
    ]

    ls_list = []

    if large_scale_time == 'same_and_earlier_time':
        for profile_string in profiles:
            if profile_string != 'Vertical wind shear':
                levels = levels_in
                if profile_string == 'Dry static energy':  # dont take 990hPa of s because it's correlated to RH_990
                    levels = levels_in[:-1]
                # if profile_string == 'Horizontal wind U component':
                #     levels = levels_in[2]
            else:
                levels = levels_in[:-1] + [965]
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
                if profile_string == 'Dry static energy':  # dont take 990hPa of s because it's correlated to RH_990
                    levels = levels_in[:-1]
                # if profile_string == 'Horizontal wind U component':
                #     levels = levels_in[2]
            else:
                levels = levels_in[:-1] + [965]
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
            if not l_profiles_as_eof:
                if profile_string != 'Vertical wind shear':
                    levels = levels_in
                    if profile_string == 'Dry static energy':  # dont take 990hPa of s because it's correlated to RH_990
                        levels = levels_in[:-1]
                else:
                    levels = levels_in[:-1] + [965]

                ls_list.append(
                    large_scale.where(large_scale['long_name'] == profile_string + string_selection,
                                      drop=True).sel(lev=levels)
                )
            else:
                if profile_string in ['Relative humidity', 'Horizontal r advection']:
                    levels = slice(1, None) # slice(1, 4)
                else:
                    levels = slice(None, None) # slice(None, 3)
                ls_list.append(
                    large_scale.where(large_scale['long_name'] == profile_string+string_selection,
                                      drop=True).sel(number=levels)
                )

        for scalar_string in scalars:
            ls_list.append(
                large_scale.where(large_scale['long_name'] == scalar_string+string_selection,
                                  drop=True)
            )

    return xr.concat(ls_list, dim=height_dim)
