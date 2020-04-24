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
        c2 = xr.concat([
              ds_largescale.cin
            , ds_largescale.cape
            , ds_largescale.cld_low
            , ds_largescale.lw_dn_srf
            , ds_largescale.wspd_srf
            , ds_largescale.v_srf
            , ds_largescale.r_srf
            , ds_largescale.lw_net_toa
            , ds_largescale.SH
            , ds_largescale.LWP
        ])

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
        names_list = []
        names_list.append(ds_largescale.cin       .long_name + '            ')
        names_list.append(ds_largescale.cape      .long_name + '            ')
        names_list.append(ds_largescale.cld_low   .long_name + '            ')
        names_list.append(ds_largescale.lw_dn_srf .long_name + '            ')
        names_list.append(ds_largescale.wspd_srf  .long_name + '            ')
        names_list.append(ds_largescale.v_srf     .long_name + '            ')
        names_list.append(ds_largescale.r_srf     .long_name + '            ')
        names_list.append(ds_largescale.lw_net_toa.long_name + '            ')
        names_list.append(ds_largescale.SH        .long_name + '            ')
        names_list.append(ds_largescale.LWP       .long_name + '            ')
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


def subselect_ls_vars(large_scale, levels=None, large_scale_time=None):

    if large_scale_time not in ['same_time', 'same_and_earlier_time', 'only_earlier_time', 'only_later_time']:
        raise ValueError("String large_scale_time to select large-scale time steps does not match or is not provided.")

    # select a few levels of a few variables which might be relevant to explain ROME
    profiles = [
        'vertical velocity',
        'Horizontal temperature Advection',
        'Horizontal r advection',
        'd(dry static energy)/dt',
        'd(water vapour mixing ratio)/dt',
        'Relative humidity',
        'Horizontal wind U component',
        'Horizontal wind V component',
    ]

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
    ]

    ls_list = []

    if large_scale_time == 'same_and_earlier_time':
        for profile_string in profiles:
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
