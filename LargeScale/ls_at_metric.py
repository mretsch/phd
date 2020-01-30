import numpy as np
import xarray as xr

def large_scale_at_metric_times(ds_largescale, timeseries,
                                chosen_vars=None,
                                l_take_scalars=False,
                                l_take_same_time=False,
                                l_take_only_predecessor_time=False):
    """Returns a concatenated array of the large-scale variables and the time series, at times of both being present.
    chosen_vars selects some variables out of the large-scale state dataset."""

    if chosen_vars is None:
        chosen_vars = ['omega', 'div', 'T_adv_h', 'T_adv_v', 'r_adv_h', 'r_adv_v',
                       's_adv_h', 's_adv_v', 'dsdt', 'drdt', 'RH']
    var_list =     [ds_largescale[var]       [:, :-1] for var in chosen_vars] # ! bottom level has redundant information
    var_list.append(ds_largescale['dwind_dz'][:, :-2]) # ! two bottom levels filled with NaN

    c1 = xr.concat(var_list, dim='lev')

    if l_take_scalars:
        c2 = xr.concat([
              ds_largescale.cin
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
    for var in chosen_vars:
        names_list.extend([ds_largescale[var].long_name for _ in range(len(ds_largescale[var][:, :-1].lev))])
        variable_size.append(len(names_list) - sum(variable_size))
    names_list   .extend([ds_largescale['dwind_dz'].long_name for _ in range(len(ds_largescale['dwind_dz'][:, :-2].lev))])
    variable_size.append(len(names_list) - sum(variable_size))

    c1.coords['long_name'] = ('lev', names_list)

    if l_take_scalars:
        c2_r = c2.rename({'concat_dims': 'lev'})
        c2_r.coords['lev'] = np.arange(len(c2))
        names_list = []
        names_list.append(ds_largescale.cin       .long_name)
        names_list.append(ds_largescale.cld_low   .long_name)
        names_list.append(ds_largescale.lw_dn_srf .long_name)
        names_list.append(ds_largescale.wspd_srf  .long_name)
        names_list.append(ds_largescale.v_srf     .long_name)
        names_list.append(ds_largescale.r_srf     .long_name)
        names_list.append(ds_largescale.lw_net_toa.long_name)
        names_list.append(ds_largescale.SH        .long_name)
        names_list.append(ds_largescale.LWP       .long_name)
        c2_r.coords['long_name'] = ('lev', names_list)

        var = xr.concat([c1, c2_r], dim='lev')
    else:
        var = c1

    # large scale variables only where timeseries is defined
    var_metric = var.where(timeseries.notnull(), drop=True)

    # boolean for the large scale variables without any NaN anywhere
    l_var_nonull = var_metric.notnull().all(dim='lev')

    # large-scale state variables at same time as timeseries, or not
    if l_take_same_time:
        predictor = var_metric[{'time': l_var_nonull}]
        target = timeseries.sel(time=predictor.time)

    else:
        var_nonull = var_metric[l_var_nonull]
        var_nonull_6earlier = var_nonull.time - np.timedelta64(6, 'h')
        times = []
        for t in var_nonull_6earlier:
            try:
                _ = var_metric.sel(time=t)
                times.append(t.values)
            except KeyError:
                continue
        var_6earlier = var_metric.sel(time=times)
        var_6earlier_nonull = var_6earlier[var_6earlier.notnull().all(dim='lev')]

        if l_take_only_predecessor_time:
            # timeseries 6h later is necessarily a value, because back at times of var_metric, where timeseries is a number.
            target = timeseries.sel(time=(var_6earlier_nonull.time + np.timedelta64(6, 'h')).values)
            predictor = var.sel(time=target.time - np.timedelta64(6, 'h'))
        else:
            var_6later_nonull = var_nonull.sel(time=(var_6earlier_nonull.time + np.timedelta64(6, 'h')))
            # first 'create' the right array with the correct 'late' time steps
            var_both_times = xr.concat([var_6later_nonull, var_6later_nonull], dim='lev')
            half = int(len(var_both_times.lev) / 2)
            # fill one half with values from earlier time step
            var_both_times[:, half:] = var_6earlier_nonull.values
            var_both_times['long_name'][half:] = \
                [name.item() + ', 6h earlier' for name in var_both_times['long_name'][half:]]

            target = timeseries.sel(time=var_both_times.time.values)
            predictor = var_both_times

    return predictor, target, variable_size
