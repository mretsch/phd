import numpy as np
import xarray as xr

def large_scale_at_metric_times(ds_largescale, timeseries,
                                chosen_vars=None,
                                l_normalise_input=True,
                                l_take_scalars=False,
                                l_take_same_time=False,
                                l_take_only_predecessor_time=False,
                                l_take_also_predecessor_time=False,
                                l_take_only_successor_time=False):
    """Returns a concatenated array of the large-scale variables and the time series, at times of both being present.
    chosen_vars selects some variables out of the large-scale state dataset."""

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
    for var in chosen_vars:
        if var != 'dwind_dz':
            # add extra length to variable name, such that additional info can be added later
            names_list.extend([ds_largescale[var].long_name + '            ' for _ in range(len(ds_largescale[var][:, :-1].lev))])
            variable_size.append(len(names_list) - sum(variable_size))
        else:
            names_list   .extend([ds_largescale['dwind_dz'].long_name for _ in range(len(ds_largescale['dwind_dz'][:, :-2].lev))])
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
            # timeseries at +6h is necessarily a value,
            # because it is back at times of var_metric, where timeseries is a number.
            target = timeseries.sel(time=(var_6earlier_nonull.time + np.timedelta64(6, 'h')).values)
            predictor = var.sel(time=target.time - np.timedelta64(6, 'h'))
            predictor['long_name'][:] = \
                [name.item().replace( '            ', ', 6h earlier') for name in predictor['long_name']]

        if l_take_also_predecessor_time:
            # choose times of var_nonull which follow a time in var_6earlier_nonull
            var_time0_nonull = var_nonull.sel(time=(var_6earlier_nonull.time + np.timedelta64(6, 'h')))
            # first create the right array with the correct 'time=0' time steps
            var_both_times = xr.concat([var_time0_nonull, var_time0_nonull], dim='lev')
            half = int(len(var_both_times.lev) / 2)
            # fill one half with values from earlier time step
            var_both_times[:, half:] = var_6earlier_nonull.values
            var_both_times['long_name'][half:] = \
                [name.item().replace( '            ', ', 6h earlier') for name in var_both_times['long_name'][half:]]

            target = timeseries.sel(time=var_both_times.time.values)
            predictor = var_both_times

        if l_take_only_successor_time:
            var_nonull_6later = var_nonull.time + np.timedelta64(6, 'h')
            times = []
            for t in var_nonull_6later:
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
                [name.item().replace( '            ', ', 6h   later') for name in predictor['long_name']]

    return predictor, target, variable_size


def subselect_ls_vars(large_scale, levels=None):

    # variable_list = [
    #     'vertical velocity            ',
    #     'Horizontal temperature Advection            ',
    #     'Horizontal r advection            ',
    #     'd(dry static energy)/dt            ',
    #     'd(water vapour mixing ratio)/dt            ',
    #     'Relative humidity            ',
    #     'Horizontal wind U component            ',
    #     'Horizontal wind V component            ',
    #     'Convective Inhibition            ',
    #     'Convective Available Potential Energy            ',
    #     'Satellite-measured low cloud            ',
    #     'Surface downwelling LW            ',
    #     '10m wind speed            ',
    #     '10m V component            ',
    #     '2m water vapour mixing ratio            ',
    #     'TOA LW flux, upward positive            ',
    #     'Surface sensible heat flux, upward positive            ',
    #     'MWR-measured cloud liquid water path            ',
    #
    #     'vertical velocity, 6h earlier',
    #     'Horizontal temperature Advection, 6h earlier',
    #     'Horizontal r advection, 6h earlier',
    #     'd(dry static energy)/dt, 6h earlier',
    #     'd(water vapour mixing ratio)/dt, 6h earlier',
    #     'Relative humidity, 6h earlier',
    #     'Horizontal wind U component, 6h earlier',
    #     'Horizontal wind V component, 6h earlier',
    #     'Convective Inhibition, 6h earlier',
    #     'Convective Available Potential Energy, 6h earlier',
    #     'Satellite-measured low cloud, 6h earlier',
    #     'Surface downwelling LW, 6h earlier',
    #     '10m wind speed, 6h earlier',
    #     '10m V component, 6h earlier',
    #     '2m water vapour mixing ratio, 6h earlier',
    #     'TOA LW flux, upward positive, 6h earlier',
    #     'Surface sensible heat flux, upward positive, 6h earlier',
    #     'MWR-measured cloud liquid water path, 6h earlier',
    # ]

    # select a few levels of a few variables which might be relevant to explain ROME

    # var1  = large_scale.where(large_scale['long_name'] == 'vertical velocity            ',
    #                           drop=True).sel(lev=levels)
    # var2  = large_scale.where(large_scale['long_name'] == 'Horizontal temperature Advection            ',
    #                           drop=True).sel(lev=levels)
    # var3  = large_scale.where(large_scale['long_name'] == 'Horizontal r advection            ',
    #                           drop=True).sel(lev=levels)
    # var4  = large_scale.where(large_scale['long_name'] == 'd(dry static energy)/dt            ',
    #                           drop=True).sel(lev=levels)
    # var5  = large_scale.where(large_scale['long_name'] == 'd(water vapour mixing ratio)/dt            ',
    #                           drop=True).sel(lev=levels)
    # var6  = large_scale.where(large_scale['long_name'] == 'Relative humidity            ',
    #                           drop=True).sel(lev=levels)
    # var7  = large_scale.where(large_scale['long_name'] == 'Horizontal wind U component            ',
    #                           drop=True).sel(lev=levels)
    # var8  = large_scale.where(large_scale['long_name'] == 'Horizontal wind V component            ',
    #                           drop=True).sel(lev=levels)
    #
    # var9  = large_scale.where(large_scale['long_name'] == 'Convective Inhibition            ',
    #                           drop=True)
    # var10 = large_scale.where(large_scale['long_name'] == 'Convective Available Potential Energy            ',
    #                           drop=True)
    # var11 = large_scale.where(large_scale['long_name'] == 'Satellite-measured low cloud            ',
    #                           drop=True)
    # var12 = large_scale.where(large_scale['long_name'] == 'Surface downwelling LW            ',
    #                           drop=True)
    # var13 = large_scale.where(large_scale['long_name'] == '10m wind speed            ',
    #                           drop=True)
    # var14 = large_scale.where(large_scale['long_name'] == '10m V component            ',
    #                           drop=True)
    # var15 = large_scale.where(large_scale['long_name'] == '2m water vapour mixing ratio            ',
    #                           drop=True)
    # var16 = large_scale.where(large_scale['long_name'] == 'TOA LW flux, upward positive            ',
    #                           drop=True)
    # var17 = large_scale.where(large_scale['long_name'] == 'Surface sensible heat flux, upward positive            ',
    #                           drop=True)
    # var18 = large_scale.where(large_scale['long_name'] == 'MWR-measured cloud liquid water path            ',
    #                           drop=True)

    # var19 = large_scale.where(large_scale['long_name'] == 'vertical velocity, 6h earlier',
    #                           drop=True).sel(lev=levels)
    # var20 = large_scale.where(large_scale['long_name'] == 'Horizontal temperature Advection, 6h earlier',
    #                           drop=True).sel(lev=levels)
    # var21 = large_scale.where(large_scale['long_name'] == 'Horizontal r advection, 6h earlier',
    #                           drop=True).sel(lev=levels)
    # var22 = large_scale.where(large_scale['long_name'] == 'd(dry static energy)/dt, 6h earlier',
    #                           drop=True).sel(lev=levels)
    # var23 = large_scale.where(large_scale['long_name'] == 'd(water vapour mixing ratio)/dt, 6h earlier',
    #                           drop=True).sel(lev=levels)
    # var24 = large_scale.where(large_scale['long_name'] == 'Relative humidity, 6h earlier',
    #                           drop=True).sel(lev=levels)
    # var25 = large_scale.where(large_scale['long_name'] == 'Horizontal wind U component, 6h earlier',
    #                           drop=True).sel(lev=levels)
    # var26 = large_scale.where(large_scale['long_name'] == 'Horizontal wind V component, 6h earlier',
    #                           drop=True).sel(lev=levels)
    #
    # var27 = large_scale.where(large_scale['long_name'] == 'Convective Inhibition, 6h earlier',
    #                           drop=True)
    # var28 = large_scale.where(large_scale['long_name'] == 'Convective Available Potential Energy, 6h earlier',
    #                           drop=True)
    # var29 = large_scale.where(large_scale['long_name'] == 'Satellite-measured low cloud, 6h earlier',
    #                           drop=True)
    # var30 = large_scale.where(large_scale['long_name'] == 'Surface downwelling LW, 6h earlier',
    #                           drop=True)
    # var31 = large_scale.where(large_scale['long_name'] == '10m wind speed, 6h earlier',
    #                           drop=True)
    # var32 = large_scale.where(large_scale['long_name'] == '10m V component, 6h earlier',
    #                           drop=True)
    # var33 = large_scale.where(large_scale['long_name'] == '2m water vapour mixing ratio, 6h earlier',
    #                           drop=True)
    # var34 = large_scale.where(large_scale['long_name'] == 'TOA LW flux, upward positive, 6h earlier',
    #                           drop=True)
    # var35 = large_scale.where(large_scale['long_name'] == 'Surface sensible heat flux, upward positive, 6h earlier',
    #                           drop=True)
    # var36 = large_scale.where(large_scale['long_name'] == 'MWR-measured cloud liquid water path, 6h earlier',
    #                           drop=True)

    var37 = large_scale.where(large_scale['long_name'] == 'vertical velocity, 6h   later',
                            drop=True).sel(lev=levels)
    var38 = large_scale.where(large_scale['long_name'] == 'Horizontal temperature Advection, 6h   later',
                            drop=True).sel(lev=levels)
    var39 = large_scale.where(large_scale['long_name'] == 'Horizontal r advection, 6h   later',
                            drop=True).sel(lev=levels)
    var40 = large_scale.where(large_scale['long_name'] == 'd(dry static energy)/dt, 6h   later',
                            drop=True).sel(lev=levels)
    var41 = large_scale.where(large_scale['long_name'] == 'd(water vapour mixing ratio)/dt, 6h   later',
                            drop=True).sel(lev=levels)
    var42 = large_scale.where(large_scale['long_name'] == 'Relative humidity, 6h   later',
                            drop=True).sel(lev=levels)
    var43 = large_scale.where(large_scale['long_name'] == 'Horizontal wind U component, 6h   later',
                            drop=True).sel(lev=levels)
    var44 = large_scale.where(large_scale['long_name'] == 'Horizontal wind V component, 6h   later',
                            drop=True).sel(lev=levels)

    var45 = large_scale.where(large_scale['long_name'] == 'Convective Inhibition, 6h   later',
                            drop=True)
    var46 = large_scale.where(large_scale['long_name'] == 'Convective Available Potential Energy, 6h   later',
                            drop=True)
    var47 = large_scale.where(large_scale['long_name'] == 'Satellite-measured low cloud, 6h   later',
                            drop=True)
    var48 = large_scale.where(large_scale['long_name'] == 'Surface downwelling LW, 6h   later',
                            drop=True)
    var49 = large_scale.where(large_scale['long_name'] == '10m wind speed, 6h   later',
                            drop=True)
    var50 = large_scale.where(large_scale['long_name'] == '10m V component, 6h   later',
                            drop=True)
    var51 = large_scale.where(large_scale['long_name'] == '2m water vapour mixing ratio, 6h   later',
                            drop=True)
    var52 = large_scale.where(large_scale['long_name'] == 'TOA LW flux, upward positive, 6h   later',
                            drop=True)
    var53 = large_scale.where(large_scale['long_name'] == 'Surface sensible heat flux, upward positive, 6h   later',
                            drop=True)
    var54 = large_scale.where(large_scale['long_name'] == 'MWR-measured cloud liquid water path, 6h   later',
                            drop=True)

    return xr.concat([
        # var1, var2, var3, var4,
        # var5, var6, var7, var8,
        # var9, var10, var11, var12,
        # var13, var14, var15, var16,
        # var17, var18,
        # var19, var20,
        # var21, var22, var23, var24,
        # var25, var26, var27, var28,
        # var29, var30, var31, var32,
        # var33, var34, var35, var36,
        var37, var38,
        var39, var40, var41, var42,
        var43, var44, var45, var46,
        var47, var48, var49, var50,
        var51, var52, var53, var54,
    ], dim='lev')
