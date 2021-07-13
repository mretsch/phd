from os.path import expanduser
home = expanduser("~")
import timeit
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
import cartopy.io as cio
from Plotscripts.colors_solarized import sol


def smallregion_in_tropics(tropic_wide_field, region, surface_type, fillvalue):


    land_sea = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/land_sea_avg.nc')

    land_sea['lat'] = tropic_wide_field['lat']
    land_sea['lon'] = tropic_wide_field['lon']

    if surface_type == 'ocean':
        surface_mask = land_sea < 0.2
        field_ocean  = tropic_wide_field.where(surface_mask, other=fillvalue)
    if surface_type == 'coast':
        surface_mask = (0.2 < land_sea) & (land_sea < 0.8)
        field_coast  = tropic_wide_field.where(surface_mask, other=fillvalue)
    if surface_type == 'land':
        surface_mask = land_sea > 0.8
        field_land   = tropic_wide_field.where(surface_mask, other=fillvalue)

    field = tropic_wide_field

    if 'North Pacific' in region:
        pac = field_ocean.where(((160 < field_ocean['lon']) | (field_ocean['lon'] < -90)) &
                                (   0 < field_ocean['lat']),
                                other=fillvalue)
        small_region = pac
    if 'Pacific Region 1' in region:
        pa1 = field.where((( 170 < field['lon']) | (field['lon'] < -178)) &
                          ((   6 < field['lat']) & (field['lat'] <    8)),
                          other=fillvalue)
        small_region = pa1
    if 'Pacific Region 2' in region:
        pa2 = field.where(((-145 < field['lon']) & (field['lon'] < -133)) &
                          ((   6 < field['lat']) & (field['lat'] <    8)),
                          other=fillvalue)
        small_region = pa2
    if 'Pacific Region 3' in region:
        pa3 = field.where(((-145 < field['lon']) & (field['lon'] < -139)) &
                          ((  14 < field['lat']) & (field['lat'] <   20)),
                          other=fillvalue)
        small_region = pa3
    if 'Amazon delta' in region:
        ama = field.where((-52 < field['lon']) & (field['lon'] < -44) &
                          ( -1 < field['lat']) & (field['lat'] <   5),
                          other=fillvalue)
        small_region = ama
    if 'NW Australia' in region:
        aus = field_coast.where((120 < field_coast['lon']) & (field_coast['lon'] < 134) &
                                (-20 < field_coast['lat']) & (field_coast['lat'] < -10),
                                other=fillvalue)
        small_region = aus
    if 'South of India' in region:
        ind = field.where(( 77 < field['lon']) & (field['lon'] < 82) &
                          (-12 < field['lat']) & (field['lat'] < -6),
                          other=fillvalue)
        small_region = ind

    return small_region


start = timeit.default_timer()
plt.rc('font'  , size=20)

rome_highfreq = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/rome_10mmhour.nc')
rome_3h = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/rome_10mmhour_3hmaxavg.nc')
rh_3h = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/rh500_3hmaxavg_by_rome10mm.nc')
div = xr.open_dataarray(home+'/Documents/Data/Simulation/r2b10/Divergence900/div900_avg.nc').transpose('time', 'lon', 'lat')

rome = rome_3h.sel(time=div['time'].values)
rh   = rh_3h.  sel(time=div['time'].values)

rome_thresh = np.nanpercentile(rome_highfreq, 90)
dry_thresh = 40
moist_thresh = 70

# rome_rh_mask = (rome > rome_thresh) & ((rh < dry_thresh) | (moist_thresh < rh))
rome_dry_mask = (rome > rome_thresh) &  (rh < dry_thresh)
rome_moist_mask = (rome > rome_thresh) &  (moist_thresh < rh)

div_vector_list = []
i=-1
fig, ax = plt.subplots(1, 1)
for r_mask, region in zip([rome_dry_mask], ['NW Australia']):

    i += 1

    l_whole_tropics = False
    if l_whole_tropics:
        short_rome_mask = r_mask.isel({'time': slice(4, -4)}).values

        rome_and_rh    = xr.where(r_mask, rome, np.nan)
        div_at_rome    = xr.where(r_mask, div , np.nan)
        rome_rh_div    = rome_and_rh.where(div_at_rome.notnull())
        rh_div         = xr.         where(div_at_rome.notnull(), rh, np.nan)

        l_concat_to_vectors = False
        if l_concat_to_vectors:
            div_flat  = xr.DataArray(np.ravel(div_at_rome))
            rome_flat = xr.DataArray(np.ravel(rome_rh_div))
            rh_flat   = xr.DataArray(np.ravel(rh_div))

            div_vector_list.append(div_flat[div_flat.notnull()])
            rome_vector = rome_flat[rome_flat.notnull()]
            rh_vector   = rh_flat[rh_flat.notnull()]

            ax.hist(div_vector_list[-1], bins=np.linspace(-5e-5, 2.5e-5, 40), density=True)

    else: # select by region
        region_mask = smallregion_in_tropics(tropic_wide_field=r_mask, region=region, surface_type='coast',
                                             fillvalue=bool(0))
        short_rome_mask = region_mask.isel({'time': slice(4, -4)}).values

    avg_7timesteps, std_7timesteps = [], []

    slices_int = [
        slice(None,   -8),
        slice(   1,   -7),
        slice(   2,   -6),
        slice(   3,   -5),
        slice(   4,   -4),
        slice(   5,   -3),
        slice(   6,   -2),
        slice(   7,   -1),
        slice(   8, None),
    ]

    for time_ints in slices_int:
        div_timeshifted = div.isel({'time': time_ints})
        avg_7timesteps.append(div_timeshifted.where(short_rome_mask).mean())
        std_7timesteps.append(div_timeshifted.where(short_rome_mask).std() )

    if i==0:
        sol_col = sol['red']
    else:
        sol_col = sol['blue']
    plt.plot(np.array(avg_7timesteps), label=region, lw=2, color=sol_col)
    plt.fill_between(x=np.arange(len(avg_7timesteps)),
                     y1=np.array(avg_7timesteps)-np.array(std_7timesteps),
                     y2=np.array(avg_7timesteps)+np.array(std_7timesteps),
                     alpha=0.1,
                     color=sol_col)

plt.axvline(x=len(avg_7timesteps)//2, color='lightgrey', zorder=0)
plt.axhline(y=0, color='lightgrey', zorder=0)
plt.legend()
plt.ylim(-2e-5, 0.5e-5)
plt.ylabel('Divergence [1/s]')
plt.xlim(0, len(avg_7timesteps)-1)
plt.xlabel('Time around high ROME [h]')
plt.xticks(ticks=np.arange(len(avg_7timesteps)), labels=['-12', '-9', '-6', '-3', '0', '3', '6', '9', '12'])
plt.savefig(home+'/Desktop/div_composite_tropics.pdf', bbox_inches='tight')
plt.show()

# earliest_div = div.isel({'time': slice(None, -4)})
# early_div    = div.isel({'time': slice(   1, -3)})
# same_div     = div.isel({'time': slice(   2, -2)})
# late_div     = div.isel({'time': slice(   3, -1)})
# latest_div   = div.isel({'time': slice(   4, None)})
#
# earliest_div_avg = earliest_div.where(short_rome_mask).mean()
# early_div_avg = early_div.where(short_rome_mask).mean()
# same_div_avg = same_div.where(short_rome_mask).mean()
# late_div_avg = late_div.where(short_rome_mask).mean()
# latest_div_avg = latest_div.where(short_rome_mask).mean()


stop = timeit.default_timer()
print(f'Time used: {stop - start}')