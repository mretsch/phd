from os.path import expanduser
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import timeit
import cartopy.crs as ccrs
import cartopy.io as cio
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

home = expanduser("~")
start = timeit.default_timer()

plt.rc('font'  , size=21)
plt.rc('legend', fontsize=21)

ds_steiner = xr.open_mfdataset(home+'/Data/Steiner/*season0910*', chunks=40)
steiner = ds_steiner.steiner_echo_classification

# create mask to have line around nan-region
radar_mask = xr.where(steiner[28].isnull(), 1, 0)
lons = steiner.lon[:]
lats = steiner.lat[:]


def make_map(projection=ccrs.PlateCarree()):
    fig, ax = plt.subplots(figsize=(4,4),
                           subplot_kw=dict(projection=projection))
    #gl = ax.gridlines(draw_labels=True)
    #gl.xlabels_top = gl.ylabels_right = False
    #gl.xformatter = LONGITUDE_FORMATTER
    #gl.yformatter = LATITUDE_FORMATTER
    return fig, ax


fig, ax = make_map(projection=ccrs.PlateCarree())

# The actual part of the world we'll see in the plot
width = 0 # degrees east,west,north,south
extent = [min(lons) - width, max(lons) + width, min(lats) - width, max(lats) + width] # Darwin [lon,lon,lat,lat]
ax.set_extent(extent)

# The land data:
shp = cio.shapereader.Reader('/Users/mret0001/Data/OpenStreetMaps/land-polygons-complete-4326/Australia_polygons')
for record, geometry in zip(shp.records(), shp.geometries()):
    ax.add_geometries([geometry], ccrs.PlateCarree(), facecolor='lightgray', edgecolor='black')

ax.contour(lons, lats, radar_mask, colors='k', linewidths=2, levels=1, transform=ccrs.PlateCarree())

ax.set_yticks(- np.arange(11,14,0.5))
#ax.set_yticklabels(labels=np.arange(11,14,0.5))
ax.set_yticklabels(labels=['11', '', '12', '', '13', ''])
ax.set_ylabel('Latitude [$^\circ$S]')
#ax.set_xticks([130., 131., 132.])
ax.axes.set_xticks([130, 130.5, 131, 131.5, 132])
ax.axes.set_xticklabels(labels=['130', '', '131', '', '132'])
ax.axes.set_xlabel('Longitude [$^\circ$E]')

fig.savefig(home+'/Desktop/land_sea_CPOL.pdf', transparent=True, bbox_inches='tight')
plt.show()

stop = timeit.default_timer()
print('Script needed %.1f seconds.' % (stop - start))

