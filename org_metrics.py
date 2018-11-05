import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import timeit
import skimage.measure as skm
# from dask.distributed import Client
import artificial_fields as af


class Pairs:

    def __init__(self, pairlist):
        self.pairlist = pairlist
        if pairlist:
            self.partner1, self.partner2 = zip(*pairlist)
        else:
            self.partner1, self.partner2 = [], []

    def distance(self):
        """The distance in units of pixels between two centroids of cloud-objects found by .regionprops."""

        dist_x = np.array([c.centroid[1] for c in self.partner1]) - \
                 np.array([c.centroid[1] for c in self.partner2])
        dist_y = np.array([c.centroid[0] for c in self.partner1]) - \
                 np.array([c.centroid[0] for c in self.partner2])
        return np.sqrt(dist_x**2 + dist_y**2)


def gen_shortlist(start, inlist):
    """List of iterator items starting at 'start', not 0."""
    for j in range(start, len(inlist)):
        yield inlist[j]


def gen_tuplelist(inlist):
    """List of tuples of all possible unique pairs in an iterator."""
    for i, item1 in enumerate(inlist):
        for item2 in gen_shortlist(start=i + 1, inlist=inlist):
            yield item1, item2


def conv_org_pot(pairs):
    """The Convective Organisation Potential according to [White et al. 2018]"""
    if not pairs.pairlist:
        return np.nan
    diameter_1 = np.array([c.equivalent_diameter for c in pairs.partner1])
    diameter_2 = np.array([c.equivalent_diameter for c in pairs.partner2])
    v = np.array(0.5 * (diameter_1 + diameter_2) / pairs.distance())
    return np.sum(v) / len(pairs.pairlist)


def metric_1(clouds):
    if not clouds:
        return np.nan
    c_area = xr.DataArray([c.area for c in clouds])
    a_max = c_area.max()
    a_all = c_area.sum()
    return a_max / a_all * a_max


def n_objects(clouds):
    if not clouds:
        return np.nan
    return len(clouds)


def avg_area(clouds):
    if not clouds:
        return np.nan
    return xr.DataArray([c.area for c in clouds]).mean()


def run_metrics(artificial=False, file="Steiner/CPOL_STEINER_ECHO_CLASSIFICATION_threedays.nc"):
    # c = Client()

    if artificial:
        conv_0 = af.art
    else:
        ds_st = xr.open_mfdataset("/Users/mret0001/Data/"+file, chunks={'time': 100})

        stein  = ds_st.steiner_echo_classification
        conv   = stein.where(stein == 2)
        conv_0 = conv.fillna(0.)

    props = []
    labeled = np.zeros_like(conv_0).astype(int)
    for i, scene in enumerate(conv_0):  # conv has dimension (time, lat, lon). A scene is a lat-lon slice.
        labeled[i, :, :] = skm.label(scene, background=0)  # , connectivity=1)
        props.append(skm.regionprops(labeled[i, :, :]))

    all_pairs = [Pairs(pairlist=list(gen_tuplelist(cloudlist))) for cloudlist in props]

    # compute the metrics
    cop      = xr.DataArray([conv_org_pot(pairs=p) for p in all_pairs])

    m1       = xr.DataArray([metric_1(clouds=cloudlist) for cloudlist in props])

    o_number = xr.DataArray([n_objects(clouds=cloudlist) for cloudlist in props])

    o_area   = xr.DataArray([avg_area(clouds=cloudlist) for cloudlist in props])

    # put together a dataset from the different metrices
    ds_m = xr.Dataset({'cop': cop,
                       'm1': m1,
                       'o_number': o_number,
                       'o_area': o_area
                       })

    # get metrics a time dimension.
    ds_m.coords['time'] = ('dim_0', conv_0.time)
    ds_m = ds_m.rename({'dim_0': 'time'})

    return ds_m


if __name__ == '__main__':
    start = timeit.default_timer()

    # compute the metrics
    ds_metric = run_metrics(artificial=False,
                            file="Steiner/CPOL_STEINER_ECHO_CLASSIFICATION_season*.nc")

    # a quick histrogram
    ds_metric.cop.plot.hist(bins=55)
    plt.title('COP distribution, sample size: ' + str(ds_metric.cop.notnull().sum().values))
    plt.show()

    # save metrics as netcdf-files
    for var in ds_metric.variables:
        xr.Dataset({var: ds_metric[var]}).to_netcdf('/Users/mret0001/Data/Analysis/'+var+'_new.nc')

#    # set bins to group metric-dataset into its seasons
#    #TODO does binning work with these time stamps? No!
#    season_bins = ['07-2009', '07-2010', '07-2011', '07-2012', '07-2013', '07-2014', '07-2015', '07-2016', '07-2017']
#    _, ds_seasons = zip(*ds_metric.groupby_bins('time', season_bins))
#
#    # save as netcdf-files
#    years = ['0910', '1011', '1112', '1213', '1314', '1415', '1516', '1617']
#    paths = []  # ['/Users/mret0001/Data/Analysis/metrics_season'+season+'_new.nc' for season in years]
#    for i in range(len(ds_seasons)):
#        paths.append('/Users/mret0001/Data/Analysis/metrics_season'+years[i]+'_new.nc')
#    xr.save_mfdataset(ds_seasons, paths)

    stop = timeit.default_timer()
    print('This script needed {} seconds.'.format(stop-start))
