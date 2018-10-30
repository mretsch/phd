import math as m
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


def run_metrics(artificial=False):
    # c = Client()

    if artificial:
        conv_0 = af.art
    else:
        files = "Steiner/CPOL_STEINER_ECHO_CLASSIFICATION_season0910.nc"
        ds_st = xr.open_mfdataset("/Users/mret0001/Data/"+files, chunks={'time': 100})

        stein  = ds_st.steiner_echo_classification
        conv   = stein.where(stein == 2)
        conv_0 = conv.fillna(0.)

    props = []
    labeled = np.zeros_like(conv_0).astype(int)
    for i, scene in enumerate(conv_0):  # conv has dimension (time, lat, lon). A scene is a lat-lon slice.
        labeled[i, :, :] = skm.label(scene, background=0)  # , connectivity=1)
        props.append(skm.regionprops(labeled[i, :, :]))

    all_pairs = [Pairs(pairlist=list(gen_tuplelist(cloudlist))) for cloudlist in props]

    cop = xr.DataArray([conv_org_pot(pairs=p) for p in all_pairs])

    m1  = xr.DataArray([metric_1(clouds=cloudlist) for cloudlist in props])

    # get cop a time dimension.
    cop.coords['time'] = ('dim_0', conv_0.time)
    cop = cop.rename({'dim_0': 'time'})
    m1.coords['time'] = ('dim_0', conv_0.time)
    m1 = m1.rename({'dim_0': 'time'})

    return cop, m1


if __name__ == '__main__':
    start = timeit.default_timer()

    # compute the metrics
    cop, m1 = run_metrics(artificial=False)

    # a quick histrogram
    cop.plot.hist(bins=55)
    plt.title('COP distribution, sample size: '+str(cop.notnull().sum().values))
    plt.show()
    #t = cop.time.where(cop > 0.4, drop=True)
    #highcop = conv.sel(time=t)
    #highcop[0, :, :].plot()
    #plt.show()

    # save as netcdf-files
    xr.save_mfdataset([xr.Dataset({'cop': cop}), xr.Dataset({'m1': m1})],
                      ['../../Data/Analysis/cop_new.nc', '../../Data/Analysis/m1_new.nc'])

    stop = timeit.default_timer()
    print('This script needed {} seconds.'.format(stop-start))
