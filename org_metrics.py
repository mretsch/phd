import math as m
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import timeit
import skimage.measure as skm
from dask.distributed import Client


class Pairs:

    def __init__(self, pairlist):
        self.pairlist = pairlist

    def distance(self):
        """The distance in units of pixels between two centroids of cloud-objects found by .regionprops."""
        # pairlist is a list of tuples
        partner1 = list(list(zip(*self.pairlist))[0])
        partner2 = list(list(zip(*self.pairlist))[1])

        dist_x = np.array(list(map(lambda c: c.centroid[1], partner1))) - \
                 np.array(list(map(lambda c: c.centroid[1], partner2)))
        dist_y = np.array(list(map(lambda c: c.centroid[0], partner1))) - \
                 np.array(list(map(lambda c: c.centroid[0], partner2)))
        return np.sqrt(dist_x**2 + dist_y**2)


def gen_shortlist(start, inlist):
    """List of iterator items starting at 'start', not 0."""
    for j in range(start, len(inlist)):
        yield inlist[j]


def gen_tuplelist(inlist):
    """List of tuples of all possible unique pairs in an iterator."""
    for i, item1 in enumerate(inlist):
        for item2 in gen_shortlist(start=i + 1, inlist=inlist):
            yield (item1, item2)


def conv_org_pot(pairs):
    """The Convective Organisation Potential according to [White et al. 2018]"""
    if pairs.pairlist == []: return np.nan
    partner1 = list(list(zip(*pairs.pairlist))[0])
    partner2 = list(list(zip(*pairs.pairlist))[1])
    diameter_1 = np.array(list(map(lambda c: c.equivalent_diameter, partner1)))
    diameter_2 = np.array(list(map(lambda c: c.equivalent_diameter, partner2)))

    v = np.array(0.5 * (diameter_1 + diameter_2) / pairs.distance())
    return np.sum(v) / len(pairs.pairlist)


if __name__ == '__main__':

    start = timeit.default_timer()

    files = "Steiner/CPOL_STEINER_ECHO_CLASSIFICATION_season0910.nc"
    ds_st = xr.open_mfdataset("/Users/mret0001/Data/"+files, chunks={'time': 1000})

    c = Client()
    stein  = ds_st.steiner_echo_classification
    conv   = stein.where(stein == 2)
    conv_0 = conv.fillna(0.)

    props = []
    labeled = np.zeros_like(conv_0).astype(int)
    for i, scene in enumerate(conv_0):  # conv has dimension (time, lat, lon). A scene is a lat-lon slice.
        labeled[i, :, :] = skm.label(scene, background=0)
        props.append(skm.regionprops(labeled[i, :, :]))

    all_pairs = [Pairs(pairlist=list(gen_tuplelist(cloudlist))) for cloudlist in props]

    cop = xr.DataArray(list(map(conv_org_pot, all_pairs)))
    # TODO: get cop a time dimension. And do a hist plot of cop. And a 2D-hist plot for basic_stats.
    cop.coords['time'] = ('dim_0', conv.time)
    cop = cop.rename({'dim_0': 'time'})

    cop.plot.hist()
    plt.show()
    t = cop.time.where(cop > 0.4, drop=True)
    highcop = conv.sel(time=t)
    highcop[0, :, :].plot()
    plt.show()

    stop = timeit.default_timer()
    print('This script needed {} seconds.'.format(stop-start))
