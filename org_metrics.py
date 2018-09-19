import math as m
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import timeit
import skimage.measure as skm

start = timeit.default_timer()

files = "Steiner/CPOL_STEINER_ECHO_CLASSIFICATION_oneday.nc"
ds_st = xr.open_mfdataset("/Users/mret0001/Data/"+files, chunks={'time': 1000})

stein  = ds_st.steiner_echo_classification
conv   = stein.where(stein == 2)
conv_0 = conv.fillna(0.)

props = []
labeled = np.zeros_like(conv_0).astype(int)
for i, scene in enumerate(conv_0):  # conv has dimension (time, lat, lon). A scene is a lat-lon slice.
    labeled[i, :, :] = skm.label(scene, background=0)
    props.append(skm.regionprops(labeled[i, :, :]))


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


all_pairs = [Pairs(pairlist=list(gen_tuplelist(cloudlist))) for cloudlist in props]

test = all_pairs[28].distance()

########################################################################################################

def v_potential(pairs):
    if pairs.pairlist == []: return np.nan
    partner1 = list(list(zip(*pairs.pairlist))[0])
    partner2 = list(list(zip(*pairs.pairlist))[1])
    diameter_1 = np.array(list(map(lambda c: c.equivalent_diameter, partner1)))
    diameter_2 = np.array(list(map(lambda c: c.equivalent_diameter, partner2)))

    v = np.array(0.5 * diameter_1 + diameter_2 / pairs.distance())
    return np.sum(v) / len(pairs.pairlist)


def cop(allpairs):
    """The Convective Organisation Potential according to [White et al. 2018]"""

    # get the 'interaction potential', v, for every pair of objects/clouds
    v = list(map(v_potential, allpairs))

    return v


v_pot = cop(all_pairs)

stop = timeit.default_timer()
print('This script needed {} seconds.'.format(stop-start))
