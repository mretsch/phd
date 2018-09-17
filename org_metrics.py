import math as m
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import timeit
import skimage.measure as skm

start = timeit.default_timer()

files = "Steiner/CPOL_STEINER_ECHO_CLASSIFICATION_oneday.nc"
ds_st = xr.open_mfdataset("/Users/mret0001/Data/"+files, chunks={'time': 1000})

stein = ds_st.steiner_echo_classification
conv = stein.where(stein == 2)

props = []
labeled = np.zeros_like(conv).astype(int)
for i, scene in enumerate(conv):  # conv has dimension (time, lat, lon). A scene is a lat-lon slice.
    labeled[i, :, :] = skm.label(scene)
    props.append(skm.regionprops(labeled[i, :, :]))

# props[28][1].centroid
# TODO: DONE above command give error, maybe cause of conflicting metadata inherited from 'labeled' initialisation


class Partners:

    def __init__(self, partnerlist):
        self.partnerlist = partnerlist

    def distance(self):
        """The distance in units of pixels between two centroids of objects found by .regionprops."""
        # partnerlist is a list of tuples
        partner1 = list(list(zip(*self.partnerlist))[0])
        partner2 = list(list(zip(*self.partnerlist))[1])

        dist_x = np.array(list(map(lambda o: o.centroid[1], partner1))) - \
                 np.array(list(map(lambda o: o.centroid[1], partner2)))
        dist_y = np.array(list(map(lambda o: o.centroid[0], partner1))) - \
                 np.array(list(map(lambda o: o.centroid[0], partner2)))
        return np.sqrt(dist_x**2 + dist_y**2)


#def get_partners(list):
#    pass
p = Partners(partnerlist=[(props[28][1],props[28][8]),(props[28][2],props[28][3]),(props[28][4],props[28][5])])
d = p.distance()

# For COP [White et al. 2018] just use (props.equivalent_diameter / 2.) to get the radius needed for COP.
#def cop():
#    dist = distance()
#    pass


stop = timeit.default_timer()
print('This script needed {} seconds.'.format(stop-start))
