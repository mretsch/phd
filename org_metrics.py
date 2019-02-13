import collections
import functools
import math as m
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import timeit
import skimage.measure as skm
# import artificial_fields as af
# import random_fields as rf
import scipy as sp
import shapely.geometry as spg
import shapely.ops as spo


class Pairs:

    def __init__(self, pairlist):
        self.pairlist = pairlist
        if pairlist:
            self.partner1, self.partner2 = zip(*pairlist)
        else:
            self.partner1, self.partner2 = [], []

    def __len__(self):
        return len(self.pairlist)

    def distance_regionprops(self):
        """The distance in units of pixels between two centroids of cloud-objects found by skm.regionprops."""

        dist_x = np.array([c.centroid[1] for c in self.partner1]) - \
                 np.array([c.centroid[1] for c in self.partner2])
        dist_y = np.array([c.centroid[0] for c in self.partner1]) - \
                 np.array([c.centroid[0] for c in self.partner2])
        return np.sqrt(dist_x**2 + dist_y**2)

    def distance_shapely(self):
        """The shortest distance in units of pixels between edges of cloud-objects given by shapely.MultiPolygon."""

        return np.array([self.pairlist[i][0].distance(self.pairlist[i][1]) for i in range(len(self.pairlist))])


def gen_shortlist(start, inlist):
    """Iterator items starting at 'start', not 0."""
    for j in range(start, len(inlist)):
        yield inlist[j]


def gen_tuplelist(inlist):
    """Tuples of all possible unique pairs in an iterator. For one element only in iterator, yields tuple with two
    times the same item."""
    if len(inlist) == 1:
        yield inlist[0], inlist[0]
    else:
        for i, item1 in enumerate(inlist):
            for item2 in gen_shortlist(start=i + 1, inlist=inlist):
                yield item1, item2


def _gen_shapely_objects(in_func):
    """Decorator returning shapely objects in each slice of 3D-array."""

    @functools.wraps(in_func)
    def wrapper(array):
        for scene in array:  # array has dimension (time, lat, lon). A scene is a lat-lon slice.
            labeled = skm.label(scene, background=0)  # , connectivity=1)
            objects = skm.regionprops(labeled)

            objects = in_func(objects)

            # apply find_contours on each regionprop object, then make shapely out of it. To have correct order.
            polys = []
            for o in objects:
                # the layout and properties of the regionprops-object
                layout = o.image.astype(int)
                bounds = o.bbox
                y_length = bounds[2] - bounds[0]
                x_length = bounds[3] - bounds[1]
                # prepare bed to put layout in
                bed = np.zeros(shape=(y_length + 2, x_length + 2))
                bed[1:-1, 1:-1] = layout
                # get the contour needed for shapely
                contour = skm.find_contours(bed, level=0.5, fully_connected='high')
                # increase coordinates to get placement inside of original input array right
                contour[0][:, 0] += bounds[0]  # increase y-values
                contour[0][:, 1] += bounds[1]  # increase x-values
                # sugar coating needed for shapely (contour only consists of 1 object...anyways)
                coordinates = [[[tuple(coord) for coord in c], []] for c in contour]
                m_poly = spg.MultiPolygon(coordinates)
                polys.append(m_poly[0])
            yield polys

    return wrapper


@_gen_shapely_objects
def gen_shapely_objects_all(_):
    """Shapely objects including objects touching radar boundary."""
    return _

@_gen_shapely_objects
def gen_shapely_objects(r_objects):
    """Shapely objects without objects touching radar boundary. Decorated by function which finds all objects."""
    box_areas = np.array([o.bbox_area for o in r_objects])
    outer_index = box_areas.argmax()
    del r_objects[outer_index]
    return r_objects


def gen_regionprops_objects_all(array):
    """skimage.regionprops objects including objects touching radar boundary."""
    for scene in array:  # array has dimension (time, lat, lon). A scene is a lat-lon slice.
        labeled = skm.label(scene, background=0)  # , connectivity=1)
        yield skm.regionprops(labeled)


def gen_regionprops_objects(array):
    """skimage.regionprops objects without objects touching radar boundary."""
    for scene in array:  # array has dimension (time, lat, lon). A scene is a lat-lon slice.
        labeled = skm.label(scene, background=0)  # , connectivity=1)
        objects = skm.regionprops(labeled)
        box_areas = np.array([o.bbox_area for o in objects])
        outer_index = box_areas.argmax()
        del objects[outer_index]
        yield objects


def conv_org_pot(pairs):
    """The Convective Organisation Potential according to [White et al. 2018]"""
    if not pairs.pairlist:
        return np.nan
    if len(pairs) == 1:
        if pairs.partner1 == pairs.partner2:
            return np.array(0.5 * pairs.partner1.equivalent_diameter)
    diameter_1 = np.array([c.equivalent_diameter for c in pairs.partner1])
    diameter_2 = np.array([c.equivalent_diameter for c in pairs.partner2])
    v = np.array(0.5 * (diameter_1 + diameter_2) / pairs.distance_regionprops())
    return np.sum(v) / len(pairs.pairlist)


def cop_mod(pairs):
    """Modified COP to account for different areas of objects."""
    if not pairs.pairlist:
        return np.nan
    if len(pairs) == 1:
        if pairs.partner1 == pairs.partner2:
            return np.array(0.5 * pairs.partner1.equivalent_diameter)
    diameter_1 = np.array([c.equivalent_diameter for c in pairs.partner1])
    diameter_2 = np.array([c.equivalent_diameter for c in pairs.partner2])
    v = np.array(0.5 * (diameter_1 + diameter_2) / pairs.distance_regionprops())

    # weight mean by the larger area of an object-pair
    areas = np.zeros(shape=(2, len(v)))
    areas[0, :] = [c.area for c in pairs.partner1]
    areas[1, :] = [c.area for c in pairs.partner2]
    weights = areas.max(0)
    mod_v = v * weights
    return np.sum(mod_v) / np.sum(weights)  # Bethan's proposal
    # return np.sum(mod_v) / len(mod_v)  # my modification


def _shape_independent_cop(in_func):
    """Decorator for metrics SIC and ESO."""

    @functools.wraps(in_func)
    def wrapper(s_pairs, r_pairs=None):
        if not s_pairs.pairlist:
            return np.nan
        # + 0.5 because shapely contours 'skip' edges of pixels
        area_1 = np.array([c.area for c in s_pairs.partner1]) + 0.5
        area_2 = np.array([c.area for c in s_pairs.partner2]) + 0.5

        large_area = np.maximum(area_1, area_2)
        small_area = np.minimum(area_1, area_2)
        # TODO do not hardcode total area
        radar_area = 9841  # pixels

        if len(s_pairs) == 1:
            if s_pairs.partner1 == s_pairs.partner2:
                #return np.array(area_1)
                return np.array(area_1 / radar_area)

        if r_pairs:
            # modify area_1 and area_2. SIC --> ESO.
            ma_mi_1, ma_mi_2 = in_func(r_pairs)
            v = np.array((area_1 * ma_mi_1 + area_2 * ma_mi_2) / s_pairs.distance_shapely()**2)
        else:
            # v = np.array((area_1           + area_2          ) / s_pairs.distance_shapely()**2)
            # new version of SIC
            v = np.array((area_1           + area_2          ) / s_pairs.distance_shapely()**2)
            single_connection = (large_area / radar_area) * (1 - small_area / large_area * v / radar_area)
        #return np.mean(v)
        return single_connection.prod()**(1./len(large_area))

    return wrapper


@_shape_independent_cop
def shape_independent_cop():
    """COP-analogous metric independent of shape and accounting for different areas of objects. The area sum of
    two objects divided by their shortest distance."""
    pass


@_shape_independent_cop
def elliptic_shape_organisation(r_pairs):
    """Decorated by SIC, to compute ESO. Multiply object area of SIC with its major-minor axis ratio first."""
    major, minor = [], []
    for c in r_pairs.partner1:
        major.append(c.major_axis_length)
        minor.append(c.minor_axis_length)
    ma, mi = np.array(major), np.array(minor)
    ma = np.where(ma == 0., 1., ma)
    mi = np.where(mi == 0., 1., mi)
    ma_mi_1 = ma / mi
    major, minor = [], []
    for c in r_pairs.partner2:
        major.append(c.major_axis_length)
        minor.append(c.minor_axis_length)
    ma, mi = np.array(major), np.array(minor)
    ma = np.where(ma == 0., 1., ma)
    mi = np.where(mi == 0., 1., mi)
    ma_mi_2 = ma / mi
    return ma_mi_1, ma_mi_2


def i_org(pairs, objects):
    """I_org according to [Tompkins et al. 2017]"""
    if not pairs.pairlist:
        return np.nan

    distances = np.array(pairs.distance_regionprops())
    dist_min = []

    faster = True
    if faster:
        for cloud in objects:
            dist_min.append(np.array([distances[i] for i, pair in enumerate(pairs.pairlist) if cloud in pair]).min())
    else:
        # maybe faster
        for cloud in objects:
            pair_in = np.isin(np.array(pairs.pairlist), cloud, assume_unique=True)
            pair_one = pair_in[:, 0] + pair_in[:, 1]
            dist_min.append(distances[pair_one].min())

    # the theoretical Weibull-distribution for n particles
    u_dist_min, u_dist_min_counts = np.unique(dist_min, return_counts=True)
    lamda = len(objects) / 9841.  # one radar scan contains 9841 pixels
    weib_cdf = 1 - np.exp(- lamda * m.pi * u_dist_min**2)

    # the CDF from the actual data
    data_cdf = np.cumsum(u_dist_min_counts / np.sum(u_dist_min_counts))

    # compute the integral between Weibull CDF and data CDF
    weib_cdf = np.append(0, weib_cdf   )
    weib_cdf = np.append(   weib_cdf, 1)
    data_cdf = np.append(0, data_cdf   )
    data_cdf = np.append(   data_cdf, 1)
    cdf_integral = sp.integrate.cumtrapz(data_cdf, weib_cdf, initial=0)
    return cdf_integral[-1]


def metric_1(clouds):
    """First own metric."""
    if not clouds:
        return np.nan
    c_area = xr.DataArray([c.area for c in clouds])
    a_max = c_area.max()
    a_all = c_area.sum()
    return a_max / a_all * a_max


def n_objects(clouds):
    """Number of objects in one scene."""
    if not clouds:
        return np.nan
    return len(clouds)


def avg_area(clouds):
    """Average area of objects in one scene."""
    if not clouds:
        return np.nan
    return xr.DataArray([c.area for c in clouds]).mean()


def max_area(clouds):
    """Area of largest objects in one scene."""
    if not clouds:
        return np.nan
    return xr.DataArray([c.area for c in clouds]).max()


def max_area_id(clouds):
    """skm.regionprops-ID of largest objects in one scene."""
    if not clouds:
        return np.nan
    area = xr.DataArray([c.area for c in clouds])
    da_clouds = xr.DataArray(clouds)
    area_max = da_clouds.where(area == area.max(), drop=True)
    return list(area_max.values)


def run_metrics(file="", switch={}):
    """Compute different organisation metrics on classified data."""
    switch = collections.defaultdict(lambda: False, switch)

    # -------------------------------
    # prepare input and find objects
    # -------------------------------

    if switch['artificial']:
        conv_0 = af.art
    elif switch['random']:
        conv_0 = rf.rand_objects
    else:
        ds_st  = xr.open_mfdataset(file, chunks={'time': 40})
        stein  = ds_st.steiner_echo_classification  # .sel(time=slice('2015-11-11T09:10:00', '2015-11-11T09:20:00'))
        if switch['boundary']:
            conv   = stein.where(stein == 2)
            conv_0 = conv.fillna(0.)
        else:
            # fill surrounding with convective pixels
            conv_0 = stein.fillna(2.)
            conv_0 = conv_0.where(conv_0 != 1, other=0)

    # find objects via skm.label, to use skm.regionprops
    if switch['cop'] or switch['cop_mod'] or switch['iorg'] or switch['basics'] or switch['eso']:
        if switch['boundary']:
            props = list(gen_regionprops_objects_all(conv_0))
        else:
            props = list(gen_regionprops_objects    (conv_0))
        all_r_pairs = [Pairs(pairlist=list(gen_tuplelist(cloudlist))) for cloudlist in props]

    # find objects via skm.find_contours, to use shapely
    if switch['sic'] or switch['eso']:
        if switch['boundary']:
            props = list(gen_shapely_objects_all(conv_0))
        else:
            props = list(gen_shapely_objects    (conv_0))
        all_s_pairs = [Pairs(pairlist=list(gen_tuplelist(cloudlist))) for cloudlist in props]

    # --------------------
    # compute the metrics
    # --------------------

    # conv_org_pot needs 94% of time (tested with splitter=True for data of one day). Because pairs.distance.
    cop = xr.DataArray([conv_org_pot(pairs=p) for p in all_r_pairs]) if switch['cop'] else np.nan

    cop_m = xr.DataArray([cop_mod(pairs=p) for p in all_r_pairs]) if switch['cop_mod'] else np.nan

    sic = xr.DataArray([shape_independent_cop(s_pairs=p) for p in all_s_pairs]) if switch['sic'] else np.nan

    eso = xr.DataArray([elliptic_shape_organisation(s_pairs=s_p, r_pairs=r_p)
                        for s_p, r_p in list(zip(all_s_pairs, all_r_pairs))]) if switch['eso'] else np.nan

    iorg = xr.DataArray([i_org(pairs=all_r_pairs[i], objects=props[i])
                         for i in range(len(all_r_pairs))]) if switch['iorg'] else np.nan

    m1, o_number, o_area, o_area_max = [], [], [], []
    if switch['basics']:
        for cloudlist in props:
            m1.append        (metric_1 (clouds=cloudlist))
            o_number.append  (n_objects(clouds=cloudlist))
            o_area.append    (avg_area (clouds=cloudlist))
            o_area_max.append(max_area (clouds=cloudlist))
    else:
        m1 = np.nan
        o_number = np.nan
        o_area = np.nan
        o_area_max = np.nan

    # ---------------
    # create dataset
    # ---------------

    m1 = xr.DataArray(m1)
    o_number = xr.DataArray(o_number)
    o_area = xr.DataArray(o_area)
    o_area_max = xr.DataArray(o_area_max)

    # put together a dataset from the different metrices
    ds_m = xr.Dataset({'cop': cop,
                       'cop_mod': cop_m,
                       'sic': sic,
                       'eso': eso,
                       'm1': m1,
                       'iorg': iorg,
                       'o_number': o_number,
                       'o_area': o_area,
                       'o_area_max': o_area_max,
                       })

    # get metrics a time dimension.
    ds_m.coords['time'] = ('dim_0', conv_0.time)
    ds_m = ds_m.rename({'dim_0': 'time'})

    return ds_m


if __name__ == '__main__':
    # c = Client()
    start = timeit.default_timer()

    switch = {'artificial': False, 'random': False,
              'cop': False, 'cop_mod': False, 'sic': True, 'eso': False, 'iorg': False, 'basics': True,
              'boundary': False}

    # compute the metrics
    ds_metric = run_metrics(switch=switch,
                            # file="/Users/matthiasretsch/Google Drive File Stream/My Drive/Data/steiner*")
                            file="/Users/mret0001/Data/Steiner/steiner_testdata_*2013*")

    # save metrics as netcdf-files
    save = False
    if save:
        for var in ds_metric.variables:
            xr.Dataset({var: ds_metric[var]}).to_netcdf('/Users/mret0001/Desktop/'+var+'_new.nc')

    stop = timeit.default_timer()
    print('This script needed {} seconds.'.format(stop-start))
