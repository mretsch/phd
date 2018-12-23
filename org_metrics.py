import collections
import functools
import math as m
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import timeit
import skimage.measure as skm
import artificial_fields as af
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
    """Tuples of all possible unique pairs in an iterator."""
    for i, item1 in enumerate(inlist):
        for item2 in gen_shortlist(start=i + 1, inlist=inlist):
            yield item1, item2


def gen_shapely_objects_all(array):
    """Shapely objects including objects touching radar boundary."""
    for field in array:
        # get contours to create shapely polygons
        contours = skm.find_contours(field, level=1, fully_connected='high')

        circs = np.array([len(c) for c in contours])
        if circs.any():
            coordinates = [[[tuple(coord) for coord in poly], []] for poly in contours]

            # fill object holes, to avoid false polygons, pairs & and distances, by taking union of all polygons
            in_poly = spo.unary_union(spg.MultiPolygon(coordinates))
        else:
            in_poly = []

        # get rid of non-iterable Polygon class, i.e. single objects, which fail for generators later
        yield in_poly if type(in_poly) == spg.MultiPolygon else []


def gen_shapely_objects(array):
    """Shapely objects without objects touching radar boundary."""
    for field in array:
        # get contours to create shapely polygons
        contours = skm.find_contours(field, level=1, fully_connected='high')

        circs = np.array([len(c) for c in contours])
        if circs.any():
            # single out outer contour with longest circumference
            oc_index = circs.argmax()
            outer_contour = contours.pop(oc_index)
            oc_poly = spg.Polygon(outer_contour)

            coordinates = [[[tuple(coord) for coord in poly], []] for poly in contours]
            m_poly = spg.MultiPolygon(coordinates)

            # fill object holes, to avoid false polygons, pairs & and distances, by taking union of all polygons
            in_poly = spo.unary_union([p for p in m_poly if p.within(oc_poly)])
        else:
            in_poly = []

        # get rid of non-iterable Polygon class, i.e. single objects, which fail for generators later
        yield in_poly if type(in_poly) == spg.MultiPolygon else []


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
    diameter_1 = np.array([c.equivalent_diameter for c in pairs.partner1])
    diameter_2 = np.array([c.equivalent_diameter for c in pairs.partner2])
    v = np.array(0.5 * (diameter_1 + diameter_2) / pairs.distance_regionprops())
    return np.sum(v) / len(pairs.pairlist)


def cop_mod(pairs):
    """Modified COP to account for different areas of objects."""
    if not pairs.pairlist:
        return np.nan
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

    @functools.wraps(in_func)
    def wrapper(s_pairs, r_pairs=None):
        if not s_pairs.pairlist:
            return np.nan
        area_1 = np.array([c.area for c in s_pairs.partner1])
        area_2 = np.array([c.area for c in s_pairs.partner2])

        if r_pairs:
            # modify area_1 and area_2. SIC --> ESO.
            ma_mi_1, ma_mi_2 = in_func(r_pairs)
            v = np.array((area_1 * ma_mi_1 + area_2 * ma_mi_2) / s_pairs.distance_shapely()**2)
        else:
            v = np.array((area_1           + area_2          ) / s_pairs.distance_shapely()**2)

        return np.mean(v)
    return wrapper


@_shape_independent_cop
def shape_independent_cop():
    """COP-analogous metric independent of shape and accounting for different areas of objects. The area sum of
    two objects divided by their shortest distance."""
    pass


@_shape_independent_cop
def elliptic_shape_organisation(r_pairs):
    """Decorated with SIC, to compute ESO. Multiply object area of SIC with its major-minor axis ratio first."""
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

    if switch['artificial']:
        conv_0 = af.art
    else:
        ds_st  = xr.open_mfdataset(file, chunks={'time': 40})
        stein  = ds_st.steiner_echo_classification
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

    sic = xr.DataArray([shape_independent_cop(p) for p in all_s_pairs]) if switch['sic'] else np.nan

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

    switch = {'artificial': False,
              'cop': False, 'cop_mod': False, 'sic': False, 'eso': True, 'iorg': False, 'basics': False,
              'boundary': False}

    # compute the metrics
    ds_metric = run_metrics(switch=switch,
                            file="/Users/mret0001/Data/Steiner/CPOL_STEINER_ECHO_CLASSIFICATION_season*.nc")

    # a quick histrogram
    # ds_metric.cop_mod.plot.hist(bins=55)
    # plt.title('COP distribution, sample size: ' + str(ds_metric.cop.notnull().sum().values))
    # plt.show()

    # save metrics as netcdf-files
    save = False
    if save:
        for var in ds_metric.variables:
            xr.Dataset({var: ds_metric[var]}).to_netcdf('/Users/mret0001/Desktop/'+var+'_new.nc')

    stop = timeit.default_timer()
    print('This script needed {} seconds.'.format(stop-start))
