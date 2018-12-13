import math as m
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import timeit
import skimage.measure as skm
# from dask.distributed import Client
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
    v = np.array(0.5 * (diameter_1 + diameter_2) / pairs.distance_regionprops())
    return np.sum(v) / len(pairs.pairlist)


def cop_mod(pairs, scaling):
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


def cop_shape(pairs):
    """COP-analogous metric independent of shape and accounting for different areas of objects. The area sum of
    two objects divided by their shortest distance."""
    if not pairs.pairlist:
        return np.nan
    area_1 = np.array([c.area for c in pairs.partner1])
    area_2 = np.array([c.area for c in pairs.partner2])
    v = np.array((area_1 + area_2) / pairs.distance_shapely())
    return np.sum(v) / len(v)


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


def run_metrics(file="", artificial=False):
    """Compute different organisation metrics on classified data."""

    get_cop = True
    get_cop_mod = False
    get_cop_shape = True
    get_iorg = False
    get_others = False

    if artificial:
        conv_0 = af.art
    else:
        ds_st  = xr.open_mfdataset(file, chunks={'time': 40})
        stein  = ds_st.steiner_echo_classification
        # conv   = stein.where(stein == 2)
        # conv_0 = conv.fillna(0.)
        conv_0 = stein.fillna(2.)
        conv_0 = conv_0.where(conv_0 != 1, other=0)

    # # find objects via skm.label, to use skm.regionprops
    # props = []
    # labeled = np.zeros_like(conv_0).astype(int)
    # for i, scene in enumerate(conv_0):  # conv has dimension (time, lat, lon). A scene is a lat-lon slice.
    #     labeled[i, :, :] = skm.label(scene, background=0)  # , connectivity=1)
    #     props.append(skm.regionprops(labeled[i, :, :]))

    # all_pairs = [Pairs(pairlist=list(gen_tuplelist(cloudlist))) for cloudlist in props]

    # find objects via skm.find_contours, to use shapely
    if get_cop_shape:
        props, m_poly, in_poly = [], [], []
        for i, scene in enumerate(conv_0):  # conv has dimension (time, lat, lon). A scene is a lat-lon slice.
            # get contours to create shapely polygons
            contours = skm.find_contours(scene, level=1, fully_connected='high')

            coordinates = [[[tuple(coord) for coord in poly], []] for poly in contours]

            # fill holes in objects, to avoid false polygons, pairs & and distances, by taking union of all polygons
            # m_poly.append(spo.unary_union(spg.MultiPolygon(polygons)))
            m_poly.append(spg.MultiPolygon(coordinates))

            # get rid of objects touching the boundary of radar area
            circs = np.array([len(p.exterior.coords) for p in m_poly[-1]])
            if circs.any():
                oc_index = circs.argmax()
                outer_contour = m_poly[-1][oc_index]
                try:
                    m_poly_less = spg.MultiPolygon([m_poly[-1][:oc_index], m_poly[-1][oc_index+1:]])
                except IndexError:
                    m_poly_less = spg.MultiPolygon([m_poly[-1][:oc_index]])
                #TODO sort m_poly via the Within class (shapely docu) and the get rid of last (or first) object.
                # Sort takes too long...!
                in_poly.append([p for p in m_poly_less if p.within(outer_contour)])
            else:
                in_poly.append([])


            # get rid of non-iterable Polygon class, which fails for generators later
            props = list((p if type(p) == spg.MultiPolygon else [] for p in m_poly))

        all_pairs_s = [Pairs(pairlist=list(gen_tuplelist(cloudlist))) for cloudlist in props]

    # compute the metrics

    # conv_org_pot needs 94% of time (tested with splitter=True for data of one day). Because pairs.distance.
    cop = xr.DataArray([conv_org_pot(pairs=p) for p in all_pairs]) if get_cop else np.nan

    cop_m = xr.DataArray([cop_mod(pairs=p, scaling=1) for p in all_pairs]) if get_cop_mod else np.nan

    cop_s = xr.DataArray([cop_shape(pairs=p) for p in all_pairs_s]) if get_cop_shape else np.nan

    iorg = xr.DataArray([i_org(pairs=all_pairs[i], objects=props[i])
                         for i in range(len(all_pairs))]) if get_iorg else np.nan

    m1, o_number, o_area, o_area_max = [], [], [], []
    if get_others:
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
                       'cop_shape': cop_s,
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

    # compute the metrics
    ds_metric = run_metrics(artificial=False,
                            file="/Users/mret0001/Data/Steiner/CPOL_STEINER_ECHO_CLASSIFICATION_oneday.nc")

    # a quick histrogram
    # ds_metric.cop_mod.plot.hist(bins=55)
    # plt.title('COP distribution, sample size: ' + str(ds_metric.cop.notnull().sum().values))
    # plt.show()

    # save metrics as netcdf-files
    save = True
    if save:
        for var in ds_metric.variables:
            xr.Dataset({var: ds_metric[var]}).to_netcdf('/Users/mret0001/Desktop/'+var+'_new.nc')

    stop = timeit.default_timer()
    print('This script needed {} seconds.'.format(stop-start))
