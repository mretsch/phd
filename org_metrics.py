import math as m
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import timeit
import skimage.measure as skm
# from dask.distributed import Client
import artificial_fields as af
import scipy as sp


class Pairs:

    def __init__(self, pairlist):
        self.pairlist = pairlist
        if pairlist:
            self.partner1, self.partner2 = zip(*pairlist)
        else:
            self.partner1, self.partner2 = [], []

    def distance(self):
        """The distance in units of pixels between two centroids of cloud-objects found by skm.regionprops."""

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


def cop_mod(pairs, scaling):
    """Modified COP to account for different areas of objects."""
    if not pairs.pairlist:
        return np.nan
    diameter_1 = np.array([c.equivalent_diameter for c in pairs.partner1])
    diameter_2 = np.array([c.equivalent_diameter for c in pairs.partner2])
    v = np.array(0.5 * (diameter_1 + diameter_2) / pairs.distance())

    # weight mean by the larger area of an object-pair
    areas = np.zeros(shape=(2, len(v)))
    areas[0, :] = [c.area for c in pairs.partner1]
    areas[1, :] = [c.area for c in pairs.partner2]
    weights = areas.max(0)
    mod_v = v * weights
    return np.sum(mod_v) / np.sum(weights)


def cop_largest(pairs, max_id):
    """COP computed only for largest object."""
    if not pairs.pairlist:
        return np.nan
    diameter_1 = np.array([c.equivalent_diameter for c in pairs.partner1])
    diameter_2 = np.array([c.equivalent_diameter for c in pairs.partner2])
    v = np.array(0.5 * (diameter_1 + diameter_2) / pairs.distance())

    # weight mean by the larger area of an object-pair
    areas = np.zeros(shape=(2, len(v)))
    areas[0, :] = [c.area for c in pairs.partner1]
    areas[1, :] = [c.area for c in pairs.partner2]
    weights = areas.max(0)
    mod_v = v * weights
    return np.sum(mod_v) / np.sum(weights)


def i_org(pairs, objects):
    """I_org according to [Tompkins et al. 2017]"""
    if not pairs.pairlist:
        return np.nan

    distances = np.array(pairs.distance())
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

def run_metrics(artificial=False, file=""):
    """Compute different organisation metrics on classified data."""

    if artificial:
        conv_0 = af.art
    else:
        ds_st = xr.open_mfdataset(file)

        stein  = ds_st.steiner_echo_classification
        conv   = stein.where(stein == 2)
        conv_0 = conv.fillna(0.)

    props = []
    labeled = np.zeros_like(conv_0).astype(int)
    for i, scene in enumerate(conv_0):  # conv has dimension (time, lat, lon). A scene is a lat-lon slice.
        labeled[i, :, :] = skm.label(scene, background=0, connectivity=1)
        props.append(skm.regionprops(labeled[i, :, :]))

    all_pairs = [Pairs(pairlist=list(gen_tuplelist(cloudlist))) for cloudlist in props]

    # compute the metrics
    # conv_org_pot needs 94% of time (tested with splitter=True for data of one day). Because pairs.distance.
    get_cop = False
    cop = xr.DataArray([conv_org_pot(pairs=p) for p in all_pairs]) if get_cop else np.nan

    get_cop_mod = False
    cop_m = xr.DataArray([cop_mod(pairs=p) for p in all_pairs]) if get_cop_mod else np.nan

    get_cop_largest = True
    if get_cop_largest:
        o_max_id = [max_area_id(clouds=cloudlist) for cloudlist in props]
        cop_l = xr.DataArray([cop_largest(pairs=p, max_id=o_max_id[i]) for i, p in enumerate(all_pairs)])

    get_iorg = False
    iorg = xr.DataArray([i_org(pairs=all_pairs[i], objects=props[i])
                         for i in range(len(all_pairs))]) if get_iorg else np.nan

    m1, o_number, o_area, o_area_max = [], [], [], []
    for cloudlist in props:
        m1.append(metric_1(clouds=cloudlist))
        o_number.append(n_objects(clouds=cloudlist))
        o_area.append(avg_area(clouds=cloudlist))
        o_area_max.append(max_area(clouds=cloudlist))

    m1 = xr.DataArray(m1)
    o_number = xr.DataArray(o_number)
    o_area = xr.DataArray(o_area)
    o_area_max = xr.DataArray(o_area_max)

    # put together a dataset from the different metrices
    ds_m = xr.Dataset({'cop': cop,
                       'cop_mod': cop_m,
                       'm1': m1,
                       'Iorg': iorg,
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
                            file="/Users/mret0001/Data/Steiner/CPOL_STEINER_ECHO_CLASSIFICATION_threedays.nc")

    # a quick histrogram
    # ds_metric.cop.plot.hist(bins=55)
    # plt.title('COP distribution, sample size: ' + str(ds_metric.cop.notnull().sum().values))
    # plt.show()

    # save metrics as netcdf-files
    save = True
    if save:
        for var in ds_metric.variables:
            xr.Dataset({var: ds_metric[var]}).to_netcdf('/Users/mret0001/Desktop/'+var+'_new.nc')

    stop = timeit.default_timer()
    print('This script needed {} seconds.'.format(stop-start))
