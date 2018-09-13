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
labeled = xr.zeros_like(conv).load().astype(int)
for i, scene in enumerate(conv):  # conv has dimension (time, lat, lon). A scene is a lat-lon slice.
    labeled[i, :, :] = skm.label(scene)
    props.append(skm.regionprops(labeled))

stop = timeit.default_timer()
print('This script needed {} seconds.'.format(stop-start))
