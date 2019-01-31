import matplotlib.pyplot as plt
import math as m
import numpy as np
import xarray as xr
import seaborn as sns

sic = xr.open_dataarray('/Users/mret0001/Data/Analysis/No_Boundary/sic.nc')

# bins = np.linspace(start=0., stop=sic.max(), num=40+1)  # 50
bins = np.linspace(start=0., stop=m.sqrt(sic.max()), num=18)**2

sns.distplot(sic[sic.notnull()], bins=bins, kde=False, hist_kws={'log': True} , norm_hist=True)
# plt.xscale('log')
plt.show()
