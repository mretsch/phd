import matplotlib.pyplot as plt
import math as m
import numpy as np
import xarray as xr
import seaborn as sns


plt.rc('font'  , size=12)
plt.rc('legend', fontsize=12)
#font = {'fontname': 'Helvetica'}
#plt.rc('font',**{'family':'serif','serif':['Times']})
# plt.rc('text.latex' , preamble=r'\usepackage{cmbright}')

ds = xr.open_mfdataset(['/Users/mret0001/Data/Analysis/No_Boundary/sic.nc',
                        '/Users/mret0001/Data/Analysis/No_Boundary/eso.nc',
                        '/Users/mret0001/Data/Analysis/No_Boundary/cop.nc',
                        ])

fig, ax = plt.subplots()
linestyle = ['dashed', 'solid', 'dotted']
for i, metric in enumerate([ds.eso, ds.sic, ds.cop]):

    # bins = np.linspace(start=0., stop=sic.max(), num=40+1)  # 50
    bins = np.linspace(start=m.sqrt(metric.min()), stop=m.sqrt(metric.max()), num=18)**2

    # sns.distplot(metric[metric.notnull()], bins=bins, kde=False, norm_hist=True)  # hist_kws={'log': True})

    total = metric.notnull().sum().values
    metric_clean = metric.fillna(-1)
    h, edges = np.histogram(metric_clean, bins=bins)  # , density=True)

    bin_centre = 0.5* (edges[1:] + edges[:-1])
    dx         =       edges[1:] - edges[:-1]
    dlogx      = dx / (bin_centre * m.log(10))

    #plt.plot(bin_centre, h)
    #plt.plot(bin_centre, h / dx)
    #plt.plot(bin_centre, h / dx / total) # equals density=True
    #plt.plot(bin_centre, h / dlogx)
    #plt.plot(bin_centre, h / dlogx / total) # equals density=True
    plt.plot(bin_centre, h / dlogx / total * 100,  # equals density=True in percent
             color='k',
             linewidth=2.,
             linestyle=linestyle[i]
             )

plt.xscale('log')

plt.ylabel('d$\mathcal{P}$ / dlog($\mathcal{M}$)  [% $\cdot 1^{-1}$]')
plt.xlabel('Metric $\mathcal{M}$  [1]')#, **font)
plt.legend(['ESO', 'SIC', 'COP'])

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_position('zero')
#ax.yaxis.set_ticks_position('none')  # 'left', 'right'
ax.tick_params(axis='y', direction='in')
ax.tick_params(axis='x', length=5)

plt.show()
