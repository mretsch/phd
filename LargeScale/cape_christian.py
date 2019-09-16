import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import warnings
warnings.filterwarnings('ignore')

# =======================================================================================
# Set some constants we need
g = 9.81
Rd = 287.047
Rv = 461.5
cp = 1005
L = 2500000.
eps = Rd / Rv
a = 253000000000.
b = -5420


# =======================================================================================
# Read and scale the data
# data_all = np.loadtxt('/Users/mret0001/Desktop/assignment_data.txt')
Tzero = 273.16

ds = xr.open_dataset('/Users/mret0001/Data/LargeScaleState/CPOL_large-scale_forcing.nc')
n_lev = len(ds.lev)

# check first timestep of LargeScale data
data_all = np.zeros((n_lev, 3))
print(data_all.shape)

cin_all = xr.zeros_like(ds.T[:1])
cape_all = xr.zeros_like(ds.T[:1])

for i, _ in enumerate(ds.time[:1]):
    data_all[:, 0] = ds.lev.values
    data_all[0, 0] = ds.p_srf_aver[i].values
    data_all[:, 1] = ds.T[i, :].values
    data_all[0, 1] = ds.T_srf[i].values + Tzero

    p = data_all[:, 0] * 100
    T = data_all[:, 1]

    r    = ds.r[i, :].values / 1000.
    r[0] = ds.r_srf[i].values / 1000.
    e = r * p / (eps + r)
    data_all[:, 2] = b / np.log(e / a)

    Td = data_all[:, 2]     # dewpoint ambient air

    qsp = np.zeros((n_lev)) # saturation specific humidity in parcel
    Tp  = np.zeros((n_lev)) # temperature of parcel


    # =======================================================================================
    # First calculate q from Td
    q = np.zeros((n_lev))
    theta = np.zeros((n_lev))
    thetae = np.zeros((n_lev))
    thetae2 = np.zeros((n_lev))
    for nl in range(0, n_lev):
        e = a * np.exp(b / Td[nl])
        q[nl] = (eps * e) / p[nl]
        theta[nl] = T[nl] * (100000. / p[nl]) ** (Rd / cp)
        thetae[nl] = theta[nl] * np.exp((L * q[nl]) / (cp * T[nl]))
        thetae2[nl] = theta[nl] * np.exp((L * q[nl]) / (cp * Td[nl]))

    # plt.plot(1000 * q, z)
    # plt.show()
    # plt.plot(theta, z)
    # plt.show()
    # plt.plot(thetae, z)
    # plt.plot(thetae2, z)
    # plt.show()


    tpw = 0. # total precipitable water
    tda = T[0]
    for nl in range(0, n_lev - 1):
        tpw = tpw + q[nl] * (p[nl] - p[nl + 1]) / g
    print('Total precipitable water:', tpw)


    # =======================================================================================
    # Calculate the height of each pressure level assuming the lowest is the surface
    z    = np.zeros((n_lev))
    dTdz = np.zeros((n_lev - 1))

    for nl in range(1, n_lev):
        z[nl] = z[nl - 1] - ((Rd * T[nl - 1]) / p[nl - 1]) * (p[nl] - p[nl - 1]) / g

    # plt.plot(p, z)
    # plt.show()
    # plt.plot(1000 * q, z)
    # plt.show()


    # =======================================================================================
    # Calculate the LCL, first using an approximate formula, then by lifting a parcel, then calculate the temperature
    # along the moist adiabat
    lcl_a = 125 * (T[0] - Td[0])
    print('Estimated LCL: ', lcl_a)

    es = a * np.exp(b / T[0])
    qsp[0] = eps * es / p[0]
    qp0 = q[0]
    Tp[0] = T[0]
    theta[0] = Tp[0] * (100000. / p[0]) ** (Rd / cp)
    thetae[0] = theta[0] * np.exp((L * qsp[0]) / (cp * Tp[0]))

    # lift the parcel
    nlnb = -99
    l_positive = False
    for nl in range(1, n_lev):
        # go up the dry adiabat whilst unsaturated
        if qsp[nl - 1] > qp0:
            Tp[nl] = Tp[nl - 1] - (g / cp) * (z[nl] - z[nl - 1])
            es = a * np.exp(b / Tp[nl])
            qsp[nl] = eps * es / p[nl]
            nlcl = nl
        # go up the moist adiabat after that
        else:
            num = 1 + ((L * qsp[nl - 1]) / (Rd * Tp[nl - 1]))
            den = 1 + ((L * L * qsp[nl - 1]) / (cp * Rv * Tp[nl - 1] * Tp[nl - 1]))
            Tp[nl] = Tp[nl - 1] - (num / den) * (g / cp) * (z[nl] - z[nl - 1])
            es = a * np.exp(b / Tp[nl])
            qsp[nl] = eps * es / p[nl]
            # the level of neutral buoyancy only occurs after there was some buoyancy to begin with
            if Tp[nl] > T[nl]:
                l_positive = True
            if Tp[nl] < T[nl]:
                print('T_parcel below T_ambient at level {0} with height {1:.1f} meters.'.format(nl, z[nl]))
                if (nlnb == -99) and l_positive:
                    nlnb = nl

    print('LCL: {0}, LCL-height: {1:.1f} meters, LCL-temperature: {2:.1f}'.format(nlcl, z[nlcl], T[nlcl] - 273.16))

    # plt.plot(Tp-T,z)
    # plt.plot(Tp, z)
    # plt.plot(T, z)
    # plt.show()
    # print(z[12])


    # =======================================================================================
    # Calculate CIN
    # the level of free convection is at/above the lifting condensation level
    nlfc = -99
    for nl in range(nlcl, nlnb):
        if ((Tp[nl] - T[nl]) > 0.) and nlfc == -99:
            nlfc = nl

    # nlbot=nlcl
    # nlfc=12

    cin = 0.
    for nl in range(0, nlfc):
        cin = cin + Rd * (Tp[nl] - T[nl]) * (p[nl + 1] - p[nl]) / p[nl]
    cin_all[i] = cin
    print('CIN: ', cin)

    # print(Tp - T)


    # =======================================================================================
    # Calculate CAPE
    cape = 0.
    # nlnb=54
    for nl in range(nlfc, nlnb):
        cape = cape + Rd * (Tp[nl] - T[nl]) * (p[nl] - p[nl + 1]) / p[nl]
    cape_all[i] = cape
    print('CAPE: ', cape)
