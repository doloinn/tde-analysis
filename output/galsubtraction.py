import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import Counter

from scipy.optimize import curve_fit

import random
import time

import my_scatter_matrix2

ZEROFLUX = 3.631e-11  # W m-2 nm-1

cadence = [37, 26, 15]

def flux2mag(flux):
    return -2.5 * np.log10(flux / ZEROFLUX)

def mag2flux(mag):
    return ZEROFLUX * 10**(-mag / 2.5)

def add_mags(m1, m2):
    return -2.5 * np.log10(10**(-m1/2.5) + 10**(-m2/2.5))

def sub_mags(m1, m2):
    return -2.5 * np.log10(10**(-m1/2.5) - 10**(-m2/2.5))

def decayfun(t, a, b):
    return a * t**(-5/3) + b


print(mag2flux(20) / mag2flux(16))    
exit()


tde_data = pickle.load(open('tdes.pickle', 'rb'))
tde_sim = pd.read_csv('tde_parsed.csv', index_col=0)

tde_sim.replace(-1.0000000150474662e+30, np.nan, inplace=True)

tde_gal_data = [{}, {}]
for key, val in tde_data.items():
    # if key in tde_sim.index.values:
    tde_gal_data[key % 2][key - (key % 2)] = val

res_drop = ['multiple_matches', 'pix_sub-pix_pos', 'offset', 'ac_offset', 'al_offset', 'theta', 'ra', 'dec']
gal_drop = ['ra', 'dec']

# plt.rc('font',**{'family':'serif','serif':['Palatino']})
# # plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# plt.rc('text', usetex=True)

simulated_data = {}
for i, j in tde_sim.iloc[::2].iterrows():
    temp_gal_dic = tde_gal_data[0][i]['data']
    for key in gal_drop:
        temp_gal_dic.pop(key)
    # temp_gal_dic.pop('dec')
    temp_gal_dic_in = {'in %s' % key: temp_gal_dic[key] for key in temp_gal_dic.keys()}

    temp_tde_dic = tde_gal_data[1][i]['data']
    temp_tde_dic_in = {'in %s' % key: temp_tde_dic[key] for key in temp_tde_dic.keys()}

    temp_res_dic = j.to_dict()
    for key in res_drop:
        temp_res_dic.pop(key)
    # temp_res_dic.pop('pix_sub-pix_pos')
    temp_res_dic_out = {'out %s' % key: temp_res_dic[key] for key in temp_res_dic.keys()}

    simulated_data[i] = {**temp_gal_dic_in, **temp_tde_dic_in, **temp_res_dic_out, 'tde source_g_mag': tde_sim.get_value(i + 1, 'source_g_mag')}
    # print(simulated_data)

tdedf = pd.DataFrame.from_dict(simulated_data, orient='index')

tdedf.rename(columns={'out source_g_mag': 'galaxy source_g_mag'}, inplace=True)

tdedf['total source_g_mag'] = tdedf.apply(lambda row: add_mags(row['tde source_g_mag'], row['galaxy source_g_mag']), axis=1)

rename = {'in galaxy mag': '$V_G$', 'in galaxy radius': '$R_G$', 'in snapshot mag': '$G_T$', 'in point time': '$t$', 'in peak mag': '$G_P$', 'total source_g_mag': '$G_{G+T}$'}
# tdedf.rename(columns=rename, inplace=True)


tdedf_detected = tdedf.dropna()


plotdf = tdedf.loc[tdedf['in galaxy mag'] == 16].loc[tdedf['in galaxy radius'] == 0.5].loc[tdedf['in peak mag'] == 14]
plotdf_detected = plotdf.dropna()
subtracted_mag = sub_mags(plotdf_detected['out found_mag'], 16.1)
# print(plotdf_detected, subtracted_mag)
# exit()

# print(max(plotdf_detected['out found_mag']))

# plt.plot(plotdf['in point time'], plotdf['total source_g_mag'], '.', alpha=0.8)
plt.plot(plotdf_detected['in point time'], plotdf_detected['out found_mag'], '.', alpha=0.8)
plt.plot(plotdf_detected['in point time'], subtracted_mag, '.', alpha=0.8)
# plt.plot(plotdf_detected['in point time'].iloc[70:], sub_mags(plotdf_detected['out found_mag'].iloc[70:], 16.1), '.', alpha=0.8)
# yerr = np.sqrt(mag2flux(subtracted_mag))
# print(yerr)
# upyerr = sub_mags(flux2mag(mag2flux(subtracted_mag) + yerr), subtracted_mag)
# downyerr = sub_mags(subtracted_mag, flux2mag(mag2flux(subtracted_mag) - yerr))
# print(subtracted_mag, downyerr)
# asymerr = [downyerr, upyerr]
# plt.errorbar(plotdf_detected['in point time'], subtracted_mag, yerr=asymerr, fmt='.')
# x_ref = plotdf_detected['in point time'].iloc[70:]
# y_ref = flux2mag(mag2flux(8) * x_ref**-(5/3))
# plt.plot(x_ref, y_ref)

x = plotdf_detected['in point time'].iloc[70:]
y = mag2flux(subtracted_mag[70:])
popt, pcov = curve_fit(decayfun, x, y, sigma=np.sqrt(y), absolute_sigma=True, method='trf')
popt2, pcov2 = curve_fit(decayfun, x, mag2flux(plotdf_detected['out found_mag'][70:]))

fit = decayfun(x, *popt)

plt.plot(x, flux2mag(fit))
plt.plot(x, flux2mag(decayfun(x, *popt2)))
print(flux2mag(popt2[1]))
# print(popt, pcov)

# def decayerr(t, pcov):
#     ader = t**(-5/3)
#     bder = 1
#     # print((ader**2) * pcov[0, 0], (bder**2) * pcov[1, 1], 2 * ader * bder * pcov[0, 1])
#     return np.sqrt((ader**2) * pcov[0, 0] + (bder**2) * pcov[1, 1] + 2 * ader * bder * pcov[0, 1])

# fiterr = decayerr(x, pcov)

# # print(fit, fiterr)

# plt.fill_between(x, flux2mag(fit - fiterr), flux2mag(fit + fiterr), alpha=0.4)



plt.gca().invert_yaxis()




# Whole light curve plots
# for gm in tdedf['in galaxy mag'].unique()[:1]:
#     for gr in tdedf['in galaxy radius'].unique()[:1]:
#         for pm in tdedf['in peak mag'].unique()[:1]:
#             plotdf = tdedf.loc[tdedf['in galaxy mag'] == gm].loc[tdedf['in galaxy radius'] == gr].loc[tdedf['in peak mag'] == pm]
#             plotdf_detected = tdedf_detected.loc[tdedf_detected['in galaxy mag'] == gm].loc[tdedf_detected['in galaxy radius'] == gr].loc[tdedf_detected['in peak mag'] == pm]
#             x = plotdf['in point time']
#             y = plotdf['total source_g_mag']
#             x_det = plotdf_detected['in point time']
#             # y_det = plotdf_detected['total source_g_mag']
#             y_det = plotdf_detected['out found_mag']
#             plt.figure()
#             plt.plot(x, y, '.', alpha=0.2, label='Simulated')
#             # plt.plot(x_det, y_det - 1, '.', alpha=0.2, label='Detected')
#             plt.plot(x_det, y_det, 'xC3', alpha=0.2, label='Detected')
#             plt.xlabel('Time [days]')
#             plt.ylabel('Brightness [$G$ mag]')
#             # plt.title('Galaxy radius: %s, galaxy magnitude: %s, TDE peak magnitude: %s' % (gr, gm, pm))
#             plt.gca().invert_yaxis()
#             plt.legend(loc='best')
#             fig = plt.gcf()
#             fig.canvas.set_window_title('%s, %s, %s' %(gr, gm, pm))




plt.show()
