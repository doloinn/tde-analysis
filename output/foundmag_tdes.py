import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import Counter

import random
import time

import my_scatter_matrix

cadence = [37, 26, 15]

def add_mags(m1, m2):
    return -2.5 * np.log10(10**(-m1/2.5) + 10**(-m2/2.5))

tde_data = pickle.load(open('tdes.pickle', 'rb'))
tde_sim = pd.read_csv('tde_parsed.csv', index_col=0)

tde_sim.replace(-1.0000000150474662e+30, np.nan, inplace=True)

tde_gal_data = [{}, {}]
for key, val in tde_data.items():
    # if key in tde_sim.index.values:
    tde_gal_data[key % 2][key - (key % 2)] = val

res_drop = ['multiple_matches', 'pix_sub-pix_pos', 'offset', 'ac_offset', 'al_offset', 'theta', 'ra', 'dec']
gal_drop = ['ra', 'dec']

plt.rc('font',**{'family':'serif','serif':['Palatino']})
# plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text', usetex=True)

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


# Histogram matrix
smdrop = ['in ra', 'in dec', 'galaxy source_g_mag', 'tde source_g_mag', 'out found_mag']
tdedf_sm = tdedf.drop(smdrop, 1)
tdedf_detected_sm = tdedf_detected.drop(smdrop, 1)

# sm = my_scatter_matrix2.scatter_matrix(tdedf_detected_sm, tdedf_sm)


# plt.rc('font',**{'family':'serif','serif':['Palatino']})
# # plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# plt.rc('text', usetex=True)

# sm = my_scatter_matrix2.scatter_matrix(tdedf_detected_sm, tdedf_sm)
# sm2 = my_scatter_matrix2.scatter_matrix(spdf_detected_sm, spdf_sm)

# print(tdedf.loc['in galaxy radius'] == 5)
#     for ba in tdedf['in b/a'].unique():
#         magdf = tdedf.loc[tdedf['in galaxy radius'] == r].loc[tdedf['in b/a'] == ba]
#         magdf_detected = magdf.dropna()
#         uni = np.concatenate([magdf['total source_g_mag'].unique(), magdf['out found_mag'].unique()])
#         # spc = 12
#         bins = 50
#         bins = np.linspace(min(uni), max(uni), bins)
#         # histout = np.histogram(magdf['in theta'], bins=bins)
#         # histout2 = np.histogram(magdf_detected['in theta'], bins=bins)

#         # histvals = histout2[0] / histout[0]
#         plt.figure()
#         # plt.bar(histout[1][:-1] + spc / 2, histvals, spc, edgecolor='black')

# for r in tdedf['in galaxy radius'].unique():
#     dfr = tdedf.loc[tdedf['in galaxy radius'] == r]
#     dfr_detected = tdedf_detected.loc[tdedf_detected['in galaxy radius'] == r]
#     uni = np.concatenate([tdedf['total source_g_mag'].unique(), tdedf['out found_mag'].unique()])
#     # spc = 12
#     bins = 50
#     bins = np.linspace(min(uni), max(uni), bins)

#     plt.figure()
#     plt.hist(dfr['total source_g_mag'], bins=bins, color='C2', edgecolor='black')#, label='Simulated input $G$ mag')
#     plt.hist(dfr_detected['total source_g_mag'], bins=bins, color='C0', edgecolor='black')#, label='Detected input $G$ mag')
#     # for i in dfr_detected['in galaxy mag'].unique():
#     #     plt.hist(dfr_detected['out found_mag'], bins=bins, color='C3', edgecolor='black', alpha=0.6, label='Detected $G$ mag')
#     for c, v in enumerate(tdedf['in galaxy mag'].unique()):
#         plt.hist(dfr_detected.loc[dfr_detected['in galaxy mag'] == v]['out found_mag'], bins=bins, color='C%s' %(c+3), edgecolor='black', alpha=0.6, label='$v = %s$' %v)

#     plt.title('r=%s' %r)
#     plt.xlabel('$G$ magnitude')
#     plt.legend()



# uni = np.concatenate([tdedf['total source_g_mag'].unique(), tdedf['tde source_g_mag'].unique()])
# # spc = 12
# bins = 50
# bins = np.linspace(min(uni), max(uni), bins)


# plt.hist(tdedf['galaxy source_g_mag'], alpha=0.5, edgecolor='k', bins=bins, label='galaxy')
# plt.hist(tdedf['tde source_g_mag'], alpha=0.5, edgecolor='k', bins=bins, label='tde')
# plt.hist(tdedf['total source_g_mag'], alpha=0.8, edgecolor='k', bins=bins, label='total')
# plt.legend()


#         plt.title('r = %s, b/a = %s' %(r, ba))
#         # plt.title('r = %s' %r)
# plt.rc('font',**{'family':'serif','serif':['Palatino']})
# plt.rc('text', usetex=True)

# uni = np.concatenate([tdedf['total source_g_mag'].unique(), tdedf_detected['out found_mag'].unique()])
# bins = np.linspace(min(uni), max(uni), 25)

# plt.hist(tdedf['total source_g_mag'], bins=bins, color='C2', edgecolor='k')
# plt.hist(tdedf_detected['total source_g_mag'], bins=bins, color='C0', edgecolor='k')
# plt.hist(tdedf_detected['out found_mag'], bins=bins, color='C3', edgecolor='k', alpha=0.6)

f, axarr = plt.subplots(3, 1, sharex=True)

for i, r in enumerate(tdedf['in galaxy radius'].unique()):
    magdf = tdedf.loc[tdedf['in galaxy radius'] == r]
    magdf_detected = magdf.dropna()
    uni = np.concatenate([magdf['total source_g_mag'].unique(), magdf['out found_mag'].unique()])
    # spc = 12
    bins = 50
    bins = np.linspace(min(uni), max(uni), bins)
    # histout = np.histogram(magdf['in theta'], bins=bins)
    # histout2 = np.histogram(magdf_detected['in theta'], bins=bins)

    # histvals = histout2[0] / histout[0]
    # plt.figure()
    # plt.bar(histout[1][:-1] + spc / 2, histvals, spc, edgecolor='black')

    axarr[i].hist(magdf['total source_g_mag'], bins=bins, color='C2', edgecolor='black')
    axarr[i].hist(magdf_detected['total source_g_mag'], bins=bins, color='C0', edgecolor='black')
    for c, ba in enumerate(tdedf['in galaxy mag'].unique()):
        axarr[i].hist(magdf_detected.loc[tdedf['in galaxy mag'] == ba]['out found_mag'], bins=bins, color='C%s' %(c+3), edgecolor='black', alpha=0.6, label='$V_G = %s$' %ba)

    axarr[i].set_title("Radius $= %s''$" %r)
    # plt.title('r = %s, b/a = %s' %(r, ba))
    # plt.title('r = %s' %r)
    # plt.legend(loc='best')


    # axarr[2].set_xlabel('$G$ magnitude')
    # axarr[0].set_xlabel('Time [days]')
    # axarr[0].set_ylabel('Brightness in G [mags]')
    # # axarr[0].legend(loc='best')

    # axarr[1].semilogx(time, mag, label='TDE lightcurve')
    # axarr[1].semilogx(x_ref, y_ref, label='$t^{-5/3}$')
    # axarr[1].set_xlabel('Time [days]')
    # axarr[1].set_ylabel('Brightness in G [mags]')
    # axarr[1].legend(loc='best')
    # axarr[1].invert_yaxis()
    # axarr[i].legend()

legstuff = axarr[2].get_legend_handles_labels()
axarr[2].set_xlabel('$G$ magnitude')
plt.figlegend(legstuff[0][2:], legstuff[1][2:], loc='best')
plt.tight_layout()

plt.show()