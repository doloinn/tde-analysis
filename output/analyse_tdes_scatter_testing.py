import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from palettable.colorbrewer.sequential import *

import random
import time

import my_scatter_matrix2

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

tdedf_detected = tdedf.dropna()


plt.rc('font',**{'family':'serif','serif':['Palatino']})
# plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text', usetex=True)

f, axarr = plt.subplots(3, sharex=True)

values = tdedf_detected['in dec'].values
allvalues = tdedf['in dec'].values


uni = np.unique(values)
# bins = 10

histout = np.histogram(allvalues, bins=3)#, bins=bins, **hist_kwds)
axarr[0].bar(np.arange(len(histout[0])) + 0.5, histout[0], width=1, color='C2', edgecolor='black') 
# axarr[0].bar(np.arange(len(histout[0])), histout[0], width=1) 

histout = np.histogram(values, bins=3)#, bins=bins, **hist_kwds)
x = axarr[0].bar(np.arange(len(histout[0])) + 0.5, histout[0], width=1, color='C0', edgecolor='black') 
# x = axarr[0].bar(np.arange(len(histout[0])), histout[0], width=1) 
# print(x)

axarr[0].set_xticks([0.5 + k for k, _ in enumerate(histout[0])])
axarr[0].set_xticklabels(['%.3g' % k for k in histout[1]])


uni = [tdedf_detected['total source_g_mag'].unique(), tdedf_detected['in dec'].unique()]
# bins = [10, 10]
# for k, c in enumerate(uni):
#     # print(a, b)
#     if len(c) < 4 and c[0] < c[1]:
#         bins[k] = np.append(c, c[-1] + 0.1)
#     elif len(c) < 4 and c[0] > c[1]:
#         bins[k] = np.insert(c, 0, c[0] + 0.1)
#         bins[k] = bins[k][::-1]

histout = np.histogram2d(tdedf_detected['total source_g_mag'], tdedf_detected['in dec'], bins=(10, 3))#, bins=bins)

# if a[:3] == 'out' or b[:3] == 'out':
#     axarr[1].imshow(histout[0], interpolation='none', aspect='auto', origin='lower', cmap=plt.get_cmap('Reds'), extent=[0, len(histout[2]) - 1, 0, len(histout[1]) - 1])
# else:
allhistout = np.histogram2d(tdedf['total source_g_mag'], tdedf['in dec'], bins=(10, 3))
# print(histout[0], allhistout[0])
# axarr[1].imshow(histout[0] / allhistout[0], interpolation='none', aspect='auto', origin='lower', cmap=plt.get_cmap('Blues'), vmin=0, vmax=1, extent=[0, len(histout[2]) - 1, 0, len(histout[1]) - 1])
axarr[1].imshow(allhistout[0], interpolation=None, aspect='auto', origin='lower', cmap=Greens_9.mpl_colormap, extent=[0, len(histout[2]) - 1, 0, len(histout[1]) - 1])
axarr[1].set_xticks([0.5 + k for k, l in enumerate(histout[0][0, :])])
# axarr[1].set_xticklabels(['%.3g' % k for k in histout[2]], rotation='vertical')
axarr[1].set_xticklabels(list(range(-500, 501, 500)), rotation='vertical')
axarr[1].set_yticks([0.5 + k for k, l in enumerate(histout[0][:, 0])])
axarr[1].set_yticklabels(['%.3g' % k for k in allhistout[1]])

# axarr[1].set_xlabel('Input declination')
axarr[1].set_ylabel('$G_{G+T}$ (simulated)')


axarr[2].imshow(histout[0], interpolation=None, aspect='auto', origin='lower', cmap=Blues_9.mpl_colormap, extent=[0, len(histout[2]) - 1, 0, len(histout[1]) - 1])


axarr[2].set_xticks([0.5 + k for k, l in enumerate(histout[0][0, :])])
# axarr[2].set_xticklabels(['%.3g' % k for k in histout[2]], rotation='vertical')
axarr[2].set_xticklabels(list(range(-500, 501, 500)), rotation='vertical')
axarr[2].set_yticks([0.5 + k for k, l in enumerate(histout[0][:, 0])])
axarr[2].set_yticklabels(['%.3g' % k for k in histout[1]])

axarr[2].set_xlabel(r'Input $\zeta$ coordinate')
axarr[2].set_ylabel('$G_{G+T}$ (detected)')


plt.tight_layout()

# plt.savefig('coordbias.pdf')

plt.show()
