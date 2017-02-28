import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import Counter

import random
import time

import my_scatter_matrix2

galaxy_data = pickle.load(open('galaxies.pickle', 'rb'))
galaxy_sim = pd.read_csv('galaxy_parsed.csv', index_col=0)

galaxy_sim.replace(-1.0000000150474662e+30, np.nan, inplace=True)

# types = ['elliptical', 'spiral']
gal_data = [{}, {}]
for key, val in galaxy_data.items():
    # print(key, val)
    # time.sleep(0.1)
    # if key in tde_sim.index.values:
    if val['galaxy type'] == 'elliptical':
        gal_data[0][key] = val
    elif val['galaxy type'] == 'spiral':
        gal_data[1][key] = val

res_drop = ['multiple_matches', 'pix_sub-pix_pos', 'offset', 'ac_offset', 'al_offset', 'theta', 'ra', 'dec']
# res_drop = []

dfs = []
for k in range(2):
    simulated_data = {}
    for i, j in gal_data[k].items():
        temp_gal_dic = gal_data[k][i]['data']
        temp_gal_dic_in = {'in %s' % key: temp_gal_dic[key] for key in temp_gal_dic.keys()}

        temp_res_dic = galaxy_sim.loc[i].to_dict()
        for key in res_drop:
            temp_res_dic.pop(key)
        temp_res_dic_out = {'out %s' % key: temp_res_dic[key] for key in temp_res_dic.keys()}

        simulated_data[i] = {**temp_gal_dic_in, **temp_res_dic_out}

    dfs.append(pd.DataFrame.from_dict(simulated_data, orient='index'))
    dfs[k].rename(columns={'out source_g_mag': 'source_g_mag'}, inplace=True)

eldf = dfs[0].loc[:, (dfs[0] != dfs[0].iloc[0]).any()] 
spdf = dfs[1].loc[:, (dfs[1] != dfs[1].iloc[0]).any()] 

rename = {'in b/a': '$b/a$', 'in radius': '$R_B$', 'in v mag': '$V$', 'in theta': r'$\theta$', 'in v-i': '$V-I$'}
# eldf.rename(columns=rename, inplace=True)

eldf_detected = eldf.dropna()
spdf_detected = spdf.dropna()

smdrop = ['in ra', 'in dec', 'source_g_mag', 'out found_mag']
eldf_sm = eldf.drop(smdrop, 1)
eldf_detected_sm = eldf_detected.drop(smdrop, 1)
spdf_sm = spdf.drop(smdrop, 1)
spdf_detected_sm = spdf_detected.drop(smdrop, 1)

# print(len(eldf_detected) / len(eldf), len(spdf_detected) / len(spdf), (len(eldf_detected) + len(spdf_detected)) / (len(eldf) + len(spdf)))

plt.rc('font',**{'family':'serif','serif':['Palatino']})
# plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text', usetex=True)

# sm = my_scatter_matrix2.scatter_matrix(eldf_detected_sm, eldf_sm)
# sm2 = my_scatter_matrix2.scatter_matrix(spdf_detected_sm, spdf_sm)

# raddf = eldf.loc[eldf['in radius'] == 2].loc[eldf['in v mag'] == 15]
# raddf_detected = raddf.dropna()

# radnan = pd.isnull(raddf).any(1).nonzero()[0] // 57

# cnt = Counter(radnan)
# print(cnt)
# # print(radnan)
# # exit()

# # dec = -500 hist everything

# for i in raddf.columns:
#     plt.figure()
#     try:
#         plt.hist(raddf[i], edgecolor='k')
#     except:
#         pass
#     plt.hist(raddf_detected[i], edgecolor='k')
#     plt.title(i)

raddf = eldf.loc[eldf['in radius'] == 2].loc[eldf['in v mag'] == 15]
raddf_detected = raddf.dropna()
raddf2 = eldf.loc[eldf['in radius'] == 5].loc[eldf['in v mag'] == 15]
raddf_detected2 = raddf2.dropna()
raddf3 = eldf.loc[eldf['in radius'] == 0.5].loc[eldf['in v mag'] == 15]
raddf_detected3 = raddf3.dropna()

# plt.figure()
# plt.hist(raddf['in dec'], color='C2', edgecolor='k')
# plt.hist(raddf_detected['in dec'], color='C0', edgecolor='k')
# plt.xlabel('Input $\zeta$ coordinate')

f, axarr = plt.subplots(3, 1, sharex=True)

histout = np.histogram(raddf['in dec'], bins=3)#, bins=bins, **hist_kwds)
axarr[1].bar(np.arange(len(histout[0])) + 0.5, histout[0], width=1, color='C2', edgecolor='black') 
# axarr[1].bar(np.arange(len(histout[0])), histout[0], width=1) 
histout = np.histogram(raddf_detected['in dec'], bins=3)#, bins=bins, **hist_kwds)
axarr[1].bar(np.arange(len(histout[0])) + 0.5, histout[0], width=1, color='C0', edgecolor='black') 
print((500 - histout[0]) / 1500)
# x = axarr[1].bar(np.arange(len(histout[0])), histout[0], width=1) 
# print(x)
axarr[1].set_xticks([0.5 + k for k, _ in enumerate(histout[0])])
axarr[1].set_xticklabels([-500, 0, 500])
# axarr[1].set_xticklabels(['%.3g' % k for k in histout[1]])
axarr[1].set_title("Radius = $2''$")

histout = np.histogram(raddf2['in dec'], bins=3)#, bins=bins, **hist_kwds)
axarr[2].bar(np.arange(len(histout[0])) + 0.5, histout[0], width=1, color='C2', edgecolor='black') 
# axarr[2].bar(np.arange(len(histout[0])), histout[0], width=1) 

histout = np.histogram(raddf_detected2['in dec'], bins=3)#, bins=bins, **hist_kwds)
axarr[2].bar(np.arange(len(histout[0])) + 0.5, histout[0], width=1, color='C0', edgecolor='black') 
# x = axarr[2].bar(np.arange(len(histout[0])), histout[0], width=1) 
print((500 - histout[0]) / 1500)
axarr[2].set_xticks([0.5 + k for k, _ in enumerate(histout[0])])
axarr[2].set_xticklabels([-500, 0, 500])
# axarr[2].set_xticklabels(['%.3g' % k for k in histout[1]])

axarr[2].set_title("Radius = $5''$")

histout = np.histogram(raddf3['in dec'], bins=3)#, bins=bins, **hist_kwds)
axarr[0].bar(np.arange(len(histout[0])) + 0.5, histout[0], width=1, color='C2', edgecolor='black') 
# axarr[0].bar(np.arange(len(histout[0])), histout[0], width=1) 

histout = np.histogram(raddf_detected3['in dec'], bins=3)#, bins=bins, **hist_kwds)
axarr[0].bar(np.arange(len(histout[0])) + 0.5, histout[0], width=1, color='C0', edgecolor='black') 
# x = axarr[0].bar(np.arange(len(histout[0])), histout[0], width=1) 
# print(x)
axarr[0].set_xticks([0.5 + k for k, _ in enumerate(histout[0])])
axarr[0].set_xticklabels([-500, 0, 500])
# axarr[0].set_xticklabels(['%.3g' % k for k in histout[1]])

axarr[0].set_title("Radius = $0.5''$")

axarr[2].set_xlim([0, 3])
axarr[2].set_xlabel('Input $\zeta$ coordinate')

plt.tight_layout()

# axarr[1].set_xlim([0, 3])
# axarr[1].set_xlabel('Input $\zeta$ coordinate')


# plt.title('r=2, v=15')

# decdf = eldf.loc[eldf['in dec'] == -500]
# decdf_detected = decdf.dropna()

# trueeffdf = eldf.loc[eldf['in dec'] == 500]
# trueeffdf_detected = trueeffdf.dropna()
# print(len(trueeffdf_detected) / len(trueeffdf))
# exit()

# for i in decdf.columns:
#     plt.figure()
#     try:
#         plt.hist(decdf[i], edgecolor='k')
#     except:
#         pass
#     plt.hist(decdf_detected[i], edgecolor='k')
#     plt.title(i)



# for r in eldf['in radius'].unique():
#     magdf = eldf.loc[eldf['in radius'] == r]
#     magdf_detected = magdf.dropna()
#     uni = np.concatenate([magdf['source_g_mag'].unique(), magdf['out found_mag'].unique()])
#     # spc = 12
#     bins = 50
#     bins = np.linspace(min(uni), max(uni), bins)
#     # histout = np.histogram(magdf['in theta'], bins=bins)
#     # histout2 = np.histogram(magdf_detected['in theta'], bins=bins)

#     # histvals = histout2[0] / histout[0]
#     plt.figure()
#     # plt.bar(histout[1][:-1] + spc / 2, histvals, spc, edgecolor='black')

#     plt.hist(magdf['source_g_mag'], bins=bins, color='C2', edgecolor='black')
#     plt.hist(magdf_detected['source_g_mag'], bins=bins, color='C0', edgecolor='black')
#     for c, ba in enumerate(eldf['in b/a'].unique()):
#         plt.hist(magdf_detected.loc[eldf['in b/a'] == ba]['out found_mag'], bins=bins, color='C%s' %(c+3), edgecolor='black', alpha=0.6, label='$b/a = %s$' %ba)


#     # plt.title('r = %s, b/a = %s' %(r, ba))
#     # plt.title('r = %s' %r)
#     plt.legend(loc='best')
#     plt.xlabel('$G$ magnitude')


plt.show()