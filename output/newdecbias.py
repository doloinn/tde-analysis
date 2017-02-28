import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import Counter

import random
import time

import my_scatter_matrix2

def add_mags(m1, m2):
    return -2.5 * np.log10(10**(-m1/2.5) + 10**(-m2/2.5))

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

print(len(tdedf_detected) / len(tdedf))
print((len(tdedf_detected.loc[tdedf_detected['in dec'] == 500]) - len(tdedf_detected.loc[tdedf_detected['in dec'] == -500])) / (3 * len(tdedf_detected.loc[tdedf_detected['in dec'] == 500])))
exit()


plt.rc('font',**{'family':'serif','serif':['Palatino']})
# plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text', usetex=True)




dfs = [eldf, spdf, tdedf]
dfs_detected = [eldf_detected, spdf_detected, tdedf_detected]
dfs_str = ['Elliptical simulations', 'Spiral simulations', 'TDE simulations']

print('Elliptical: total efficiency = %s, corrected efficiency = %s' %(len(eldf_detected) / len(eldf), len(eldf_detected.loc[eldf_detected['in dec'] == 500]) / len(eldf.loc[eldf['in dec'] == 500])))
print('Spiral: total efficiency = %s, corrected efficiency = %s' %(len(spdf_detected) / len(spdf), len(spdf_detected.loc[spdf_detected['in dec'] == 500]) / len(spdf.loc[spdf['in dec'] == 500])))

for i, j in enumerate(dfs_detected):
    # plt.figure()
    fig, ax1 = plt.subplots()
    # plt.hist(dfs[i]['in dec'], color='C2', edgecolor='k')
    # histout = plt.hist(j['in dec'], color='C0', edgecolor='k')
    uni = dfs[i]['in dec'].unique()
    # bins = 10

    histout = np.histogram(dfs[i]['in dec'], bins=3)#, bins=bins, **hist_kwds)
    ax1.bar(np.arange(len(histout[0])) + 0.5, histout[0], width=1, color='C2', edgecolor='black') 
    # axarr[0].bar(np.arange(len(histout[0])), histout[0], width=1) 

    histout = np.histogram(j['in dec'], bins=3)#, bins=bins, **hist_kwds)
    ax1.bar(np.arange(len(histout[0])) + 0.5, histout[0], width=1, color='C0', edgecolor='black') 
    # x = axarr[0].bar(np.arange(len(histout[0])), histout[0], width=1) 
    # print(x)
    ax1.set_xticks([0.5 + k for k, _ in enumerate(histout[0])])
    ax1.set_xticklabels([-500, 0, 500])
    # axarr[0].set_xticklabels(['%.3g' % k for k in histout[1]])

    ax1.set_xlim([0, 3])
    ax1.set_xlabel('Input $\zeta$ coordinate')
    
    ax2 = ax1.twinx()
    ax2.axhline(max(histout[0]), color='k')
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticks([max(histout[0])])
    ax2.set_yticklabels(['Expected\ndetections'])
    # plt.text(3.02, max(histout[0]), 'Expected\nefficiency', color='k', verticalalignment='center', horizontalalignment='left')



    fig.tight_layout()














    # plt.title(dfs_str[i])

# decdf = eldf.loc[eldf['in dec'] == -500]
# decdf_detected = decdf.dropna()

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