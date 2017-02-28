import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

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
plt.rc('text', usetex=True)

# sm = my_scatter_matrix2.scatter_matrix(eldf_detected_sm, eldf_sm)
# sm2 = my_scatter_matrix2.scatter_matrix(spdf_detected_sm, spdf_sm)

# print(eldf.loc['in radius'] == 5)


# for j, v in enumerate(eldf['in v mag'].unique()):
#     f, axarr = plt.subplots(3, 1, sharex=True)
#     for i, r in enumerate(eldf['in radius'].unique()):
#         # for ba in eldf['in b/a'].unique()[0:1]:
#         thetadf = eldf.loc[eldf['in v mag'] == v].loc[eldf['in radius'] == r].loc[eldf['in b/a'] == 0.3]
#         thetadf_detected = thetadf.dropna()

#         spc = 12
#         bins = np.arange(0, 361, spc)
#         histout = np.histogram(thetadf['in theta'], bins=bins)
#         histout2 = np.histogram(thetadf_detected['in theta'], bins=bins)

#         histvals = histout2[0] / histout[0]
#         # plt.figure()
#         axarr[i].bar(histout[1][:-1] + spc / 2, histvals, spc, edgecolor='black')
#         axarr[i].set_title("Radius = $%s''$" %r)

#         for k in [0, 180, 360]:
#             axarr[i].axvline(k, color='C3')

#     axarr[2].set_xlabel(r'$\theta$')
#     axarr[2].set_xlim([0, 360])
#     f.canvas.set_window_title('%s' %(v))
#     f.tight_layout()

for j, v in enumerate(spdf['in v mag'].unique()):
    f, axarr = plt.subplots(3, 1, sharex=True)
    for i, r in enumerate(spdf['in disk radius'].unique()):
        # for ba in spdf['in b/a'].unique()[0:1]:
        thetadf = spdf.loc[spdf['in v mag'] == v].loc[spdf['in disk radius'] == r].loc[spdf['in bulge b/a'] == 0.3]
        thetadf_detected = thetadf.dropna()

        spc = 12
        bins = np.arange(0, 361, spc)
        histout = np.histogram(thetadf['in theta'], bins=bins)
        histout2 = np.histogram(thetadf_detected['in theta'], bins=bins)

        histvals = histout2[0] / histout[0]
        # plt.figure()
        axarr[i].bar(histout[1][:-1] + spc / 2, histvals, spc, edgecolor='black')
        axarr[i].set_title("Radius = $%s''$" %r)
        axarr[i].axhline(histvals.mean())

        for k in [0, 180, 360]:
            axarr[i].axvline(k, color='C3')

    axarr[2].set_xlabel(r'$\theta$')
    axarr[2].set_xlim([0, 360])
    f.canvas.set_window_title('%s' %(v))
    f.tight_layout()


plt.show()