import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import pickle
from palettable.colorbrewer.sequential import *
from palettable.colorbrewer.diverging import *

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

# 114 031

# uni = np.unique(values)

# x = eldf_detected['in v mag']
# y = eldf_detected['in radius']

# histout = np.histogram2d(y, x, bins=(3, 3))#, bins=bins)
# histout[0][1, 0] *= 1.114
# histout[0][2, 0] *= 1.031

# # plt.imshow(histout[0], interpolation=None, aspect='auto', origin='lower', cmap=Blues_9.mpl_colormap, extent=[0, len(histout[2]) - 1, 0, len(histout[1]) - 1])
# plt.imshow(histout[0] / 1500, interpolation='gaussian', aspect='auto', origin='lower', cmap=Blues_9.mpl_colormap, extent=[0, len(histout[2]) - 1, 0, len(histout[1]) - 1])


# # plt.xticks([0.5 + k for k, l in enumerate(histout[0][0, :])], x.unique())
# plt.xticks([0, 1.5, 3], x.unique())
# # plt.set_xticklabels(['%.3g' % k for k in histout[2]], rotation='vertical')
# # plt.xticklabels(list(range(-500, 501, 500)), rotation='vertical')
# # plt.yticks([0.5 + k for k, l in enumerate(histout[0][:, 0])], y.unique())
# plt.yticks([0, 1.5, 3], y.unique())
# # plt.yticklabels(['%.3g' % k for k in histout[1]])
# # for j in range(3):
# #     for i in range(3):
# #         text = plt.text(0.5 + i, 0.5 + j, '%i' %histout[0][j, i], color='black', verticalalignment='center', horizontalalignment='center')
# #         text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
# #                        path_effects.Normal()])

# plt.xlabel(r'$V$ magnitude')
# plt.ylabel('Radius')

x = spdf_detected['in v mag']
y = spdf_detected['in b/t']

histout = np.histogram2d(y, x, bins=(3, 3))#, bins=bins)

plt.imshow(histout[0], interpolation=None, aspect='auto', origin='lower', cmap=Blues_9.mpl_colormap, extent=[0, len(histout[2]) - 1, 0, len(histout[1]) - 1])


plt.xticks([0.5 + k for k, l in enumerate(histout[0][0, :])], x.unique())
# plt.xticks([0, 1.5, 3], x.unique())
# plt.set_xticklabels(['%.3g' % k for k in histout[2]], rotation='vertical')
# plt.xticklabels(list(range(-500, 501, 500)), rotation='vertical')
plt.yticks([0.5 + k for k, l in enumerate(histout[0][:, 0])], y.unique())
# plt.yticks([0, 1.5, 3], y.unique())
# plt.yticklabels(['%.3g' % k for k in histout[1]])

for j in range(3):
    for i in range(3):
        text = plt.text(0.5 + i, 0.5 + j, '%i' %histout[0][j, i], color='black', verticalalignment='center', horizontalalignment='center')
        text.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
                       path_effects.Normal()])

plt.xlabel(r'$V$ magnitude')
plt.ylabel('$B/T$')

plt.colorbar()

plt.tight_layout()

# plt.savefig('coordbias.pdf')

plt.show()
