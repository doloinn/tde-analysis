import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import Counter

import random
import time

import my_scatter_matrix2

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
mi = min(tdedf['in point time'].unique())
ma = max(tdedf['in point time'].unique())
bins = np.linspace(mi, ma, 20)
histout = np.histogram(tdedf['in point time'], bins=bins)
histout_detected = np.histogram(tdedf_detected['in point time'], bins=bins)

histvals = histout_detected[0] / histout[0]
spc = bins[1] - bins[0]
plt.bar(bins[:-1] + spc / 2, histvals, spc, edgecolor='black')
print(histvals[-1] - min(histvals))

plt.xlim([mi, ma])
plt.xlabel('Time [days]')

# print(tdedf_detected['in dec'].unique())

# Whole light curve plots
# for gm in tdedf['in galaxy mag'].unique()[:]:
#     for gr in tdedf['in galaxy radius'].unique()[:]:
#         for pm in tdedf['in peak mag'].unique()[:]:
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


# for gm in tdedf['in galaxy mag'].unique()[:]:
#     for gr in tdedf['in galaxy radius'].unique()[:]:
#         for pm in tdedf['in peak mag'].unique()[:]:
#             plotdf = tdedf.loc[tdedf['in galaxy mag'] == gm].loc[tdedf['in galaxy radius'] == gr].loc[tdedf['in peak mag'] == pm]
#             plotdf_detected = tdedf_detected.loc[tdedf_detected['in galaxy mag'] == gm].loc[tdedf_detected['in galaxy radius'] == gr].loc[tdedf_detected['in peak mag'] == pm]
#             plt.figure()
#             plt.hist(tdedf['in dec'])
#             histout = plt.hist(tdedf_detected['in dec'])
#             plt.axhline(max(histout[0]))


# Sampled light curve plots
# cadence = [37, 26, 15]
# lc = []
# tdedf['light curve'] = 0

# for gm in tdedf['in galaxy mag'].unique():
#     for gr in tdedf['in galaxy radius'].unique():
#         for pm in tdedf['in peak mag'].unique():
#             df = tdedf.loc[tdedf['in galaxy mag'] == gm].loc[tdedf['in galaxy radius'] == gr].loc[tdedf['in peak mag'] == pm]
#             for cad in cadence:
#                 for p in range(cad):
#                     lidx = [p + i * cad for i in range(len(df) // cad)]
#                     lc.append(df.iloc[lidx])

# # print(len(lc))

# for i in random.sample(lc, 10):
#     df_det = i.dropna()
#     gr = i.iloc[0]['in galaxy radius']
#     gm = i.iloc[0]['in galaxy mag']
#     pm = i.iloc[0]['in peak mag']

#     plt.figure()
#     plt.plot(i['in point time'], i['total source_g_mag'], 'b.', label='Simulated')
#     plt.plot(df_det['in point time'], df_det['out found_mag'], 'rx', label='Detected')
#     plt.axhline(i.iloc[0]['galaxy source_g_mag'], label='Galaxy magnitude', color='purple')
#     plt.axhline(20.7, label='Gaia cut off', color='green')
#     plt.xlabel('Time [days]')
#     plt.ylabel('Brightness [mag]')
#     plt.title('Galaxy radius: %s, galaxy V magnitude: %s, TDE peak magnitude: %s' % (gr, gm, pm))
#     plt.gca().invert_yaxis()
#     plt.legend(loc='best')


plt.show()
