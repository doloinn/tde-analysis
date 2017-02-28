import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

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

# Whole light curve plots
for gm in tdedf['in galaxy mag'].unique()[:]:
    for gr in tdedf['in galaxy radius'].unique()[:]:
        for pm in tdedf['in peak mag'].unique()[:]:
            plotdf = tdedf.loc[tdedf['in galaxy mag'] == gm].loc[tdedf['in galaxy radius'] == gr].loc[tdedf['in peak mag'] == pm]
            plotdf_detected = tdedf_detected.loc[tdedf_detected['in galaxy mag'] == gm].loc[tdedf_detected['in galaxy radius'] == gr].loc[tdedf_detected['in peak mag'] == pm]
            x = plotdf['in point time']
            y = plotdf['total source_g_mag']
            x_det = plotdf_detected['in point time']
            y_det = plotdf_detected['out found_mag']
            plt.figure()
            plt.plot(x, y, '.', alpha=0.2, label='Simulated')
            plt.plot(x_det, y_det, 'xC3', alpha=0.2, label='Detected')
            plt.xlabel('Time [days]')
            plt.ylabel('Brightness [$G$ mag]')
            plt.gca().invert_yaxis()
            plt.legend(loc='best')
            fig = plt.gcf()
            fig.canvas.set_window_title('%s, %s, %s' %(gr, gm, pm))

plt.show()
