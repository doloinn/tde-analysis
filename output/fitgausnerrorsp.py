import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from scipy.optimize import curve_fit
from scipy.stats import norm
from matplotlib import mlab

import random
import time

import my_scatter_matrix2

def gaus(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

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

smdrop = ['in ra', 'in dec', 'source_g_mag', 'out found_mag']
eldf_sm = eldf.drop(smdrop, 1)
eldf_detected_sm = eldf_detected.drop(smdrop, 1)
spdf_sm = spdf.drop(smdrop, 1)
spdf_detected_sm = spdf_detected.drop(smdrop, 1)

# print(len(eldf_detected) / len(eldf), len(spdf_detected) / len(spdf), (len(eldf_detected) + len(spdf_detected)) / (len(eldf) + len(spdf)))

# plt.rc('font',**{'family':'serif','serif':['Palatino']})
# # plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# plt.rc('text', usetex=True)

# sm = my_scatter_matrix2.scatter_matrix(eldf_detected_sm, eldf_sm)
# sm2 = my_scatter_matrix2.scatter_matrix(spdf_detected_sm, spdf_sm)

# print(eldf.loc['in radius'] == 5)
# for r in eldf['in radius'].unique():
#     for ba in eldf['in b/a'].unique():
#         magdf = eldf.loc[eldf['in radius'] == r].loc[eldf['in b/a'] == ba]
#         magdf_detected = magdf.dropna()
#         uni = np.concatenate([magdf['source_g_mag'].unique(), magdf['out found_mag'].unique()])
#         # spc = 12
#         bins = 50
#         bins = np.linspace(min(uni), max(uni), bins)
#         # histout = np.histogram(magdf['in theta'], bins=bins)
#         # histout2 = np.histogram(magdf_detected['in theta'], bins=bins)

#         # histvals = histout2[0] / histout[0]
#         plt.figure()
#         # plt.bar(histout[1][:-1] + spc / 2, histvals, spc, edgecolor='black')

#         plt.hist(magdf['source_g_mag'], bins=bins, edgecolor='black')
#         plt.hist(magdf_detected['source_g_mag'], bins=bins, edgecolor='black')
#         plt.hist(magdf_detected['out found_mag'], bins=bins, edgecolor='black', alpha=0.6)


#         plt.title('r = %s, b/a = %s' %(r, ba))
#         # plt.title('r = %s' %r)
plt.rc('font',**{'family':'serif','serif':['Palatino']})
plt.rc('text', usetex=True)

# uni = np.concatenate([eldf['source_g_mag'].unique(), eldf_detected['out found_mag'].unique()])
# bins = np.linspace(min(uni), max(uni), 25)

# plt.hist(eldf['source_g_mag'], bins=bins, color='C2', edgecolor='k')
# plt.hist(eldf_detected['source_g_mag'], bins=bins, color='C0', edgecolor='k')
# plt.hist(eldf_detected['out found_mag'], bins=bins, color='C3', edgecolor='k', alpha=0.6)

# f, axarr = plt.subplots(3, 1, sharex=True)

def pdf(x, mu, sigma):
    return (1 / np.sqrt(2 * sigma**2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))

for i, r in enumerate(spdf['in disk radius'].unique()):
    for j, v in enumerate(spdf['in v mag'].unique()):
        magdf = spdf.loc[spdf['in disk radius'] == r].loc[spdf['in v mag'] == v]
        magdf_detected = magdf.dropna()
        uni = np.concatenate([magdf['source_g_mag'].unique(), magdf['out found_mag'].unique()])
        # spc = 12
        bins = 50
        bins = np.linspace(min(uni), max(uni), bins)
        # histout = np.histogram(magdf['in theta'], bins=bins)
        # histout2 = np.histogram(magdf_detected['in theta'], bins=bins)

        # histvals = histout2[0] / histout[0]
        # plt.figure()
        # plt.bar(histout[1][:-1] + spc / 2, histvals, spc, edgecolor='black')

        # plt.hist(magdf[ 'source_g_mag'], bins=bins, color='C2', edgecolor='black')


        for c, ba in enumerate(magdf['in bulge b/a'].unique()):
            for d, bt in enumerate(magdf['in b/t'].unique()):
                fig = plt.figure()
                # bins = np.linspace(min(magdf_detected.loc[magdf['in b/a'] == ba]['out found_mag'].unique()), max(magdf_detected.loc[magdf['in b/a'] == ba]['out found_mag'].unique()), 10)
                # x = np.linspace(min(bins), max(bins), 1000)
                # plt.hist(magdf_detected['source_g_mag'], bins=bins, color='C0', edgecolor='black')
                histout = plt.hist(magdf_detected.loc[magdf['in bulge b/a'] == ba].loc[magdf['in b/t'] == bt]['out found_mag'], normed=1, color='C3', edgecolor='black', alpha=0.6, label='$b/a = %s$' %ba)

                x = np.linspace(min(histout[1]), max(histout[1]), 1000)
                mu, sigma = norm.fit(magdf_detected.loc[magdf['in bulge b/a'] == ba].loc[magdf['in b/t'] == bt]['out found_mag'])
                y = mlab.normpdf(x, mu, sigma)
                # plt.plot(x, y, 'C4')
                # plt.axvline(mu, color='C4')
                xfit = (histout[1][1:] + histout[1][:-1]) / 2
                # print(xfit, histout[0], mu, sigma)
                shift = []
                dshift = []
                try:
                    popt, pcov = curve_fit(pdf, xfit, histout[0], p0=[mu, sigma])
                    pstd = np.sqrt(np.diag(pcov))
                    # print('mu', popt[0])
                    # print('v', magdf_detected.loc[magdf_detected['in b/a'] == ba]['in v mag'].iloc[0])
                    shift.append(round(popt[0], 2))
                    dshift.append(round(popt[1], 2))
                    # print(pstd)
                    # print('%s, %s, %s, %s' %(r, v, ba, bt), shift[-1], dshift[-1])
                    plt.plot(x, pdf(x, *popt))
                except:
                    shift.append(np.nan)
                    dshift.append(np.nan)
                    # print('nul')
                    # print('%s, %s, %s, %s' %(r, v, ba, bt), '-', '-')



                fig.canvas.set_window_title(" %s, %s, %s, %s" %(r, v, ba, bt))
                plt.xlabel('$G$ magnitude')
                plt.ylabel('Normalised counts')
        # plt.title('r = %s, b/a = %s' %(r, ba))
        # plt.title('r = %s' %r)
        # plt.legend(loc='best')

        # print('%s & %s \\' %(ba, ' & '.join([str(i) for i in shift])))


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

# legstuff = axarr[2].get_legend_handles_labels()
# axarr[2].set_xlabel('$G$ magnitude')
# plt.figlegend(legstuff[0][2:], legstuff[1][2:], loc='best')
plt.tight_layout()

plt.show()