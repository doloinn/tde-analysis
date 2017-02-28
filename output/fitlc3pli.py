import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import pickle
from scipy.optimize import curve_fit
from scipy.odr.odrpack import *

import random
import time

import my_scatter_matrix2

cadence = [37, 26, 15]
ZEROFLUX = 3.631e-11  # W m-2 nm-1

def flux2mag(flux):
    return -2.5 * np.log10(flux / ZEROFLUX)

def mag2flux(mag):
    return ZEROFLUX * 10**(-mag / 2.5)

def add_mags(m1, m2):
    return -2.5 * np.log10(10**(-m1/2.5) + 10**(-m2/2.5))

def sub_mags(m1, m2):
    return -2.5 * np.log10(10**(-m1/2.5) - 10**(-m2/2.5))



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

# Histogram matrix
smdrop = ['in ra', 'in dec', 'galaxy source_g_mag', 'tde source_g_mag', 'out found_mag']
tdedf_sm = tdedf.drop(smdrop, 1)
tdedf_detected_sm = tdedf_detected.drop(smdrop, 1)

# sm = my_scatter_matrix2.scatter_matrix(tdedf_detected_sm, tdedf_sm)

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


# plt.rc('font',**{'family':'serif','serif':['Palatino']})
# # plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# plt.rc('text', usetex=True)

# Sampled light curve plots
cadence = [37, 26, 15]
lc = []
tdedf['light curve'] = 0

for gm in tdedf['in galaxy mag'].unique():
    for gr in tdedf['in galaxy radius'].unique():
        for pm in tdedf['in peak mag'].unique():
            df = tdedf.loc[tdedf['in galaxy mag'] == gm].loc[tdedf['in galaxy radius'] == gr].loc[tdedf['in peak mag'] == pm]
            for cad in cadence:
                for p in range(cad):
                    lidx = [p + i * cad for i in range(len(df) // cad)]
                    lc.append(df.iloc[lidx])

# print(len(lc))

# for i in random.sample(lc, 10):
#     df_det = i.dropna()
#     gr = i.iloc[0]['in galaxy radius']
#     gm = i.iloc[0]['in galaxy mag']
#     pm = i.iloc[0]['in peak mag']

#     fig = plt.figure()
#     plt.plot(i['in point time'], i['total source_g_mag'], 'b.', label='Simulated')
#     plt.plot(df_det['in point time'], df_det['out found_mag'], 'rx', label='Detected')
#     # plt.axhline(i.iloc[0]['galaxy source_g_mag'], label='Galaxy magnitude', color='purple')
#     # plt.axhline(20.7, label='Gaia cut off', color='green')
#     plt.xlabel('Time [days]')
#     plt.ylabel('Brightness [ $G$ mag]')
#     # plt.title('Galaxy radius: %s, galaxy V magnitude: %s, TDE peak magnitude: %s' % (gr, gm, pm))
#     plt.gca().invert_yaxis()
#     plt.legend(loc='best')
#     fig.canvas.set_window_title('%s, %s, %s' %(gr, gm, pm))

galintermags = {0.5: [16.126666666666665, 18.120000000000001, 20.113333333333333], 2: [17.733333333333334, 19.699999999999999, 21.666666666666664], 5: [19.009999999999998, 20.670000000000002, 22.330000000000002]}
df = lc[0]
# print((df.iloc[0]['in galaxy mag'] - 16) / 2)
# galmag = galintermags[df.iloc[0]['in galaxy radius']][int((df.iloc[0]['in galaxy mag'] - 16) / 2)]
# print(galmag)
# exit()

def decayfun(t, a, b, c, d):
    return a * (t + b)**c + d
    # return a * t + b

# def linefun(x, a, b, c, d):
#     return a * x + b

# def expfun(x, a, b, c, d):
#     return a * np.exp(x) + b

# def odrdec(B, x):
#     return decayfun(x, *B)

# def odrline(B, x):
#     return linefun(x, *B)

# def odrexp(B, x):
#     return expfun(x, *B)

# odrfunctions = [odrdec,]# odrline, odrexp]
# cffunctions = [decayfun,]# linefun, expfun]

# Fitting lc
# lcme = [i for i in lc if i.iloc[0]['in galaxy mag'] == 16 and i.iloc[0]['in galaxy radius'] == 0.5]
# lc = lcme
# fitlc = [lc[0]]

# funs = [decayfun, linefun, expfun]
# fitdata = [[], [], []]
# cntr = [0, 0, 0]
pli = []
# fitlc = [i.dropna() for i in random.sample(lc, 10)]
# for i in random.sample(lc, 10):
for i in lc:
# for i in lc[:1]:

    galmag = galintermags[i.iloc[0]['in galaxy radius']][int((i.iloc[0]['in galaxy mag'] - 16) / 2)]
    galflux = mag2flux(galmag)
    # print(galmag)

    lcmags = i.dropna()['out found_mag'].as_matrix()
    lcfluxes = mag2flux(lcmags)
    lctimes = i.dropna()['in point time'].as_matrix()

    peakidx = lcfluxes.argmax()

    # for j, fun in enumerate(funs):
        # plt.figure()
    odrfun = lambda B, x: decayfun(x, *B, galflux)
    temppli = []
    tempres = []
    for k in range(2):
            startidx = peakidx + k

            fittimes = lctimes[startidx:] - lctimes[startidx] 
            fitfluxes = lcfluxes[startidx:]

            try:
                # popt, pcov = curve_fit(cffunctions[j], fittimes, fitfluxes)
                popt, pcov = curve_fit(lambda x, a, b, c: decayfun(x, a, b, c, galflux), fittimes, fitfluxes)
            except:
                popt = np.ones(3)

            fitfun = Model(odrfun)
            mydata = RealData(fittimes, fitfluxes)
            # print(popt)
            myodr = ODR(mydata, fitfun, beta0=popt)

            myoutput = myodr.run()
            # myoutput.pprint()
            # print(j, myoutput.stopreason, np.sqrt(myoutput.res_var) / lcfluxes[0])
            # print(myoutput.__dict__['info'], myoutput.stopreason, myoutput.inv_condnum)
            # if myoutput.
            if myoutput.info <= 3:
                # tempfitdata.append('fucked')
            # else:
                # tempfitdata.append(np.sqrt(myoutput.res_var) / lcfluxes[0])
                temppli.append(myoutput.beta[2])
                tempres.append(myoutput.res_var)

            # fitplotx = np.linspace(fittimes.min(), fittimes.max(), 100)
            # fitploty = odrfun(myoutput.beta, fitplotx)
            # plt.plot(fittimes + lctimes[startidx], fitfluxes, '.')
            # plt.plot(fitplotx + lctimes[startidx], fitploty)

    if len(temppli) > 0:
        pli.append(temppli[np.nanargmin(tempres)])
    else:
        pli.append(np.nan)
    # if tempfitdata == []:
    #     cntr[j] += 1
    #     fitdata[j].append(np.nan)
    # else:
    #     fitdata[j].append(min(tempfitdata))



# plt.xlabel('Time [days]')
# plt.ylabel('Brightness [$G$ mag]')
# plt.gca().invert_yaxis()

# print(pli)


plt.rc('font',**{'family':'serif','serif':['Palatino']})
# plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text', usetex=True)

print(np.nanmedian(pli))

# for i, d in enumerate(fitdata):
#     for j in d:
#         j == ['fucked'] * 2
#         cntr[i] += 1

sourcedf = pd.DataFrame({'gal mag': [i.iloc[0]['in galaxy mag'] for i in lc], 'gal rad': [i.iloc[0]['in galaxy radius'] for i in lc], 'peak mag': [i.iloc[0]['in peak mag'] for i in lc], 'cadence': [i.iloc[1]['in point time'] - i.iloc[0]['in point time'] for i in lc]})
fitdf = pd.DataFrame({'powerlaw': pli})
outdf = pd.concat([fitdf, sourcedf], axis=1)

# print(outdf)

for i in outdf.columns.drop('powerlaw', 1):
    for j in outdf[i].unique():
        meanval = outdf.loc[outdf[i] == j]['powerlaw'].mean()
        stdval = outdf.loc[outdf[i] == j]['powerlaw'].std()
        print('%s = %s' %(i, j), meanval, stdval, outdf.loc[outdf[i] == j]['powerlaw'].isnull().sum() / len(outdf.loc[outdf[i] == j]['powerlaw']), outdf.loc[outdf[i] == j]['powerlaw'].isnull().sum(), len(outdf.loc[outdf[i] == j]['powerlaw']))
        plt.figure()
        plt.hist(outdf.loc[outdf[i] == j]['powerlaw'].dropna(), bins=20, ec='k')
        plt.axvline(-5./3., color='C3')
        # plt.axvline(meanval, color='C4')
        # plt.axvspan(meanval - stdval, meanval + stdval, color='C4', alpha=0.5)
        plt.xlabel('Best fit $n$')
        # plt.title('%s=%s' %(i, j))
        plt.tight_layout()
        plt.gcf().canvas.set_window_title('%s=%s' %(i, j))
        # plt.savefig('indexhist/%s%s.png' %(i, j))

print(outdf.loc[outdf['cadence'] == 15].loc[outdf['gal mag'] == 20].loc[outdf['gal rad'] == 5].loc[outdf['peak mag'] == 14]['powerlaw'].mean())
print(outdf.loc[outdf['cadence'] == 37].loc[outdf['gal mag'] == 14].loc[outdf['gal rad'] == 0.5].loc[outdf['peak mag'] == 18]['powerlaw'].mean())

# cols = outdf.columns.drop('powerlaw', 1)
# ticks = np.arange(0.5, 3)
# for ci in cols:
#     for cj in cols:
#         if ci == cj:
#             continue
#         histvals = []
#         for i in outdf[ci].unique():
#             histvals.append([])
#             for j in outdf[cj].unique():
#                 # print('%s=%s, %s=%s' %(ci, i, cj, j), outdf.loc[outdf[ci] == i].loc[outdf[cj] == j]['powerlaw'].mean(), outdf.loc[outdf[ci] == i].loc[outdf[cj] == j]['powerlaw'].std(), outdf.loc[outdf[ci] == i].loc[outdf[cj] == j]['powerlaw'].isnull().sum() / len(outdf.loc[outdf[ci] == i].loc[outdf[cj] == j]['powerlaw']), outdf.loc[outdf[ci] == i].loc[outdf[cj] == j]['powerlaw'].isnull().sum(), len(outdf.loc[outdf[ci] == i].loc[outdf[cj] == j]['powerlaw']))
#                 try:
#                     histvals[-1].append(outdf.loc[outdf[ci] == i].loc[outdf[cj] == j].loc[outdf['peak mag'] != 18]['powerlaw'].isnull().sum() / len(outdf.loc[outdf[ci] == i].loc[outdf[cj] == j].loc[outdf['peak mag'] != 18]['powerlaw']))
#                     print('%s=%s, %s=%s' %(ci, i, cj, j), outdf.loc[outdf[ci] == i].loc[outdf[cj] == j].loc[outdf['peak mag'] != 18]['powerlaw'].isnull().sum() / len(outdf.loc[outdf[ci] == i].loc[outdf[cj] == j].loc[outdf['peak mag'] != 18]['powerlaw']))
#                 except ZeroDivisionError:
#                     histvals[-1].append(1)
#         # print(histvals)
#         plt.figure()
#         plt.imshow(histvals, interpolation=None, aspect='auto', cmap=plt.get_cmap('Blues'), origin='lower', vmin=0, vmax=1, extent=[0, 3, 0, 3])
#         for l in range(3):
#             for k in range(3):
#                 text = plt.text(0.5 + l, 0.5 + k, '%.2f' %histvals[k][l], color='black', verticalalignment='center', horizontalalignment='center')
#                 text.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
#                                path_effects.Normal()])


#         text.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white'),
#                        path_effects.Normal()])
#         plt.xticks(ticks, outdf[cj].unique())
#         plt.yticks(ticks, outdf[ci].unique())
#         plt.title('Fraction non-convergent')
#         plt.xlabel(cj)
#         plt.ylabel(ci)
#         plt.colorbar()
#         plt.savefig('indexhist/nopm18_%s%s_%s=%s.png' %(ci, i, cj, j))


plt.show()

# print(fitdf)
# print(sourcedf)
# print(outdf)
# outdf.to_csv('fittdes.csv')
# print(fitdf.mean())

# print(fitdata)
# print(cntr)
# plt.show()
