# David Lynn
# Code to get galaxy data and histogram efficiencies

# Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Tweaked Pandas' scatter_matrix to use 2D histograms
import my_scatter_matrix

# Load input galaxy data and parsed output
galaxy_data = pickle.load(open('galaxies.pickle', 'rb'))
galaxy_sim = pd.read_csv('galaxy_parsed.csv', index_col=0)

# Non-detections look like this, replace with NaN
galaxy_sim.replace(-1.0000000150474662e+30, np.nan, inplace=True)

# Separate into ellipticals and spirals
gal_data = [{}, {}]
for key, val in galaxy_data.items():
    if val['galaxy type'] == 'elliptical':
        gal_data[0][key] = val
    elif val['galaxy type'] == 'spiral':
        gal_data[1][key] = val

# Drop these fields from results as they're not very useful
res_drop = ['multiple_matches', 'pix_sub-pix_pos', 'offset', 'ac_offset', 'al_offset', 'theta', 'ra', 'dec']

# List wit elliptical dataframe and spiral dataframe
dfs = []
for k in range(2):

    # Put stuff into dictionary to convert to dataframe later
    simulated_data = {}

    # Loop over inputs, make temporary dictionary with data
    for i, j in gal_data[k].items():
        temp_gal_dic = gal_data[k][i]['data']
        temp_gal_dic_in = {'in %s' % key: temp_gal_dic[key] for key in temp_gal_dic.keys()}

        # Convert results to dictionary
        temp_res_dic = galaxy_sim.loc[i].to_dict()

        # Drop useless fields
        for key in res_drop:
            temp_res_dic.pop(key)
        temp_res_dic_out = {'out %s' % key: temp_res_dic[key] for key in temp_res_dic.keys()}

        # Concatenate dictionaries
        simulated_data[i] = {**temp_gal_dic_in, **temp_res_dic_out}

    # Add data to list
    dfs.append(pd.DataFrame.from_dict(simulated_data, orient='index'))

    # Drop "out" for source mag as there is a value for all sources
    dfs[k].rename(columns={'out source_g_mag': 'source_g_mag'}, inplace=True)

# Break into elliptical and spiral dataframes
eldf = dfs[0].loc[:, (dfs[0] != dfs[0].iloc[0]).any()] 
spdf = dfs[1].loc[:, (dfs[1] != dfs[1].iloc[0]).any()] 

# Drop NaNs for detected sources
eldf_detected = eldf.dropna()
spdf_detected = spdf.dropna()

# Preparing data for scatter_matrix
smdrop = ['in ra', 'in dec', 'source_g_mag', 'out found_mag']
eldf_sm = eldf.drop(smdrop, 1)
eldf_detected_sm = eldf_detected.drop(smdrop, 1)
spdf_sm = spdf.drop(smdrop + ['in bulge radius'], 1)
spdf_detected_sm = spdf_detected.drop(smdrop + ['in bulge radius'], 1)

# Rename parameters for plot
elrename = {'in b/a': '$b/a$', 'in radius': '$R_B$', 'in v mag': '$V$', 'in theta': r'$\theta$', 'in v-i': '$V-I$'}
eldf_sm.rename(columns=elrename, inplace=True)
eldf_detected_sm.rename(columns=elrename, inplace=True)
sprename = {'in b/t': '$B/T$', 'in disk radius': '$R_D$',  'in bulge b/a': '$(b/a)_B$', 'in v mag': '$V$', 'in theta': r'$\theta$', 'in v-i': '$V-I$'}
spdf_sm.rename(columns=sprename, inplace=True)
spdf_detected_sm.rename(columns=sprename, inplace=True)

# Produce histogram matrices using modified scatter_matrix
# Takes input and detected dataframes and compares them
sm = my_scatter_matrix.scatter_matrix(eldf_detected_sm, eldf_sm)
sm2 = my_scatter_matrix.scatter_matrix(spdf_detected_sm, spdf_sm)

plt.show()
