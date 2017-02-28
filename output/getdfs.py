# David Lynn
# Functions to get galaxy and TDE data

# Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Tweaked Pandas' scatter_matrix to use 2D histograms
import my_scatter_matrix

def add_mags(m1, m2):
    return -2.5 * np.log10(10**(-m1/2.5) + 10**(-m2/2.5))

def get_galaxy_dfs():
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

    return eldf, spdf


def get_tde_df():
    # Load input data and parsed output
    tde_data = pickle.load(open('tdes.pickle', 'rb'))
    tde_sim = pd.read_csv('tde_parsed.csv', index_col=0)

    # Non-detections, replace with NaNs
    tde_sim.replace(-1.0000000150474662e+30, np.nan, inplace=True)

    # Separate into galaxies and tdes
    tde_gal_data = [{}, {}]
    for key, val in tde_data.items():
        tde_gal_data[key % 2][key - (key % 2)] = val

    # Drop useless fields from results
    res_drop = ['multiple_matches', 'pix_sub-pix_pos', 'offset', 'ac_offset', 'al_offset', 'theta', 'ra', 'dec']
    gal_drop = ['ra', 'dec']

    # Combine input and output, galaxies and TDEs, make dictionary to
    # convert to dataframe later
    simulated_data = {}
    for i, j in tde_sim.iloc[::2].iterrows():

        # Galaxy input dictionary
        temp_gal_dic = tde_gal_data[0][i]['data']
        for key in gal_drop:
            temp_gal_dic.pop(key)
        temp_gal_dic_in = {'in %s' % key: temp_gal_dic[key] for key in temp_gal_dic.keys()}

        # TDE input dictionary
        temp_tde_dic = tde_gal_data[1][i]['data']
        temp_tde_dic_in = {'in %s' % key: temp_tde_dic[key] for key in temp_tde_dic.keys()}

        # Results dictionary
        temp_res_dic = j.to_dict()
        for key in res_drop:
            temp_res_dic.pop(key)
        temp_res_dic_out = {'out %s' % key: temp_res_dic[key] for key in temp_res_dic.keys()}

        # Concatenate dictionaries
        simulated_data[i] = {**temp_gal_dic_in, **temp_tde_dic_in, **temp_res_dic_out, 'tde source_g_mag': tde_sim.get_value(i + 1, 'source_g_mag')}

    # Convert dictionary to dataframe
    tdedf = pd.DataFrame.from_dict(simulated_data, orient='index')

    # Drop "out" for source mag as there is a value for all sources
    tdedf.rename(columns={'out source_g_mag': 'galaxy source_g_mag'}, inplace=True)
    tdedf['total source_g_mag'] = tdedf.apply(lambda row: add_mags(row['tde source_g_mag'], row['galaxy source_g_mag']), axis=1)

    return tdedf
