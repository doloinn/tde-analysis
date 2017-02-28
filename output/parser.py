# David Lynn
# Script to parse simulation outputs
# Calls IDL

# Won't be able to run this without simulation outputs and IDL

# Numpy for numbers
import numpy as np

# Glob for finding all files in directory
import glob

# Pandas for data analytics
import pandas as pd

# Pickle for loading in input pickle file
import pickle

# pidly for calling IDL
import pidly

# Initialise pidly
idl = pidly.IDL('/usr/local/bin/idl')

# Path to Diana Harrison's IDL GIBIS parsing library
path = r'/home/dlynn/sim17/GIBIS_IDL/IDLAstro/pro:/home/dlynn/sim17/GIBIS_IDL/FITS_IMAGE:/home/dlynn/sim17/GIBIS_IDL/PSF_UTILS:/home/dlynn/sim17/GIBIS_IDL/GIBIS_SIMU:/home/dlynn/sim17/GIBIS_IDL/READ_ALL:/home/dlynn/sim17/GIBIS_IDL/OTHER:/home/dlynn/sim17/GIBIS_IDL/QA:/home/dlynn/sim17/GIBIS_IDL/IMAGE:/home/dlynn/sim17/GIBIS_IDL/READ:'
idl("!path = '%s' + !path" % path)

# Loop over object type
for j in ['tde', 'galaxy']:
    files = glob.glob('/home/dlynn/sim17/results2/davidlynn_%s_results_2/*.pix' %j)

    # Making a list of dataframes
    df_list = []
    # These are the parameters which GIBIS outputs
    columns = ['theta', 'ra', 'dec', 'source_g_mag', 'offset', 'found_mag', 'multiple_matches', 'pix_sub-pix_pos', 'al_offset', 'ac_offset']
    # Loop over files
    for i, fn in enumerate(files):

        # I'll call the simulation the file name
        sim_name = fn.split('/')[-1]
        path_to_file = fn + '/'

        # Open observation file, get indices of sources
        with open('%stransit1/%s_tr1_6.obs' %(path_to_file, sim_name), 'r') as f:
            x = f.readlines()[1:]
            idx_array = [int(i.split()[0].split('-')[-1]) for i in x]

        # Setting these in IDL
        idl.path_to_file = path_to_file
        idl.sim_name = sim_name
        idl.idx_array = idx_array

        # Run the parser
        idl("gather_vpa_data_4_davy, path_to_file, sim_name, 1, idx_array, transits2process=transits2process, vpa_data=vpa_data, per_transit_data=per_transit_data")

        # Make dataframe of transposed data and add to list
        output = idl.per_transit_data[:, :, 0].T
        df_list.append(pd.DataFrame(output[:, :-1], index=np.int_(output[:, -1]), columns=columns))
        
    # Concatenate all the lists
    df = pd.concat(df_list)

    # Output to csv
    df.to_csv('results/%s_parsed.csv' %j)
