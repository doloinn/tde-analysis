# David Lynn
# Code to histogram TDE detection efficiencies

# Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Tweaked Pandas' scatter_matrix to use 2D histograms
import my_scatter_matrix

# Get TDE dataframe
from getdfs import get_tde_df
tdedf = get_tde_df()

# Drop NaNs for detected sources
tdedf_detected = tdedf.dropna()

# Prepare dataframes for histogram matrix
smdrop = ['in ra', 'in dec', 'galaxy source_g_mag', 'tde source_g_mag', 'out found_mag']
tdedf_sm = tdedf.drop(smdrop, 1)
tdedf_detected_sm = tdedf_detected.drop(smdrop, 1)

# Rename fields to true values so they're neater in plot
rename = {'in galaxy mag': '$V_G$', 'in galaxy radius': '$R_G$', 'in snapshot mag': '$G_T$', 'in point time': '$t$', 'in peak mag': '$G_P$', 'total source_g_mag': '$G_{G+T}$'}
tdedf_sm.rename(columns=rename, inplace=True)
tdedf_detected_sm.rename(columns=rename, inplace=True)

# Produce histogram matrices using modified scatter_matrix
# Takes input and detected dataframes and compares them
sm = my_scatter_matrix.scatter_matrix(tdedf_detected_sm, tdedf_sm)

plt.show()
