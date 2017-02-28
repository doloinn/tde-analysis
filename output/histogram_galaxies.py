# David Lynn
# Code to get galaxy data and histogram efficiencies

# Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Tweaked Pandas' scatter_matrix to use 2D histograms
import my_scatter_matrix

# Get galaxy dataframes
from getdfs import get_galaxy_dfs
eldf, spdf = get_galaxy_dfs()

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
